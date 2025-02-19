from sklearn.ensemble import RandomForestGroupDebate
import threading
from warnings import warn

import numpy as np
from scipy.sparse import issparse

from ._base import _partition_estimators
from ._forest import _get_n_samples_bootstrap
from ..exceptions import DataConversionWarning
from ..tree import (
    DecisionTreeRegressor,
    ContinuedDecisionTreeRegressor
)
from ..tree._tree import DOUBLE, DTYPE
from ..utils import check_random_state

from ..utils.parallel import Parallel, delayed
from ..utils.validation import (
    _check_sample_weight,
    check_is_fitted,
    check_array
)

__all__ = [
    "SharedKnowledgeRandomForestRegressor",
]

def _store_prediction(predict, X, out, lock, tree_index):
    """
    --------------------------------------------------------------------------
    This is a utility function for joblib's Parallel.

    It can't go locally in ForestClassifier or ForestRegressor, because joblib
    complains that it cannot pickle it when placed there.
    --------------------------------------------------------------------------

    Store each tree's prediction in the 2D array `out`.
    """
    prediction = predict(X, check_input=False)
    with lock:
        out[0][tree_index] = prediction   # Store predictions in the column corresponding to the tree

# ----------------------------------- Alternativa D (idea Ramiro) -----------------------------------

class SharedKnowledgeRandomForestRegressor(RandomForestGroupDebate):
    def __init__(
        self,
        n_estimators=100,
        *,
        criterion="squared_error",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=1.0,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=True,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        ccp_alpha=0.0,
        max_samples=None,
        monotonic_cst=None,
        group_size=10,
        initial_max_depth=5, # New parameter
    ):
        super().__init__(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            ccp_alpha=ccp_alpha,
            max_samples=max_samples,
            monotonic_cst=monotonic_cst,
            group_size=group_size,
        )

        self.initial_max_depth = initial_max_depth
        self.extended_grouped_estimators_ = []

        if self.max_depth != None and self.initial_max_depth >= self.max_depth:
            raise ValueError(
                "The initial_max_depth must be strictly less than max_depth."
            )

    def _original_fit_validations(self, X, y, sample_weight):
        # Validate or convert input data
        if issparse(y):
            raise ValueError("sparse multilabel-indicator for y is not supported.")

        X, y = self._validate_data(
            X,
            y,
            multi_output=True,
            accept_sparse="csc",
            dtype=DTYPE,
            ensure_all_finite=False,
        )

        estimator = type(self.estimator)(criterion=self.criterion)
        missing_values_in_feature_mask = (
            estimator._compute_missing_values_in_feature_mask(
                X, estimator_name=self.__class__.__name__
            )
        )

        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X)

        if issparse(X):
            # Pre-sort indices to avoid that each individual tree of the
            # ensemble sorts the indices.
            X.sort_indices()

        y = np.atleast_1d(y)
        if y.ndim == 2 and y.shape[1] == 1:
            warn(
                (
                    "A column-vector y was passed when a 1d array was"
                    " expected. Please change the shape of y to "
                    "(n_samples,), for example using ravel()."
                ),
                DataConversionWarning,
                stacklevel=2,
            )

        if y.ndim == 1:
            # reshape is necessary to preserve the data contiguity against vs
            # [:, np.newaxis] that does not.
            y = np.reshape(y, (-1, 1))

        if self.criterion == "poisson":
            if np.any(y < 0):
                raise ValueError(
                    "Some value(s) of y are negative which is "
                    "not allowed for Poisson regression."
                )
            if np.sum(y) <= 0:
                raise ValueError(
                    "Sum of y is not strictly positive which "
                    "is necessary for Poisson regression."
                )

        self._n_samples, self.n_outputs_ = y.shape

        y, expanded_class_weight = self._validate_y_class_weight(y)

        if getattr(y, "dtype", None) != DOUBLE or not y.flags.contiguous:
            y = np.ascontiguousarray(y, dtype=DOUBLE)

        if expanded_class_weight is not None:
            if sample_weight is not None:
                sample_weight = sample_weight * expanded_class_weight
            else:
                sample_weight = expanded_class_weight

        if not self.bootstrap and self.max_samples is not None:
            raise ValueError(
                "`max_sample` cannot be set if `bootstrap=False`. "
                "Either switch to `bootstrap=True` or set "
                "`max_sample=None`."
            )
        elif self.bootstrap:
            n_samples_bootstrap = _get_n_samples_bootstrap(
                n_samples=X.shape[0], max_samples=self.max_samples
            )
        else:
            n_samples_bootstrap = None

        random_state = check_random_state(self.random_state)

        # Decapsulate classes_ attributes
        if hasattr(self, "classes_") and self.n_outputs_ == 1:
            self.n_classes_ = self.n_classes_[0]
            self.classes_ = self.classes_[0]

        return X, y, sample_weight, missing_values_in_feature_mask, random_state

    def fit(self, X, y, sample_weight=None):
        n_samples = X.shape[0]

        # Save the extended tree max_depth
        original_max_depth = self.max_depth

        # Ensure max_depth is set to initial_max_depth
        self.max_depth = self.initial_max_depth

        # Call to original fit method
        super().fit(X, y)

        # max_depth back to None as no limit for extended trees
        self.max_depth = original_max_depth

        # Divide the trees and their samples into groups
        initial_grouped_trees = self.group_split(self.estimators_)
        grouped_samples = self.group_split(self.estimators_samples_)

        grouped_new_X = []

        # For each group of trees and samples
        for i, trees_group in enumerate(initial_grouped_trees):

            samples_group = grouped_samples[i]
            extended_trees_group = []

            # For each tree in the group
            for j, tree in enumerate(trees_group):
                # Samples used by the j-th tree
                j_tree_samples = samples_group[j]

                # Original X, y for the j-th tree
                j_tree_original_X = X[j_tree_samples]
                j_tree_y = y[j_tree_samples]

                # Initialize a list to store predictions from all other trees in the group
                other_tree_predictions = np.zeros_like(j_tree_y, dtype=float)
                num_other_trees_in_group = 0

                # For each tree in the group (except the j-th tree)
                for k, other_tree in enumerate(trees_group):
                    if k != j:
                        # Predict the samples used by tree j using the current tree k
                        predictions = other_tree.predict(j_tree_original_X)
                        other_tree_predictions += predictions
                        num_other_trees_in_group += 1

                # Compute the average prediction of the other members in the group
                avg_other_tree_predictions = other_tree_predictions / num_other_trees_in_group if num_other_trees_in_group > 0 else np.zeros_like(j_tree_y)

                # Define a new column with the averaged predictions
                avg_predictions_column = avg_other_tree_predictions.reshape(-1, 1)

                # Concatenate the predictions of the other trees averaged, with j-th tree's original features
                j_tree_pp_X = np.hstack((j_tree_original_X, avg_predictions_column))

                # Validate training data
                j_tree_original_X, _, _, _, _ = self._original_fit_validations(j_tree_original_X, j_tree_y, sample_weight)
                j_tree_pp_X, j_tree_y, sample_weight, missing_values_in_feature_mask, random_state = self._original_fit_validations(j_tree_pp_X, j_tree_y, sample_weight)

                # Fit the extended tree with the new features based on the original tree
                extended_tree = ContinuedDecisionTreeRegressor(initial_tree=tree, random_state=random_state, max_depth=self.max_depth)
                extended_tree.fit(j_tree_original_X, j_tree_pp_X, j_tree_y, sample_weight, missing_values_in_feature_mask, False)

                # Add fitted extended tree to the group
                extended_trees_group.append(extended_tree)

            # Add group of trees to the list of extended grouped estimators
            self.extended_grouped_estimators_.append(extended_trees_group)

    def predict(self, X):

        check_is_fitted(self)

        # X = self._validate_X_predict(X)
        X = check_array(X, dtype=np.float32)

        n_jobs, _, _ = _partition_estimators(self.n_estimators, self.n_jobs)
        lock = threading.Lock()

        if self.n_outputs_ > 1:
            raise ValueError("Multiprediction not available in this implementation of Random Forest.")
        else:
            initial_predictions = np.zeros((self.n_estimators, X.shape[0]), dtype=np.float64)

        # Compute predictions on initial trees with original features in X
        Parallel(n_jobs=n_jobs, verbose=self.verbose, require="sharedmem")(
            delayed(_store_prediction)(e.predict, X, [initial_predictions], lock, i)
            for i, e in enumerate(self.estimators_)
        )

        # Divide the predictions into groups
        grouped_predictions = self.group_split_predictions(initial_predictions)

        # Initialize vector for the new predictions
        new_grouped_predictions = np.zeros_like(grouped_predictions)
        group_averages = np.empty((self._n_groups, X.shape[0]))

        for i, group_preds in enumerate(grouped_predictions):
            for j, tree_preds in enumerate(group_preds):
                # Initialize a list to store predictions from all other trees in the group
                other_tree_predictions = np.zeros_like(tree_preds, dtype=float)
                num_other_trees = 0

                # For each tree in the group (except the j-th tree)
                for k, other_tree_preds in enumerate(group_preds):
                    if k != j:
                        other_tree_predictions += other_tree_preds
                        num_other_trees += 1

                # Convert to np.array
                avg_other_tree_predictions = (
                    other_tree_predictions / num_other_trees if num_other_trees > 0 else np.zeros_like(tree_preds)
                )

                # Define the new column with the averaged predictions
                avg_predictions_column = avg_other_tree_predictions.reshape(-1, 1)

                # Concatenate the shared predictions with the original features
                new_X = np.hstack((X, avg_predictions_column))

                # Validate the input data
                new_X = self._validate_X_predict(new_X)

                # Predict the samples for the current extended complete tree
                predictions = self.extended_grouped_estimators_[i][j].predict(new_X)

                # Store the predictions in corresponding group and tree
                new_grouped_predictions[i, j, :] = predictions

            # Calculate the group average
            group_averages[i] = np.mean(new_grouped_predictions[i], axis=0)

        # Compute the final prediction as the average of the group averages
        y_hat = np.mean(group_averages, axis=0)

        return y_hat