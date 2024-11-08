from sklearn.ensemble import RandomForestRegressor, RandomForestGroupDebate
import threading
from abc import ABCMeta, abstractmethod
from numbers import Integral, Real
from warnings import catch_warnings, simplefilter, warn

import numpy as np
import pandas as pd
from scipy.sparse import hstack as sparse_hstack
from scipy.sparse import issparse
from scipy.stats import mstats
from sklearn.model_selection import train_test_split

from ..base import (
    ClassifierMixin,
    MultiOutputMixin,
    RegressorMixin,
    TransformerMixin,
    _fit_context,
    is_classifier,
)
from ..exceptions import DataConversionWarning
from ..metrics import accuracy_score, r2_score
from ..preprocessing import OneHotEncoder
from ..tree import (
    DecisionTreeRegressor,
    ContinuedDecisionTreeRegressor
)
from ..tree._tree import DOUBLE, DTYPE
from ..utils import check_random_state, compute_sample_weight
from ..utils._param_validation import Interval, RealNotInt, StrOptions
from ..utils._tags import _safe_tags
from ..utils.multiclass import check_classification_targets, type_of_target
from ..utils.parallel import Parallel, delayed
from ..utils.validation import (
    _check_feature_names_in,
    _check_sample_weight,
    _num_samples,
    check_is_fitted,
    check_array
)
from ._base import BaseEnsemble, _partition_estimators

from ._forest import _get_n_samples_bootstrap
__all__ = [
    "SharedKnowledgeRandomForestRegressor",
]

def _store_prediction(predict, X, out, lock, tree_index):
    # AGREGAR DISCLAIMER MISMO DE LA DOC ORIGINAL
    """
    Store each tree's prediction in the 2D array `out`.
    Now we store the predictions in the tree's corresponding column.
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
        self.initial_estimators_ = []

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
            n_samples_bootstrap = _get_n_samples_bootstrap(  # TODO: hay que importarla desde donde esté
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
        
        # TODO: creo que tenemos que devolver random_state para que se entrenen los árboles con misma seed

    def fit(self, X, y, sample_weight=None):
        n_samples = X.shape[0]

        # Save the extended tree max_depth
        original_max_depth = self.max_depth

        # Ensure max_depth is set to initial_max_depth
        self.max_depth = self.initial_max_depth

        # Call to original fit method
        super().fit(X, y)

        print("Finished fitting original trees")
        
        # max_depth back to None as no limit for extended trees
        self.max_depth = original_max_depth

        # We don't need to bootstrap the data for the extended trees training
        # self.bootstrap = False

        # Divide the trees into groups
        initial_grouped_trees = self.group_split(self.estimators_)
        grouped_samples = self.group_split(self.estimators_samples_)

        grouped_new_X = []

        # For each group of trees and samples
        for i, trees_group in enumerate(initial_grouped_trees):

            samples_group = grouped_samples[i]
            extended_trees_group = []

            # For each tree in the group
            for j, tree in enumerate(trees_group):
                # Initialize a list to store predictions from all other trees
                other_tree_predictions = []

                # Samples used by the j-th tree
                j_tree_samples = samples_group[j]

                # Original X, y for the j-th tree
                j_tree_original_X = X[j_tree_samples]
                j_tree_y = y[j_tree_samples]
                
                # print("Samples usado por el árbol", j, ":", j_tree_samples)
                # print("Original X for the tree", j, ":", j_tree_original_X.shape)
                # print(j_tree_original_X)

                # For each tree in the group (except the j-th tree)
                for k, other_tree in enumerate(trees_group):
                    if k != j:
                        # Predict the samples used by tree j using the current tree k
                        predictions = other_tree.predict(j_tree_original_X)
                        other_tree_predictions.append(predictions)
                
                # Define new columns for the j-th tree
                new_columns = np.array(other_tree_predictions).T

                # Concatenate the predictions of the other trees with j-th tree's original features
                j_tree_pp_X = np.hstack((j_tree_original_X, new_columns))
                
                # print("New X for the tree", j, ":", j_tree_pp_X.shape)
                # print(j_tree_pp_X)

                # Validate training data
                j_tree_original_X, _, _, _, _ = self._original_fit_validations(j_tree_original_X, j_tree_y, sample_weight)
                j_tree_pp_X, j_tree_y, sample_weight, missing_values_in_feature_mask, random_state = self._original_fit_validations(j_tree_pp_X, j_tree_y, sample_weight)
                
                # Fit initial tree
                initial_tree = DecisionTreeRegressor(random_state=random_state, max_depth=self.initial_max_depth)
                initial_tree.fit(j_tree_original_X, j_tree_y, sample_weight=sample_weight, check_input=False)
                self.initial_estimators_.append(initial_tree)

                # Fit the extended tree with the new features based on the original tree
                extended_tree = ContinuedDecisionTreeRegressor(initial_tree=initial_tree, random_state=random_state, max_depth=self.max_depth)
                # extended_tree.fit(j_tree_pp_X, j_tree_y, sample_weight=sample_weight)
                extended_tree.fit_v2(j_tree_original_X, j_tree_pp_X, j_tree_y, sample_weight, missing_values_in_feature_mask, False)

                # Add fitted extended tree to the group
                extended_trees_group.append(extended_tree)
            
            # Add group of trees to the list of extended grouped estimators
            self.extended_grouped_estimators_.append(extended_trees_group)
        
        print("Finished fitting extended trees")

    def predict(self, X):
        
        check_is_fitted(self) #fijarse que no estamos ocultando las features?

        # X = self._validate_X_predict(X)
        X = check_array(X, dtype=np.float32)

        n_jobs, _, _ = _partition_estimators(self.n_estimators, self.n_jobs)
        lock = threading.Lock()

        if self.n_outputs_ > 1:
            initial_predictions = np.zeros((self.n_estimators, X.shape[0], self.n_outputs_), dtype=np.float64)
        else:
            initial_predictions = np.zeros((self.n_estimators, X.shape[0]), dtype=np.float64)

        # Compute predictions on original trees with original features in X
        Parallel(n_jobs=n_jobs, verbose=self.verbose, require="sharedmem")(
            delayed(_store_prediction)(e.predict, X, [initial_predictions], lock, i)
            for i, e in enumerate(self.estimators_) 
        )

        # Divide the predictions into groups
        grouped_predictions = self.random_group_split(initial_predictions)
        # TODO: a chequear pero por como están estos métodos de split se dividen las predicciones de 
        # igual forma que se dividieron en el fit los árboles

        # Initialize vector for the new predictions
        new_grouped_predictions = np.zeros_like(grouped_predictions)
        group_averages = np.empty((self._n_groups, X.shape[0]))

        for i, group_preds in enumerate(grouped_predictions):
            # print(f"Group {i}")
            for j, tree_preds in enumerate(group_preds):
                # print(f"Tree {j}")
                # Remove the j-th tree's predictions
                shared_predictions = []

                # For each tree in the group (except the j-th tree)
                for k, other_tree_preds in enumerate(group_preds):
                    if k != j:
                        # print("Getting shared predictions of tree", k)
                        shared_predictions.append(other_tree_preds)
                
                # Convert to np.array
                shared_predictions = np.array(shared_predictions)
                
                # Concatenate the shared predictions with the original features
                new_X = np.hstack((X, shared_predictions.T))

                # Validate the input data
                new_X = self._validate_X_predict(new_X)

                # print("Completed creating new X")
                # print(new_X.shape)
                # print(self.extended_grouped_estimators_[i][j].tree_.max_depth)
                # print(self.extended_grouped_estimators_[i][j].tree_.children_left)
                # Predict the samples for the current extended complete tree
                predictions = self.extended_grouped_estimators_[i][j].predict(new_X)

                # Store the predictions in corresponding group and tree
                new_grouped_predictions[i, j, :] = predictions

            # Calculate the group average
            group_averages[i] = np.mean(new_grouped_predictions[i], axis=0)

        y_hat = np.mean(group_averages, axis=0)

        return y_hat