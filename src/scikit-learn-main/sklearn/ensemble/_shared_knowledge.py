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
    BaseDecisionTree,
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    ExtraTreeClassifier,
    ExtraTreeRegressor,
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
)
from ._base import BaseEnsemble, _partition_estimators

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

def get_tree_data(tree):
    """
    Extract the necessary data from a trained DecisionTreeRegressor
    """

    t = tree.tree_
    n_nodes = t.node_count

    is_lefts = [1 if i in t.children_left else 0 for i in range(n_nodes)]
    is_leafs = [1 if t.children_left[i] == -1 else 0 for i in range(n_nodes)]
    features = list(t.feature)
    thresholds = list(t.threshold)
    impurities = list(t.impurity)
    n_node_samples = list(t.n_node_samples)
    weighted_n_node_samples = list(t.weighted_n_node_samples)
    missing_go_to_lefts = [0] * n_nodes 

    parents = [-1] * n_nodes

    for parent, (left, right) in enumerate(zip(t.children_left, t.children_right)):
        if left != -1:  # If there is a left child
            parents[left] = parent
        if right != -1:  # If there is a right child
            parents[right] = parent

    return {
        "parents": parents,
        "is_lefts": is_lefts,
        "is_leafs": is_leafs,
        "features": features,
        "thresholds": thresholds,
        "impurities": impurities,
        "n_node_samples": n_node_samples,
        "weighted_n_node_samples": weighted_n_node_samples,
        "missing_go_to_lefts": missing_go_to_lefts,
    }


class SharedKnowledgeeRandomForestRegressor(RandomForestGroupDebate):
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
        self.initial_grouped_trees = []

    def fit(self, X, y):

        # original RandomForestsRegressor with max_depth=self.initial_max_depth
        rf = RandomForestRegressor(random_state=self.random_state, max_depth=self.initial_max_depth)
        
        # Fit the model with original algorithm
        rf.fit(X, y)

        # Get the trees and samples
        initial_trees = rf.estimators_
        trees_samples = rf.estimators_samples_

        # Divide the trees into groups
        self.initial_grouped_trees = self.group_split(initial_trees)
        grouped_samples = self.group_split(trees_samples)
        
        grouped_new_columns = []

        # For each group of trees and samples
        for i, trees_group in enumerate(self.initial_grouped_trees):

            samples_group = grouped_samples[i]
            group_new_columns = []

            # For each tree in the group
            for j, tree in enumerate(trees_group):
                # Initialize a list to store predictions from all other trees
                other_tree_predictions = []

                # For each tree in the group (except the j-th tree)
                for k, other_tree in enumerate(trees_group):
                    if k != j:
                        # Predict the samples for the current tree
                        predictions = other_tree.predict(X[samples_group[j]])
                        other_tree_predictions.append(predictions)
            
                # Append the predictions for this tree
                group_new_columns.append(np.array(other_tree_predictions).T)
        
            grouped_new_columns.append(group_new_columns)
        
        # Print test
        print(f"new columns for tree 0 in group 0: {grouped_new_columns[0][0]}")
        print(f"Shape new columns for tree 0 in group 0: {grouped_new_columns[0][0].shape}")
        print(f"Count of samples for tree 0 in group 0: {len(grouped_samples[0][0])}")
        print(f"Shape of X for tree 0 in group 0: {X[grouped_samples[0][0]].shape}")
        print(f"Ratio Silvio: {(self.group_size-1)/X[grouped_samples[0][0]].shape[1]}")
        # Concatenate the new columns with the original features
        new_X = np.hstack((X[grouped_samples[0][0]], grouped_new_columns[0][0]))
        print(f"Shape of X after hstack for tree 0 in group 0: {new_X.shape}")

        for i, trees_group in enumerate(self.initial_grouped_trees):
            for j, tree in enumerate(trees_group):
                
                # Extract the necessary data from the tree
                tree_data = get_tree_data(tree)
                
                # # Concatenate the other tree's predictions with the original features
                new_X = np.hstack(X[samples_group[i][j]], grouped_new_columns[i][j])

                # # Fit the extended tree with the new features based on the original tree
                # new_tree = ContinuedDecisionTreeRegressor(initial_tree_data=tree_data)
                # new_tree.fit(new_X, y[samples_group[i][j]])

                # # Add fitted tree to the estimators_ list
                # self.estimators_.append(new_tree)
        
        # # Divide the trees into groups
        # self.estimators_ = self.group_split(self.estimators_)

    def predict(self, X):
        
        check_is_fitted(self) #fijarse que no estamos ocultando las features?

        X = self._validate_X_predict(X)

        n_jobs, _, _ = _partition_estimators(self.n_estimators, self.n_jobs)
        lock = threading.Lock()

        if self.n_outputs_ > 1:
            initial_predictions = np.zeros((self.n_estimators, X.shape[0], self.n_outputs_), dtype=np.float64)
        else:
            initial_predictions = np.zeros((self.n_estimators, X.shape[0]), dtype=np.float64)

        Parallel(n_jobs=n_jobs, verbose=self.verbose, require="sharedmem")(
            delayed(_store_prediction)(e.predict, X, [initial_predictions], lock, i)
            for i, e in enumerate(self.initial_grouped_trees) # En vez de self.estimators_ uso los initial_grouped_trees (aun no están las predicciones de los otros árboles)
        ) # initial_grouped_trees es lista de lista asi que hay que hacer un for anidado/ aplanar

        grouped_predictions = self.random_group_split(initial_predictions)

        # Initialize the new grouped predictions
        new_grouped_predictions = np.zeros_like(grouped_predictions)
        group_averages = np.empty((self._n_groups, X.shape[0]))

        for i, group in enumerate(grouped_predictions):
            for j, tree in enumerate(group):
                # Remove the j-th tree's predictions
                shared_predictions = np.delete(grouped_predictions[i], j, axis=0)

                # Concatenate the shared predictions with the original features
                new_X = np.hstack((X, shared_predictions.T))

                # Predict the samples for the current complete tree
                predictions = self.estimators_[i][j].predict(new_X)

                # Store the predictions in corresponding group and tree
                new_grouped_predictions[i, j, :] = predictions

            # Calculate the group average
            group_averages[i] = np.mean(new_grouped_predictions[i], axis=0)

        y_hat = np.mean(group_averages, axis=0)

        return y_hat