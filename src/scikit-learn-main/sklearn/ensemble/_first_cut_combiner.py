from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestGroupDebate
import threading
from abc import ABCMeta, abstractmethod
from numbers import Integral, Real
from warnings import catch_warnings, simplefilter, warn

import numpy as np
import pandas as pd
from scipy.sparse import hstack as sparse_hstack
from scipy.sparse import issparse
from scipy.stats import mstats
from sklearn.metrics import mean_squared_error #agregado para calcular el mse de cada arbol en sus oob y sacar su peso en la prediccion 
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

class RFRegressorFirstCutCombiner(RandomForestGroupDebate):
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

        self.combined_trees = []
    
    def fit(self, X, y, sample_weights=None):
        # Fit a random forest regressor with max_depth = 1 for each tree.
        rf = RandomForestRegressor(random_state=self.random_state, max_depth=1)
        rf.fit(X, y)

        # Get the trees and samples used
        initial_trees, trees_samples = rf.estimators_, rf.estimators_samples_

        # Divide the trees into groups
        self.initial_grouped_trees = self.group_split(initial_trees)
        grouped_samples = self.group_split(trees_samples)

        for i, tree_group in enumerate(self.initial_grouped_trees):
            group_samples_used = grouped_samples[i]

            samplesidx = set()
            for samples in group_samples_used:
                for sampleidx in samples:
                    samplesidx.add(sampleidx)
            samplesidx = list(samplesidx)

            X_union = X.iloc[samplesidx].to_numpy()
            y_union = y.iloc[samplesidx].to_numpy()

            group_combined_tree = DecisionTreeRegressorCombiner(X_union, y_union, tree_group)
            self.combined_trees.append(group_combined_tree)
        
    def predict(self, X):
        check_is_fitted(self)

        X = self._validate_X_predict(X)

        n_jobs, _, _ = _partition_estimators(self._n_groups, self.n_jobs)
        lock = threading.Lock()

        if self.n_outputs_ > 1:
            group_predictions = np.zeros((self._n_groups, X.shape[0], self.n_outputs_), dtype=np.float64)
        else:
            group_predictions = np.zeros((self._n_groups, X.shape[0]), dtype=np.float64)

        Parallel(n_jobs=n_jobs, verbose=self.verbose, require="sharedmem")(
            delayed(_store_prediction)(e.predict, X, [group_predictions], lock, i)
            for i, e in enumerate(self.combined_trees) # Uso las predicciones de los arboles agrupados
        )

        y_hat = np.mean(group_predictions, axis=0)

        return y_hat