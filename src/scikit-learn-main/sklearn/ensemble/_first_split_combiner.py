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
    DecisionTreeRegressorCombiner,
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

def _accumulate_prediction(predict, X, out, lock):
    """
    This is a utility function for joblib's Parallel.

    It can't go locally in ForestClassifier or ForestRegressor, because joblib
    complains that it cannot pickle it when placed there.
    """
    prediction = predict(X, check_input=False)
    with lock:
        if len(out) == 1:
            out[0] += prediction
        else:
            for i in range(len(out)):
                out[i] += prediction[i]

class RFRegressorFirstSplitCombiner(RandomForestGroupDebate):
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
        # Set max_depth = 1. We only care about the first split.
        self.max_depth = 1

        # Call to original fit method
        super().fit(X, y)

        # Divide the trees into groups
        self.initial_grouped_trees = self.group_split(self.estimators_)
        grouped_samples = self.group_split(self.estimators_samples_)

        for i, tree_group in enumerate(self.initial_grouped_trees):
            group_samples_used = grouped_samples[i]

            samplesidx = set()
            for samples in group_samples_used:
                for sampleidx in samples:
                    samplesidx.add(sampleidx)
            samplesidx = list(samplesidx)

            # TODO: @fede chquear como hacer esto asegurando que funcionen los indices 
            # creo que en algun momento X siempre se convierte a numpy por lo que el iloc no tiene sentido
            # se lo saqué (el iloc) y me tiraba error también
            X_union = X.iloc[samplesidx].to_numpy() 
            y_union = y.iloc[samplesidx].to_numpy()

            group_combined_tree = DecisionTreeRegressorCombiner(initial_trees=tree_group)
            group_combined_tree.fit(X_union, y_union)
            self.combined_trees.append(group_combined_tree)
        
    def predict(self, X):
        check_is_fitted(self)

        X = self._validate_X_predict(X)

        n_jobs, _, _ = _partition_estimators(self._n_groups, self.n_jobs)
        lock = threading.Lock()

        # avoid storing the output of every estimator by summing them here
        if self.n_outputs_ > 1:
            y_hat = np.zeros((X.shape[0], self.n_outputs_), dtype=np.float64)
        else:
            y_hat = np.zeros((X.shape[0]), dtype=np.float64)

        # Parallel loop
        lock = threading.Lock()
        Parallel(n_jobs=n_jobs, verbose=self.verbose, require="sharedmem")(
            delayed(_accumulate_prediction)(e.predict, X, [y_hat], lock)
            for e in self.combined_trees
        )

        y_hat /= len(self.combined_trees) # promedia las estimaciones de los arboles combinados

        return y_hat