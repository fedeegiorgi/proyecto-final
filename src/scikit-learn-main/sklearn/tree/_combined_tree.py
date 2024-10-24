"""
Extensión del módulo de árboles de decisión de scikit-learn para la Alternativa C
de continuación de entrenamiento de árboles en base a otro inicial con menos datos.
"""

# import originales (chequear cuáles no se usan)

import copy
import numbers
from abc import ABCMeta, abstractmethod
from math import ceil
from numbers import Integral, Real

import numpy as np
from scipy.sparse import issparse

from ..base import (
    BaseEstimator,
    ClassifierMixin,
    MultiOutputMixin,
    RegressorMixin,
    _fit_context,
    clone,
    is_classifier,
)
from ..utils import Bunch, check_random_state, compute_sample_weight
from ..utils._param_validation import Hidden, Interval, RealNotInt, StrOptions
from ..utils.multiclass import check_classification_targets
from ..utils.validation import (
    _assert_all_finite_element_wise,
    _check_sample_weight,
    assert_all_finite,
    check_is_fitted,
)
from . import _criterion, _splitter, _tree
from ._criterion import Criterion
from ._splitter import Splitter
from ._tree import (
    BestFirstTreeBuilder,
    DepthFirstTreeBuilder,
    Tree,
    _build_pruned_tree_ccp,
    ccp_pruning_path,
)
from ._utils import _any_isnan_axis0

# import nuevos

from ._classes import DecisionTreeRegressor

class DecisionTreeRegressorCombiner(DecisionTreeRegressor):

    def __init__(
        self,
        *,
        criterion="squared_error",
        splitter="best",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=None,
        random_state=None,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        ccp_alpha=0.0,
        monotonic_cst=None,
        intial_trees=None, # New parameter of type list of DecisionTreeRegressor
    ):
        super().__init__(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            random_state=random_state,
            min_impurity_decrease=min_impurity_decrease,
            ccp_alpha=ccp_alpha,
            monotonic_cst=monotonic_cst,
        )

        self.initial_trees = intial_trees

    def _get_single_tree_data(self, tree):
        """
        Extract the necessary data from an initial trained DecisionTreeRegressor
        """

        t = tree.tree_

        if t.node_count != 3:
            raise ValueError("No valid split available")

        n_nodes = t.node_count
        features = t.feature
        thresholds = t.threshold
        impurities = t.impurity
        n_node_samples = np.zeros(n_nodes)
        weighted_n_node_samples = np.ones(n_nodes)
        missing_go_to_lefts = np.zeros(n_nodes)  # Create a numpy array of zeros

        return {
            "features": features,
            "thresholds": thresholds,
            "impurities": impurities,
            "n_node_samples": n_node_samples,
            "weighted_n_node_samples": weighted_n_node_samples,
            "missing_go_to_lefts": missing_go_to_lefts,
        }

    def _get_initial_trees_data(self):
        features = np.array((3, len(self.initial_trees)))
        thresholds, impurities, n_node_samples, weighted_n_node_samples, missing_go_to_lefts = np.empty_like(features), np.empty_like(features), np.empty_like(features), np.empty_like(features), np.empty_like(features)

        for i, tree in enumerate(self.initial_trees):
            tree_attributes = self._get_single_tree_data(tree)
            features[i] = (tree_attributes['features'])
            thresholds[i] = (tree_attributes['thresholds'])
            impurities[i] = (tree_attributes['impurities'])
            n_node_samples[i] = (tree_attributes['n_node_samples'])
            weighted_n_node_samples[i] = (tree_attributes['weighted_n_node_samples'])
            missing_go_to_lefts[i] = (tree_attributes['missing_go_to_lefts'])

        return features, thresholds, impurities, n_node_samples, weighted_n_node_samples, missing_go_to_lefts

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y, sample_weight=None, check_input=False): # No agregamos check_input porque desde RandomForestRegressor siempre se llama en False
        """
        Continues the training of the DecisionTreeRegressor with the new provided data.
        """

        ########################## Copia del fit original ##########################

        # Determine output settings
        n_samples, self.n_features_in_ = X.shape

        y = np.atleast_1d(y)

        if y.ndim == 1:
            # reshape is necessary to preserve the data contiguity against vs
            # [:, np.newaxis] that does not.
            y = np.reshape(y, (-1, 1))

        self.n_outputs_ = y.shape[1]

        if getattr(y, "dtype", None) != DOUBLE or not y.flags.contiguous:
            y = np.ascontiguousarray(y, dtype=DOUBLE)

        if len(y) != n_samples:
            raise ValueError(
                "Number of labels=%d does not match number of samples=%d"
                % (len(y), n_samples)
            )

        if sample_weight is not None:
            raise ValueError("sample_weight not available in DecisionTreeRegressorCombiner.")

        # Build Tree
        features, thresholds, impurities, n_node_samples, weighted_n_node_samples, missing_go_to_lefts = self._get_initial_trees_data()
        # builder = ...
        # builder.build(_tree, features, thresholds, n_node_samples, weighted_n_node_samples, missing_go_to_lefts)