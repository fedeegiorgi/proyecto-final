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
from ._tree_comb import TreeCombiner
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
            raise ValueError("No valid split available.")

        # n_nodes = t.node_count
        feature = t.feature[0]
        threshold = t.threshold[0]
        impurity = t.impurity[0]
        # n_node_samples = np.zeros(n_nodes)
        # weighted_n_node_samples = np.ones(n_nodes)
        # missing_go_to_lefts = np.zeros(n_nodes)  # Create a numpy array of zeros

        return feature, threshold, impurity

    def _get_initial_trees_data(self):
        features, thresholds, impurities, n_node_samples, weighted_n_node_samples, missing_go_to_lefts = [], [], [], [], [], []
        
        for tree in self.initial_trees:
            tree_attributes = self._get_single_tree_data(tree)
            features.append(tree_attributes[0])
            thresholds.append(tree_attributes[1])
            impurities.append(tree_attributes[2])
            n_node_samples.append(0)
            weighted_n_node_samples.append(1)
            missing_go_to_lefts.append(0)

        return np.array(features), np.array(thresholds), np.array(impurities), np.array(n_node_samples), np.array(weighted_n_node_samples), np.array(missing_go_to_lefts)

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
        
        self.tree_ = TreeCombiner(
            self.n_features_in_,
            # TODO: tree shouldn't need this in this case
            np.array([1] * self.n_outputs_, dtype=np.intp),
            self.n_outputs_,
            features,
            thresholds,
            impurities,
            n_node_samples,
            weighted_n_node_samples,
            missing_go_to_lefts,
        )
        
        self.tree_.combiner()
        out = self.tree_.apply(X)
        self.tree_.recompute_values(out, y)