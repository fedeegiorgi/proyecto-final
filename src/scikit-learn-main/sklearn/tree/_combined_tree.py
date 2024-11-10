"""
Extensión del módulo de árboles de decisión de scikit-learn para la Alternativa C
de continuación de entrenamiento de árboles en base a otro inicial con menos datos.
------------------------------------------------------------------------------------
Extension of scikit-learn's decision trees module for Alternative C
of continuing tree training based on an initial one with less data.
"""

# original imports

import numpy as np

import numbers
from numbers import Integral, Real

from ..base import _fit_context
from ..utils import check_random_state
from . import _criterion, _splitter, _tree
from ._criterion import Criterion
from ._splitter import Splitter
from ._tree import Tree
from ._tree_comb import DepthFirstTreeCombinerBuilder
from ..utils.validation import check_array

# =============================================================================
# Types and constants
# =============================================================================

DTYPE = _tree.DTYPE
DOUBLE = _tree.DOUBLE

CRITERIA_CLF = {
    "gini": _criterion.Gini,
    "log_loss": _criterion.Entropy,
    "entropy": _criterion.Entropy,
}
CRITERIA_REG = {
    "squared_error": _criterion.MSE,
    "friedman_mse": _criterion.FriedmanMSE,
    "absolute_error": _criterion.MAE,
    "poisson": _criterion.Poisson,
}

# new imports

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
        initial_trees=None, # New parameter of type list of DecisionTreeRegressor
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

        self.initial_trees = initial_trees

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
    def fit(self, X, y, sample_weight=None, check_input=False): # We don't add check_input because from RandomForestRegressor it is always called in False
        """
        Continues the training of the DecisionTreeRegressor with the new provided data.
        """

        ########################## Copia del fit original ##########################

        random_state = check_random_state(self.random_state)

        # Determine output settings
        n_samples, self.n_features_in_ = X.shape

        y = np.atleast_1d(y)
        expanded_class_weight = None

        if y.ndim == 1:
            # reshape is necessary to preserve the data contiguity against vs
            # [:, np.newaxis] that does not.
            y = np.reshape(y, (-1, 1))

        self.n_outputs_ = y.shape[1]

        if getattr(y, "dtype", None) != DOUBLE or not y.flags.contiguous:
            y = np.ascontiguousarray(y, dtype=DOUBLE)

        max_depth = np.iinfo(np.int32).max if self.max_depth is None else self.max_depth

        if isinstance(self.min_samples_leaf, numbers.Integral):
            min_samples_leaf = self.min_samples_leaf
        else:  # float
            min_samples_leaf = int(ceil(self.min_samples_leaf * n_samples))

        if isinstance(self.min_samples_split, numbers.Integral):
            min_samples_split = self.min_samples_split
        else:  # float
            min_samples_split = int(ceil(self.min_samples_split * n_samples))
            min_samples_split = max(2, min_samples_split)

        min_samples_split = max(min_samples_split, 2 * min_samples_leaf)

        if isinstance(self.max_features, str):
            if self.max_features == "sqrt":
                max_features = max(1, int(np.sqrt(self.n_features_in_)))
            elif self.max_features == "log2":
                max_features = max(1, int(np.log2(self.n_features_in_)))
        elif self.max_features is None:
            max_features = self.n_features_in_
        elif isinstance(self.max_features, numbers.Integral):
            max_features = self.max_features
        else:  # float
            if self.max_features > 0.0:
                max_features = max(1, int(self.max_features * self.n_features_in_))
            else:
                max_features = 0

        self.max_features_ = max_features

        # Hardcodeado a -1 porque usamos el builder DepthFirst
        # max_leaf_nodes = -1 if self.max_leaf_nodes is None else self.max_leaf_nodes
        self.max_leaf_nodes = None

        if len(y) != n_samples:
            raise ValueError(
                "Number of labels=%d does not match number of samples=%d"
                % (len(y), n_samples)
            )

        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X, DOUBLE)

        if expanded_class_weight is not None:
            if sample_weight is not None:
                sample_weight = sample_weight * expanded_class_weight
            else:
                sample_weight = expanded_class_weight

        # Set min_weight_leaf from min_weight_fraction_leaf
        if sample_weight is None:
            min_weight_leaf = self.min_weight_fraction_leaf * n_samples
        else:
            min_weight_leaf = self.min_weight_fraction_leaf * np.sum(sample_weight)

        # Build tree
        criterion = self.criterion
        if not isinstance(criterion, Criterion):
            criterion = CRITERIA_REG[self.criterion](self.n_outputs_, n_samples)
        else:
            # Make a deepcopy in case the criterion has mutable attributes that
            # might be shared and modified concurrently during parallel fitting
            criterion = copy.deepcopy(criterion)

        if self.monotonic_cst is None:  # siempre va a ser None para nosotros
            monotonic_cst = None
        else:
            raise ValueError("This implementation does not support monotonic constraints")

        ##############################################################################
        
        # Acá arranca lo importante!!

        # Splitter instantiation
        splitter = Splitter(
            criterion,
            self.max_features_,
            min_samples_leaf,
            min_weight_leaf,
            self.random_state,
            monotonic_cst,
        )
        
        # Normal tree instantiation
        self.tree_ = Tree(
            self.n_features_in_,
            # TODO: tree shouldn't need this in this case
            np.array([1] * self.n_outputs_, dtype=np.intp),
            self.n_outputs_,
        )
        # Build Tree
        features, thresholds, _, _, _, _ = self._get_initial_trees_data()
        
        builder = DepthFirstTreeCombinerBuilder(
            splitter,
            min_samples_split,
            min_samples_leaf,
            min_weight_leaf,
            max_depth,
            self.min_impurity_decrease,
        )

        builder.combiner(self.tree_, features, thresholds)

        X = check_array(X, dtype=np.float32)

        out = self.tree_.apply(X) # Finds the terminal region (=leaf node) for each sample in X.
        
        builder.recompute_values(self.tree_, out, y) # Recompute the values of the leaf nodes