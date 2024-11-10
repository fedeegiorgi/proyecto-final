"""
Extensión del módulo de árboles de decisión de scikit-learn para la Alternativa D
de continuación de entrenamiento de árboles en base a otro inicial con menos datos.
"""

# import originales (chequear cuáles no se usan)

import copy
import numbers
from math import ceil

import numpy as np
from scipy.sparse import issparse

from ..base import (
    _fit_context,
    is_classifier,
)

from ..utils import check_random_state
from ..utils.validation import _check_sample_weight
from . import _criterion, _splitter, _tree
from ._criterion import Criterion
from ._splitter import Splitter
from ._tree import Tree

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

# import nuevos

from ._classes import DecisionTreeRegressor
from ._splitter import AddOnBestSplitter
from ._extended_tree import DepthFirstTreeExtensionBuilder

class ContinuedDecisionTreeRegressor(DecisionTreeRegressor):

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
        initial_tree=None, # new parameter of type DecisionTreeRegressor
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

        self.initial_tree = initial_tree
    
    def _get_initial_tree_data(self, tree):
        """
        Extract the necessary data from an initial trained DecisionTreeRegressor
        """

        t = tree.tree_
        n_nodes = t.node_count

        is_lefts = np.array([1 if i in t.children_left else 0 for i in range(n_nodes)])
        is_leafs = np.array([1 if t.children_left[i] == -1 else 0 for i in range(n_nodes)])
        features = t.feature
        thresholds = t.threshold
        impurities = t.impurity
        impurities = np.where(impurities < 1e-10, 0, impurities)  # Replace very small impurities with 0
        n_node_samples = t.n_node_samples
        weighted_n_node_samples = t.weighted_n_node_samples 
        missing_go_to_lefts = np.zeros(n_nodes)  # Create a numpy array of zeros

        parents = np.full(n_nodes, -1)  # Initialize parents array with -1
        depths = np.full(n_nodes, 0) # Initialize depths in 0

        for parent, (left, right) in enumerate(zip(t.children_left, t.children_right)):
            if left != -1:  # If there is a left child
                parents[left] = parent
                depths[left] = depths[parent] + 1 # Increment depth for left child
            if right != -1:  # If there is a right child
                parents[right] = parent
                depths[right] = depths[parent] + 1 # Increment depth for right child

        return {
            "parents": parents,
            "depths": depths,
            "is_lefts": is_lefts,
            "is_leafs": is_leafs,
            "features": features,
            "thresholds": thresholds,
            "impurities": impurities,
            "n_node_samples": n_node_samples,
            "weighted_n_node_samples": weighted_n_node_samples,
            "missing_go_to_lefts": missing_go_to_lefts,
        }

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y, sample_weight=None, missing_values_in_feature_mask=None, check_input=False): # No agregamos check_input porque desde RandomForestRegressor siempre se llama en False
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

        # Our splitter instantiation
        splitter = AddOnBestSplitter(
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

        # Extended builder
        builder = DepthFirstTreeExtensionBuilder(
            splitter,
            min_samples_split,
            min_samples_leaf,
            min_weight_leaf,
            max_depth,
            self.min_impurity_decrease,
        )

        # Get initial tree data
        initial_tree_data = self._get_initial_tree_data(self.initial_tree)

        print("Instanciated the splitter, tree and builder inside the fit method of ContinuedDecisionTreeRegressor")
        
        # Continue training
        builder.build_extended(
            self.tree_, 
            X, 
            y, 
            initial_tree_data["parents"],
            initial_tree_data["depths"],
            initial_tree_data["is_lefts"],
            initial_tree_data["is_leafs"],
            initial_tree_data["features"],
            initial_tree_data["thresholds"],
            initial_tree_data["impurities"],
            initial_tree_data["n_node_samples"],
            initial_tree_data["weighted_n_node_samples"],
            initial_tree_data["missing_go_to_lefts"],
            sample_weight, 
            missing_values_in_feature_mask,
        )
    
    @_fit_context(prefer_skip_nested_validation=True)
    def fit_v2(
        self,
        X_original,
        X_peer_prediction,
        y,
        sample_weight=None, 
        missing_values_in_feature_mask=None,
        check_input=False
    ): # No agregamos check_input porque desde RandomForestRegressor siempre se llama en False
        """
        Trains a new tree using X_original until initial_tree depth and then continues training with X_peer_prediction.
        """
        ########################## Copia del fit original ##########################

        random_state = check_random_state(self.random_state)

        # Determine output settings
        n_samples, self.n_features_in_ = X_peer_prediction.shape # X_peer_prediction porque X_peer_prediction[1] es group_size - 1 mas grande que X_original[1] y quedan valores empty pero no faltan index
        is_classification = is_classifier(self)

        y = np.atleast_1d(y)
        expanded_class_weight = None

        if y.ndim == 1:
            # reshape is necessary to preserve the data contiguity against vs
            # [:, np.newaxis] that does not.
            y = np.reshape(y, (-1, 1))

        self.n_outputs_ = y.shape[1]

        if is_classification:
            check_classification_targets(y)
            y = np.copy(y)

            self.classes_ = []
            self.n_classes_ = []

            if self.class_weight is not None:
                y_original = np.copy(y)

            y_encoded = np.zeros(y.shape, dtype=int)
            for k in range(self.n_outputs_):
                classes_k, y_encoded[:, k] = np.unique(y[:, k], return_inverse=True)
                self.classes_.append(classes_k)
                self.n_classes_.append(classes_k.shape[0])
            y = y_encoded

            if self.class_weight is not None:
                expanded_class_weight = compute_sample_weight(
                    self.class_weight, y_original
                )

            self.n_classes_ = np.array(self.n_classes_, dtype=np.intp)

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

        max_leaf_nodes = -1 if self.max_leaf_nodes is None else self.max_leaf_nodes

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
            if is_classification:
                criterion = CRITERIA_CLF[self.criterion](
                    self.n_outputs_, self.n_classes_
                )
            else:
                criterion = CRITERIA_REG[self.criterion](self.n_outputs_, n_samples)
        else:
            # Make a deepcopy in case the criterion has mutable attributes that
            # might be shared and modified concurrently during parallel fitting
            criterion = copy.deepcopy(criterion)
        
        if self.monotonic_cst is None:  # siempre va a ser None para nosotros
            monotonic_cst = None
        else:
            raise ValueError("This implementation does not support monotonic constraints")

        ######################################################################

        # Our splitter instantiation
        splitter = AddOnBestSplitter(
            criterion,
            self.max_features_,
            min_samples_leaf,
            min_weight_leaf,
            random_state,
            monotonic_cst,
        )
        
        # Normal tree instantiation
        self.tree_ = Tree(
            self.n_features_in_,
            # TODO: tree shouldn't need this in this case
            np.array([1] * self.n_outputs_, dtype=np.intp),
            self.n_outputs_,
        )

        # Extended builder
        builder = DepthFirstTreeExtensionBuilder(
            splitter,
            min_samples_split,
            min_samples_leaf,
            min_weight_leaf,
            max_depth,
            self.min_impurity_decrease,
        )

        # Get initial tree depth (initial_max_depth)
        initial_max_depth = self.initial_tree.get_depth()

        builder.build_extended_2(
            self.tree_, 
            X_original, 
            X_peer_prediction,
            initial_max_depth,
            y,
            None, # sample_weight
            missing_values_in_feature_mask,
        )