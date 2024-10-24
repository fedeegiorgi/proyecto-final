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

from ..utils.validation import _check_sample_weight
from ._criterion import Criterion
from ._splitter import Splitter
from ._tree import Tree

# import nuevos

from ._classes import DecisionTreeRegressor

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
        intial_tree=None, # new parameter of type DecisionTreeRegressor
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

        self.initial_tree = intial_tree
    
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
    def fit(self, X, y, sample_weight=None, check_input=False): # No agregamos check_input porque desde RandomForestRegressor siempre se llama en False
        """
        Continues the training of the DecisionTreeRegressor with the new provided data.
        """

        ########################## Copia del fit original ##########################

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

        SPLITTERS = SPARSE_SPLITTERS if issparse(X) else DENSE_SPLITTERS # chusmear esto

        splitter = self.splitter
        if self.monotonic_cst is None:  # siempre va a ser None para nosotros
            monotonic_cst = None
        else:
            if self.n_outputs_ > 1:
                raise ValueError(
                    "Monotonicity constraints are not supported with multiple outputs."
                )
            # Check to correct monotonicity constraint' specification,
            # by applying element-wise logical conjunction
            # Note: we do not cast `np.asarray(self.monotonic_cst, dtype=np.int8)`
            # straight away here so as to generate error messages for invalid
            # values using the original values prior to any dtype related conversion.
            monotonic_cst = np.asarray(self.monotonic_cst)
            if monotonic_cst.shape[0] != X.shape[1]:
                raise ValueError(
                    "monotonic_cst has shape {} but the input data "
                    "X has {} features.".format(monotonic_cst.shape[0], X.shape[1])
                )
            valid_constraints = np.isin(monotonic_cst, (-1, 0, 1))
            if not np.all(valid_constraints):
                unique_constaints_value = np.unique(monotonic_cst)
                raise ValueError(
                    "monotonic_cst must be None or an array-like of -1, 0 or 1, but"
                    f" got {unique_constaints_value}"
                )
            monotonic_cst = np.asarray(monotonic_cst, dtype=np.int8)
            if is_classifier(self):
                if self.n_classes_[0] > 2:
                    raise ValueError(
                        "Monotonicity constraints are not supported with multiclass "
                        "classification"
                    )
                # Binary classification trees are built by constraining probabilities
                # of the *negative class* in order to make the implementation similar
                # to regression trees.
                # Since self.monotonic_cst encodes constraints on probabilities of the
                # *positive class*, all signs must be flipped.
                monotonic_cst *= -1

        if not isinstance(self.splitter, Splitter): # vamos a usar el único splitter que modificamos
            splitter = SPLITTERS[self.splitter](
                criterion,
                self.max_features_,
                min_samples_leaf,
                min_weight_leaf,
                random_state,
                monotonic_cst,
            )

        ##############################################################################
        
        self.tree_ = Tree(
            self.n_features_in_,
            # TODO: tree shouldn't need this in this case
            np.array([1] * self.n_outputs_, dtype=np.intp),
            self.n_outputs_,
        )

        # Acá llamar al nuevo builder