"""
Extensión del módulo de árboles de decisión de scikit-learn para la Alternativa C
de continuación de entrenamiento de árboles en base a otro inicial con menos datos.
------------------------------------------------------------------------------------
Extension of scikit-learn's decision trees module for Alternative C
of continuing tree training based on an initial one with less data.
"""

# original imports

import numpy as np

from ..base import _fit_context
from . import _tree
from ._tree_comb import TreeCombiner

DTYPE = _tree.DTYPE
DOUBLE = _tree.DOUBLE

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

        ########################## Copy of the original fit ##########################

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
            self.n_outputs_
        )
        
        self.tree_.combiner(features,
            thresholds,
            impurities,
            n_node_samples,
            weighted_n_node_samples,
            missing_go_to_lefts)

        # print(f"tree count: {self.tree_.node_count}")
        # print(f"capacity: {self.tree_.capacity}")
        # print(f"max_depth: {self.tree_.max_depth}")
        # print("Primer nivel")
        # print(self.tree_.feature[0], features[0])
        # print(self.tree_.threshold[0], thresholds[0])

        # print("Segundo nivel")
        # print(self.tree_.feature[1], features[1])
        # print(self.tree_.threshold[1], thresholds[1])

        # print("Tercer nivel")
        # print(self.tree_.feature[2], features[2])
        # print(self.tree_.threshold[2], thresholds[2])
        
        out = self.apply(X) # Finds the terminal region (=leaf node) for each sample in X.

        print(f"hice el apply:", out)
        
        self.tree_.recompute_values(out, y) # Recompute the values of the leaf nodes