from sklearn.ensemble import RandomForestGroupDebate
import threading
import numpy as np

from ..tree import DecisionTreeRegressorCombiner

from ..utils.parallel import Parallel, delayed
from ..utils.validation import check_is_fitted
from ._base import _partition_estimators

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

        self.combined_trees_ = []

        if max_depth is not None:
            raise ValueError("max_depth not available in FirstSplitCombiner.")

    
    def _compute_group_sample(self, group_samples_used, n_samples):
        """
        Compute the sample of each group of trees.
        """
        # Create a mask with False for all sample
        sample_mask = np.zeros(n_samples, dtype=bool)

        # Loop over the the samples used in the group
        for samples in group_samples_used:
            for sampleidx in samples:
                # Set the mask to True for the samples used
                sample_mask[sampleidx] = True
    
        return sample_mask

    def fit(self, X, y, sample_weights=None):
        # Set max_depth = 1. We only care about the first split.
        self.max_depth = 1

        # Call to original fit method
        super().fit(X, y)

        n_samples = X.shape[0]

        # Divide the trees into groups
        self.initial_grouped_trees = self.group_split(self.estimators_)
        grouped_samples = self.group_split(self.estimators_samples_)
        
        # For each group of trees
        for i, tree_group in enumerate(self.initial_grouped_trees):
            # Get the samples used in the group
            group_samples_used = grouped_samples[i]

            # Compute the sample mask for the combined group tree
            sample_mask = self._compute_group_sample(group_samples_used, n_samples)

            # Select only the observations that have True value, the samples used in the group
            X_union = X[sample_mask] 
            y_union = y[sample_mask]

            # Combine the trees in the group
            group_combined_tree = DecisionTreeRegressorCombiner(initial_trees=tree_group)
            group_combined_tree.fit(X_union, y_union)

            # Store the combined tree
            self.combined_trees_.append(group_combined_tree)
        
    def predict(self, X):
        check_is_fitted(self)

        X = self._validate_X_predict(X)

        n_jobs, _, _ = _partition_estimators(self._n_groups, self.n_jobs)
        lock = threading.Lock()

        # Avoid storing the output of every estimator by summing them here
        if self.n_outputs_ > 1:
            raise ValueError("Multiprediction not available in this implementation of Random Forest.")
        else:
            y_hat = np.zeros((X.shape[0]), dtype=np.float64)

        # Parallel loop
        lock = threading.Lock()
        Parallel(n_jobs=n_jobs, verbose=self.verbose, require="sharedmem")(
            delayed(_accumulate_prediction)(e.predict, X, [y_hat], lock)
            for e in self.combined_trees_
        )

        y_hat /= len(self.combined_trees_) # Averages the estimates from the combined trees.

        return y_hat