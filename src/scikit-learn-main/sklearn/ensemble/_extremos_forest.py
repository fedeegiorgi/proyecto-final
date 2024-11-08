from sklearn.ensemble import RandomForestGroupDebate
import threading
import numpy as np

from ..utils.parallel import Parallel, delayed
from ..utils.validation import check_is_fitted
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

# --------------------------------------------------------- Alternativa A --------------------------------------------------------------------

class IQRRandomForestRegressor(RandomForestGroupDebate):

    def predict(self, X):
        check_is_fitted(self) #fijarse que no estamos ocultando las features?

        X = self._validate_X_predict(X)

        n_jobs, _, _ = _partition_estimators(self.n_estimators, self.n_jobs)
        lock = threading.Lock()

        if self.n_outputs_ > 1:
            all_predictions = np.zeros((self.n_estimators, X.shape[0], self.n_outputs_), dtype=np.float64)
        else:
            all_predictions = np.zeros((self.n_estimators, X.shape[0]), dtype=np.float64)

        Parallel(n_jobs=n_jobs, verbose=self.verbose, require="sharedmem")(
            delayed(_store_prediction)(e.predict, X, [all_predictions], lock, i)
            for i, e in enumerate(self.estimators_)
        )

        grouped_predictions = self.group_split_predictions(all_predictions)

        group_averages = np.empty((self._n_groups, X.shape[0]))

        # For each group
        for i in range(self._n_groups):
            # Extract the current group
            group_predictions = grouped_predictions[i]

            # Calculate Q1 and Q3 for the current group
            Q1 = np.percentile(group_predictions, 25, axis=0)
            Q3 = np.percentile(group_predictions, 75, axis=0)

            # Calculate IQR
            IQR = Q3 - Q1

            # Define the exclusion range
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # Filter values outside the exclusion range
            filtered_predictions = np.where((group_predictions >= lower_bound) & 
                                            (group_predictions <= upper_bound), 
                                            group_predictions, np.nan)

            # Calculate the mean of the filtered predictions (ignoring NaNs)
            group_averages[i] = np.nanmean(filtered_predictions, axis=0)

        y_hat = np.mean(group_averages, axis=0)

        return y_hat

class PercentileTrimmingRandomForestRegressor(RandomForestGroupDebate):
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
        percentile=10,  # New parameter for percentile trimming
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

        # Initialize the new parameter specific to this class
        self.percentile = percentile

        if not (0 <= self.percentile < 50):
            raise ValueError("The percentile must be between 0 and 50.")

    def predict(self, X):

        check_is_fitted(self)
        X = self._validate_X_predict(X)

        n_jobs, _, _ = _partition_estimators(self.n_estimators, self.n_jobs)
        lock = threading.Lock()

        if self.n_outputs_ > 1:
            all_predictions = np.zeros((self.n_estimators, X.shape[0], self.n_outputs_), dtype=np.float64)
        else:
            all_predictions = np.zeros((self.n_estimators, X.shape[0]), dtype=np.float64)

        Parallel(n_jobs=n_jobs, verbose=self.verbose, require="sharedmem")(
            delayed(_store_prediction)(e.predict, X, [all_predictions], lock, i)
            for i, e in enumerate(self.estimators_)
        )

        grouped_predictions = self.group_split_predictions(all_predictions)

        group_averages = np.empty((self._n_groups, X.shape[0]))


        for i in range(self._n_groups):
            # Extract the current group
            group_predictions = grouped_predictions[i]

            # definimos los percentiles de exclusión
            lower_percentile = self.percentile
            upper_percentile = 100 - self.percentile

            # calculamos los valores de corte
            lower_bound = np.percentile(group_predictions, lower_percentile, axis=0)
            upper_bound = np.percentile(group_predictions, upper_percentile, axis=0)

            # filtramos los valores fuera de los percentiles
            filtered_predictions = np.where((group_predictions >= lower_bound) & 
                                            (group_predictions <= upper_bound), 
                                            group_predictions, np.nan)
            
            # Calculate the mean of the filtered predictions (ignoring NaNs)
            group_averages[i] = np.nanmean(filtered_predictions, axis=0)

        # calculamos la media de las predicciones filtradas (ignorando NaNs)
        y_hat = np.mean(group_averages, axis=0)

        return y_hat