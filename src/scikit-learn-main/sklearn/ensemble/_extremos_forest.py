from sklearn.ensemble import RandomForestGroupDebate
import threading
import numpy as np
import pandas as pd
from scipy.sparse import hstack as sparse_hstack
from scipy.sparse import issparse
from scipy.stats import zscore #agregado para descartar extremos
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

# --------------------------------------------------------- Alternativa A --------------------------------------------------------------------

class ZscoreRandomForestRegressor(RandomForestGroupDebate):

    def predict(self, X):
        """
        Predict regression target for X.

        The predicted regression target of an input sample is computed as the
        mean predicted regression targets of the trees in the forest, after
        discarding extreme values based on Z-score.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csr_matrix``.

        Returns
        -------
        y : ndarray of shape (n_samples,) or (n_samples, n_outputs)
            The predicted values.
        """
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

        # calculamos el z-score para cada prediccion
        z_scores = zscore(all_predictions, axis=0)

        # definimos el threshole
        threshold = 2.0  

        # filtramos las predicciones que superan el threshole
        filtered_predictions = np.where(np.abs(z_scores) <= threshold, all_predictions, np.nan)

        # calculamos la media con las predicciones que pasaron el filtro
        y_hat = np.nanmean(filtered_predictions, axis=0)

        return y_hat

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

        grouped_trees = self.random_group_split(all_predictions)

        group_averages = np.empty((self._n_groups, X.shape[0]))

        # For each group
        for i in range(self._n_groups):
            # Extract the current group
            group_predictions = grouped_trees[i]

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

        # definimos los percentiles de exclusión
        lower_percentile = self.percentile
        upper_percentile = 100 - self.percentile

        # calculamos los valores de corte
        lower_bound = np.percentile(all_predictions, lower_percentile, axis=0)
        upper_bound = np.percentile(all_predictions, upper_percentile, axis=0)

        # filtramos los valores fuera de los percentiles
        filtered_predictions = np.where((all_predictions >= lower_bound) & 
                                        (all_predictions <= upper_bound), 
                                        all_predictions, np.nan)

        # calculamos la media de las predicciones filtradas (ignorando NaNs)
        y_hat = np.nanmean(filtered_predictions, axis=0)

        return y_hat