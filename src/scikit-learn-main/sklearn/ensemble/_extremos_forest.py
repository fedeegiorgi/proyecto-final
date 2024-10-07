from sklearn.ensemble import RandomForestRegressor
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

def _accumulate_prediction(predict, X, out, lock=None):
    """
    This is a utility function for joblib's Parallel.
    
    It can’t go locally in ForestClassifier or ForestRegressor, because joblib
    complains that it cannot pickle it when placed there.
    """
    prediction = predict(X, check_input=False)
    
    if lock is None:
        # If no lock is provided, directly accumulate the predictions
        if len(out) == 1:
            out[0] += prediction
        else:
            for i in range(len(out)):
                out[i] += prediction[i]
    else:
        # Use the lock for thread safety
        with lock:
            if len(out) == 1:
                out[0] += prediction
            else:
                for i in range(len(out)):
                    out[i] += prediction[i]

# --------------------------------------------------------- Alternativa A --------------------------------------------------------------------

class ZscoreRandomForestRegressor(RandomForestRegressor):

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
        lock = threading.Lock() if n_jobs != 1 else None

        if self.n_outputs_ > 1:
            all_predictions = np.zeros((self.n_estimators, X.shape[0], self.n_outputs_), dtype=np.float64)
        else:
            all_predictions = np.zeros((self.n_estimators, X.shape[0]), dtype=np.float64)

        Parallel(n_jobs=n_jobs, verbose=self.verbose, require="sharedmem")(
            delayed(_accumulate_prediction)(e.predict, X, [all_predictions[i]], None)
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

class IQRRandomForestRegressor(RandomForestRegressor):

    def predict(self, X):
        check_is_fitted(self)
        X = self._validate_X_predict(X)

        n_jobs, _, _ = _partition_estimators(self.n_estimators, self.n_jobs)
        lock = threading.Lock() if n_jobs != 1 else None

        if self.n_outputs_ > 1:
            all_predictions = np.zeros((self.n_estimators, X.shape[0], self.n_outputs_), dtype=np.float64)
        else:
            all_predictions = np.zeros((self.n_estimators, X.shape[0]), dtype=np.float64)

        Parallel(n_jobs=n_jobs, verbose=self.verbose, require="sharedmem")(
            delayed(_accumulate_prediction)(e.predict, X, [all_predictions[i]], None)
            for i, e in enumerate(self.estimators_)
        )

        # calculamos los cuartiles Q1 y Q3
        Q1 = np.percentile(all_predictions, 25, axis=0)
        Q3 = np.percentile(all_predictions, 75, axis=0)
        #calculamos IQR
        IQR = Q3 - Q1

        # definimos el rango de exclusión
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # filtramos los valores fuera del rango IQR
        filtered_predictions = np.where((all_predictions >= lower_bound) & 
                                        (all_predictions <= upper_bound), 
                                        all_predictions, np.nan)

        # calculamos la media de las predicciones filtradas (ignorando NaNs)
        y_hat = np.nanmean(filtered_predictions, axis=0)

        # manejo los casos donde todas las predicciones son NaN
        if np.isnan(y_hat).any():
            y_hat = np.nan_to_num(y_hat, nan=np.nanmedian(all_predictions, axis=0))

        return y_hat

class PercentileTrimmingRandomForestRegressor(RandomForestRegressor):
    def predict(self, X):
        check_is_fitted(self)
        X = self._validate_X_predict(X)

        n_jobs, _, _ = _partition_estimators(self.n_estimators, self.n_jobs)
        lock = threading.Lock() if n_jobs != 1 else None

        if self.n_outputs_ > 1:
            all_predictions = np.zeros((self.n_estimators, X.shape[0], self.n_outputs_), dtype=np.float64)
        else:
            all_predictions = np.zeros((self.n_estimators, X.shape[0]), dtype=np.float64)

        Parallel(n_jobs=n_jobs, verbose=self.verbose, require="sharedmem")(
            delayed(_accumulate_prediction)(e.predict, X, [all_predictions[i]], None)
            for i, e in enumerate(self.estimators_)
        )

        # definimos los percentiles de exclusión
        lower_percentile = 5
        upper_percentile = 95

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