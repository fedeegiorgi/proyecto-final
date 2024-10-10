from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestGroupDebate
import threading
from abc import ABCMeta, abstractmethod
from numbers import Integral, Real
from warnings import catch_warnings, simplefilter, warn

import numpy as np
import pandas as pd
from scipy.sparse import hstack as sparse_hstack
from scipy.sparse import issparse
from scipy.stats import mstats
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

    
# ------------------------------------------------------- Alternativa B -----------------------------------------------------------------------------

# ------- version 1 (1/MSE) --------
class OOBRandomForestRegressor(RandomForestRegressor):
    
    def fit(self, X, y):

        # convertimos X a un array numpy si es un DataFrame, para no tener los feature names
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        super().fit(X, y) #utilizamos el fit original de BaseForest
        
        n_samples = X.shape[0]
        self.tree_weights = []

        # calculamos pesos OOB para cada árbol
        for i, tree in enumerate(self.estimators_):
            oob_sample_mask = np.ones(n_samples, dtype=bool) #inicializo una mascara con 1's

            # asignamos false a las muestras que el arbol utilizo para entrenar, ya que no son OOB
            oob_sample_mask[self.estimators_samples_[i]] = False
            
            oob_samples_X = X[oob_sample_mask] # solo se seleccionan las observaciones que tienen valor True, las OOB observations
            oob_samples_y = y[oob_sample_mask]
            
            if len(oob_samples_X) == 0: #si no hay muestras oob, asignamos a todos los arboles el mismo peso
                self.tree_weights.append(1 / self.n_estimators) 
                continue
            
            oob_pred = tree.predict(oob_samples_X)
            peso = 1 / mean_squared_error(oob_samples_y, oob_pred) #utilizamos la inverse del MSE para que arboles con mayor MSE, tengan menor peso
            self.tree_weights.append(peso)

        # normalizar pesos para que sumen 1
        self.tree_weights = np.array(self.tree_weights)
        self.tree_weights /= self.tree_weights.sum()

    def predict(self, X):
        check_is_fitted(self)
        
        # convertimos X a un array numpy si es un DataFrame para que no se releven las features
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        X = self._validate_X_predict(X)
        
        n_jobs, _, _ = _partition_estimators(self.n_estimators, self.n_jobs)

        # predicciones para cada árbol
        all_predictions = np.zeros((X.shape[0],), dtype=np.float64)

        # sumar las predicciones con los pesos OOB
        for i, tree in enumerate(self.estimators_):
            tree_prediction = tree.predict(X)
            all_predictions += tree_prediction * self.tree_weights[i] #ponderamos la prediccion de cada arbol con su peso correspondiente
        
        return all_predictions
    
# -------------- version 1 (con grupos)------------------------

class OOBRandomForestRegressorGroups(RandomForestGroupDebate):

    def fit(self, X, y):
        # Call to original fit method
        super().fit(X, y)

        n_samples = X.shape[0]
        self.tree_weights = []

        # Calculate OOB MSE for each tree
        for i, tree in enumerate(self.estimators_):
            # Create a mask with True for all samples
            oob_sample_mask = np.ones(n_samples, dtype=bool) 

            # Assign False to the samples that the tree used for training, as they are not OOB
            oob_sample_mask[self.estimators_samples_[i]] = False
            
            # Select only the observations that have True value, the OOB observations
            oob_samples_X = X[oob_sample_mask] 
            oob_samples_y = y[oob_sample_mask]
            
            # If no OOB samples, assign the same weight to all trees?
            if len(oob_samples_X) == 0: 
                self.tree_weights = [1] * self.n_estimators
                print("No OOB samples")
                break
            
            oob_pred = tree.predict(oob_samples_X)
            mse = mean_squared_error(oob_samples_y, oob_pred)

            # Use the inverse of the MSE so that trees with higher MSE have lower weight
            self.tree_weights.append(1/mse)

        # Reshape groups to match predictions shape
        self.tree_weights = np.array(self.tree_weights).astype(float) 
        self.tree_weights = self.tree_weights.reshape(self._n_groups, self.group_size, 1)

        # Normalize weights so that they sum up to 1
        sums = np.sum(self.tree_weights, axis=1, keepdims=True)  # Sum along the group_size dimension
        self.tree_weights /= sums
   
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

        grouped_trees = self.random_group_split(all_predictions)

        # Multiply using broadcasting to get the weighted group predictions
        group_predictions = np.sum(grouped_trees * self.tree_weights, axis=1)

        # Final prediction is the mean of the group predictions
        y_hat = np.mean(group_predictions, axis=0)

        return y_hat

# ------------- version 2 (funcion sigmoidea) -------------------------------

class OOBRandomForestRegressorSigmoid(RandomForestRegressor):

    def fit(self, X, y):

        # convertimos X a un array numpy si es un DataFrame, para no tener los feature names
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        super().fit(X, y)
        
        n_samples = X.shape[0]
        self.tree_weights = []
        mse_oob_values = []  # Aquí vamos a guardar los MSE de cada árbol

        # calculamos pesos OOB para cada árbol
        for i, tree in enumerate(self.estimators_):
            oob_sample_mask = np.ones(n_samples, dtype=bool) #inicializo una mascara con 1's

            # asignamos false a las muestras que el arbol utilizo para entrenar, ya que no son OOB
            oob_sample_mask[self.estimators_samples_[i]] = False
            
            oob_samples_X = X[oob_sample_mask] # solo se seleccionan las observaciones que tienen valor True, las OOB observations
            oob_samples_y = y[oob_sample_mask]


            if len(oob_samples_X) == 0:
                self.tree_weights.append(1 / self.n_estimators)  # le damos el mismo peso a los arboles que no tienen muestras OOB
                continue

            oob_pred = tree.predict(oob_samples_X)
            mse_oob = mean_squared_error(oob_samples_y, oob_pred)
            mse_oob_values.append(mse_oob)

        if mse_oob_values:
            mse_oob_array = np.array(mse_oob_values)

            # Parámetros de suavización
            mu = np.mean(mse_oob_array)  # media de los MSE
            sigma = np.std(mse_oob_array)  # desviación estándar de los MSE
            sigmoid_weights = 1 / (1 + np.exp((mse_oob_array - mu) / sigma)) # --> ponderación suavizada con función sigmoidea
            
            #log_mse = -np.log(mse_oob_array + 1e-8)  # Logaritmo negativo con pequeño epsilon para evitar -inf

            # normalizamos los pesos
            tree_weights = sigmoid_weights / np.sum(sigmoid_weights)
            self.tree_weights = list(tree_weights)

        # Si no hay pesos definidos, usar pesos uniformes
        if not self.tree_weights:
            self.tree_weights = [1 / self.n_estimators] * self.n_estimators


    def predict(self, X):
        check_is_fitted(self)
        
        # convertimos X a un array numpy si es un DataFrame para que no se releven las features
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        X = self._validate_X_predict(X)
        
        n_jobs, _, _ = _partition_estimators(self.n_estimators, self.n_jobs)

        # inicializamos las predicciones para cada árbol
        all_predictions = np.zeros((X.shape[0],), dtype=np.float64)

        # sumar las predicciones con los pesos OOB
        for i, tree in enumerate(self.estimators_):
            tree_prediction = tree.predict(X)
            all_predictions += tree_prediction * self.tree_weights[i] #ponderamos la prediccion de cada arbol con su peso correspondiente
        
        return all_predictions

#----- version 3 (Tangente hiperbólica) --------

class OOBRandomForestRegressorTanh(RandomForestRegressor):

    def fit(self, X, y):

        # convertimos X a un array numpy si es un DataFrame, para no tener los feature names
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        super().fit(X, y)
        
        n_samples = X.shape[0]
        self.tree_weights = []
        mse_oob_values = []  # Aquí vamos a guardar los MSE de cada árbol

        # calculamos pesos OOB para cada árbol
        for i, tree in enumerate(self.estimators_):
            oob_sample_mask = np.ones(n_samples, dtype=bool) #inicializo una mascara con 1's

            # asignamos false a las muestras que el arbol utilizo para entrenar, ya que no son OOB
            oob_sample_mask[self.estimators_samples_[i]] = False
            
            oob_samples_X = X[oob_sample_mask] # solo se seleccionan las observaciones que tienen valor True, las OOB observations
            oob_samples_y = y[oob_sample_mask]


            if len(oob_samples_X) == 0:
                self.tree_weights.append(1 / self.n_estimators)  # le damos el mismo peso a los arboles que no tienen muestras OOB
                continue

            oob_pred = tree.predict(oob_samples_X)
            mse_oob = mean_squared_error(oob_samples_y, oob_pred)
            mse_oob_values.append(mse_oob)

        if mse_oob_values:
            mse_oob_array = np.array(mse_oob_values)

            # Parámetros de suavización Tanh
            mu = np.mean(mse_oob_array)  # media de los MSE
            sigma = np.std(mse_oob_array)  # desviación estándar de los MSE
            tanh_weights = 0.5 * (1 + np.tanh((mse_oob_array - mu) / sigma))

            # normalizamos los pesos
            tanh_weights_normalized = tanh_weights / np.sum(tanh_weights)
            self.tree_weights = list(tanh_weights_normalized)

        # Si no hay pesos definidos, usar pesos uniformes
        if not self.tree_weights:
            self.tree_weights = [1 / self.n_estimators] * self.n_estimators


    def predict(self, X):
        check_is_fitted(self)
        
        # convertimos X a un array numpy si es un DataFrame para que no se releven las features
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        X = self._validate_X_predict(X)
        
        n_jobs, _, _ = _partition_estimators(self.n_estimators, self.n_jobs)

        # inicializamos las predicciones para cada árbol
        all_predictions = np.zeros((X.shape[0],), dtype=np.float64)

        # sumar las predicciones con los pesos OOB
        for i, tree in enumerate(self.estimators_):
            tree_prediction = tree.predict(X)
            all_predictions += tree_prediction * self.tree_weights[i] #ponderamos la prediccion de cada arbol con su peso correspondiente
        
        return all_predictions

#-------- version 4 (softplus) -----------
class OOBRandomForestRegressorSoftPlus(RandomForestRegressor):

    def fit(self, X, y):

        # convertimos X a un array numpy si es un DataFrame, para no tener los feature names
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        super().fit(X, y)
        
        n_samples = X.shape[0]
        self.tree_weights = []
        mse_oob_values = []  # Aquí vamos a guardar los MSE de cada árbol

        # calculamos pesos OOB para cada árbol
        for i, tree in enumerate(self.estimators_):
            oob_sample_mask = np.ones(n_samples, dtype=bool) #inicializo una mascara con 1's

            # asignamos false a las muestras que el arbol utilizo para entrenar, ya que no son OOB
            oob_sample_mask[self.estimators_samples_[i]] = False
            
            oob_samples_X = X[oob_sample_mask] # solo se seleccionan las observaciones que tienen valor True, las OOB observations
            oob_samples_y = y[oob_sample_mask]


            if len(oob_samples_X) == 0:
                self.tree_weights.append(1 / self.n_estimators)  # le damos el mismo peso a los arboles que no tienen muestras OOB
                continue

            oob_pred = tree.predict(oob_samples_X)
            mse_oob = mean_squared_error(oob_samples_y, oob_pred)
            mse_oob_values.append(mse_oob)

        if mse_oob_values:
            mse_oob_array = np.array(mse_oob_values)

            # Parámetros de suavización Tanh
            mu = np.mean(mse_oob_array)  # media de los MSE
            sigma = np.std(mse_oob_array)  # desviación estándar de los MSE
            softplus_weights = np.log(1 + np.exp(-(mse_oob_array - mu) / sigma))


            # normalizamos los pesos
            softplus_weights_normalized = softplus_weights / np.sum(softplus_weights)
            self.tree_weights = list(softplus_weights_normalized)

        # Si no hay pesos definidos, usar pesos uniformes
        if not self.tree_weights:
            self.tree_weights = [1 / self.n_estimators] * self.n_estimators


    def predict(self, X):
        check_is_fitted(self)
        
        # convertimos X a un array numpy si es un DataFrame para que no se releven las features
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        X = self._validate_X_predict(X)
        
        n_jobs, _, _ = _partition_estimators(self.n_estimators, self.n_jobs)

        # inicializamos las predicciones para cada árbol
        all_predictions = np.zeros((X.shape[0],), dtype=np.float64)

        # sumar las predicciones con los pesos OOB
        for i, tree in enumerate(self.estimators_):
            tree_prediction = tree.predict(X)
            all_predictions += tree_prediction * self.tree_weights[i] #ponderamos la prediccion de cada arbol con su peso correspondiente
        
        return all_predictions

class OOBRandomForestRegressorGroupsSoftPlus(RandomForestGroupDebate):

    def fit(self, X, y):
        # Call to original fit method
        super().fit(X, y)

        n_samples = X.shape[0]
        self.tree_weights = []

        # Calculate OOB MSE for each tree
        for i, tree in enumerate(self.estimators_):
            # Create a mask with True for all samples
            oob_sample_mask = np.ones(n_samples, dtype=bool) 

            # Assign False to the samples that the tree used for training, as they are not OOB
            oob_sample_mask[self.estimators_samples_[i]] = False
            
            # Select only the observations that have True value, the OOB observations
            oob_samples_X = X[oob_sample_mask] 
            oob_samples_y = y[oob_sample_mask]
            
            # If no OOB samples, assign the same weight to all trees?
            if len(oob_samples_X) == 0: 
                self.tree_weights = [1 * self.n_estimators]
                print("No OOB samples")
                break
            
            oob_pred = tree.predict(oob_samples_X)
            mse = mean_squared_error(oob_samples_y, oob_pred)

            # Use the inverse of the MSE so that trees with higher MSE have lower weight
            self.tree_weights.append(1/mse)

        # Reshape groups to match predictions shape
        self.tree_weights = np.array(self.tree_weights).astype(float) 
        self.tree_weights = self.tree_weights.reshape(self._n_groups, self.group_size, 1)

        # Aplly softplus function to weights
        means = np.mean(self.tree_weights, axis=1, keepdims=True)
        stds = np.std(self.tree_weights, axis=1, keepdims=True)
        self.tree_weights = np.log(1 + np.exp(-(self.tree_weights - means) / stds))

        # Normalize weights so that they sum up to 1
        sums = np.sum(self.tree_weights, axis=1, keepdims=True)  # Sum along the group_size dimension
        self.tree_weights /= sums
   
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

        grouped_trees = self.random_group_split(all_predictions)

        # Multiply using broadcasting to get the weighted group predictions
        group_predictions = np.sum(grouped_trees * self.tree_weights, axis=1)

        # Final prediction is the mean of the group predictions
        y_hat = np.mean(group_predictions, axis=0)

        return y_hat