from sklearn.ensemble import RandomForestRegressor
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

    
# ------------------------------------------------------- Alternativa B -----------------------------------------------------------------------------

# ------- version 1 (1/MSE) --------
class OOBRandomForestRegressor(RandomForestRegressor):
    
    def fit(self, X, y):

        # convertimos X a un array numpy si es un DataFrame, para no tener los feature names
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        super().fit(X, y)
        
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
    

#------- new validation set ----------------

class NewValRandomForestRegressor(RandomForestRegressor):
    
    def fit(self, X, y):

        # Convertir X a un array numpy si es un DataFrame
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Separar el 10% de los datos de entrenamiento para calcular el MSE
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

        # Ajustar el modelo con el 80% de los datos
        super().fit(X_train, y_train)
        
        self.tree_weights = []

        # Calcular pesos para cada árbol usando el 10% del set de validación
        for i, tree in enumerate(self.estimators_):
            val_pred = tree.predict(X_val)
            mse_val = mean_squared_error(y_val, val_pred)
            
            # Manejar la división por cero para MSE muy bajos
            peso = 1 / (mse_val + 1e-5)  
            self.tree_weights.append(peso)

        # Normalizar pesos para que sumen 1
        self.tree_weights = np.array(self.tree_weights)
        self.tree_weights /= self.tree_weights.sum()

    def predict(self, X):
        check_is_fitted(self)
        
        # Convertir X a un array numpy si es un DataFrame
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        X = self._validate_X_predict(X)
        
        # Recoger predicciones para cada árbol
        all_predictions = np.zeros((X.shape[0],), dtype=np.float64)

        # Sumar las predicciones con los pesos calculados
        for i, tree in enumerate(self.estimators_):
            tree_prediction = tree.predict(X)
            all_predictions += tree_prediction * self.tree_weights[i]
        
        return all_predictions
    
# class IntersectionOOBRandomForestRegressor(RandomForestRegressor):
    
#     def fit(self, X, y):

#         # Convertir X a un array numpy si es un DataFrame
#         if isinstance(X, pd.DataFrame):
#             X = X.values
        
#         super().fit(X, y)
        
#         n_samples = X.shape[0]
#         self.tree_weights = []

#         # Crear un mask de True para todas las muestras (ninguna OOB)
#         oob_intersection_mask = np.ones(n_samples, dtype=bool)

#         # Encontrar la intersección de las OOB en todos los árboles
#         for i in range(self.n_estimators):
#             oob_sample_mask = np.ones(n_samples, dtype=bool)
#             oob_sample_mask[self.estimators_samples_[i]] = False
#             oob_intersection_mask &= oob_sample_mask

#         # Extraer las muestras y etiquetas de la intersección OOB
#         oob_samples_X = X[oob_intersection_mask]
#         oob_samples_y = y[oob_intersection_mask]

#         # Si no hay muestras en la intersección OOB, salir
#         if len(oob_samples_X) == 0:
#             raise ValueError("No hay muestras en la intersección de las observaciones OOB.")

#         # Calcular pesos usando las muestras de la intersección OOB
#         for i, tree in enumerate(self.estimators_):
#             oob_pred = tree.predict(oob_samples_X)
#             mse_oob = mean_squared_error(oob_samples_y, oob_pred)
            
#             # Manejar la división por cero para MSE muy bajos
#             peso = 1 / (mse_oob + 1e-5)  
#             self.tree_weights.append(peso)

#         # Normalizar pesos para que sumen 1
#         self.tree_weights = np.array(self.tree_weights)
#         self.tree_weights /= self.tree_weights.sum()

#     def predict(self, X):
#         check_is_fitted(self)
        
#         # Convertir X a un array numpy si es un DataFrame
#         if isinstance(X, pd.DataFrame):
#             X = X.values
        
#         X = self._validate_X_predict(X)
        
#         # Recoger predicciones para cada árbol
#         all_predictions = np.zeros((X.shape[0],), dtype=np.float64)

#         # Sumar las predicciones con los pesos OOB
#         for i, tree in enumerate(self.estimators_):
#             tree_prediction = tree.predict(X)
#             all_predictions += tree_prediction * self.tree_weights[i]
        
#         return all_predictions