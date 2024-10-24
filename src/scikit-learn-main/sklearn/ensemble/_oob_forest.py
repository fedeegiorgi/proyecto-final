from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestGroupDebate
import threading

import numpy as np
from sklearn.metrics import mean_squared_error #agregado para calcular el mse de cada arbol en sus oob y sacar su peso en la prediccion 

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

    
# ------------------------------------------------------- Alternativa B -----------------------------------------------------------------------------

# ------- version 1 (1/MSE) --------
class OOBRandomForestRegressor(RandomForestRegressor):
    
    def fit(self, X, y):

        # convertimos X a un array numpy si es un DataFrame, para no tener los feature names
        # if isinstance(X, pd.DataFrame):
        #     X = X.values
        
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
            
            if len(oob_samples_X) == 0: 
                self.tree_weights = [1] * self.n_estimators
                print("No OOB samples")
                break
            
            oob_pred = tree.predict(oob_samples_X)
            mse = mean_squared_error(oob_samples_y, oob_pred) 
            self.tree_weights.append(1/mse) # utilizamos la inverse del MSE para que arboles con mayor MSE, tengan menor peso

        # normalizar pesos para que sumen 1
        self.tree_weights = np.array(self.tree_weights)
        self.tree_weights /= self.tree_weights.sum()

    def predict(self, X):
        
        check_is_fitted(self)

        # convertimos X a un array numpy si es un DataFrame, para no tener los feature names
        # if isinstance(X, pd.DataFrame):
        #     X = X.values

        X = self._validate_X_predict(X)
        
        # Assign chunk of trees to jobs
        n_jobs, _, _ = _partition_estimators(self.n_estimators, self.n_jobs)
        lock = threading.Lock()

        if self.n_outputs_ > 1:
            all_predictions = np.zeros((self.n_estimators, X.shape[0], self.n_outputs_), dtype=np.float64)
        else:
            all_predictions = np.zeros((self.n_estimators, X.shape[0]), dtype=np.float64)

        # Parallel loop
        Parallel(n_jobs=n_jobs, verbose=self.verbose, require="sharedmem")(
            delayed(_store_prediction)(e.predict, X, [all_predictions], lock, i)
            for i, e in enumerate(self.estimators_)
        )

        weighted_predictions = np.zeros(X.shape[0], dtype=np.float64)

        for i in range(self.n_estimators):
            weighted_predictions += all_predictions[i] * self.tree_weights[i] 
    
        return weighted_predictions
    
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

# ------------- version 2 (Función Sigmoidea) -------------------------------

class OOBRandomForestRegressorGroupsSigmoid(RandomForestGroupDebate):

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

        # Aplly softplus function to weights
        means = np.mean(self.tree_weights, axis=1, keepdims=True)
        stds = np.std(self.tree_weights, axis=1, keepdims=True)
        self.tree_weights = 1 / (1 + np.exp((self.tree_weights - means) / stds))

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

#----- version 3 (Tangente hiperbólica) --------

class OOBRandomForestRegressorGroupsTanh(RandomForestGroupDebate):

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

        # Aplly softplus function to weights
        means = np.mean(self.tree_weights, axis=1, keepdims=True)
        stds = np.std(self.tree_weights, axis=1, keepdims=True)
        self.tree_weights = 0.5 * (1 + np.tanh((self.tree_weights - means) / stds))
        
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

#-------- version 4 (softplus) -----------

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

        # Aplly softplus function to weights
        means = np.mean(self.tree_weights, axis=1, keepdims=True)
        stds = np.std(self.tree_weights, axis=1, keepdims=True)
        self.tree_weights = np.log(1 + np.exp((self.tree_weights - means) / stds))

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