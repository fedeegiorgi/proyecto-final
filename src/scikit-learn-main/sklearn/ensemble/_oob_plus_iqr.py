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


class OOB_plus_IQR(RandomForestGroupDebate):

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

        # # Normalize weights so that they sum up to 1
        # sums = np.sum(self.tree_weights, axis=1, keepdims=True)  # Sum along the group_size dimension
        # self.tree_weights /= sums

   
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

            final_predictions = []

            # For each group
            for i in range(self._n_groups):
                
                # Extract the current group
                group_predictions = grouped_trees[i]
                
                # Extract the weights of the current group
                group_weights = self.tree_weights[i]

                # Calculate Q1 and Q3 for the current group
                Q1 = np.percentile(group_predictions, 25, axis=0)
                Q3 = np.percentile(group_predictions, 75, axis=0)

                # Calculate IQR
                IQR = Q3 - Q1

                # Define the exclusion range
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                # Filtrar predicciones fuera de los límites
                mask = (group_predictions >= lower_bound) & (group_predictions <= upper_bound)

                # Aplicar máscara a las predicciones y pesos (mantener solo las predicciones válidas y sus pesos correspondientes)
                valid_predictions = np.where(mask, group_predictions, np.nan)
                valid_weights = np.where(mask, group_weights, np.nan)

                # Normalizar los pesos solo para las predicciones válidas
                weight_sums = np.nansum(valid_weights, axis=0, keepdims=True)  # Sumar pesos de árboles válidos
                normalized_weights = np.nan_to_num(valid_weights / weight_sums)  # Normalizar pesos válidos

                # Calcular predicción ponderada utilizando los pesos normalizados
                weighted_predictions = np.nansum(valid_predictions * normalized_weights, axis=0)
                
                final_predictions.append(weighted_predictions)

            # Convertir a numpy array y calcular la media final para cada observación
            final_predictions = np.array(final_predictions)
            y_hat = np.mean(final_predictions, axis=0)

            return y_hat