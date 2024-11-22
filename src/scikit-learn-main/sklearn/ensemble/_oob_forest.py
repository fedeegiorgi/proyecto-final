from sklearn.ensemble import RandomForestGroupDebate
import threading

import numpy as np

# New, to calculate weight of tree according to MSE in OOB samples.
from sklearn.metrics import mean_squared_error

from ..utils.parallel import Parallel, delayed
from ..utils.validation import check_is_fitted
from ._base import _partition_estimators

def _store_prediction(predict, X, out, lock, tree_index):
    """
    --------------------------------------------------------------------------
    This is a utility function for joblib's Parallel.

    It can't go locally in ForestClassifier or ForestRegressor, because joblib
    complains that it cannot pickle it when placed there.
    --------------------------------------------------------------------------

    Store each tree's prediction in the 2D array `out`.
    """
    prediction = predict(X, check_input=False)
    with lock:
        out[0][tree_index] = prediction   # Store predictions in the column corresponding to the tree


class OOBRandomForestRegressor(RandomForestGroupDebate):

    def fit(self, X, y):
        # Call to original fit method
        super().fit(X, y)

        n_samples = X.shape[0]
        self.tree_weights = []

        if not self.bootstrap:
            raise ValueError("bootstrap must be set to True to have OOB samples.")

        # Calculate OOB MSE for each tree
        for i, tree in enumerate(self.estimators_):
            # Create a mask with True for all samples
            oob_sample_mask = np.ones(n_samples, dtype=bool) 

            # Assign False to the samples that the tree used for training, as they are not OOB
            oob_sample_mask[self.estimators_samples_[i]] = False

            # Select only the observations that have True value, the OOB observations
            oob_samples_X = X[oob_sample_mask] 
            oob_samples_y = y[oob_sample_mask]

            # If no OOB samples even with bootstrap=True, raise an error (shouldn't happen!)
            if len(oob_samples_X) == 0: 
                raise ValueError("No OOB samples detected.")
            
            oob_pred = tree.predict(oob_samples_X)
            mse = mean_squared_error(oob_samples_y, oob_pred)

            # Use the inverse of the MSE so that trees with higher MSE have lower weight
            self.tree_weights.append(1 / mse)
            
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
            raise ValueError("Multiprediction not available in this implementation of Random Forest.")
        else:
            all_predictions = np.zeros((self.n_estimators, X.shape[0]), dtype=np.float64)

        Parallel(n_jobs=n_jobs, verbose=self.verbose, require="sharedmem")(
            delayed(_store_prediction)(e.predict, X, [all_predictions], lock, i)
            for i, e in enumerate(self.estimators_)
        )

        grouped_predictions = self.group_split_predictions(all_predictions)

        # Multiply using broadcasting to get the weighted group predictions
        group_predictions = np.sum(grouped_predictions * self.tree_weights, axis=1)

        # Final prediction is the mean of the group predictions
        y_hat = np.mean(group_predictions, axis=0)

        return y_hat


class OOBPlusIQRRandomForestRegressor(RandomForestGroupDebate):

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
            
            # If no OOB samples, raise Exeption
            if oob_samples_X.shape[0] == 0: 
                raise ValueError("No out-of-bag samples available for some tree. Use more estimators or larger dataset.")
            
            oob_pred = tree.predict(oob_samples_X)
            mse = mean_squared_error(oob_samples_y, oob_pred)

            # Use the inverse of the MSE so that trees with higher MSE have lower weight
            self.tree_weights.append(1/mse)

        # Reshape groups to match predictions shape
        self.tree_weights = np.array(self.tree_weights).astype(float) 
        self.tree_weights = self.tree_weights.reshape(self._n_groups, self.group_size, 1)

   
    def predict(self, X):
        check_is_fitted(self) #fijarse que no estamos ocultando las features?

        X = self._validate_X_predict(X)

        n_jobs, _, _ = _partition_estimators(self.n_estimators, self.n_jobs)
        lock = threading.Lock()

        if self.n_outputs_ > 1:
            raise ValueError("Multiprediction not available in this implementation of Random Forest.")
        else:
            all_predictions = np.zeros((self.n_estimators, X.shape[0]), dtype=np.float64)

        Parallel(n_jobs=n_jobs, verbose=self.verbose, require="sharedmem")(
            delayed(_store_prediction)(e.predict, X, [all_predictions], lock, i)
            for i, e in enumerate(self.estimators_)
        )

        grouped_predictions = self.group_split_predictions(all_predictions)

        final_predictions = []

        # For each group
        for i in range(self._n_groups):
            
            # Extract the current group
            group_predictions = grouped_predictions[i]
            
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

            # FIlter predictions that are within the exclusion range
            mask = (group_predictions >= lower_bound) & (group_predictions <= upper_bound)

            # Apply mask to predictions and weights (keep only valid predictions and their corresponding weights)
            valid_predictions = np.where(mask, group_predictions, np.nan)
            valid_weights = np.where(mask, group_weights, np.nan)

            # Normalize weights only for valid predictions
            weight_sums = np.nansum(valid_weights, axis=0, keepdims=True)  # Sum of weights for valid trees' predictions
            normalized_weights = np.nan_to_num(valid_weights / weight_sums)  # Normalize weights (those with excluded will have weight 0)

            # Calcular predicción ponderada utilizando los pesos normalizados
            weighted_predictions = np.nansum(valid_predictions * normalized_weights, axis=0)
            
            final_predictions.append(weighted_predictions)

        # Final predictions to numpy array and calculate the final mean for each observation
        final_predictions = np.array(final_predictions)
        y_hat = np.mean(final_predictions, axis=0)

        return y_hat