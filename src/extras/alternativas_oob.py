# De querer usarse, insertar en sklearn/ensemble

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
    Now we store the predictions in the tree's corresponding column.
    """
    prediction = predict(X, check_input=False)
    with lock:
        out[0][tree_index] = prediction   # Store predictions in the column corresponding to the tree

# ------------- version 2 (Función Sigmoidea) -------------------------------

class OOBRandomForestRegressorSigmoid(RandomForestGroupDebate):

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

        grouped_trees = self.group_split_predictions(all_predictions)

        # Multiply using broadcasting to get the weighted group predictions
        group_predictions = np.sum(grouped_trees * self.tree_weights, axis=1)

        # Final prediction is the mean of the group predictions
        y_hat = np.mean(group_predictions, axis=0)

        return y_hat

#----- version 3 (Tangente hiperbólica) --------

class OOBRandomForestRegressorTanh(RandomForestGroupDebate):

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

        grouped_trees = self.group_split_predictions(all_predictions)

        # Multiply using broadcasting to get the weighted group predictions
        group_predictions = np.sum(grouped_trees * self.tree_weights, axis=1)

        # Final prediction is the mean of the group predictions
        y_hat = np.mean(group_predictions, axis=0)

        return y_hat

#-------- version 4 (softplus) -----------

class OOBRandomForestRegressorSoftPlus(RandomForestGroupDebate):

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

        grouped_trees = self.group_split_predictions(all_predictions)

        # Multiply using broadcasting to get the weighted group predictions
        group_predictions = np.sum(grouped_trees * self.tree_weights, axis=1)

        # Final prediction is the mean of the group predictions
        y_hat = np.mean(group_predictions, axis=0)

        return y_hat