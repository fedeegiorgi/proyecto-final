from sklearn.model_selection import GridSearchCV, train_test_split
from tqdm import tqdm
from sklearn.ensemble import IQRRandomForestRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd
from scipy.io import arff
import os
from itertools import product

file_path = 'distribucion/datasets/train_data/Carbon_Emission_train.csv'
dataset_name = 'Carbon_Emission'
data = pd.read_csv(file_path)
target_column = 'CarbonEmission'

SEED = 14208

train_df, validation_df = train_test_split(data, test_size=0.2, random_state=SEED)
y_train, y_valid = train_df[target_column], validation_df[target_column]
X_train, X_valid = train_df.drop(target_column, axis=1), validation_df.drop(target_column, axis=1)

print(X_valid.shape)

# Encode categorical features
X_train = pd.get_dummies(X_train)
X_valid = pd.get_dummies(X_valid)

# Align train and validation datasets
X_train, X_valid = X_train.align(X_valid, join='left', axis=1, fill_value=0)

rf_test = IQRRandomForestRegressor(random_state=SEED, n_estimators=450, group_size=5, max_depth=None)
rf_test.fit(X_train.values, y_train.values)

predictions = rf_test.predict(X_valid.values)
mse = mean_squared_error(y_valid, predictions)

print(mse)