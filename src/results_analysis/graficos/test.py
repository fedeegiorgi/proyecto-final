import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.ensemble import (
    IQRRandomForestRegressor, PercentileTrimmingRandomForestRegressor, 
    OOBRandomForestRegressor, OOB_plus_IQR, RFRegressorFirstSplitCombiner, SharedKnowledgeRandomForestRegressor)
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
from scipy.io import arff
import os


# Replace 'your_file.npy' with the path to your .npy file
file_path = 'Carbon_Emission_IQR_max_depth.npy'
data_carbon = np.load(file_path)

# Define column names
columns = ['max_depth', 'MSE']

# Create a DataFrame
df_carbon = pd.DataFrame(data_carbon, columns=columns)

sorted_df_carbon = df_carbon.sort_values(by='MSE', ascending=True)
# sorted_df_carbon['Rank'] = range(1, len(sorted_df_carbon) + 1)

top_5_carbon = sorted_df_carbon[:5]
print('Top 5 max_depths:')
print(top_5_carbon)

DATASETS_COLUMNS = {
    'Carbon_Emission': 'CarbonEmission',
    'Wind': 'WIND',
    'House_8L': 'price',
}

SEED = 14208
dataset_name = 'Carbon_Emission'
file_path = 'Carbon_Emission_train.csv'

data = pd.read_csv(file_path)

# Preprocess data
target_column = DATASETS_COLUMNS[dataset_name]
train_df, validation_df = train_test_split(data, test_size=0.2, random_state=SEED)

y_train, y_valid = train_df[target_column], validation_df[target_column]
X_train, X_valid = train_df.drop(target_column, axis=1), validation_df.drop(target_column, axis=1)

# Encode categorical features
X_train = pd.get_dummies(X_train)
X_valid = pd.get_dummies(X_valid)

# Align train and validation datasets
X_train, X_valid = X_train.align(X_valid, join='left', axis=1, fill_value=0)

rf_iqr = IQRRandomForestRegressor(max_depth=20, random_state=SEED, n_jobs=3)
rf_iqr.fit(X_train.values, y_train)

y_pred = rf_iqr.predict(X_valid.values)
mse = mean_squared_error(y_valid, y_pred)
print(f'MSE haciendo rf_iqr = IQRRandomForestRegressor(max_depth=20, random_state=SEED, n_jobs=3): {mse}')