from sklearn.ensemble import RandomForestRegressor, ZscoreRandomForestRegressor, IQRRandomForestRegressor, PercentileTrimmingRandomForestRegressor
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.io import arff
import os
from scipy.stats import lognorm
from scipy.stats import norm

SEED = 14208
# Reemplazar con el PATH del dataset descargado
dataset_filepath = 'titanic_fare_test.arff'

# Para cargar el dataset
data = arff.loadarff(dataset_filepath) 
df = pd.DataFrame(data[0])

# Reemplazar con la columna a predecir
pred_col_name = 'Fare'

# Separacion en Train, Validation ('X' e 'y' para cada split)
train_df, validation_df = train_test_split(df, test_size=0.2, random_state=SEED)
y_train, y_valid = train_df[pred_col_name], validation_df[pred_col_name]
X_train, X_valid = train_df.drop(pred_col_name, axis=1), validation_df.drop(pred_col_name, axis=1)

#DEFAULT--------------------------------------------------------------------------------------------------------------------
model_default = RandomForestRegressor(n_estimators=100, random_state=42)
model_default.fit(X_train, y_train)

# Make predictions
predictions_default = model_default.predict(X_valid)

mse_default = mean_squared_error(y_valid, predictions_default)
print(f"MSE model default: {mse_default:.4f}")

# ZSCORE--------------------------------------------------------------------------------------------------------------------
model_zscore = ZscoreRandomForestRegressor(n_estimators=100, random_state=42)
model_zscore.fit(X_train, y_train)

# Make predictions
predictions_zscore = model_zscore.predict(X_valid)

mse_zscore = mean_squared_error(y_valid, predictions_zscore)
print(f"MSE model Zscore: {mse_zscore:.4f}")

# IQR-----------------------------------------------------------------------------------------------------------------------
model_iqr = IQRRandomForestRegressor(n_estimators=100, random_state=42)
model_iqr.fit(X_train, y_train)

# Make predictions
predictions_iqr = model_iqr.predict(X_valid)

mse_iqr = mean_squared_error(y_valid, predictions_iqr)
print(f"MSE model IQR: {mse_iqr:.4f}")

# PERCENTILE TRIMMING------------------------------------------------------------------------------------------------------
model_trim = PercentileTrimmingRandomForestRegressor(n_estimators=100, random_state=42)
model_trim.fit(X_train, y_train)

# Make predictions
predictions_trim = model_trim.predict(X_valid)

mse_trim = mean_squared_error(y_valid, predictions_trim)
print(f"MSE model Percentile Trimming: {mse_trim:.4f}")

