from sklearn.ensemble import CustomRandomForestRegressor
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.io import arff
import os
from scipy.stats import lognorm
from scipy.stats import norm

SEED = 21415
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

# Fit the model
model = CustomRandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_valid)

mse = mean_squared_error(y_valid, predictions)
print(f"Mean Squared Error of the model: {mse:.4f}")

