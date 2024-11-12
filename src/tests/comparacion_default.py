from sklearn.ensemble import RandomForestRegressor, IQRRandomForestRegressor, OOBRandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd

SEED = 14208

df = pd.read_csv('distribucion/datasets/train_data/Carbon_Emission_train.csv')

# Preprocesamiento de datos
train_df, validation_df = train_test_split(df, test_size=0.2, random_state=SEED)
y_train, y_valid = train_df['CarbonEmission'], validation_df['CarbonEmission']
X_train, X_valid = train_df.drop('CarbonEmission', axis=1), validation_df.drop('CarbonEmission', axis=1)
X_train = pd.get_dummies(X_train)
X_valid = pd.get_dummies(X_valid)
X_train, X_valid = X_train.align(X_valid, join='left', axis=1, fill_value=0)

# Inicialización del modelo
model_default = RandomForestRegressor(random_state=SEED)
model_default.fit(X_train.values, y_train)

predictions_default = model_default.predict(X_valid.values)
mse_default = mean_squared_error(y_valid, predictions_default)
print(f"MSE RF Defult:", mse_default)

# Inicialización del modelo
model_IQR = IQRRandomForestRegressor(n_estimators = 150, group_size = 3, max_depth = 17, random_state=SEED)
model_IQR.fit(X_train.values, y_train)

predictions_iqr = model_IQR.predict(X_valid.values)
mse_iqr = mean_squared_error(y_valid, predictions_iqr)
print(f"MSE IQR:", mse_iqr)

# Inicialización del modelo
model_OOB = OOBRandomForestRegressor(n_estimators = 160, group_size = 5, max_depth = 17, random_state=SEED)
model_OOB.fit(X_train.values, y_train)

predictions_oob = model_OOB.predict(X_valid.values)
mse_OOB = mean_squared_error(y_valid, predictions_oob)
print(f"MSE OOB:", mse_OOB)

if mse_OOB < mse_default:
    print('fui mejor que el default')