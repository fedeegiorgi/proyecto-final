from sklearn.ensemble import ContinueTrainRandomForestRegressor
from sklearn.model_selection import train_test_split
import pandas as pd

SEED = 14208

df = pd.read_csv('datasets/salary_football.csv')

# Preprocesamiento de datos
train_df, validation_df = train_test_split(df, test_size=0.2, random_state=SEED)
y_train, y_valid = train_df['Wage'], validation_df['Wage']
X_train, X_valid = train_df.drop('Wage', axis=1), validation_df.drop('Wage', axis=1)
X_train = pd.get_dummies(X_train)
X_valid = pd.get_dummies(X_valid)
X_train, X_valid = X_train.align(X_valid, join='left', axis=1, fill_value=0)

# Inicialización del modelo
model = ContinueTrainRandomForestRegressor(initial_max_depth=5, random_state=SEED)
model.fit(X_train.values, y_train)