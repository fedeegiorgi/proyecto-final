from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
print("Importe lo q no agregamos")
from sklearn.tree import DecisionTreeRegressorCombiner


print("Logré importar todo")

SEED = 14208

df = pd.read_csv('distribucion/datasets/salary_football.csv')

# Preprocesamiento de datos
train_df, validation_df = train_test_split(df, test_size=0.2, random_state=SEED)
y_train, y_valid = train_df['Wage'], validation_df['Wage']
X_train, X_valid = train_df.drop('Wage', axis=1), validation_df.drop('Wage', axis=1)
X_train = pd.get_dummies(X_train)
X_valid = pd.get_dummies(X_valid)
X_train, X_valid = X_train.align(X_valid, join='left', axis=1, fill_value=0)

# Inicialización del modelo
rf = RandomForestRegressor(random_state=SEED, n_estimators=20, max_depth=1)
rf.fit(X_train, y_train)

trees = rf.estimators_
samples_used = rf.estimators_samples_

print("Ya cree los arboles con RF")

samplesidx = set()
for samples in samples_used:
    for sampleidx in samples:
        samplesidx.add(sampleidx)
samplesidx = list(samplesidx)

X_union = X_train.iloc[samplesidx].to_numpy()
y_union = y_train.iloc[samplesidx].to_numpy()

print("Llamo al decision combiner... reza malena")
combinador = DecisionTreeRegressorCombiner(initial_trees=trees)
combinador.fit(X_union, y_union)
