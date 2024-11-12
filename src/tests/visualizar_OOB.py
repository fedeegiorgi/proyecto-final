import numpy as np
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.io import arff


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

# convertir los dataFrames a NumPy arrays para sacar los nombres de las features
X_train = X_train.values 
X_valid = X_valid.values

#RandomForest con oob_score = True
rf = RandomForestRegressor(oob_score=True, n_estimators=100, random_state=SEED)
rf.fit(X_train, y_train)

n_samples = X_train.shape[0]

peso_arboles = []

# Recorrer cada árbol en el RandomForest
for i, tree in enumerate(rf.estimators_):
    
    oob_sample_mask = np.ones(n_samples, dtype=bool) #inicializo una mascara con 1's
    
    # asignamos false a las muestras que el arbol utilizo para entrenar, ya que no son OOB
    oob_sample_mask[rf.estimators_samples_[i]] = False 
    
    # obtener las observaciones OOB para este árbol
    oob_samples_X = X_train[oob_sample_mask]# solo se seleccionan las features de las observaciones que tienen valor True, las OOB observations
    oob_samples_y = y_train[oob_sample_mask] #solo se selecciona el valor objetivo de las observaciones que tienen valor True, las OOB observations
    
    # print(f"Árbol {i} - OOB observations (X): \n{X_train[oob_sample_mask]}") 
    # print(f"Árbol {i} - OOB observations (y): \n{oob_samples_y}")

    # predicciones del árbol sobre sus observaciones OOB
    oob_pred = tree.predict(oob_samples_X)
    
    # calcular el error MSE para estas predicciones
    mse_oob = mean_squared_error(oob_samples_y, oob_pred)
    peso = 1 / (mse_oob) #sumamos un epsilon por las moscas?
    
    # Añadir el peso a la lista de pesos
    peso_arboles.append(peso)

    #print(f"Árbol {i} - MSE sobre OOB samples: {mse_oob}, Peso sin normalizar: {peso}")

# normalizar los pesos para que sumen 1
peso_arboles = np.array(peso_arboles) #lo paso a un numpy array
suma = peso_arboles.sum()
print(suma)
peso_arboles /= peso_arboles.sum() # a cada peso lo divido por la suma de los pesos
print(peso_arboles.sum())

print(f"Suma de los pesos normalizados: {sum(peso_arboles)}")

for i, peso in enumerate(peso_arboles):
     print(f"Árbol {i} - Peso normalizado: {peso}")

#obtenemos todas las predicciones
all_predictions = np.zeros((X_valid.shape[0],), dtype=np.float64)

# sumamos las predicciones de cada arbol junto con su peso
for i, tree in enumerate(rf.estimators_):
    tree_prediction = tree.predict(X_valid)
    all_predictions += tree_prediction * peso_arboles[i]

# calculamos el MSE
mse_final = mean_squared_error(y_valid, all_predictions)

print(f"MSE de la prediccion final: {mse_final}")
