import os
import time
import pandas as pd
from scipy.io import arff
from sklearn.ensemble import RandomForestRegressor, ZscoreRandomForestRegressor, IQRRandomForestRegressor, OOBRandomForestRegressor, OOBRandomForestRegressorGroups, OOBRandomForestRegressorGroupsSigmoid, OOBRandomForestRegressorGroupsTanh, OOBRandomForestRegressorGroupsSoftPlus
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

SEED = 14208
DATASETS_COLUMNS = {'Height': 'childHeight'}
DATASETS_FOLDER = 'datasets/'
SAVE_CSV = True

def process_dataset(filepath, extension, dataset_name):
    # Carga de datos
    if extension == '.arff':
        data = arff.loadarff(filepath)
        df = pd.DataFrame(data[0])
    elif extension == '.csv':
        df = pd.read_csv(filepath)
    else:
        raise ValueError("Tipo de archivo no soportado. Ingresar un archivo de extension 'arff' o 'csv'.")

    # Preprocesamiento de datos
    train_df, validation_df = train_test_split(df, test_size=0.2, random_state=SEED)
    y_train, y_valid = train_df[DATASETS_COLUMNS[dataset_name]], validation_df[DATASETS_COLUMNS[dataset_name]]
    X_train, X_valid = train_df.drop(DATASETS_COLUMNS[dataset_name], axis=1), validation_df.drop(DATASETS_COLUMNS[dataset_name], axis=1)
    X_train = pd.get_dummies(X_train)
    X_valid = pd.get_dummies(X_valid)
    X_train, X_valid = X_train.align(X_valid, join='left', axis=1, fill_value=0)

    # Evaluacion modelo default
    default_model = RandomForestRegressor(random_state=SEED)
    default_start_time = time.time()
    default_model.fit(X_train, y_train)
    default_end_time = time.time()
    default_predictions = default_model.predict(X_valid)
    default_mse = mean_squared_error(y_valid, default_predictions)
    default_r2 = r2_score(y_valid, default_predictions)
    default_time = default_end_time - default_start_time

    # Evaluacion modelo alternativa A [IQR]
    iqr_model = IQRRandomForestRegressor(random_state=SEED)
    iqr_start_time = time.time()
    iqr_model.fit(X_train, y_train)
    iqr_end_time = time.time()
    iqr_predictions = iqr_model.predict(X_valid)
    iqr_mse = mean_squared_error(y_valid, iqr_predictions)
    iqr_r2 = r2_score(y_valid, iqr_predictions)
    iqr_time = iqr_end_time - iqr_start_time

    # Evaluacion modelo alternativa B [OOB]
    oob_model = OOBRandomForestRegressor(random_state=SEED)
    oob_start_time = time.time()
    oob_model.fit(X_train.values, y_train)
    oob_end_time = time.time()
    oob_predictions = oob_model.predict(X_valid.values)
    oob_mse = mean_squared_error(y_valid, oob_predictions)
    oob_r2 = r2_score(y_valid, oob_predictions)
    oob_time = oob_end_time - oob_start_time

    # Evaluacion modelo alternativa B [OOBGroups]
    oobgroup_model = OOBRandomForestRegressorGroups(group_size=10, n_estimators=100, random_state=SEED)
    oobgroup_start_time = time.time()
    oobgroup_model.fit(X_train.values, y_train)
    oobgroup_end_time = time.time()
    oobgroup_predictions = oobgroup_model.predict(X_valid.values)
    oobgroup_mse = mean_squared_error(y_valid, oobgroup_predictions)
    oobgroup_r2 = r2_score(y_valid, oobgroup_predictions)
    oobgroup_time = oobgroup_end_time - oobgroup_start_time

    # Evaluacion modelo alternativa B [OOBGroups con suavización Sigmoidea]
    oob_s_model = OOBRandomForestRegressorGroupsSigmoid(random_state=SEED)
    oob_s_start_time = time.time()
    oob_s_model.fit(X_train.values, y_train)
    oob_s_end_time = time.time()
    oob_s_predictions = oob_s_model.predict(X_valid.values)
    oob_s_mse = mean_squared_error(y_valid, oob_s_predictions)
    oob_s_r2 = r2_score(y_valid, oob_s_predictions)
    oob_s_time = oob_s_end_time - oob_s_start_time

    # Evaluacion modelo alternativa B [OOBGroups con función Tangente hiperbólica]
    oob_t_model = OOBRandomForestRegressorGroupsTanh(random_state=SEED)
    oob_t_start_time = time.time()
    oob_t_model.fit(X_train.values, y_train)
    oob_t_end_time = time.time()
    oob_t_predictions = oob_t_model.predict(X_valid.values)
    oob_t_mse = mean_squared_error(y_valid, oob_t_predictions)
    oob_t_r2 = r2_score(y_valid, oob_t_predictions)
    oob_t_time = oob_t_end_time - oob_t_start_time

    # Evaluacion modelo alternativa B [OOBGroups con función SoftPlus]
    oob_sp_model = OOBRandomForestRegressorGroupsSoftPlus(random_state=SEED)
    oob_sp_start_time = time.time()
    oob_sp_model.fit(X_train.values, y_train)
    oob_sp_end_time = time.time()
    oob_sp_predictions = oob_sp_model.predict(X_valid.values)
    oob_sp_mse = mean_squared_error(y_valid, oob_sp_predictions)
    oob_sp_r2 = r2_score(y_valid, oob_sp_predictions)
    oob_sp_time = oob_sp_end_time - oob_sp_start_time


    return default_mse, default_r2, default_time, iqr_mse, iqr_r2, iqr_time, oob_mse, oob_r2, oob_time, oobgroup_mse, oobgroup_r2, oobgroup_time, oob_s_mse, oob_s_r2, oob_s_time, oob_t_mse, oob_t_r2, oob_t_time, oob_sp_mse, oob_sp_r2, oob_sp_time

# Inicializa un DataFrame vacío
results_df = pd.DataFrame()

for filename in os.listdir(DATASETS_FOLDER):
    filepath = os.path.join(DATASETS_FOLDER, filename)
    dataset_name, file_extension = os.path.splitext(filename)
    if file_extension not in [".arff", ".csv"]:
        continue
    else:
        default_mse, default_r2, default_time, iqr_mse, iqr_r2, iqr_time, oob_mse, oob_r2, oob_time, oobgroup_mse, oobgroup_r2, oobgroup_time, oob_s_mse, oob_s_r2, oob_s_time, oob_t_mse, oob_t_r2, oob_t_time, oob_sp_mse, oob_sp_r2, oob_sp_time= process_dataset(filepath=filepath, extension=file_extension, dataset_name=dataset_name)

        # Crea un DataFrame temporal con los resultados del dataset actual
        df_temp = pd.DataFrame({
            'Metric': ['Default MSE', 'Default R2', 'Default Time', 'IQR MSE', 'IQR R2', 'IQR Time', 'OOB MSE', 'OOB R2', 'OOB Time', 'OOBGroups MSE', 'OOBGroups R2', 'OOBGroups Time',
                       'OOBGroups Sigmoid MSE', 'OOBGroups Sigmoid R2', 'OOBGroups Sigmoid Time', 'OOBGroups Tanh MSE', 'OOBGroups Tanh R2', 'OOBGroups Tanh Time',
                       'OOBGroups SoftPlus MSE', 'OOBGroups SoftPlus R2', 'OOBGroups SoftPlus Time'],
            dataset_name: [default_mse, default_r2, default_time, iqr_mse, iqr_r2, iqr_time, oob_mse, oob_r2, oob_time, oobgroup_mse, oobgroup_r2, oobgroup_time,
                           oob_s_mse, oob_s_r2, oob_s_time, oob_t_mse, oob_t_r2, oob_t_time, oob_sp_mse, oob_sp_r2, oob_sp_time]
        })
        
        # Une los resultados del dataset actual al DataFrame principal
        results_df = pd.merge(results_df, df_temp, on='Metric', how='outer') if not results_df.empty else df_temp

if SAVE_CSV:
    results_df.to_csv("benchmarking.csv", index=False)
