import os
import time
import pandas as pd
from scipy.io import arff
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

SEED = 21415
DATASETS_COLUMNS = {'titanic_fare_test': 'Fare'}
DATASETS_FOLDER = 'src/datasets/'

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

    # Evaluacion modelo alternativa A
    # z_score_model = ZScoreExcluderRFRegressor(random_state=SEED)
    z_score_model = RandomForestRegressor(random_state=SEED)
    z_score_start_time = time.time()
    z_score_model.fit(X_train, y_train)
    z_score_end_time = time.time()
    z_score_predictions = z_score_model.predict(X_valid)
    z_score_mse = mean_squared_error(y_valid, z_score_predictions)
    z_score_r2 = r2_score(y_valid, z_score_predictions)
    z_score_time = z_score_end_time - z_score_start_time

    return default_mse, default_r2, default_time, z_score_mse, z_score_r2, z_score_time


for filename in os.listdir(DATASETS_FOLDER):
    filepath = os.path.join(DATASETS_FOLDER, filename)
    dataset_name, file_extension = os.path.splitext(filename)
    if file_extension not in [".arrf", ".csv"]:
        next
    else:
        default_mse, default_r2, default_time, z_score_mse, z_score_r2, z_score_time = process_dataset(filepath, file_extension, filename)

    df_for_csf = {filename: {'Default MSE': default_mse, "Default R2": default_r2, "Default Time": default_time,
                             'Z-Score MSE': z_score_mse, "Z-Score R2": z_score_r2, "Z-Score Time": z_score_time}}
    df_for_csf = pd.DataFrame(df_for_csf)
    df_for_csf.to_csv("benchmarking.csv")
