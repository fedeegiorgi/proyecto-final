import os
import time
import pandas as pd
from scipy.io import arff
from sklearn.ensemble import RandomForestRegressor, ZscoreRandomForestRegressor, IQRRandomForestRegressor, OOBRandomForestRegressor, NewValRandomForestRegressor
#, IntersectionOOBRandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

SEED = 14208
DATASETS_COLUMNS = {'titanic_fare_test': 'Fare'}
DATASETS_FOLDER = '/home/marustina/Documents/2024_2S/TD8/proyecto-final/src/datasets'
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

    # Evaluacion modelo alternativa A [Z-score]
    # z_score_model = ZscoreRandomForestRegressor(random_state=SEED)
    # z_score_start_time = time.time()
    # z_score_model.fit(X_train, y_train)
    # z_score_end_time = time.time()
    # z_score_predictions = z_score_model.predict(X_valid)
    # z_score_mse = mean_squared_error(y_valid, z_score_predictions)
    # z_score_r2 = r2_score(y_valid, z_score_predictions)
    # z_score_time = z_score_end_time - z_score_start_time

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
    oob_model.fit(X_train, y_train)
    oob_end_time = time.time()
    oob_predictions = oob_model.predict(X_valid)
    oob_mse = mean_squared_error(y_valid, oob_predictions)
    oob_r2 = r2_score(y_valid, oob_predictions)
    oob_time = oob_end_time - oob_start_time

    # Evaluacion modelo alternativa B [New Validation]
    # new_val_model = NewValRandomForestRegressor(random_state=SEED)
    # new_val_start_time = time.time()
    # new_val_model.fit(X_train, y_train)
    # new_val_end_time = time.time()
    # new_val_predictions = new_val_model.predict(X_valid)
    # new_val_mse = mean_squared_error(y_valid, new_val_predictions)
    # new_val_r2 = r2_score(y_valid, new_val_predictions)
    # new_val_time = new_val_end_time - new_val_start_time

    # Evaluacion modelo alternativa B [Intersection OOB]
    # intOOB_model = IntersectionOOBRandomForestRegressor(random_state=SEED)
    # intOOB_start_time = time.time()
    # intOOB_model.fit(X_train, y_train)
    # intOOB_end_time = time.time()
    # intOOB_predictions = intOOB_model.predict(X_valid)
    # intOOB_mse = mean_squared_error(y_valid, intOOB_predictions)
    # intOOB_r2 = r2_score(y_valid, intOOB_predictions)
    # intOOB_time = intOOB_end_time - intOOB_start_time

    return default_mse, default_r2, default_time, iqr_mse, iqr_r2, iqr_time, oob_mse, oob_r2, oob_time
#, new_val_mse, new_val_r2, new_val_time
#, intOOB_mse, intOOB_r2, intOOB_time

for filename in os.listdir(DATASETS_FOLDER):
    filepath = os.path.join(DATASETS_FOLDER, filename)
    dataset_name, file_extension = os.path.splitext(filename)
    if file_extension not in [".arff", ".csv"]:
        next
    else:
        default_mse, default_r2, default_time, iqr_mse, iqr_r2, iqr_time, oob_mse, oob_r2, oob_time = process_dataset(filepath=filepath, extension=file_extension, dataset_name=dataset_name)

    df_for_csf = {filename: {'Default MSE': default_mse, "Default R2": default_r2, "Default Time": default_time,
                             'IQR MSE': iqr_mse, "IQR R2": iqr_r2, "IQR Time": iqr_time,
                             'OOB MSE': oob_mse, "OOB R2": oob_r2, "OOB Time": oob_time
                             }}
    df_for_csf = pd.DataFrame(df_for_csf)

    if SAVE_CSV:
        df_for_csf.to_csv("benchmarking.csv")