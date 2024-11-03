import os
import time
import pandas as pd
from tqdm import tqdm
from scipy.io import arff
from sklearn.ensemble import (
    RandomForestRegressor, IQRRandomForestRegressor, OOBRandomForestRegressor,
    OOBRandomForestRegressorGroups, OOBRandomForestRegressorGroupsSigmoid,
    OOBRandomForestRegressorGroupsTanh, OOBRandomForestRegressorGroupsSoftPlus, OOB_plus_IQR,
    RFRegressorFirstSplitCombiner
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

SEED = 14208
DATASETS_COLUMNS = { # Acá se agregan todos los datasets que están en la carpeta datasets
    'Height': 'childHeight',
    'salary_football': 'Wage',
    'boston_housing': 'MEDV',
    'Carbon_Emission_transformed': 'CarbonEmission',
} 
NEW_DATASET_NAME = 'Carbon_Emission_transformed.csv' # Nombre del nuevo dataset a evaluar
DATASETS_FOLDER = 'datasets/'
MODEL_CLASSES = [
    RandomForestRegressor, IQRRandomForestRegressor, OOBRandomForestRegressor,
    OOBRandomForestRegressorGroups, OOBRandomForestRegressorGroupsSigmoid,
    OOBRandomForestRegressorGroupsTanh, OOBRandomForestRegressorGroupsSoftPlus,
    OOB_plus_IQR, RFRegressorFirstSplitCombiner
]
NEW_CLASS = RFRegressorFirstSplitCombiner

# Decision variables
RUN_NEW_DATASET = True
RUN_NEW_ALGORITHM = False

def process_dataset(filepath, extension, dataset_name):
    """
    Load and preprocess a dataset from a file.

    Parameters:
        filepath (str): The path to the dataset file.
        extension (str): The file extension (e.g., '.arff', '.csv').
        dataset_name (str): The name of the dataset.

    Returns:
        tuple: A tuple containing the training and validation data.
    """
    # Carga de datos
    if extension == '.arff':
        data = arff.loadarff(filepath)
        df = pd.DataFrame(data[0])
    elif extension == '.csv':
        df = pd.read_csv(filepath)
    else:
        raise ValueError("Unsupported file type. Use 'arff' or 'csv'.")

    # Preprocesamiento de datos
    train_df, validation_df = train_test_split(df, test_size=0.2, random_state=SEED)
    y_train, y_valid = train_df[DATASETS_COLUMNS[dataset_name]], validation_df[DATASETS_COLUMNS[dataset_name]]
    X_train, X_valid = train_df.drop(DATASETS_COLUMNS[dataset_name], axis=1), validation_df.drop(DATASETS_COLUMNS[dataset_name], axis=1)
    X_train = pd.get_dummies(X_train)
    X_valid = pd.get_dummies(X_valid)
    X_train, X_valid = X_train.align(X_valid, join='left', axis=1, fill_value=0)

    return X_train, X_valid, y_train, y_valid

def evaluate_model(model_class, X_train, y_train, X_valid, y_valid, random_state=SEED):
    """
    Evaluates a regression model by training it and calculating metrics on validation data.
    
    Parameters:
        model_class (class): The model class to initialize and train (e.g., RandomForestRegressor).
        X_train (array-like): Training features.
        y_train (array-like): Training target.
        X_valid (array-like): Validation features.
        y_valid (array-like): Validation target.
        random_state (int): Seed for reproducibility.
    
    Returns:
        dict: A dictionary containing the model, MSE, R2 score, and training time.
    """
    # Initialize the model with the provided random_state
    model = model_class(random_state=random_state)
    
    # Measure the training time
    start_time = time.time()
    model.fit(X_train.values, y_train)
    end_time = time.time()
    
    # Generate predictions
    predictions = model.predict(X_valid.values)
    
    # Calculate metrics
    mse = mean_squared_error(y_valid, predictions)
    r2 = r2_score(y_valid, predictions)
    elapsed_time = end_time - start_time
    
    # Return the results
    return mse, r2, elapsed_time

def ensure_columns(results_df, dataset_name):
    """
    Ensure the DataFrame has the necessary columns for the dataset.
    
    Parameters:
        results_df (DataFrame): The DataFrame to update.
        dataset_name (str): The name of the dataset.
        
    Returns:
        DataFrame: The updated DataFrame.
    """
    for metric in ['MSE', 'R2', 'Time']:
        column_name = f"{dataset_name}_{metric}"
        if column_name not in results_df.columns:
            results_df[column_name] = None
    return results_df

def update_results_df(existing_df, new_results, dataset_name):
    """
    Update the DataFrame with new results.
    
    Parameters:
        existing_df (DataFrame): The existing DataFrame.
        new_results (dict): The new results to add.
        dataset_name (str): The name of the dataset.
        
    Returns:
        DataFrame: The updated DataFrame.
    """
    model_name = new_results.pop('Model')
    if model_name in existing_df['Model'].values:
        # Update the row for the existing model
        existing_df.loc[existing_df['Model'] == model_name, new_results.keys()] = new_results.values()
    else:
        # Add a new row for the new model
        new_row = pd.Series({'Model': model_name, **new_results})
        existing_df = pd.concat([existing_df, new_row.to_frame().T], ignore_index=True)

    return existing_df

if __name__ == "__main__":
    file_path = 'benchmarking.csv'
    if os.path.exists(file_path):
        results_df = pd.read_csv(file_path)
    else:
        results_df = pd.DataFrame(columns=['Model'])

    if RUN_NEW_ALGORITHM:
        # Evaluate the new algorithm across all datasets
        for filename in tqdm(os.listdir(DATASETS_FOLDER), desc="Processing datasets", unit="file"):
            filepath = os.path.join(DATASETS_FOLDER, filename)
            dataset_name, file_extension = os.path.splitext(filename)
            if file_extension not in ['.arff', '.csv']:
                continue

            # Ensure the DataFrame has the necessary columns for this dataset
            results_df = ensure_columns(results_df, dataset_name)

            # Process dataset and evaluate the model
            X_train, X_valid, y_train, y_valid = process_dataset(filepath, file_extension, dataset_name)
            mse, r2, elapsed_time = evaluate_model(NEW_CLASS, X_train, y_train, X_valid, y_valid)

            # Update results in the DataFrame
            new_results = {
                'Model': NEW_CLASS.__name__,
                f"{dataset_name}_MSE": mse,
                f"{dataset_name}_R2": r2,
                f"{dataset_name}_Time": elapsed_time
            }
            results_df = update_results_df(results_df, new_results, dataset_name)

    elif RUN_NEW_DATASET:
        # Evaluation of new dataset over all algorithms
        filepath = os.path.join(DATASETS_FOLDER, NEW_DATASET_NAME)
        dataset_name, file_extension = os.path.splitext(filepath)
        dataset_name = dataset_name.split('/')[-1]
        if file_extension not in [".arff", ".csv"]:
            raise ValueError("Unsupported file type. Use 'arff' or 'csv'.")
        else:
            # Ensure the DataFrame has the necessary columns for this dataset
            results_df = ensure_columns(results_df, dataset_name)

            X_train, X_valid, y_train, y_valid = process_dataset(filepath, file_extension, dataset_name)

            for model_class in tqdm(MODEL_CLASSES, desc=f"Evaluating {dataset_name}", unit="model"):
                
                mse, r2, elapsed_time = evaluate_model(model_class, X_train, y_train, X_valid, y_valid)
                
                new_results = {
                    'Model': model_class.__name__,
                    f"{dataset_name}_MSE": mse,
                    f"{dataset_name}_R2": r2,
                    f"{dataset_name}_Time": elapsed_time
                }
                
                results_df = update_results_df(results_df, new_results, dataset_name)
            
    else:
        # Evaluation of all datasets over all algorithms
        for filename in tqdm(os.listdir(DATASETS_FOLDER), desc="Processing datasets", unit="file"):
            filepath = os.path.join(DATASETS_FOLDER, filename)
            dataset_name, file_extension = os.path.splitext(filename)
            if file_extension not in [".arff", ".csv"]:
                continue
            else:
                # Ensure the DataFrame has the necessary columns for this dataset
                results_df = ensure_columns(results_df, dataset_name)

                X_train, X_valid, y_train, y_valid = process_dataset(filepath, file_extension, dataset_name)

                for model_class in tqdm(MODEL_CLASSES, desc=f"Evaluating {dataset_name}", unit="model", leave=False):

                    mse, r2, elapsed_time = evaluate_model(model_class, X_train, y_train, X_valid, y_valid)
                    
                    new_results = {
                        'Model': model_class.__name__,
                        f"{dataset_name}_MSE": mse,
                        f"{dataset_name}_R2": r2,
                        f"{dataset_name}_Time": elapsed_time
                    }
                    
                    results_df = update_results_df(results_df, new_results, dataset_name)

    results_df.to_csv(file_path, index=False)