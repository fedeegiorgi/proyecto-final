from sklearn.model_selection import GridSearchCV, train_test_split
from tqdm import tqdm
from sklearn.ensemble import (
    RandomForestRegressor, IQRRandomForestRegressor, OOBRandomForestRegressor,
    OOBRandomForestRegressorGroups, OOBRandomForestRegressorGroupsSigmoid,
    OOBRandomForestRegressorGroupsTanh, OOBRandomForestRegressorGroupsSoftPlus, OOB_plus_IQR,
    RFRegressorFirstSplitCombiner
)
from sklearn.metrics import mean_squared_error
import pandas as pd
from scipy.io import arff
import os
from itertools import product

SEED = 14208
results = []

# Mapping dataset names to target columns
DATASETS_COLUMNS = {
    'Carbon_Emission': 'CarbonEmission',
    'Laptop': 'Price', 
    'Wind': 'WIND',
}

# Select from the terminal which dataset to use:
print("Select the dataset you would like to use:")
print("1: Carbon Emission")
print("2: Laptop")
print("3: Wind")

choice = input("Enter the number corresponding to your choice: ")

# Set the file path and dataset name based on the choice
if choice == "1":
    file_path = 'distribucion/datasets/train_data/Carbon_Emission_train.csv'
    dataset_name = 'Carbon_Emission'
elif choice == "2":
    file_path = 'distribucion/datasets/train_data/laptop_train.csv'
    dataset_name = 'Laptop'
elif choice == "3":
    file_path = 'distribucion/datasets/train_data/wind_train.csv'
    dataset_name = 'Wind'
else:
    print("Invalid choice. Please select 1, 2, or 3.")
    file_path = None

# Check the file extension and load data accordingly
if file_path:
    if file_path.endswith('.csv'):
        data = pd.read_csv(file_path)
    elif file_path.endswith('.arff'):
        data, meta = arff.loadarff(file_path)
        data = pd.DataFrame(data)
    else:
        raise ValueError("Unsupported file format. Please provide a .csv or .arff file.")
else: 
    print("No valid dataset selected.")
    exit()

# Preprocess data
target_column = DATASETS_COLUMNS[dataset_name]
train_df, validation_df = train_test_split(data, test_size=0.2, random_state=SEED)

y_train, y_valid = train_df[target_column], validation_df[target_column]
X_train, X_valid = train_df.drop(target_column, axis=1), validation_df.drop(target_column, axis=1)

# Encode categorical features
X_train = pd.get_dummies(X_train)
X_valid = pd.get_dummies(X_valid)

# Align train and validation datasets
X_train, X_valid = X_train.align(X_valid, join='left', axis=1, fill_value=0)


# Ready for model training and grid search
print("Data preprocessing complete. Ready for Hyperparameter Tuning.")

# Define Parameter Grids
param_grids = {
    "1": {
        'model': IQRRandomForestRegressor(),
        'param_grid': {
            'n_estimators': [30, 50, 100, 200, 300, 400, 500], 
            'group_size':  [3, 5, 10, 20, 30, 40, 50], 
            'max_depth': [None, 10, 20]},
        'name': "IQR"
    },
    "2": {
        'model': OOBRandomForestRegressorGroups(),
        'param_grid': {
            'n_estimators': [30, 50, 100, 200], 
            'group_size': [3, 5, 10], 
            'max_depth': [None, 10, 20]},
        'name': "OOB"
    },
    "3": {
        'model': OOB_plus_IQR(),
        'param_grid': {
            'n_estimators': [30, 50, 100, 200], 
            'group_size': [3, 5, 10], 
            'max_depth': [None, 10, 20]},
        'name': "OOB + IQR"
    },
    "4": {
        'model': RFRegressorFirstSplitCombiner(),
        'param_grid': {
            'n_estimators': [30, 50, 100, 200], 
            'group_size': [3, 5, 10], 
            'max_depth': [None, 10, 20]},
        'name': "FirstSplit"
    },
    # "5": {
    #     'model': SharedKnowledgeRandomForestRegressor(),
    #     'param_grid': {
    #         'n_estimators': [30, 50, 100, 200], 
    #         'group_size': [3, 5, 10], 
    #         'max_depth': [None, 10, 20], 
    #         'initial_max_depth': [5, 10, 15, 20, 25, 30]},
    #     'name': "SharedKnowledge"
    # },
}

# Prompt to select the model(s)
print("Select the model/models you would like to optimize (comma-separated if choosing multiple):")
print("1: IQR")
print("2: OOB")
print("3: OOB + IQR")
print("4: FirstSplit")

choices = input("Enter the numbers corresponding to your choice(s): ").split(',')

for choice in tqdm(choices):
    choice = choice.strip()
    if choice not in param_grids:
        print(f"Invalid choice: {choice}. Skipping.")
        continue

    config = param_grids[choice]
    model, param_grid, model_name = config['model'], config['param_grid'], config['name']
    
    print(f"\nRunning custom grid search for model: {model_name}")

    # Generate filtered parameter combinations
    parameters = []

    if model_name == 'SharedKnowledge':
        for i in param_grid['n_estimators']:         
            for j in param_grid['group_size']:  
                for k in param_grid['max_depth']:
                    for z in param_grid['initial_max_depth']:
                        if i % j == 0 and i > j:  # Ensures n_estimators is a multiple of group_size and that group_size < n_estimators
                            if k > z:
                                parameters.append((i, j, k, z)) 
        
        # Manual grid search
        best_mse, best_params = float('inf'), None
        for n_estimators, group_size, max_depth, initial_max_depth in tqdm(parameters):
            model_instance = model.__class__(n_estimators=n_estimators, group_size=group_size, max_depth=max_depth, initial_max_depth = initial_max_depth, random_state=SEED)
            model_instance.fit(X_train.values, y_train)
            
            y_pred = model_instance.predict(X_valid)
            mse = mean_squared_error(y_valid, y_pred)
            
            if mse < best_mse:
                best_mse = mse
                best_params = {'n_estimators': n_estimators, 'group_size': group_size, 'max_depth': max_depth, 'initial_max_depth': initial_max_depth}
        
    else: 
        for i in param_grid['n_estimators']:         
            for j in param_grid['group_size']:  
                for k in param_grid['max_depth']:
                    if i % j == 0 and i > j:  # Ensures n_estimators is a multiple of group_size and that group_size < n_estimators
                        parameters.append((i, j, k)) 

        # Manual grid search
        best_mse, best_params = float('inf'), None
        for n_estimators, group_size, max_depth in tqdm(parameters):
            model_instance = model.__class__(n_estimators=n_estimators, group_size=group_size, max_depth=max_depth, random_state=SEED)
            model_instance.fit(X_train.values, y_train)
            
            y_pred = model_instance.predict(X_valid.values)
            mse = mean_squared_error(y_valid, y_pred)
            
            if mse < best_mse:
                best_mse = mse
                best_params = {'n_estimators': n_estimators, 'group_size': group_size, 'max_depth': max_depth}

    print(f"Best Parameters for {model_name}: {best_params}")
    print(f"Validation MSE for {model_name}: {best_mse}")

    results.append({
        'Model': model_name,
        'Best Parameters': best_params,
        'Validation MSE': best_mse
    })

# Display all results
print("\nAll Results:")
for result in results:
    print(result)