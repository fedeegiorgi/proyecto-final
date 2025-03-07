from sklearn.model_selection import train_test_split
from tqdm import tqdm

from sklearn.ensemble import (
    RandomForestRegressor, IQRRandomForestRegressor, PercentileTrimmingRandomForestRegressor, 
    OOBRandomForestRegressor, OOBPlusIQRRandomForestRegressor,
    FirstSplitCombinerRandomForestRegressor, SharedKnowledgeRandomForestRegressor)

from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
from scipy.io import arff
import random
import csv

SEED = 14208
sampled_in_blocks = False
results = []
models_results = {}

# Function to get the top 50 parameter combinations for a model
def get_top_50_params(model_name):
    # Load the top 50 parameter combinations
    file_name = f"results_analysis/csvs/{model_name}_Top_150.csv"
    f = open(file_name, 'r')
    reader = csv.reader(f)
    next(reader)  # Skip the header row
    params = []
    for row in reader:
        tup = []
        for val in row:
            try:
                val = float(val)
                if val.is_integer():
                    val = int(val)
            except ValueError:
                val = None
            tup.append(val)
        
        params.append(tuple(tup))
    
    return params

# Mapping dataset names to target columns
DATASETS_COLUMNS = {
    'Carbon_Emission': 'CarbonEmission',
    'Wind': 'WIND',
    'House_8L': 'price',
    'Flight': 'Price',
    'Rainfall': 'Rainfall',
    'Abalone': 'Rings'
}

# Select from the terminal which dataset to use:
print("Select the dataset you would like to use:")
print("1: Carbon Emission")
print("2: House_8L")
print("3: Wind")
print("4: Flight")
print("5: Rainfall")
print("6: Abalone")

dataset_choice = input("Enter the number corresponding to your dataset_choice: ")

# Set the file path and dataset name based on the dataset_choice
if dataset_choice == "1":
    file_path = 'datasets/train_data/Carbon_Emission_train.csv'
    dataset_name = 'Carbon_Emission'
elif dataset_choice == "2":
    file_path = 'datasets/train_data/house_8L_train_7000.csv'
    dataset_name = 'House_8L'
elif dataset_choice == "3":
    file_path = 'datasets/train_data/wind_train.csv'
    dataset_name = 'Wind'
elif dataset_choice == "4":
    file_path = 'datasets/train_data/flight_train.csv'
    dataset_name = 'Flight'
elif dataset_choice == "5":
    file_path = 'datasets/train_data/rainfall_train.csv'
    dataset_name = 'Rainfall'
elif dataset_choice == "6":
    file_path = 'datasets/train_data/abalone_train.csv'
    dataset_name = 'Abalone'

else:
    print("Invalid dataset_choice. Please select 1, 2, 3, 4, 5, o 6.")
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
print("Data preprocessing complete. Ready for Grid Search.")


# Define Parameter Grids
param_grids = {
    "1": {
        'model': IQRRandomForestRegressor(),
        'param_grid': {
            'n_estimators': list(range(50, 301, 10)) + list(range(350, 1001, 50)) + [1250, 1500], 
            'group_size': list(range(3, 11, 1)) + list(range(15, 51, 5)), 
            'max_depth': list(range(10, 51, 1))
        },
        'name': "IQR"
    },
    "2": {
        'model': PercentileTrimmingRandomForestRegressor(),
        'param_grid': {
            'n_estimators': list(range(50, 301, 10)) + list(range(350, 1001, 50)) + [1250, 1500], 
            'group_size': list(range(3, 11, 1)) + list(range(15, 51, 5)), 
            'max_depth': list(range(10, 51, 1)),
            'percentile': list(range(1, 16, 1))
        }, 
        'name': "Percentile_Trimming"
    },
    "3": {
        'model': OOBRandomForestRegressor(),
        'param_grid': {
            'n_estimators': list(range(50, 301, 10)) + list(range(350, 1001, 50)) + [1250, 1500], 
            'group_size': list(range(3, 11, 1)) + list(range(15, 51, 5)), 
            'max_depth': list(range(10, 51, 1))
        },
        'name': "OOB"
    },
    "4": {
        'model': OOBPlusIQRRandomForestRegressor(),
        'param_grid': {
            'n_estimators': list(range(50, 301, 10)) + list(range(350, 1001, 50)) + [1250, 1500], 
            'group_size': list(range(3, 11, 1)) + list(range(15, 51, 5)), 
            'max_depth': list(range(10, 51, 1))
        },
        'name': "OOB_plus_IQR"
    },
    "5": {
        'model': FirstSplitCombinerRandomForestRegressor(),
        'param_grid': {
            'n_estimators': list(range(50, 301, 10)) + list(range(350, 1001, 50)) + [1250, 1500],
            'group_size': list(range(3, 20, 1)),
            'max_features': [-1, -2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8],
        },
        'name': "First_Splits_Combiner"
    },
    "6": {
        'model': SharedKnowledgeRandomForestRegressor(),
        'param_grid': {
            'n_estimators': [280], 
            'group_size': [7],
            'max_depth': list(range(20, 41, 1)) + [None],
            'initial_max_depth': [14]
        },
        'name': "Shared_Knowledge"
    },
    "7": {
        'model': RandomForestRegressor(),
        'param_grid': {
            'max_depth': list(range(10, 51, 1))
        },
        'name': "Random_Forest"
    }
}

# Prompt to select the model(s)
print("Select the model/models you would like to optimize (comma-separated if choosing multiple):")
print("1: IQR")
print("2: Percentile Trimming")
print("3: OOB")
print("4: OOB + IQR")
print("5: First Splits Combiner")
print("6: Shared Knowledge")
print("7: Classic Random Forest")

choices = input("Enter the numbers corresponding to your choice(s): ").split(',')

# Perform search or top_50
print("Would you like to perform a grid search or use the top 50 parameter combinations?")
print("1. Grid Search")
print("2. Top 50 Parameter Combinations")

search_choice = int(input("Enter the number corresponding to your choice: "))

for choice in tqdm(choices):
    choice = choice.strip()
    if choice not in param_grids:
        print(f"Invalid choice: {choice}. Skipping.")
        exit()

    config = param_grids[choice]
    model, param_grid, model_name = config['model'], config['param_grid'], config['name']

    # Generate filtered parameter combinations and sampled parameters combinations
    parameters = []
    sampled_params = []

    # List to hold MSE values for each parameter combination
    mse_list = []

    if model_name == 'Shared_Knowledge':
        # Manual grid search
        best_mse, best_params = float('inf'), None

        if search_choice == 1:
            for i in param_grid['n_estimators']:         
                for j in param_grid['group_size']:  
                    for k in param_grid['max_depth']:
                        for z in param_grid['initial_max_depth']:
                            if i % j == 0 and i > j:  # Ensures n_estimators is a multiple of group_size and that group_size < n_estimators
                                if not k or k > z: # Ensures max_depth > initial_max_depth
                                    parameters.append((i, j, k, z)) 
            
            max_combinations = len(parameters)

            if dataset_choice != '1':
                # Input del usuario para muestreo de combinaciones
                try:
                    n = int(input(f"\nEnter the number of parameter combinations you want to sample from the {max_combinations} combinations: "))
                    
                    if n > max_combinations: 
                        print(f"The number of parameter combinations must be less than or equal to {max_combinations}.")
                        exit()
                
                except ValueError:
                    print("Please enter a valid integer for the number of combinations.")
                    exit()
                
                sampled_params = random.sample(parameters, n)
            else:
                sampled_blocks = []
                sampled_in_blocks = True
                random.seed(SEED)

                for block_num in range(6):
                # Sample 300 unique parameter combinations
                    sampled_block = random.sample(parameters, 250)
                    sampled_blocks.append(sampled_block)
                
                    # Remove the sampled items from the original parameters to avoid duplicates
                    parameters = [param for param in parameters if param not in sampled_block]

                block_selection = int(input(f"\nEnter the block number (0, 1, 2, 3, 4, 5): "))
                sampled_params = sampled_blocks[block_selection]       
        elif search_choice == 2:
            sampled_params = get_top_50_params(model_name)
        else:
            print("Invalid choice. Please select 1 or 2.")
            exit()

        print(f"\nRunning grid search for model: {model_name}")
        
        for n_estimators, group_size, max_depth, initial_max_depth in tqdm(sampled_params):
            model_instance = model.__class__(n_estimators=n_estimators, group_size=group_size, max_depth=max_depth, initial_max_depth = initial_max_depth, random_state=SEED)
            model_instance.fit(X_train.values, y_train.values)
            
            y_pred = model_instance.predict(X_valid.values)
            mse = mean_squared_error(y_valid, y_pred)
            mse_list.append(mse)
            
            if mse < best_mse:
                best_mse = mse
                best_params = {'n_estimators': n_estimators, 'group_size': group_size, 'max_depth': max_depth, 'initial_max_depth': initial_max_depth}
    
    elif model_name == 'Percentile_Trimming':
        # Manual grid search
        best_mse, best_params = float('inf'), None

        if search_choice == 1:
            for i in param_grid['n_estimators']:         
                for j in param_grid['group_size']:  
                    for k in param_grid['max_depth']:
                        for z in param_grid['percentile']:
                            if i % j == 0 and i > j:  # Ensures n_estimators is a multiple of group_size and that group_size < n_estimators
                                if z < 50: # Ensures percentile isnt above a certain value
                                    parameters.append((i, j, k, z)) 
            
            max_combinations = len(parameters)

            # Input del usuario para muestreo de combinaciones
            try:
                n = int(input(f"\nEnter the number of parameter combinations you want to sample from the {max_combinations} combinations: "))
                
                if n > max_combinations: 
                    print(f"The number of parameter combinations must be less than or equal to {max_combinations}.")
                    exit()
            
            except ValueError:
                print("Please enter a valid integer for the number of combinations.")
                exit()
            
            sampled_params = random.sample(parameters, n)
        elif search_choice == 2:
            sampled_params = get_top_50_params(model_name)
        else:
            print("Invalid choice. Please select 1 or 2.")
            exit()

        print(f"\nRunning grid search for model: {model_name}")

        for n_estimators, group_size, max_depth, percentile in tqdm(sampled_params):
            model_instance = model.__class__(n_estimators=n_estimators, group_size=group_size, max_depth=max_depth, percentile=percentile, random_state=SEED, n_jobs=3)
            model_instance.fit(X_train.values, y_train)
            
            y_pred = model_instance.predict(X_valid.values)
            mse = mean_squared_error(y_valid, y_pred)
            mse_list.append(mse)
            
            if mse < best_mse:
                best_mse = mse
                best_params = {'n_estimators': n_estimators, 'group_size': group_size, 'max_depth': max_depth, 'percentile': percentile}
    
    elif model_name == 'First_Splits_Combiner':
        # Manual grid search
        best_mse, best_params = float('inf'), None

        if search_choice == 1:
            for i in param_grid['n_estimators']:         
                for j in param_grid['group_size']:
                    for k in param_grid['max_features']: 
                        if i % j == 0 and i > j:  # Ensures n_estimators is a multiple of group_size and that group_size < n_estimators
                            parameters.append((i, j, k))
            
            max_combinations = len(parameters)

            # Input del usuario para muestreo de combinaciones
            try:
                n = int(input(f"\nEnter the number of parameter combinations you want to sample from the {max_combinations} combinations: "))
                
                if n > max_combinations: 
                    print(f"The number of parameter combinations must be less than or equal to {max_combinations}.")
                    exit()
            
            except ValueError:
                print("Please enter a valid integer for the number of combinations.")
                exit()
            
            sampled_params = random.sample(parameters, n)
        elif search_choice == 2:
            sampled_params = get_top_50_params(model_name)
        else:
            print("Invalid choice. Please select 1 or 2.")
            exit()

        print(f"\nRunning grid search for model: {model_name}")

        for n_estimators, group_size, max_features in tqdm(sampled_params):
            if max_features == -1:
                max_features = 'sqrt'
            elif max_features == -2:
                max_features = 'log2'
                
            model_instance = model.__class__(n_estimators=n_estimators, group_size=group_size, max_features=max_features, random_state=SEED)
            model_instance.fit(X_train.values, y_train)
            
            y_pred = model_instance.predict(X_valid.values)
            mse = mean_squared_error(y_valid, y_pred)
            mse_list.append(mse)
            
            if mse < best_mse:
                best_mse = mse
                best_params = {'n_estimators': n_estimators, 'group_size': group_size, 'max_features': max_features}
    
    elif model_name == 'Random_Forest':
        # Manual grid search
        best_mse, best_params = float('inf'), None

        if search_choice == 1:
            for i in param_grid['max_depth']:
                parameters.append(i)
            
            max_combinations = len(parameters)

            # Input del usuario para muestreo de combinaciones
            try:
                n = int(input(f"\nEnter the number of parameter combinations you want to sample from the {max_combinations} combinations: "))
                
                if n > max_combinations: 
                    print(f"The number of parameter combinations must be less than or equal to {max_combinations}.")
                    exit()
            
            except ValueError:
                print("Please enter a valid integer for the number of combinations.")
                exit()
            
            sampled_params = random.sample(parameters, n)
        else:
            print("Invalid choice. Classic random only grid search.")
            exit()

        print(f"\nRunning grid search for model: {model_name}")
        print(sampled_params)

        for max_depth in tqdm(sampled_params):
            model_instance = model.__class__(max_depth=max_depth, random_state=SEED)
            model_instance.fit(X_train.values, y_train)
            
            y_pred = model_instance.predict(X_valid.values)
            mse = mean_squared_error(y_valid, y_pred)
            mse_list.append(mse)
             
            if mse < best_mse:
                best_mse = mse
                best_params = {'max_depth': max_depth}

    else: 
        # Manual grid search
        best_mse, best_params = float('inf'), None

        if search_choice == 1:
            for i in param_grid['n_estimators']:         
                for j in param_grid['group_size']:  
                    for k in param_grid['max_depth']:
                        if i % j == 0 and i > j:  # Ensures n_estimators is a multiple of group_size and that group_size < n_estimators
                            parameters.append((i, j, k)) 

            max_combinations = len(parameters)

            # Input del usuario para muestreo de combinaciones
            try:
                n = int(input(f"\nEnter the number of parameter combinations you want to sample from the {max_combinations} combinations: "))
                
                if n > max_combinations: 
                    print(f"The number of parameter combinations must be less than or equal to {max_combinations}.")
                    exit()
            
            except ValueError:
                print("Please enter a valid integer for the number of combinations.")
                exit()
            
            sampled_params = random.sample(parameters, n)
        elif search_choice == 2:
            sampled_params = get_top_50_params(model_name)
        else:
            print("Invalid choice. Please select 1 or 2.")
            exit()

        print(f"\nRunning grid search for model: {model_name}")

        for n_estimators, group_size, max_depth in tqdm(sampled_params):
            model_instance = model.__class__(n_estimators=n_estimators, group_size=group_size, max_depth=max_depth, random_state=SEED, n_jobs=3)
            model_instance.fit(X_train.values, y_train)
            
            y_pred = model_instance.predict(X_valid.values)
            mse = mean_squared_error(y_valid, y_pred)
            mse_list.append(mse)
             
            if mse < best_mse:
                best_mse = mse
                best_params = {'n_estimators': n_estimators, 'group_size': group_size, 'max_depth': max_depth}
    
    # Create a tensor of parameter combinations with their corresponding MSE values
    parameter_np = np.array(sampled_params, dtype=np.float32)
    if choice == '7':
        parameter_np = parameter_np.reshape(-1, 1)

    mse_np = np.array(mse_list, dtype=np.float32).reshape(-1, 1)  # Convert MSE list to a tensor and add a dimension for concatenation
    result_np = np.concatenate((parameter_np, mse_np), axis=1)  # Combine parameters and MSE values
    
    if search_choice == 2:
        file_path = f'resultados_top_150/{dataset_name}/{dataset_name}_{model_name}_150.npy'
    elif sampled_in_blocks:
        file_path = f'resultados_grid_search/{dataset_name}/{dataset_name}_{model_name}_{block_selection}.npy'
    else:
        file_path = f'resultados_grid_search/{dataset_name}/{dataset_name}_{model_name}.npy'

    # Guardar el resultado como un archivo .npy
    np.save(file_path, result_np)

    # Save the tensor in the dictionary with the model name as the key
    models_results[model_name] = result_np

    results.append({
        'Model': model_name,
        'Best Parameters': best_params,
        'Validation MSE': best_mse
    })

# Display all results
print("\nBest Parameters and MSE:")
for result in results:
    print(result)

if search_choice == 1:
    # Display results for each model
    for model_name, arr in models_results.items():
        if search_choice == 2:
            path = f'resultados_top_150/{dataset_name}/{dataset_name}_{model_name}_150.npy'
        elif sampled_in_blocks:
            path = f'resultados_grid_search/{dataset_name}/{dataset_name}_{model_name}_{block_selection}.npy'
        else:
            path = f'resultados_grid_search/{dataset_name}/{dataset_name}_{model_name}.npy'
        data = np.load(path)
        print(data)