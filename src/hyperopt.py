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
    # 'Laptop': 'Price', 
    'Wind': 'WIND',
    'House_8L': 'price',
}

# Select from the terminal which dataset to use:
print("Select the dataset you would like to use:")
print("1: Carbon Emission")
# print("2: Laptop")
print("2: Wind")
print("3: House_8L")

choice = input("Enter the number corresponding to your choice: ")

# Set the file path and dataset name based on the choice
if choice == "1":
    file_path = 'distribucion/datasets/train_data/Carbon_Emission_train.csv'
    dataset_name = 'Carbon_Emission'
elif choice == "2":
    file_path = 'distribucion/datasets/train_data/house_8L_train.csv'
    dataset_name = 'House_8L'
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
            'n_estimators': list(range(50, 5001, 25)), 
            'group_size': list(range(5, 101, 5)), 
            'max_depth': [None] + list(range(10, 31, 1))},
        'name': "IQR"
    },
    "2": {
        'model': OOBRandomForestRegressorGroups(),
        'param_grid': {
            'n_estimators': list(range(50, 5001, 25)), 
            'group_size': list(range(5, 101, 5)), 
            'max_depth': [None] + list(range(10, 31, 1))},
        'name': "OOB"
    },
    "3": {
        'model': OOB_plus_IQR(),
        'param_grid': {
            'n_estimators': list(range(50, 5001, 25)), 
            'group_size': list(range(5, 101, 5)), 
            'max_depth': [None] + list(range(10, 31, 1))},
        'name': "OOB + IQR"
    },
    # "4": {
    #     'model': RFRegressorFirstSplitCombiner(),
    #     'param_grid': {
    #         'n_estimators': [30, 50, 100, 200], 
    #         'group_size': [3, 5, 10], 
    #         'max_depth': [None, 10, 20]},
    #     'name': "First Splits Combiner"
    # },
    # "5": {
    #     'model': SharedKnowledgeRandomForestRegressor(),
    #     'param_grid': {
    #         'n_estimators': [30, 50, 100, 200], 
    #         'group_size': [3, 5, 10], 
    #         'max_depth': [None, 10, 20], 
    #         'initial_max_depth': [5, 10, 15, 20, 25, 30]},
    #     'name': "Shared Knowledge"
    # },
}

# Prompt to select the model(s)
print("Select the model/models you would like to optimize (comma-separated if choosing multiple):")
print("1: IQR")
print("2: OOB")
print("3: OOB + IQR")
#print("4: First Splits Combiner")
# print("5: Shared Knowledge")

choices = input("Enter the numbers corresponding to your choice(s): ").split(',')

for choice in tqdm(choices):
    choice = choice.strip()
    if choice not in param_grids:
        print(f"Invalid choice: {choice}. Skipping.")
        continue

    config = param_grids[choice]
    model, param_grid, model_name = config['model'], config['param_grid'], config['name']
    
    print(f"\nRunning grid search for model: {model_name}")

#     # Initialize GridSearchCV with the selected model and parameter grid
#     grid_search = GridSearchCV(
#         estimator=model,
#         param_grid=param_grid,
#         cv=2,
#         scoring='neg_mean_squared_error',
#         n_jobs=-1,
#         verbose=2
#     )
    
#     # Fit grid search
#     grid_search.fit(X_train.values, y_train)
    
#     # Get the best parameters and score
#     best_params = grid_search.best_params_
#     best_score = grid_search.best_score_
    
#     # Evaluate the model with the best parameters on the validation set
#     best_model = grid_search.best_estimator_
#     y_pred = best_model.predict(X_valid)
#     mse = mean_squared_error(y_valid, y_pred)
    
#     # Save the results
#     results.append({
#         "Model": model_name,
#         "Best Params": best_params,
#         "Best Cross-Validation Score (neg MSE)": best_score,
#         "Validation MSE": mse
#     })
    
#     print(f"\nResults for model: {model_name}")
#     print("Best Parameters:", best_params)
#     print("Best Cross-Validation Score (neg MSE):", best_score)
#     print("Validation MSE:", mse)

# # After all choices are processed, display the summary of results
# print("\nGrid Search Complete. Summary of Results:")
# for result in results:
#     print(result)

    # Generate filtered parameter combinations
    parameters = []

    if model_name == 'Shared Knowledge':
        for i in param_grid['n_estimators']:         
            for j in param_grid['group_size']:  
                for k in param_grid['max_depth']:
                    for z in param_grid['initial_max_depth']:
                        if i % j == 0 and i > j:  # Ensures n_estimators is a multiple of group_size and that group_size < n_estimators
                            if k > z: # Ensures max_depth > initial_max_depth
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
            #params = {'n_estimators': n_estimators, 'group_size': group_size, 'max_depth': max_depth, "MSE": mse}
            
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