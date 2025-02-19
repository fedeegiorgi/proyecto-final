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
import os

SEED = 14208

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
print("Data preprocessing complete.")

# Define Parameter Grids
param_grids = {
    "Carbon_Emission": {
        "1": {
            'model': IQRRandomForestRegressor(),
            'param_grid': {
                'n_estimators': list(range(30, 1351, 60)), 
                'group_size': [2, 3, 5, 6, 10, 15, 25, 30, 50, 75], 
                'max_depth': list(range(2, 33, 1))
            },
            'name': "IQR",
            'def_params': {'n_estimators': 150, 'group_size': 10, 'max_depth': 17}
        },
        "2": {
            'model': PercentileTrimmingRandomForestRegressor(),
            'param_grid': {
                'n_estimators': list(range(100, 1250, 50)) , 
                'group_size': [3, 5, 6, 10, 15, 25, 30, 50, 75], 
                'max_depth': list(range(29, 60, 1)),
                'percentile': list(range(1, 16, 1))
            }, 
            'name': "Percentile_Trimming",
            'def_params': {'n_estimators': 150, 'group_size': 50, 'percentile': 2, 'max_depth': 44}
        },
        "3": {
            'model': OOBRandomForestRegressor(),
            'param_grid': {
                'n_estimators': list(range(30, 1381, 60)), 
                'group_size': [2, 3, 4, 5, 6, 9, 10, 12, 15, 18, 20, 30, 36, 45, 60, 90], 
                'max_depth': list(range(5, 36, 1))
            },
            'name': "OOB",
            'def_params': {'n_estimators': 180, 'group_size': 3, 'max_depth': 20}
        },
        "4": {
            'model': OOBPlusIQRRandomForestRegressor(),
            'param_grid': {
                'n_estimators': list(range(30, 1381, 60)), 
                'group_size': [2, 3, 4, 5, 6, 9, 10, 12, 15, 18, 20, 30, 36, 45, 60, 90], 
                'max_depth': list(range(6, 47, 1))
            },
            'name': "OOB_plus_IQR",
            'def_params': {'n_estimators': 180, 'group_size': 3, 'max_depth': 21}
        },
        "5": {
            'model': FirstSplitCombinerRandomForestRegressor(),
            'param_grid': {
                'n_estimators': list(range(50, 1200, 50)),
                'group_size': [2, 4, 5, 10, 20],
                'max_features': [-1, -2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8],
            },
            'name': "First_Splits_Combiner",
            'def_params': {'n_estimators': 100, 'group_size': 10, 'max_features': 'log2'}
        },
        "6": {
            'model': SharedKnowledgeRandomForestRegressor(),
            'param_grid': {
                'n_estimators': list(range(70, 1331, 70)), 
                'group_size': [2, 4, 7, 8, 10, 14, 20, 28, 40, 70, 140],
                'max_depth': list(range(15, 46, 1)),
                'initial_max_depth': list(range(2, 19, 1))},
            'name': "Shared_Knowledge",
            'def_params': {'n_estimators': 280, 'group_size': 7, 'initial_max_depth': 14, 'max_depth': 20}
        }
    },
    "House_8L": {
        "1": {
            'model': IQRRandomForestRegressor(),
            'param_grid': {
                'n_estimators': list(range(30, 1351, 60)), 
                'group_size': [2, 3, 5, 6, 10, 15, 25, 30, 50, 75], 
                'max_depth': list(range(2, 33, 1))
            },
            'name': "IQR",
            'def_params': {'n_estimators': 150, 'group_size': 10, 'max_depth': 17}
        },
        "2": {
            'model': PercentileTrimmingRandomForestRegressor(),
            'param_grid': {
                'n_estimators': list(range(100, 1250, 50)), 
                'group_size': [3, 5, 6, 10, 15, 25, 30, 50, 75], 
                'max_depth': list(range(25, 56, 1)),
                'percentile': list(range(1, 16, 1))
            }, 
            'name': "Percentile_Trimming",
            'def_params': {'n_estimators': 150, 'group_size': 50, 'percentile': 2, 'max_depth': 40}
        },
        "3": {
            'model': OOBRandomForestRegressor(),
            'param_grid': {
                'n_estimators': list(range(30, 1381, 60)), 
                'group_size': [2, 3, 4, 5, 6, 9, 10, 12, 15, 18, 20, 30, 36, 45, 60, 90], 
                'max_depth': list(range(17, 48, 1))
            },
            'name': "OOB",
            'def_params': {'n_estimators': 180, 'group_size': 3, 'max_depth': 32}
        },
        "4": {
            'model': OOBPlusIQRRandomForestRegressor(),
            'param_grid': {
                'n_estimators': list(range(30, 1381, 60)), 
                'group_size': [2, 3, 4, 5, 6, 9, 10, 12, 15, 18, 20, 30, 36, 45, 60, 90], 
                'max_depth': list(range(27, 58, 1))
            },
            'name': "OOB_plus_IQR",
            'def_params': {'n_estimators': 180, 'group_size': 3, 'max_depth': 42}
        },
        "5": {
            'model': FirstSplitCombinerRandomForestRegressor(),
            'param_grid': {
                'n_estimators': list(range(50, 1200, 50)),
                'group_size': [2, 4, 5, 10, 20],
                'max_features': [-1, -2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8],
            },
            'name': "First_Splits_Combiner",
            'def_params': {'n_estimators': 100, 'group_size': 10, 'max_features': 'log2'}
        },
        "6": {
            'model': SharedKnowledgeRandomForestRegressor(),
            'param_grid': {
                'n_estimators': list(range(70, 1331, 70)), 
                'group_size': [2, 4, 7, 8, 10, 14, 20, 28, 40, 70, 140],
                'max_depth': list(range(15, 39, 1)),
                'initial_max_depth': list(range(2, 19, 1))},
            'name': "Shared_Knowledge",
            'def_params': {'n_estimators': 280, 'group_size': 7, 'initial_max_depth': 14, 'max_depth': 23}
        }
    },
    "Wind": {
        "1": {
            'model': IQRRandomForestRegressor(),
            'param_grid': {
                'n_estimators': list(range(30, 1351, 60)), 
                'group_size': [2, 3, 5, 6, 10, 15, 25, 30, 50, 75], 
                'max_depth': list(range(4, 35, 1))
            },
            'name': "IQR",
            'def_params': {'n_estimators': 150, 'group_size': 10, 'max_depth': 19}
        },
        "2": {
            'model': PercentileTrimmingRandomForestRegressor(),
            'param_grid': {
                'n_estimators': list(range(100, 1250, 50)), 
                'group_size': [3, 5, 6, 10, 15, 25, 30, 50, 75], 
                'max_depth': list(range(29, 60, 1)),
                'percentile': list(range(1, 16, 1))
            }, 
            'name': "Percentile_Trimming",
            'def_params': {'n_estimators': 150, 'group_size': 50, 'percentile': 2, 'max_depth': 44}
        },
        "3": {
            'model': OOBRandomForestRegressor(),
            'param_grid': {
                'n_estimators': list(range(30, 1381, 60)), 
                'group_size': [2, 3, 4, 5, 6, 9, 10, 12, 15, 18, 20, 30, 36, 45, 60, 90], 
                'max_depth': list(range(17, 48, 1))
            },
            'name': "OOB",
            'def_params': {'n_estimators': 180, 'group_size': 3, 'max_depth': 32}
        },
        "4": {
            'model': OOBPlusIQRRandomForestRegressor(),
            'param_grid': {
                'n_estimators': list(range(30, 1381, 60)), 
                'group_size': [2, 3, 4, 5, 6, 9, 10, 12, 15, 18, 20, 30, 36, 45, 60, 90], 
                'max_depth': list(range(4, 30, 1))
            },
            'name': "OOB_plus_IQR",
            'def_params': {'n_estimators': 180, 'group_size': 3, 'max_depth': 14}
        },
        "5": {
            'model': FirstSplitCombinerRandomForestRegressor(),
            'param_grid': {
                'n_estimators': list(range(50, 1200, 50)),
                'group_size': [2, 4, 5, 10, 20],
                'max_features': [-1, -2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8],
            },
            'name': "First_Splits_Combiner",
            'def_params': {'n_estimators': 100, 'group_size': 10, 'max_features': 'log2'}
        },
        "6": {
            'model': SharedKnowledgeRandomForestRegressor(),
            'param_grid': {
                'n_estimators': list(range(70, 1331, 70)), 
                'group_size': [2, 4, 7, 8, 10, 14, 20, 28, 40, 70, 140],
                'max_depth': list(range(15, 41, 1)),
                'initial_max_depth': list(range(2, 19, 1))},
            'name': "Shared_Knowledge",
            'def_params': {'n_estimators': 280, 'group_size': 7, 'initial_max_depth': 14, 'max_depth': 25}
        }
    },
    "Flight": {
        "1": {
            'model': IQRRandomForestRegressor(),
            'param_grid': {
                'max_depth': list(range(10, 51, 1)),
            },
            'name': "IQR",
            'def_params': {'n_estimators': 150, 'group_size': 10}
        },
        "2": {
            'model': PercentileTrimmingRandomForestRegressor(),
            'param_grid': {
                'max_depth': list(range(10, 51, 1))
            }, 
            'name': "Percentile_Trimming",
            'def_params': {'n_estimators': 150, 'group_size': 50, 'percentile': 2}
        },
        "3": {
            'model': OOBRandomForestRegressor(),
            'param_grid': {
                'max_depth': list(range(10, 51, 1)),
            },
            'name': "OOB",
            'def_params': {'n_estimators': 180, 'group_size': 3}
        },
        "4": {
            'model': OOBPlusIQRRandomForestRegressor(),
            'param_grid': {
                'max_depth': list(range(10, 51, 1)),
            },
            'name': "OOB_plus_IQR",
            'def_params': {'n_estimators': 180, 'group_size': 3}
        },
        "5": {
            'model': FirstSplitCombinerRandomForestRegressor(),
            'param_grid': {
                'max_features': [-1, -2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8],
            },
            'name': "First_Splits_Combiner",
            'def_params': {'n_estimators': 100, 'group_size': 10}
        },
        "6": {
            'model': SharedKnowledgeRandomForestRegressor(),
            'param_grid': {
                'max_depth': list(range(20, 51, 1))
            },
            'name': "Shared_Knowledge",
            'def_params': {'n_estimators': 280, 'group_size': 7, 'initial_max_depth': 14}
        }
    },
    "Rainfall": {
        "1": {
            'model': IQRRandomForestRegressor(),
            'param_grid': {
                'max_depth': list(range(10, 51, 1)),
            },
            'name': "IQR",
            'def_params': {'n_estimators': 150, 'group_size': 10}
        },
        "2": {
            'model': PercentileTrimmingRandomForestRegressor(),
            'param_grid': {
                'max_depth': list(range(10, 51, 1))
            }, 
            'name': "Percentile_Trimming",
            'def_params': {'n_estimators': 150, 'group_size': 50, 'percentile': 2}
        },
        "3": {
            'model': OOBRandomForestRegressor(),
            'param_grid': {
                'max_depth': list(range(10, 51, 1)),
            },
            'name': "OOB",
            'def_params': {'n_estimators': 180, 'group_size': 3}
        },
        "4": {
            'model': OOBPlusIQRRandomForestRegressor(),
            'param_grid': {
                'max_depth': list(range(10, 51, 1)),
            },
            'name': "OOB_plus_IQR",
            'def_params': {'n_estimators': 180, 'group_size': 3}
        },
        "5": {
            'model': FirstSplitCombinerRandomForestRegressor(),
            'param_grid': {
                'max_features': [-1, -2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8],
            },
            'name': "First_Splits_Combiner",
            'def_params': {'n_estimators': 100, 'group_size': 10}
        },
        "6": {
            'model': SharedKnowledgeRandomForestRegressor(),
            'param_grid': {
                'max_depth': list(range(20, 51, 1))
            },
            'name': "Shared_Knowledge",
            'def_params': {'n_estimators': 280, 'group_size': 7, 'initial_max_depth': 14}
        }
    },
    "Abalone": {
        "1": {
            'model': IQRRandomForestRegressor(),
            'param_grid': {
                'max_depth': list(range(10, 51, 1)),
            },
            'name': "IQR",
            'def_params': {'n_estimators': 150, 'group_size': 10}
        },
        "2": {
            'model': PercentileTrimmingRandomForestRegressor(),
            'param_grid': {
                'max_depth': list(range(10, 51, 1))
            }, 
            'name': "Percentile_Trimming",
            'def_params': {'n_estimators': 150, 'group_size': 50, 'percentile': 2}
        },
        "3": {
            'model': OOBRandomForestRegressor(),
            'param_grid': {
                'max_depth': list(range(10, 51, 1)),
            },
            'name': "OOB",
            'def_params': {'n_estimators': 180, 'group_size': 3}
        },
        "4": {
            'model': OOBPlusIQRRandomForestRegressor(),
            'param_grid': {
                'max_depth': list(range(10, 51, 1)),
            },
            'name': "OOB_plus_IQR",
            'def_params': {'n_estimators': 180, 'group_size': 3}
        },
        "5": {
            'model': FirstSplitCombinerRandomForestRegressor(),
            'param_grid': {
                'max_features': [-1, -2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8],
            },
            'name': "First_Splits_Combiner",
            'def_params': {'n_estimators': 100, 'group_size': 10}
        },
        "6": {
            'model': SharedKnowledgeRandomForestRegressor(),
            'param_grid': {
                'max_depth': list(range(20, 51, 1))
            },
            'name': "Shared_Knowledge",
            'def_params': {'n_estimators': 280, 'group_size': 7, 'initial_max_depth': 14}
        }
    },
}

# Prompt to select the model(s)
print("Select the model/models you would like to optimize (comma-separated if choosing multiple):")
print("1: IQR")
print("2: Percentile Trimming")
print("3: OOB")
print("4: OOB + IQR")
print("5: First Splits Combiner")
print("6: Shared Knowledge")

choices = input("Enter the numbers corresponding to your choice(s): ").split(',')

# Prompt to select the non fixed hyperparameter
print("Select the hyperparameter(s) you want to variate:")
print("1: max_depth")
print("2: n_estimators")
print("3: group_size")
print("4: percentile")
print("5: initial_max_depth")
print("6: max_features")

hp_choices = input("Enter the numbers corresponding to your choice(s): ").split(',')

for choice in tqdm(choices):
    choice = choice.strip()
    if choice not in param_grids[dataset_name]:
        print(f"Invalid choice: {choice}. Skipping.")
        exit()

    config = param_grids[dataset_name][choice]
    model, param_grid, model_name, def_params = config['model'], config['param_grid'], config['name'], config['def_params']

    for hp_choice in hp_choices:
        mse_list = []

        if hp_choice == '1':
            if model_name == 'First_Splits_Combiner':
                continue

            params, param_name = param_grid['max_depth'], 'max_depth'
            for max_depth in tqdm(params):
                if model_name in ['IQR', 'OOB', 'OOB_plus_IQR']:
                    model_instance = model.__class__(n_estimators=def_params['n_estimators'], group_size=def_params['group_size'], 
                                                     max_depth=max_depth, random_state=SEED, n_jobs=3)
                elif model_name == 'Percentile_Trimming':
                    model_instance = model.__class__(n_estimators=def_params['n_estimators'], group_size=def_params['group_size'], percentile=def_params['percentile'],
                                                     max_depth=max_depth, random_state=SEED, n_jobs=3)
                else:
                    model_instance = model.__class__(n_estimators=def_params['n_estimators'], group_size=def_params['group_size'], initial_max_depth=def_params['initial_max_depth'],
                                                     max_depth=max_depth, random_state=SEED, n_jobs=3)

                model_instance.fit(X_train.values, y_train.values)
                
                y_pred = model_instance.predict(X_valid.values)
                mse = mean_squared_error(y_valid, y_pred)
                mse_list.append(mse)

        elif hp_choice == '2':
            params, param_name = param_grid['n_estimators'], 'n_estimators'
            for n_estimators in tqdm(params):
                if model_name in ['IQR', 'OOB', 'OOB_plus_IQR']:
                    model_instance = model.__class__(n_estimators=n_estimators, group_size=def_params['group_size'], 
                                                     max_depth=def_params['max_depth'], random_state=SEED, n_jobs=3)
                elif model_name == 'Percentile_Trimming':
                    model_instance = model.__class__(n_estimators=n_estimators, group_size=def_params['group_size'], percentile=def_params['percentile'],
                                                     max_depth=def_params['max_depth'], random_state=SEED, n_jobs=3)
                elif model_name == 'First_Splits_Combiner':
                    model_instance = model.__class__(n_estimators=n_estimators, group_size=def_params['group_size'], 
                                                     max_features=def_params['max_features'], random_state=SEED, n_jobs=3)
                else:
                    model_instance = model.__class__(n_estimators=n_estimators, group_size=def_params['group_size'], initial_max_depth=def_params['initial_max_depth'],
                                                     max_depth=def_params['max_depth'], random_state=SEED, n_jobs=3)
                model_instance.fit(X_train.values, y_train.values)
                
                y_pred = model_instance.predict(X_valid.values)
                mse = mean_squared_error(y_valid, y_pred)
                mse_list.append(mse)

        elif hp_choice == '3':
            params, param_name = param_grid['group_size'], 'group_size'
            for group_size in tqdm(params):
                if model_name in ['IQR', 'OOB', 'OOB_plus_IQR']:
                    model_instance = model.__class__(n_estimators=def_params['n_estimators'], group_size=group_size, 
                                                     max_depth=def_params['max_depth'], random_state=SEED, n_jobs=3)
                elif model_name == 'Percentile_Trimming':
                    model_instance = model.__class__(n_estimators=def_params['n_estimators'], group_size=group_size, percentile=def_params['percentile'],
                                                     max_depth=def_params['max_depth'], random_state=SEED, n_jobs=3)
                elif model_name == 'First_Splits_Combiner':
                    model_instance = model.__class__(n_estimators=def_params['n_estimators'], group_size=group_size, max_features=def_params['max_features'], 
                                                     random_state=SEED)
                else:
                    model_instance = model.__class__(n_estimators=def_params['n_estimators'], group_size=group_size, initial_max_depth=def_params['initial_max_depth'],
                                                     max_depth=def_params['max_depth'], random_state=SEED)
                model_instance.fit(X_train.values, y_train.values)
                
                y_pred = model_instance.predict(X_valid.values)
                mse = mean_squared_error(y_valid, y_pred)
                mse_list.append(mse)

        elif hp_choice == '4':
            if model_name != 'Percentile_Trimming':
                continue

            params, param_name = param_grid['percentile'], 'percentile'
            for percentile in tqdm(params):
                model_instance = model.__class__(n_estimators=def_params['n_estimators'], group_size=def_params['group_size'], percentile=percentile,
                                                     max_depth=def_params['max_depth'], random_state=SEED, n_jobs=3)

                model_instance.fit(X_train.values, y_train.values)
                
                y_pred = model_instance.predict(X_valid.values)
                mse = mean_squared_error(y_valid, y_pred)
                mse_list.append(mse)
        
        elif hp_choice == '5':
            if model_name != 'Shared_Knowledge':
                continue

            params, param_name = param_grid['initial_max_depth'], 'initial_max_depth'
            for initial_max_depth in tqdm(params):
                model_instance = model.__class__(n_estimators=def_params['n_estimators'], group_size=def_params['group_size'], initial_max_depth=initial_max_depth,
                                                 max_depth=def_params['max_depth'], random_state=SEED)
                model_instance.fit(X_train.values, y_train.values)
                
                y_pred = model_instance.predict(X_valid.values)
                mse = mean_squared_error(y_valid, y_pred)
                mse_list.append(mse)

        else:
            if model_name != 'First_Splits_Combiner':
                continue

            params, param_name = param_grid['max_features'], 'max_features'
            for max_features in tqdm(params):
                if max_features == -1:
                    max_features = 'sqrt'
                elif max_features == -2:
                    max_features = 'log2'

                model_instance = model.__class__(n_estimators=def_params['n_estimators'], group_size=def_params['group_size'], max_features=max_features, random_state=SEED)
                model_instance.fit(X_train.values, y_train.values)
                
                y_pred = model_instance.predict(X_valid.values)
                mse = mean_squared_error(y_valid, y_pred)
                mse_list.append(mse)

        parameter_np = np.array(params, dtype=np.float32).reshape(-1, 1)
        mse_np = np.array(mse_list, dtype=np.float32).reshape(-1, 1)
        result_np = np.concatenate((parameter_np, mse_np), axis=1)

        # Define the file path
        file_path = f'resultados_grid_search/{dataset_name}/{dataset_name}_{model_name}_{param_name}.npy'

        # Ensure the target directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        np.save(file_path, result_np)