# import warnings
# warnings.filterwarnings("ignore")

import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error
from scipy.stats import kruskal
import scikit_posthocs as sp

from sklearn.ensemble import (
    RandomForestRegressor, IQRRandomForestRegressor, PercentileTrimmingRandomForestRegressor, 
    OOBRandomForestRegressor, OOBPlusIQRRandomForestRegressor,
    FirstSplitCombinerRandomForestRegressor, SharedKnowledgeRandomForestRegressor)

# Para cada dataset d: 
#    Para cada algoritmo a:
#       Hacer 10-fold CV (importante: deben ser siempre los mismos 10 folds).
#       Esto genera una lista de 10 resultados, que llamamos R_a.
#    Correr Kruskal-Wallis sobre todos los R_a (un único test, en simultáneo).
#    Si KW arroja p<0.05:
#       Correr Dunn's post-hoc test para ver qué pares son significativamente distintos. 

SEED = 14208

DATASETS_COLUMNS = {
    'Carbon_Emission': 'CarbonEmission',
    'Wind': 'WIND',
    'House_8L': 'price',
}


TEST = True
datasets_train = {
    'Carbon_Emission': 'distribucion/datasets/train_data/Carbon_Emission_train.csv',
    'House_8L': 'distribucion/datasets/train_data/house_8L_train_7000.csv',
    'Wind': 'distribucion/datasets/train_data/wind_train.csv'
}

datasets_test = {
    'Carbon_Emission': 'distribucion/datasets/test_data/Carbon_Emission_test.csv',
    'House_8L': 'distribucion/datasets/test_data/house_8L_test.csv',
    'Wind': 'distribucion/datasets/test_data/wind_test.csv'
}

dataset_specific_params = {
    'Carbon_Emission': {
        'IQRRandomForestRegressor': {'n_estimators': 150, 'group_size': 3, 'max_depth': 17},
        'PercentileTrimmingRandomForestRegressor': {'n_estimators': 150, 'group_size': 50,'percentile': 2, 'max_depth': 44},
        'OOBRandomForestRegressor': {'n_estimators': 180, 'group_size': 3,'max_depth': 20},
        'OOBPlusIQRRandomForestRegressor': {'n_estimators': 180, 'group_size': 3, 'max_depth': 21},
        'FirstSplitCombinerRandomForestRegressor': {'n_estimators': 100, 'group_size': 10, 'max_features': 'log2'},
        'SharedKnowledgeRandomForestRegressor': {'n_estimators': 280, 'group_size': 7, 'initial_max_depth': 14, 'max_depth': 20},
        'RandomForestRegressor': {'max_depth': 20},
    },
    'House_8L': {
        'IQRRandomForestRegressor': {'n_estimators': 150, 'group_size': 3, 'max_depth': 17},
        'PercentileTrimmingRandomForestRegressor': {'n_estimators': 150, 'group_size': 50,'percentile': 2, 'max_depth': 40},
        'OOBRandomForestRegressor': {'n_estimators': 180, 'group_size': 3, 'max_depth': 32},
        'OOBPlusIQRRandomForestRegressor': {'n_estimators': 180, 'group_size': 3, 'max_depth': 42},
        'FirstSplitCombinerRandomForestRegressor': {'n_estimators': 100, 'group_size': 10, 'max_features': 'log2'},
        'SharedKnowledgeRandomForestRegressor': {'n_estimators': 280, 'group_size': 7, 'initial_max_depth': 14, 'max_depth': 23},
        'RandomForestRegressor': {'max_depth': 17},
    },
    'Wind': {
        'IQRRandomForestRegressor': {'n_estimators': 150, 'group_size': 3, 'max_depth': 19},
        'PercentileTrimmingRandomForestRegressor': {'n_estimators': 150, 'group_size': 50,'percentile': 2, 'max_depth': 44},
        'OOBRandomForestRegressor': {'n_estimators': 180, 'group_size': 3, 'max_depth': 32},
        'OOBPlusIQRRandomForestRegressor': {'n_estimators': 180, 'group_size': 3, 'max_depth': 14},
        'FirstSplitCombinerRandomForestRegressor': {'n_estimators': 100, 'group_size': 10, 'max_features': 'log2'},
        'SharedKnowledgeRandomForestRegressor': {'n_estimators': 280, 'group_size': 7, 'initial_max_depth': 14, 'max_depth': 25},
        'RandomForestRegressor': {'max_depth': 12},
    }
}


def get_models_for_dataset():
    # Conseguimos los hiperparametros de cada modelo segun el dataset
    return [
        IQRRandomForestRegressor(),
        PercentileTrimmingRandomForestRegressor(),
        OOBRandomForestRegressor(),
        OOBPlusIQRRandomForestRegressor(),
        FirstSplitCombinerRandomForestRegressor(),
        SharedKnowledgeRandomForestRegressor(),
        RandomForestRegressor()
    ]

def kfold_cross_validation(models, dataset_name, X, y, n_splits=10):
    all_scores = []
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    params = dataset_specific_params[dataset_name]
    for model in tqdm(models, desc="Models"):
        
        for fold, (train_index, test_index) in enumerate(kf.split(X), start=1):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]  # .iloc porque KFold separa por numero de índice, X and y tienen que ser pandas DataFrames 
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]  # .iloc porque KFold separa por numero de índice
            
            model_instance = model.__class__(**params[model.__class__.__name__], random_state=SEED)
            model_instance.fit(X_train.values, y_train.values)
            
            y_pred = model_instance.predict(X_test.values)
            
            mse = mean_squared_error(y_test, y_pred)
            all_scores.append({'Model': model.__class__.__name__, 'Fold': fold, 'MSE': mse})
    
    results_df = pd.DataFrame(all_scores)
    
    results_df = results_df.sort_values(by='MSE').reset_index(drop=True)
    
    results_df['Rank'] = results_df['MSE'].rank(method="dense", ascending=True).astype(int)

    return results_df


if TEST == False: 

    for dataset_name, path in tqdm(datasets_train.items(), desc="Datasets"): 
        
        # Preprocesamiento de datos
        train_df = pd.read_csv(path)
        target_column = DATASETS_COLUMNS[dataset_name]

        train_df, validation_df = train_test_split(train_df, test_size=0.2, random_state=SEED)
        
        y_train, y_valid = train_df[target_column], validation_df[target_column]
        X_train, X_valid = train_df.drop(target_column, axis=1), validation_df.drop(target_column, axis=1)
        
        X_train, X_valid = pd.get_dummies(X_train), pd.get_dummies(X_valid)
        X_train, X_valid = X_train.align(X_valid, join='left', axis=1, fill_value=0)

        models = get_models_for_dataset()
        scores = kfold_cross_validation(models, dataset_name, X_train, y_train)
        scores.to_csv(f'score_{dataset_name}.csv', index=False)

        # Agrupamos los MSEs por modelo
        grouped_mses = [scores[scores['Model'] == model.__class__.__name__]['MSE'].values for model in models]
        #print(grouped_mses)

        # Kruskal-Wallis test
        stat, p_value = kruskal(*grouped_mses) 
        print(f"Dataset: {dataset_name}")
        print(f"Kruskal-Wallis H-statistic: {stat}") # A mayor valor de stat, mayor la diferencia de los MSEs entre los grupos.
        print(f"p-value: {p_value}")

        if p_value < 0.05:
            print(f"There is a significant difference in MSEs between models for dataset: {dataset_name}.")
            
            # Dunn's post-hoc test
            posthoc_results = sp.posthoc_dunn(grouped_mses, p_adjust='bonferroni') # ver QUE modelos tienen diferencias significativas
            print("Dunn's post-hoc test results:")
            print(posthoc_results)
            posthoc_results.to_csv(f'posthoc_results{dataset_name}.csv', index=True)

        else:
            print(f"No significant difference in MSEs between models for dataset: {dataset_name}.")

elif TEST == True:

    for dataset_name, path in tqdm(datasets_test.items(), desc="Datasets"): 

        # Preprocesamiento de datos de Test
        test_df = pd.read_csv(path)
        target_column = DATASETS_COLUMNS[dataset_name]
        y_test = test_df[target_column]
        X_test = test_df.drop(target_column, axis=1)
        X_test = pd.get_dummies(X_test)

        models = get_models_for_dataset()
        scores = kfold_cross_validation(models, dataset_name, X_test, y_test)
        scores.to_csv(f'score_{dataset_name}.csv', index=False)

        # Agrupamos los MSEs por modelo
        grouped_mses = [scores[scores['Model'] == model.__class__.__name__]['MSE'].values for model in models]
        # Kruskal-Wallis test
        stat, p_value = kruskal(*grouped_mses) 
        print(f"Dataset: {dataset_name}")
        print(f"Kruskal-Wallis H-statistic: {stat}") # A mayor valor de stat, mayor la diferencia de los MSEs entre los grupos.
        print(f"p-value: {p_value}")

        if p_value < 0.05:
            print(f"There is a significant difference in MSEs between models for dataset: {dataset_name}.")
            
            # Dunn's post-hoc test
            posthoc_results = sp.posthoc_dunn(grouped_mses, p_adjust='bonferroni')
            print("Dunn's post-hoc test results:")
            print(posthoc_results)
            posthoc_results.to_csv(f'posthoc_results_{dataset_name}.csv', index=True)

        else:
            print(f"No significant difference in MSEs between models for dataset: {dataset_name}.")