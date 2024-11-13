import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error
from scipy.stats import kruskal
from sklearn.ensemble import RandomForestRegressor, OOBRandomForestRegressor, IQRRandomForestRegressor

df = pd.read_csv('distribucion/datasets/train_data/wind_train.csv')
SEED = 14208

# Preprocesamiento de datos
train_df, validation_df = train_test_split(df, test_size=0.2, random_state=SEED)
y_train, y_valid = train_df['WIND'], validation_df['WIND']
X_train, X_valid = train_df.drop('WIND', axis=1), validation_df.drop('WIND', axis=1)
X_train = pd.get_dummies(X_train)
X_valid = pd.get_dummies(X_valid)
X_train, X_valid = X_train.align(X_valid, join='left', axis=1, fill_value=0)

def kfold_cross_validation(models, X, y, n_splits=10):
    all_scores = []
    
    for model in models:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
        
        for fold, (train_index, test_index) in enumerate(kf.split(X), start=1):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]  # .iloc porque KFold separa por numero de índice
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]  # .iloc porque KFold separa por numero de índice
            
            model.fit(X_train.values, y_train)
            
            y_pred = model.predict(X_test.values)
            
            mse = mean_squared_error(y_test, y_pred)
            all_scores.append({'Model': model.__class__.__name__, 'Fold': fold, 'MSE': mse})
    
    results_df = pd.DataFrame(all_scores)
    
    results_df = results_df.sort_values(by='MSE').reset_index(drop=True)
    
    results_df['Rank'] = results_df['MSE'].rank(method="dense", ascending=True).astype(int)

    return results_df


rf_def = RandomForestRegressor(random_state=SEED)
rf_oob = OOBRandomForestRegressor(random_state=SEED)
rf_iqr = IQRRandomForestRegressor(random_state=SEED)

models = [rf_def, rf_oob]
scores = kfold_cross_validation(models, X_train, y_train)

# Agrupamos los MSEs de cada fold por modelo
grouped_mses = [scores[scores['Model'] == model.__class__.__name__]['MSE'].values for model in models]
#print(grouped_mses)

# Kruskal-Wallis test
stat, p_value = kruskal(*grouped_mses) # unpack
print(f"Kruskal-Wallis H-statistic: {stat}") # A mayor valor de stat, mayor la diferencia de los MSEs entre los grupos.
print(f"p-value: {p_value}")

if p_value < 0.05:
    print("There is a significant difference in MSEs between models.")
else:
    print("No significant difference in MSEs between models.")