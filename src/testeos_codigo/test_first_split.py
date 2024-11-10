from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor, RFRegressorFirstSplitCombiner, RandomForestGroupDebate
from sklearn.tree import plot_tree
from sklearn.metrics import mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import arff

SEED = 14208

df = pd.read_csv('../distribucion/datasets/train_data/laptop_train.csv')

# Preprocesamiento de datos
train_df, validation_df = train_test_split(df, test_size=0.2, random_state=SEED)
y_train, y_valid = train_df['Price'], validation_df['Price']
X_train, X_valid = train_df.drop('Price', axis=1), validation_df.drop('Price', axis=1)
X_train = pd.get_dummies(X_train)
X_valid = pd.get_dummies(X_valid)
X_train, X_valid = X_train.align(X_valid, join='left', axis=1, fill_value=0)

#----------------------------------------------------------------------------------------------------------------------------

#Instanciamos Modelos
n_estimators = 9
group_size = 3
rf_default_groups = RandomForestGroupDebate(random_state=SEED, n_estimators=n_estimators, group_size=group_size, max_depth=1)
rf_default_groups.fit(X_train.values, y_train)

# Divide the trees into groups
rf_default_groups.initial_grouped_trees = rf_default_groups.group_split(rf_default_groups.estimators_)
print(rf_default_groups.initial_grouped_trees[0][1].tree_.feature)
print(rf_default_groups.initial_grouped_trees[0][1].tree_.threshold)
# grouped_samples = rf_default_groups.group_split(rf_default_groups.estimators_samples_)

rf_comb = RFRegressorFirstSplitCombiner(random_state=SEED, n_estimators=n_estimators, group_size=group_size)
rf_comb.fit(X_train.values, y_train)

print(rf_comb.combined_trees[0].tree_.feature)
print(rf_comb.combined_trees[0].tree_.threshold)

predictions = rf_comb.predict(X_valid.values)
mse = mean_squared_error(y_valid, predictions)
print(mse)

#Comparamos arboles y cortes

