from sklearn.ensemble import SharedKnowledgeRandomForestRegressor, OOBRandomForestRegressor, RandomForestRegressor, RFRegressorFirstSplitCombiner
from sklearn.model_selection import train_test_split, KFold
import pandas as pd
from sklearn.tree import DecisionTreeRegressorCombiner, plot_tree
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import make_regression
# Crear datos de ejemplo
#X, y = make_regression(n_samples=10, n_features=5, random_state=0)

SEED = 14208

df = pd.read_csv('../distribucion/datasets/test_data/Carbon_Emission_test.csv')

# Preprocesamiento de datos
# train_df, validation_df = train_test_split(df, test_size=0.2, random_state=SEED)
# y_train, y_valid = train_df['CarbonEmission'], validation_df['CarbonEmission']
# X_train, X_valid = train_df.drop('CarbonEmission', axis=1), validation_df.drop('CarbonEmission', axis=1)
# X_train = pd.get_dummies(X_train)
# X_valid = pd.get_dummies(X_valid)
# X_train, X_valid = X_train.align(X_valid, join='left', axis=1, fill_value=0)


target_column = 'CarbonEmission'
y_test = df[target_column]
y_test = y_test.values
X_test = df.drop(target_column, axis=1)
X_test = pd.get_dummies(X_test)
print(X_test)
X_test = X_test.values
print(X_test)

# print(X.shape)
rf_test = SharedKnowledgeRandomForestRegressor(random_state=SEED, n_estimators=280, group_size=7, initial_max_depth=14, max_depth=20)
# rf_test = RFRegressorFirstSplitCombiner(random_state=SEED, n_estimators=200, group_size=25, max_features=1.0)
# rf_test = RandomForestRegressor(random_state=SEED, n_estimators=30, max_depth=5)
# rf_test.fit(X, y)
# rf_test.fit(X_test.values, y_test.values)

all_scores = []
kf = KFold(n_splits=10, shuffle=True, random_state=SEED)

for fold, (train_index, test_index) in enumerate(kf.split(X_test), start=1):
    X_train, X_test = X_test[train_index], X_test[test_index]  # .iloc porque KFold separa por numero de índice, X and y tienen que ser pandas DataFrames 
    y_train, y_test = y_test[train_index], y_test[test_index]  # .iloc porque KFold separa por numero de índice
            
    rf_test.fit(X_train, y_train)
    
    y_pred = rf_test.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    all_scores.append({'Model': 'SK', 'Fold': fold, 'MSE': mse})

print(all_scores)
# print("original_X")
# print(X)
# print(rf_test.estimators_samples_[0])
# samples_used = rf_test.estimators_samples_[0]
# print("new_X")
# print(X[samples_used])

# predictions = rf_test.predict(X_valid.values)
#print(np.any(np.isnan(predictions)))
# mse = mean_squared_error(y_valid.values, predictions)

#print("***************** MSE:", mse)
# print(rf_test.extended_grouped_estimators_[0][0].tree_.max_depth)
# print(rf_test.extended_grouped_estimators_[0][0].tree_.n_node_samples)

# print(rf_test.estimators_[0].tree_.max_depth)
# print(rf_test.estimators_[0].tree_.n_node_samples)
# print(rf_test.group_size)
# print(mse)

# plt.figure(figsize=(20, 20))
# plot_tree(rf_test.estimators_[0])
# plt.savefig('initial_tree.png')
# plt.close()

# plt.figure(figsize=(20, 20))
# plot_tree(rf_test.initial_estimators_[0])
# plt.savefig('initial_tree_all_samples.png')
# plt.close()

# plt.figure(figsize=(40, 40))
# plot_tree(rf_test.extended_grouped_estimators_[0][0])
# plt.savefig('extended_tree.png')
# plt.close()

# print(X)
# print(y)
# print(X_train.columns)

# X_testeo = X[mask]

# print(X)
# print(X[:4])

# first_leave_mask = np.where((X_testeo[4] <= 4.98) & (X_testeo['CompanyName_hp'] <= 0.5) & (X_testeo['CompanyName_Dell'] <= 0.5))[0]

# y_train_first_leave = y_train[first_leave_mask]

# y_hat = np.sum(y_train_first_leave) / len(y_train_first_leave)

# print(y_hat)
# plt.figure(figsize=(20, 20))
# plot_tree(rf_test.initial_estimators_[0])
# plt.savefig('initial_tree_all_samples.png')
# plt.close()

# plt.figure(figsize=(20, 20))
# plot_tree(rf_test.extended_grouped_estimators_[0][0])
# plt.savefig('extended_tree.png')
# plt.close()

# print("***************** MSE:", mse)
# print("INICIAL:")

# print(rf_test.estimators_[0].tree_.weighted_n_node_samples)
# print(rf_test.estimators_[0].tree_.threshold)

# print("EXTENDIDO:")

# print(rf_test.extended_grouped_estimators_[0][0].tree_.weighted_n_node_samples)
# print(rf_test.extended_grouped_estimators_[0][0].tree_.threshold)

# print(rf_test.extended_grouped_estimators_[0][0].tree_.impurity)
# print(rf_test.estimators_[0].tree_.impurity)
# print(rf_test.extended_grouped_estimators_[0][0].tree_.n_node_samples)
# print("****** Original tree ******")
# print("Nodes ids:")
# print(np.arange(rf_test.estimators_[0].tree_.node_count))
# print("Children  to left:")
# print(rf_test.estimators_[0].tree_.children_left)
# print("Children  to right:")
# print(rf_test.estimators_[0].tree_.children_right)

# print("****** Extended tree ******")
# print("Nodes ids:")
# print(np.arange(rf_test.extended_grouped_estimators_[0][0].tree_.node_count))
# print("Children  to left:")
# print(rf_test.extended_grouped_estimators_[0][0].tree_.children_left)
# print("Children  to right:")
# print(rf_test.extended_grouped_estimators_[0][0].tree_.children_right)

# plt.figure(figsize=(20, 20))
# plot_tree(rf_test.estimators_[1])
# plt.savefig('initial_tree_1.png')
# plt.close()

# plt.figure(figsize=(20, 20))
# plot_tree(rf_test.estimators_[2])
# plt.savefig('initial_tree_2.png')
# plt.close()

# print(rf_test.combined_trees[9].tree_.node_count)
# print(rf_test.combined_trees[9].tree_.capacity)
# print(len(rf_test.combined_trees))