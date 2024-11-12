from sklearn.ensemble import SharedKnowledgeRandomForestRegressor, OOBRandomForestRegressor, RandomForestRegressor, RFRegressorFirstSplitCombiner
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.tree import DecisionTreeRegressorCombiner, plot_tree
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import make_regression
# Crear datos de ejemplo
X, y = make_regression(n_samples=10, n_features=5, random_state=0)

SEED = 14208

df = pd.read_csv('distribucion/datasets/train_data/Carbon_Emission_train.csv')

# Preprocesamiento de datos
train_df, validation_df = train_test_split(df, test_size=0.2, random_state=SEED)
y_train, y_valid = train_df['CarbonEmission'], validation_df['CarbonEmission']
X_train, X_valid = train_df.drop('CarbonEmission', axis=1), validation_df.drop('CarbonEmission', axis=1)
X_train = pd.get_dummies(X_train)
X_valid = pd.get_dummies(X_valid)
X_train, X_valid = X_train.align(X_valid, join='left', axis=1, fill_value=0)

# print(X.shape)
rf_test = SharedKnowledgeRandomForestRegressor(random_state=SEED, n_estimators=9, group_size=3, initial_max_depth=2, max_depth=4)
# rf_test = RFRegressorFirstSplitCombiner(random_state=SEED, n_estimators=200, group_size=25, max_features=1.0)
# rf_test = RandomForestRegressor(random_state=SEED, n_estimators=30, max_depth=5)
# rf_test.fit(X, y)
rf_test.fit(X, y)

# print("original_X")
# print(X)
# print(rf_test.estimators_samples_[0])
# samples_used = rf_test.estimators_samples_[0]
# print("new_X")
# print(X[samples_used])

# predictions = rf_test.predict(X_valid.values)
# print(np.any(np.isnan(predictions)))
# mse = mean_squared_error(y_valid.values, predictions)

# print("***************** MSE:", mse)
# print(rf_test.extended_grouped_estimators_[0][0].tree_.max_depth)
# print(rf_test.extended_grouped_estimators_[0][0].tree_.n_node_samples)

# print(rf_test.estimators_[0].tree_.max_depth)
# print(rf_test.estimators_[0].tree_.n_node_samples)
# print(rf_test.group_size)
# print(mse)

plt.figure(figsize=(20, 20))
plot_tree(rf_test.estimators_[0])
plt.savefig('initial_tree.png')
plt.close()

plt.figure(figsize=(20, 20))
plot_tree(rf_test.initial_estimators_[0])
plt.savefig('initial_tree_all_samples.png')
plt.close()

plt.figure(figsize=(40, 40))
plot_tree(rf_test.extended_grouped_estimators_[0][0])
plt.savefig('extended_tree.png')
plt.close()

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