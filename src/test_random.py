from sklearn.ensemble import RFRegressorFirstSplitCombiner, OOBRandomForestRegressorGroups
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.tree import DecisionTreeRegressorCombiner, plot_tree
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

SEED = 14208

df = pd.read_csv('distribucion/datasets/Height.csv')

# Preprocesamiento de datos
train_df, validation_df = train_test_split(df, test_size=0.2, random_state=SEED)
y_train, y_valid = train_df['childHeight'], validation_df['childHeight']
X_train, X_valid = train_df.drop('childHeight', axis=1), validation_df.drop('childHeight', axis=1)
X_train = pd.get_dummies(X_train)
X_valid = pd.get_dummies(X_valid)
X_train, X_valid = X_train.align(X_valid, join='left', axis=1, fill_value=0)

# # Inicialización del modelo
# rf = RandomForestRegressor(random_state=SEED, n_estimators=20, max_depth=1)
# rf.fit(X_train, y_train)

# trees = rf.estimators_
# samples_used = rf.estimators_samples_

# print("Ya cree los arboles con RF")

# samplesidx = set()
# for samples in samples_used:
#     for sampleidx in samples:
#         samplesidx.add(sampleidx)
# samplesidx = list(samplesidx)

# X_union = X_train.iloc[samplesidx].to_numpy()
# y_union = y_train.iloc[samplesidx].to_numpy()

# print("Llamo al decision combiner... reza malena")
# combinador = DecisionTreeRegressorCombiner(initial_trees=trees)
# combinador.fit(X_union, y_union)

rf_test = RFRegressorFirstSplitCombiner(random_state=SEED, n_estimators=30, group_size=3)
rf_test.fit(X_train.values, y_train.values)

predictions = rf_test.predict(X_valid.values)
mse = mean_squared_error(y_valid, predictions)

print(rf_test.group_size)
print(mse)

# rf_test = RFRegressorFirstSplitCombiner(random_state=SEED, n_estimators=100, group_size=5)
# rf_test.fit(X_train.values, y_train.values)

# predictions = rf_test.predict(X_valid.values)
# mse = mean_squared_error(y_valid, predictions)

# print(rf_test.group_size)
# print(mse)

plt.figure(figsize=(20, 20))
plot_tree(rf_test.combined_trees[0])
plt.savefig('first_split_tree.png')
plt.close()

plt.figure(figsize=(20, 20))
plot_tree(rf_test.estimators_[0])
plt.savefig('initial_tree_0.png')
plt.close()

plt.figure(figsize=(20, 20))
plot_tree(rf_test.estimators_[1])
plt.savefig('initial_tree_1.png')
plt.close()

plt.figure(figsize=(20, 20))
plot_tree(rf_test.estimators_[2])
plt.savefig('initial_tree_2.png')
plt.close()

print(rf_test.combined_trees[9].tree_.node_count)
print(rf_test.combined_trees[9].tree_.capacity)
print(len(rf_test.combined_trees))