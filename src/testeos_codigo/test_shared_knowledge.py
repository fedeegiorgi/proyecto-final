from sklearn.ensemble import SharedKnowledgeRandomForestRegressor, OOBRandomForestRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.tree import DecisionTreeRegressorCombiner, plot_tree
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import make_regression
# Crear datos de ejemplo
X, y = make_regression(n_samples=50, n_features=10, random_state=0)
print("Original X:", X)
SEED = 14208

df = pd.read_csv('../distribucion/datasets/laptops.csv')

# Preprocesamiento de datos
train_df, validation_df = train_test_split(df, test_size=0.2, random_state=SEED)
y_train, y_valid = train_df['Price'], validation_df['Price']
X_train, X_valid = train_df.drop('Price', axis=1), validation_df.drop('Price', axis=1)
X_train = pd.get_dummies(X_train)
X_valid = pd.get_dummies(X_valid)
X_train, X_valid = X_train.align(X_valid, join='left', axis=1, fill_value=0)

initial_max_depth = 3
# print(X.shape)
rf_test = SharedKnowledgeRandomForestRegressor(random_state=SEED, n_estimators=9, group_size=3, initial_max_depth=initial_max_depth, max_depth=5)
# rf_test = OOBRandomForestRegressorGroups(random_state=SEED, n_estimators=30, group_size=3)
# rf_test = RandomForestRegressor(random_state=SEED, n_estimators=30, max_depth=5)
# rf_test.fit(X, y)
rf_test.fit(X_train.values, y_train.values)

# print("original_X")
# print(X)
# print(rf_test.estimators_samples_[0])
# samples_used = rf_test.estimators_samples_[0]
# print("new_X")
# print(X[samples_used])

# predictions = rf_test.predict(X_valid.values)
# print(np.any(np.isnan(predictions)))
# mse = mean_squared_error(y_valid, predictions)

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
plot_tree(rf_test.initial_estimators_[5])
plt.savefig('initial_tree_all_samples.png')
plt.close()

plt.figure(figsize=(20, 20))
plot_tree(rf_test.extended_grouped_estimators_[1][1])
plt.savefig('extended_tree.png')
plt.close()

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


######################################################################

# TESTEO ALTERNATIVA D

def group_split(iterable, group_size):
    # Create a list of lists by slicing the iterable into groups of size `self._n_groups`
    grouped = []
    n = len(iterable) // group_size
    for i in range(n):
        lower_bound = i * group_size
        upper_bound = (i + 1) * group_size
        group = iterable[lower_bound:upper_bound]
        grouped.append(group)
    return grouped

grouped_initial_estimators = group_split(rf_test.initial_estimators_, 3)

for i, group in enumerate(grouped_initial_estimators):
    for j, tree in enumerate(group):
        print("Group:", i)
        print("Tree:", j)
        # print("Tree:", tree.tree_.node_count)
        # print("Tree:", tree.tree_.capacity)
        # print("Tree:", tree.tree_.max_depth)
        # print("Tree:", tree.tree_.n_node_samples)
        # print("Tree:", tree.tree_.impurity)
        # print("Tree:", tree.tree_.threshold)
        # print("Tree:", tree.tree_.children_left)
        # print("Tree:", tree.tree_.children_right)
        # print("Tree:", tree.tree_.weighted_n_node_samples)
        # print("Tree:", tree.tree_.feature)
        # print("Tree:", tree.tree_.value)

        ext_tree = rf_test.extended_grouped_estimators_[i][j].tree_
        ext_tree.children_left
        ext_tree.children_right

        ext_n_nodes = ext_tree.node_count

        parents = np.full(ext_n_nodes, -1)  # Initialize parents array with -1
        depths = np.full(ext_n_nodes, 0) # Initialize depths in 0

        for parent, (left, right) in enumerate(zip(ext_tree.children_left, ext_tree.children_right)):
            if left != -1:  # If there is a left child
                parents[left] = parent
                depths[left] = depths[parent] + 1 # Increment depth for left child
            if right != -1:  # If there is a right child
                parents[right] = parent
                depths[right] = depths[parent] + 1 # Increment depth for right child

        indices_at_initial_max_depth = np.where(depths == initial_max_depth)[0]
        
        int_tree = tree.tree_
        int_n_nodes = int_tree.node_count
        int_parents = np.full(int_n_nodes, -1)  # Initialize parents array with -1
        int_depths = np.full(int_n_nodes, 0) # Initialize depths in 0

        for parent, (left, right) in enumerate(zip(int_tree.children_left, int_tree.children_right)):
            if left != -1:  # If there is a left child
                int_parents[left] = parent
                int_depths[left] = int_depths[parent] + 1 # Increment depth for left child
            if right != -1:  # If there is a right child
                int_parents[right] = parent
                int_depths[right] = int_depths[parent] + 1 # Increment depth for right child

        indices_leaves_nodes = np.where(int_depths == initial_max_depth)[0]

        # Values
        ext_tree_val = ext_tree.value[indices_at_initial_max_depth].flatten().tolist()
        int_tree_val = int_tree.value[indices_leaves_nodes].flatten().tolist()
        assert np.allclose(ext_tree_val, int_tree_val, rtol=1e-4), f"Values do not match: {ext_tree_val} != {int_tree_val} at tree {j}, group {i}"

        # Impurity
        ext_tree_impurity = ext_tree.impurity[indices_at_initial_max_depth].tolist()
        int_tree_impurity = int_tree.impurity[indices_leaves_nodes].tolist()
        assert np.allclose(ext_tree_impurity, int_tree_impurity, rtol=1e-4), f"Impurities do not match: {ext_tree_impurity} != {int_tree_impurity}"

        # Node samples
        ext_tree_n_samples = ext_tree.n_node_samples[indices_at_initial_max_depth].tolist()
        int_tree_n_samples = int_tree.n_node_samples[indices_leaves_nodes].tolist()
        assert np.allclose(ext_tree_n_samples, int_tree_n_samples, rtol=1e-4), f"Node samples do not match: {ext_tree_n_samples} != {int_tree_n_samples}"


"""
        # Threshold
        ext_tree_threshold = ext_tree.threshold[indices_at_initial_max_depth].tolist()
        int_tree_threshold = int_tree.threshold[indices_leaves_nodes].tolist()
        assert np.allclose(ext_tree_threshold, int_tree_threshold, rtol=1e-4), f"Thresholds do not match: {ext_tree_threshold} != {int_tree_threshold}"

        # Feature
        ext_tree_feature = ext_tree.feature[indices_at_initial_max_depth].tolist()
        int_tree_feature = int_tree.feature[indices_leaves_nodes].tolist()
        assert np.allclose(ext_tree_feature, int_tree_feature, rtol=1e-4), f"Features do not match: {ext_tree_feature} != {int_tree_feature}"
"""