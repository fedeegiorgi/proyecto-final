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

SEED = 14208

df = pd.read_csv('../datasets/Carbon_Emission_transformed.csv')

# Preprocesamiento de datos
train_df, validation_df = train_test_split(df, test_size=0.2, random_state=SEED)
y_train, y_valid = train_df['CarbonEmission'], validation_df['CarbonEmission']
X_train, X_valid = train_df.drop('CarbonEmission', axis=1), validation_df.drop('CarbonEmission', axis=1)
X_train = pd.get_dummies(X_train)
X_valid = pd.get_dummies(X_valid)
X_train, X_valid = X_train.align(X_valid, join='left', axis=1, fill_value=0)

# Variables
initial_max_depth = 7
group_size = 3

rf_test = SharedKnowledgeRandomForestRegressor(random_state=SEED, n_estimators=90, group_size=group_size, initial_max_depth=initial_max_depth, max_depth=None)
# rf_test.fit(X, y)
rf_test.fit(X_train.values, y_train.values)

plt.figure(figsize=(60, 60))
plot_tree(rf_test.estimators_[0])
plt.savefig('initial_tree.png')
plt.close()

plt.figure(figsize=(60, 60))
plot_tree(rf_test.initial_estimators_[0])
plt.savefig('initial_tree_all_samples.png')
plt.close()

plt.figure(figsize=(100, 100))
plot_tree(rf_test.extended_grouped_estimators_[0][0])
plt.savefig('extended_tree.png')
plt.close()

preds = rf_test.predict(X_valid.values)
mse = mean_squared_error(y_valid, preds)
print("***************** MSE:", mse)

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

# Function to set values smaller than 0.001 to 0.0
def adjust_impurity(impurity_list, threshold=0.001):
    return [0.0 if abs(value) < threshold else value for value in impurity_list]

grouped_initial_estimators = group_split(rf_test.initial_estimators_, group_size)

for i, group in enumerate(grouped_initial_estimators):
    for j, tree in enumerate(group):
        # print("Group:", i)
        # print("Tree:", j)
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
        assert np.allclose(ext_tree_val, int_tree_val, rtol=1e-2), f"Values do not match: {ext_tree_val} != {int_tree_val} at tree {j}, group {i}"

        # Impurity
        ext_tree_impurity = adjust_impurity(ext_tree.impurity[indices_at_initial_max_depth].tolist())
        int_tree_impurity = adjust_impurity(int_tree.impurity[indices_leaves_nodes].tolist())
        assert np.allclose(ext_tree_impurity, int_tree_impurity, rtol=1e-2), f"Impurities do not match: {ext_tree_impurity} != {int_tree_impurity} at tree {j}, group {i}"

        # Node samples
        # ext_tree_n_samples = ext_tree.n_node_samples[indices_at_initial_max_depth].tolist()
        # int_tree_n_samples = int_tree.n_node_samples[indices_leaves_nodes].tolist()
        # assert np.allclose(ext_tree_n_samples, int_tree_n_samples, rtol=1e-1), f"Node samples do not match: {ext_tree_n_samples} != {int_tree_n_samples} at tree {j}, group {i}"

print("PASÉ TODO LOS TESTS!!!")