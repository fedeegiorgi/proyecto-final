from node import Node

class TreeCombiner:
    def __init__(self, trees, X_union, y_union):
        self.trees = trees
        self.root = Node()
        self.X = X_union
        self.y = y_union
    
    def combiner(self, node, depth=0):
        if depth >= len(self.trees):
            return
        
        tree = self.trees[depth].tree_

        node.feature = tree.feature[0]
        node.threshold = tree.threshold[0]
        node.left = Node()
        node.right = Node()

        self.combiner(node.left, depth + 1)
        self.combiner(node.right, depth + 1)

        if depth == len(self.trees) - 1:
            node.left.value = tree.value[1][0][0]
            node.right.value = tree.value[2][0][0]

    def _set_values_to_zero(self, node):
        if node.value is not None:
            node.value = 0
            node.len_values = 0
            return

        if node.left:
            self._set_values_to_zero(node.left)
        if node.right:
            self._set_values_to_zero(node.right)

    def set_values_to_zero(self):
        self._set_values_to_zero(self.root)
    
    def recompute_values(self):
        self.set_values_to_zero()
        
        for observation, asociated_value in zip(self.X, self.y):
            node = self.root
            while node.value is None:
                obs_ft_val  = observation[node.feature]
                if obs_ft_val <= node.threshold:
                    node = node.left
                else:
                    node = node.right
                if node.value is not None:
                    node.value = ((node.value * node.len_values) + asociated_value) / (node.len_values + 1)
                    node.len_values += 1

    def combine_trees(self):
        self.combiner(self.root, 0)
        self.recompute_values()

    def predict(self, observation):
        node = self.root
        value = node.value
        while value is None:
            obs_ft_val  = observation[node.feature]
            if obs_ft_val <= node.threshold:
                node = node.left
            else:
                node = node.right
            if node.value is not None:
                value = node.value
        return value

    def print_tree(self, node, depth=0, prefix="Root"):
        indent = "    " * depth
        
        if node.value is not None:
            print(f"{indent}{prefix} --> [Leaf Value: {node.value:.2f}]")

        else:
            print(f"{indent}{prefix} --> [X[{node.feature}] <= {node.threshold:.2f}]")

            if node.left:
                self.print_tree(node.left, depth + 1, prefix="L")
            if node.right:
                self.print_tree(node.right, depth + 1, prefix="R")
    
    def __str__(self):
        self.print_tree(self.root)
        return ''