cdef class Node:
    cdef int feature
    cdef double threshold
    cdef Node left
    cdef Node right
    cdef double value
    cdef int len_values

    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature if feature is not None else -1
        self.threshold = threshold if threshold is not None else 0.0
        self.left = left
        self.right = right
        self.value = value if value is not None else 0.0
        self.len_values = 0

cdef class TreeCombiner:
    cdef list trees
    cdef Node root
    cdef list X
    cdef list y

    def __init__(self, trees, X_union, y_union):
        self.trees = trees
        self.root = Node()
        self.X = X_union
        self.y = y_union
    
    cdef void _combiner(self, Node node, int depth=0):
        if depth >= len(self.trees):
            return
        
        cdef tree = self.trees[depth].tree_

        node.feature = tree.feature[0]
        node.threshold = tree.threshold[0]
        node.left = Node()
        node.right = Node()

        self._combiner(node.left, depth + 1)
        self._combiner(node.right, depth + 1)

        if depth == len(self.trees) - 1:
            node.left.value = tree.value[1][0][0]
            node.right.value = tree.value[2][0][0]

    cdef void _set_values_to_zero(self, Node node):
        if node.value is not None:
            node.value = 0.0
            node.len_values = 0
            return

        if node.left is not None:
            self._set_values_to_zero(node.left)
        if node.right is not None:
            self._set_values_to_zero(node.right)

    cdef void set_values_to_zero(self):
        self._set_values_to_zero(self.root)

    cdef void recompute_values(self):
        self.set_values_to_zero()
        
        cdef:
            int i
            double obs_ft_val
            double asociated_value
            Node node
        
        for i in range(len(self.X)):
            observation = self.X[i]
            asociated_value = self.y[i]
            node = self.root
            
            while node.value is None:
                obs_ft_val = observation[node.feature]
                if obs_ft_val <= node.threshold:
                    node = node.left
                else:
                    node = node.right
                
                if node.value is not None:
                    node.value = ((node.value * node.len_values) + asociated_value) / (node.len_values + 1)
                    node.len_values += 1

    cdef combine_trees(self):
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
