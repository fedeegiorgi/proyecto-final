# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

from cpython cimport Py_INCREF, PyObject, PyTypeObject

from libcpp.stack cimport stack
from libcpp cimport bool

import numpy as np
cimport numpy as cnp
cnp.import_array()

cdef extern from "numpy/arrayobject.h":
    object PyArray_NewFromDescr(PyTypeObject* subtype, cnp.dtype descr,
                                int nd, cnp.npy_intp* dims,
                                cnp.npy_intp* strides,
                                void* data, int flags, object obj)
    int PyArray_SetBaseObject(cnp.ndarray arr, PyObject* obj)

# =============================================================================
# Types and constants
# =============================================================================

from numpy import float32 as DTYPE
from numpy import float64 as DOUBLE

cdef float64_t INFINITY = np.inf
cdef float64_t EPSILON = np.finfo('double').eps

# Some handy constants (BestFirstTreeBuilder)
cdef bint IS_FIRST = 1
cdef bint IS_NOT_FIRST = 0
cdef bint IS_LEFT = 1
cdef bint IS_NOT_LEFT = 0

TREE_LEAF = -1
TREE_UNDEFINED = -2
cdef intp_t _TREE_LEAF = TREE_LEAF
cdef intp_t _TREE_UNDEFINED = TREE_UNDEFINED

############################################
from ._tree cimport Tree

cdef struct StackNode:
    intp_t depth
    intp_t parent
    bint is_left
    bint is_leaf
    intp_t feature
    float64_t threshold

cdef class DepthFirstTreeCombinerBuilder(TreeBuilder):
    def __cinit__(self, Splitter splitter, intp_t min_samples_split,
                  intp_t min_samples_leaf, float64_t min_weight_leaf,
                  intp_t max_depth, float64_t min_impurity_decrease):
        self.splitter = splitter
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_leaf = min_weight_leaf
        self.max_depth = max_depth
        self.min_impurity_decrease = min_impurity_decrease        

    cpdef combiner(self, Tree tree, cnp.ndarray features, cnp.ndarray thresholds):
        """Combines the trees into a single one using first split."""
        
        cdef intp_t final_depth = features.shape[0]
        self.features = features
        self.thresholds = thresholds

        cdef intp_t depth
        cdef intp_t parent
        cdef bint is_left
        cdef bint is_leaf
        cdef intp_t feature
        cdef float64_t threshold

        cdef bint is_child_leaf
        cdef StackNode current
        cdef intp_t node_id

        tree.max_depth = final_depth

        if tree.max_depth <= 10:
            init_capacity = <intp_t> (2 ** (tree.max_depth + 1)) - 1
        else:
            init_capacity = 2047

        tree._resize(init_capacity)

        cdef stack[StackNode] builder_stack
        depth = 0

        with nogil:
            # Push root node onto stack
            builder_stack.push({
                "depth": depth,
                "parent": _TREE_UNDEFINED,
                "is_left": 0,
                "is_leaf": 0,
                "feature": self.features[depth],
                "threshold": self.thresholds[depth]
            })

            while not builder_stack.empty():
                current = builder_stack.top()
                builder_stack.pop()
                
                depth = current.depth
                parent = current.parent
                is_left = current.is_left
                is_leaf = current.is_leaf
                feature = current.feature
                threshold = current.threshold               

                node_id = tree._add_node(parent, is_left, is_leaf, feature,
                                        threshold, 0, 0, 0, 0)

                if not is_leaf:
                    depth += 1
                    is_child_leaf = features.shape[0] == depth

                    if not is_child_leaf:
                        # Push right child on stack
                        builder_stack.push({
                            "depth": depth,
                            "parent": node_id,
                            "is_left": 0,
                            "is_leaf": is_child_leaf,
                            "feature": self.features[depth],
                            "threshold": self.thresholds[depth]
                        })

                        # Push left child on stack
                        builder_stack.push({
                            "depth": depth,
                            "parent": node_id,
                            "is_left": 1,
                            "is_leaf": is_child_leaf,
                            "feature": self.features[depth],
                            "threshold": self.thresholds[depth]
                        })

                    else:
                        # Push right child (leaf) on stack
                        builder_stack.push({
                            "depth": depth,
                            "parent": node_id,
                            "is_left": 0,
                            "is_leaf": is_child_leaf,
                            "feature": 0,
                            "threshold": 0
                        })

                        # Push left child (leaf) on stack
                        builder_stack.push({
                            "depth": depth,
                            "parent": node_id,
                            "is_left": 1,
                            "is_leaf": is_child_leaf,
                            "feature": 0,
                            "threshold": 0
                        })

    cpdef recompute_values(self, Tree tree, cnp.ndarray out, cnp.ndarray y):
        cdef float64_t[:] values = np.zeros(tree.node_count, dtype=np.float64)
        cdef intp_t[:] counts = np.zeros(tree.node_count, dtype=np.intp)

        cdef intp_t j
        for j in range(y.shape[0]):
            values[out[j]] += y[j]
            counts[out[j]] += 1

        cdef float64_t* dest
        cdef intp_t i
        for i in range(tree.node_count):
            if counts[i] > 0:
                values[i] /= counts[i]
                dest = tree.value + i * tree.value_stride
                dest[0] = values[i]
