# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

from cpython cimport Py_INCREF, PyObject, PyTypeObject

from libc.stdlib cimport free
from libc.string cimport memcpy
from libc.string cimport memset
from libc.stdint cimport INTPTR_MAX
from libc.math cimport isnan
from libcpp.vector cimport vector
from libcpp.algorithm cimport pop_heap
from libcpp.algorithm cimport push_heap
from libcpp.stack cimport stack
from libcpp cimport bool

import struct

import numpy as np
cimport numpy as cnp
cnp.import_array()

from scipy.sparse import issparse
from scipy.sparse import csr_matrix

from ._utils cimport safe_realloc
from ._utils cimport sizet_ptr_to_ndarray

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
    intp_t parent
    bint is_left
    bint is_leaf
    intp_t feature
    float64_t threshold
    float64_t impurity
    intp_t n_node_samples
    intp_t weighted_n_node_samples
    intp_t missing_go_to_left
    intp_t node_depth

cdef class TreeCombiner(Tree):
    def __cinit__(
        self, 
        intp_t n_features, 
        cnp.ndarray n_classes, 
        intp_t n_outputs
    ):
        """
        Cython constructor (__cinit__) for TreeCombiner, calling the base Tree's __cinit__.
        """
        # Call the parent Tree class's __cinit__
        Tree(n_features, n_classes, n_outputs)

    cpdef void combiner(self, cnp.ndarray features, cnp.ndarray thresholds, 
                cnp.ndarray impurities, cnp.ndarray n_node_samples, 
                cnp.ndarray weighted_n_node_samples, cnp.ndarray missing_go_to_lefts):
        """Combines the trees into a single one using first split."""
        
        cdef intp_t final_depth = features.shape[0]
        cdef bint is_child_leaf

        print(features.shape[0])
        print(final_depth)
        self.max_depth = final_depth

        if self.max_depth <= 10:
            init_capacity = <intp_t> (2 ** (self.max_depth + 1)) - 1
        else:
            init_capacity = 2047

        self._resize(init_capacity)

        cdef stack[StackNode] builder_stack
        cdef intp_t depth = 0

        builder_stack.push({
            "parent": -1,
            "is_left": 0,
            "is_leaf": 0,
            "feature": features[depth],
            "threshold": thresholds[depth],
            "impurity": impurities[depth],
            "n_node_samples": n_node_samples[depth],
            "weighted_n_node_samples": weighted_n_node_samples[depth],
            "missing_go_to_left": missing_go_to_lefts[depth],
            "node_depth": depth
        })

        while not builder_stack.empty():
            current = builder_stack.top()
            builder_stack.pop()
            
            parent = current.parent
            is_left = current.is_left
            is_leaf = current.is_leaf
            feature = current.feature
            threshold = current.threshold
            impurity = current.impurity
            n_node_samples_i = current.n_node_samples
            weighted_n_node_samples_i = current.weighted_n_node_samples
            missing_go_to_left = current.missing_go_to_left
            depth = current.node_depth

            node_id = self._add_node(parent, is_left, is_leaf, feature,
                                    threshold, impurity,
                                    n_node_samples_i, weighted_n_node_samples_i,
                                    missing_go_to_left)

            if not is_leaf:
                depth += 1
                is_child_leaf = features.shape[0] - 1 == depth

                # Push right child on stack
                builder_stack.push({
                    "parent": node_id,
                    "is_left": 0,
                    "is_leaf": is_child_leaf,
                    "feature": features[depth],
                    "threshold": thresholds[depth],
                    "impurity": impurities[depth],
                    "n_node_samples": n_node_samples[depth],
                    "weighted_n_node_samples": weighted_n_node_samples[depth],
                    "missing_go_to_left": missing_go_to_lefts[depth],
                    "node_depth": depth
                })

                # Push left child on stack
                builder_stack.push({
                    "parent": node_id,
                    "is_left": 1,
                    "is_leaf": is_child_leaf,
                    "feature": features[depth],
                    "threshold": thresholds[depth],
                    "impurity": impurities[depth],
                    "n_node_samples": n_node_samples[depth],
                    "weighted_n_node_samples": weighted_n_node_samples[depth],
                    "missing_go_to_left": missing_go_to_lefts[depth],
                    "node_depth": depth
                })

    cpdef recompute_values(self, cnp.ndarray out, cnp.ndarray y):
        cdef cnp.ndarray values = np.zeros(self.node_count, dtype=np.float64)

        for i in range(y.shape[0]):
            values[out[i]] += y[i]

        cdef cnp.ndarray counts = np.zeros(self.node_count, dtype=np.intp)

        for num in out:
            counts[num] += 1

        for i in range(self.node_count):
            if counts[i] > 0:
                values[i] /= counts[i]
                self.value[i] = values[i]
                
        print("Llegué a recomputar los values")
        # self.value = <float64_t *> values.data # Cannot convert Python object to 'float64_t *'