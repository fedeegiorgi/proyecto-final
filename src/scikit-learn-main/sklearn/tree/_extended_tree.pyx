from _tree cimport (
    DepthFirstTreeBuilder,
    Tree,
    Splitter,
    Node,
    ParentInfo,
    StackRecord,
    SplitRecord,
    intp_t, float64_t, uint8_t)

# BEGIN original imports and set-up

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

# Build the corresponding numpy dtype for Node.
# This works by casting `dummy` to an array of Node of length 1, which numpy
# can construct a `dtype`-object for. See https://stackoverflow.com/q/62448946
# for a more detailed explanation.
cdef Node dummy
NODE_DTYPE = np.asarray(<Node[:1]>(&dummy)).dtype

cdef inline void _init_parent_record(ParentInfo* record) noexcept nogil:
    record.n_constant_features = 0
    record.impurity = INFINITY
    record.lower_bound = -INFINITY
    record.upper_bound = INFINITY

# END original imports and set-up

cdef class DepthFirstTreeExtensionBuilder(DepthFirstTreeBuilder):
    """
    Extended depth-first tree builder based on initial tree with additional features .
    """

    def __cinit__(self, Splitter splitter, intp_t min_samples_split,
                  intp_t min_samples_leaf, float64_t min_weight_leaf,
                  intp_t max_depth, float64_t min_impurity_decrease,
                  cnp.ndarray[cnp.double_t] parents,
                  cnp.ndarray[cnp.double_t] is_lefts,
                  cnp.ndarray[cnp.double_t] is_leafs,
                  cnp.ndarray[cnp.double_t] features,
                  cnp.ndarray[cnp.double_t] thresholds,
                  cnp.ndarray[cnp.double_t] impurities,
                  cnp.ndarray[cnp.double_t] n_node_samples,
                  cnp.ndarray[cnp.double_t] weighted_n_node_samples,
                  cnp.ndarray[cnp.double_t] missing_go_to_lefts
                ):
        
        # Call the parent class constructor
        DepthFirstTreeBuilder.__cinit__(
            self, splitter, min_samples_split, min_samples_leaf,
            min_weight_leaf, max_depth, min_impurity_decrease
        )

        # Ensure the input arrays are one-dimensional
        if (parents.ndim != 1 or is_lefts.ndim != 1 or is_leafs.ndim != 1 or 
            features.ndim != 1 or thresholds.ndim != 1 or 
            impurities.ndim != 1 or n_node_samples.ndim != 1 or 
            weighted_n_node_samples.ndim != 1 or missing_go_to_lefts.ndim != 1):
            raise ValueError("All parameters must be one-dimensional NumPy arrays.")
        
        # Assign the memoryviews
        self.parents = parents
        self.is_lefts = is_lefts
        self.is_leafs = is_leafs
        self.features = features
        self.thresholds = thresholds
        self.impurities = impurities
        self.n_node_samples = n_node_samples
        self.weighted_n_node_samples = weighted_n_node_samples
        self.missing_go_to_lefts = missing_go_to_lefts

        # Initialize the stack to store the leafs of the initial tree
        self.initial_builder_stack = stack[StackRecord]()

        # Using the new parameters load the tree
        self._load_initial_tree()

    cpdef _load_initial_tree(
        self, 
        Tree tree
    ):
        """
        Load the tree from lists containing node information.
        """
        cdef int n_nodes = self.parents.shape[0] # Get the number of nodes from the parents array
        cdef int i
        # Defino un vector split_pos --> contiene la posicion de break del nodo
        for i in range(n_nodes):
            if not self.is_leafs[i]:
                
                # Compute the split position with the threshold and feature used
                threshold = self.thresholds[i]
                feature = self.features[i]

                # Instancio un splitter heredado de BestSplitter
                #   tiene un método recompute_node_split(t, f) --> roba de node_best_split() lines 503-524 + 374
                #
                #   Adentro tiene un partitioner heredado de DensePartitioner
                #       este Partitioner tiene cuasi copia del sort_samples_and_feature_values (sin missings)
                #       y el find_split_pos (sin missings) --> con búsqueda binaria?

                # Llamo a sort_samples_and_feature_values(feature) --> internamente feature_values se ordenan
                # Llamo a p = find_split_pos(threshold) --> internamente se busca la posicion del threshold en feature_values
                
                # split_pos.append(p)

                tree._add_node(
                    self.parents[i], self.is_lefts[i], self.is_leafs[i],
                    self.features[i], self.thresholds[i], self.impurities[i],
                    self.n_node_samples[i], self.weighted_n_node_samples[i],
                    self.missing_go_to_lefts[i]
                ) 
            else:
                

                pass
                # if self.is_lefts[i]:
                #     # Push left leaf on stack
                #     self.initial_builder_stack.push({
                #         "start": start,
                #         "end": split.pos,
                #         "depth": depth + 1, # se consgiue
                #         "parent": node_id, # se consgiue (?
                #         "is_left": 1,
                #         "impurity": split.impurity_left,
                #         "n_constant_features": parent_record.n_constant_features,
                #         "lower_bound": left_child_min,
                #         "upper_bound": left_child_max,
                #     })
                # else:
                #     # Push right leaf on stack
                #     self.initial_builder_stack.push({
                #         "start": split.pos,
                #         "end": end,
                #         "depth": depth + 1,
                #         "parent": node_id,
                #         "is_left": 0,
                #         "impurity": split.impurity_right,
                #         "n_constant_features": parent_record.n_constant_features,
                #         "lower_bound": right_child_min,
                #         "upper_bound": right_child_max,
                #     })

    cpdef build(
        self,
        Tree tree,
        object X,
        const float64_t[:, ::1] y,
        const float64_t[:] sample_weight=None,
        const uint8_t[::1] missing_values_in_feature_mask=None,
    ):
        """
        Build a decision tree with additional features.
        """

        # check input
        X, y, sample_weight = self._check_input(X, y, sample_weight)
        
        # Initial capacity
        cdef intp_t init_capacity

        if tree.max_depth <= 10:
            init_capacity = <intp_t> (2 ** (tree.max_depth + 1)) - 1
        else:
            init_capacity = 2047

        tree._resize(init_capacity)

        # Parameters
        cdef Splitter splitter = self.splitter
        cdef intp_t max_depth = self.max_depth
        cdef intp_t min_samples_leaf = self.min_samples_leaf
        cdef float64_t min_weight_leaf = self.min_weight_leaf
        cdef intp_t min_samples_split = self.min_samples_split
        cdef float64_t min_impurity_decrease = self.min_impurity_decrease

        # Recursive partition (without actual recursion)
        splitter.init(X, y, sample_weight, missing_values_in_feature_mask)

        cdef intp_t start
        cdef intp_t end
        cdef intp_t depth
        cdef intp_t parent
        cdef bint is_left
        cdef intp_t n_node_samples = splitter.n_samples
        cdef float64_t weighted_n_node_samples
        cdef SplitRecord split
        cdef intp_t node_id

        cdef float64_t middle_value
        cdef float64_t left_child_min
        cdef float64_t left_child_max
        cdef float64_t right_child_min
        cdef float64_t right_child_max
        cdef bint is_leaf
        cdef bint first = 1
        cdef intp_t max_depth_seen = -1
        cdef int rc = 0

        cdef stack[StackRecord] builder_stack
        cdef StackRecord stack_record

        cdef ParentInfo parent_record
        # probablemente hay q inicializarlo distinto
        _init_parent_record(&parent_record)