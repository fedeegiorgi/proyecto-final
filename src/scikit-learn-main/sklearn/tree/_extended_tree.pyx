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

cdef class DepthFirstTreeExtensionBuilder(TreeBuilder):
    """
    Extended depth-first tree builder based on initial tree with additional features .
    """

    def __cinit__(self, Splitter splitter, intp_t min_samples_split,
                  intp_t min_samples_leaf, float64_t min_weight_leaf,
                  intp_t max_depth, float64_t min_impurity_decrease):
        self.splitter = splitter
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_leaf = min_weight_leaf
        self.max_depth = max_depth
        self.min_impurity_decrease = min_impurity_decrease

        # Initialize the stack to store the leafs of the initial tree
        self.builder_stack = stack[StackRecord]()
   
    cdef _continue_training(self, Tree tree):
        """
        Continue training the tree with unique node loaded in builder_stack.
        """
        # Parameters
        cdef intp_t max_depth = self.max_depth
        cdef intp_t min_samples_leaf = self.min_samples_leaf
        cdef float64_t min_weight_leaf = self.min_weight_leaf
        cdef intp_t min_samples_split = self.min_samples_split
        cdef float64_t min_impurity_decrease = self.min_impurity_decrease

        cdef intp_t start
        cdef intp_t end
        cdef intp_t depth
        cdef intp_t parent
        cdef bint is_left
        cdef intp_t n_node_samples = self.splitter.n_samples
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

        cdef StackRecord stack_record

        cdef ParentInfo parent_record
        # _init_parent_record(&parent_record)

        with nogil:
            while not self.builder_stack.empty():
                stack_record = self.builder_stack.top()
                self.builder_stack.pop()

                start = stack_record.start
                end = stack_record.end
                depth = stack_record.depth
                parent = stack_record.parent
                is_left = stack_record.is_left
                parent_record.impurity = stack_record.impurity
                parent_record.n_constant_features = stack_record.n_constant_features
                parent_record.lower_bound = stack_record.lower_bound
                parent_record.upper_bound = stack_record.upper_bound

                n_node_samples = end - start
                self.splitter.node_reset(start, end, &weighted_n_node_samples)

                is_leaf = (depth >= max_depth or
                           n_node_samples < min_samples_split or
                           n_node_samples < 2 * min_samples_leaf or
                           weighted_n_node_samples < 2 * min_weight_leaf)

                # if first:
                #     with gil:
                #         print("---- Continue training root ----")
                #         print("* Parent record impurity:", parent_record.impurity)
                #         print("* Start:", start)
                #         print("* End:", end)
                #     first = 0 

                # impurity == 0 with tolerance due to rounding errors
                # with gil:
                    # print("Es leaf?:", is_leaf)

                is_leaf = is_leaf or parent_record.impurity <= EPSILON

                # with gil:
                    # print("Es leaf post chequeo epsilon?:", is_leaf)

                if not is_leaf:
                    self.splitter.node_split(
                        &parent_record,
                        &split,
                    )
                    # If EPSILON=0 in the below comparison, float precision
                    # issues stop splitting, producing trees that are
                    # dissimilar to v0.18
                    is_leaf = (is_leaf or split.pos >= end or
                               (split.improvement + EPSILON <
                                min_impurity_decrease))
                    # with gil:
                    #     print("Es leaf post chequeo split?:", is_leaf)
                    #     print(f"La feature encontrada fue: {split.feature}")
                node_id = tree._add_node(parent, is_left, is_leaf, split.feature,
                                            split.threshold, parent_record.impurity,
                                            n_node_samples, weighted_n_node_samples,
                                            split.missing_go_to_left)

                with gil:
                    if split.feature > 205:
                        print("!!!!!!!!!!!!!!!!!!!!!Use feature amiga!!!!!!!!!!!!!!!!!!!!!!:", split.feature)

                if node_id == INTPTR_MAX:
                    rc = -1
                    break

                # Store value for all nodes, to facilitate tree/model
                # inspection and interpretation
                self.splitter.node_value(tree.value + node_id * tree.value_stride)
                if self.splitter.with_monotonic_cst:
                    self.splitter.clip_node_value(tree.value + node_id * tree.value_stride, parent_record.lower_bound, parent_record.upper_bound)

                if not is_leaf:
                    if (
                        not self.splitter.with_monotonic_cst or
                        self.splitter.monotonic_cst[split.feature] == 0
                    ):
                        # Split on a feature with no monotonicity constraint

                        # Current bounds must always be propagated to both children.
                        # If a monotonic constraint is active, bounds are used in
                        # node value clipping.
                        left_child_min = right_child_min = parent_record.lower_bound
                        left_child_max = right_child_max = parent_record.upper_bound
                    elif self.splitter.monotonic_cst[split.feature] == 1:
                        # Split on a feature with monotonic increase constraint
                        left_child_min = parent_record.lower_bound
                        right_child_max = parent_record.upper_bound

                        # Lower bound for right child and upper bound for left child
                        # are set to the same value.
                        middle_value = self.splitter.criterion.middle_value()
                        right_child_min = middle_value
                        left_child_max = middle_value
                    else:  # i.e. splitter.monotonic_cst[split.feature] == -1
                        # Split on a feature with monotonic decrease constraint
                        right_child_min = parent_record.lower_bound
                        left_child_max = parent_record.upper_bound

                        # Lower bound for left child and upper bound for right child
                        # are set to the same value.
                        middle_value = self.splitter.criterion.middle_value()
                        left_child_min = middle_value
                        right_child_max = middle_value

                    # Push right child on stack
                    self.builder_stack.push({
                        "start": split.pos,
                        "end": end,
                        "depth": depth + 1,
                        "parent": node_id,
                        "is_left": 0,
                        "impurity": split.impurity_right,
                        "n_constant_features": parent_record.n_constant_features,
                        "lower_bound": right_child_min,
                        "upper_bound": right_child_max,
                    })

                    # Push left child on stack
                    self.builder_stack.push({
                        "start": start,
                        "end": split.pos,
                        "depth": depth + 1,
                        "parent": node_id,
                        "is_left": 1,
                        "impurity": split.impurity_left,
                        "n_constant_features": parent_record.n_constant_features,
                        "lower_bound": left_child_min,
                        "upper_bound": left_child_max,
                    })

                if depth > max_depth_seen:
                    max_depth_seen = depth

            if rc >= 0:
                rc = tree._resize_c(tree.node_count)

            if rc >= 0:
                tree.max_depth = max_depth_seen
        if rc == -1:
            raise MemoryError()
        

    cpdef build_extended(
        self,
        Tree tree,
        object X,
        const float64_t[:, ::1] y,
        cnp.ndarray parents,
        cnp.ndarray depths,
        cnp.ndarray is_lefts,
        cnp.ndarray is_leafs,
        cnp.ndarray features,
        cnp.ndarray thresholds,
        cnp.ndarray impurities,
        cnp.ndarray n_node_samples_vec,
        cnp.ndarray weighted_n_node_samples_vec,
        cnp.ndarray missing_go_to_lefts,
        const float64_t[:] sample_weight=None,
        const uint8_t[::1] missing_values_in_feature_mask=None,
    ):
        """
        Build a decision tree with additional features.
        """
        # Ensure the input arrays are one-dimensional
        if (parents.ndim != 1 or is_lefts.ndim != 1 or is_leafs.ndim != 1 or 
            features.ndim != 1 or thresholds.ndim != 1 or 
            impurities.ndim != 1 or n_node_samples_vec.ndim != 1 or 
            weighted_n_node_samples_vec.ndim != 1 or missing_go_to_lefts.ndim != 1):
            raise ValueError("All parameters must be one-dimensional NumPy arrays.")
        
        # Assign the cnp.ndarray objects to the class attributes
        self.parents = parents
        self.depths = depths
        self.is_lefts = is_lefts
        self.is_leafs = is_leafs
        self.features = features
        self.thresholds = thresholds
        self.impurities = impurities
        self.n_node_samples_vec = n_node_samples_vec
        self.weighted_n_node_samples_vec = weighted_n_node_samples_vec
        self.missing_go_to_lefts = missing_go_to_lefts

        #####################################################################################

        # check input
        X, y, sample_weight = self._check_input(X, y, sample_weight)
        
        # print("Passed the check_input")

        # Initial capacity
        cdef intp_t init_capacity

        if tree.max_depth <= 10:
            init_capacity = <intp_t> (2 ** (tree.max_depth + 1)) - 1
        else:
            init_capacity = 2047

        tree._resize(init_capacity)

        # print("Passed the resize process")

        # Parameters
        cdef intp_t max_depth = self.max_depth
        cdef intp_t min_samples_leaf = self.min_samples_leaf
        cdef float64_t min_weight_leaf = self.min_weight_leaf
        cdef intp_t min_samples_split = self.min_samples_split
        cdef float64_t min_impurity_decrease = self.min_impurity_decrease

        # Recursive partition (without actual recursion)
        self.splitter.init(X, y, sample_weight, missing_values_in_feature_mask)

        # print("Passed the splitter init")

        cdef intp_t start
        cdef intp_t end
        cdef intp_t pos
        # cdef intp_t depth
        # cdef intp_t parent
        # cdef bint is_left
        # cdef intp_t n_node_samples = self.splitter.n_samples
        cdef float64_t weighted_n_node_samples
        cdef SplitRecord split
        cdef intp_t node_id

        # cdef float64_t middle_value
        # cdef float64_t left_child_min
        # cdef float64_t left_child_max
        # cdef float64_t right_child_min
        # cdef float64_t right_child_max
        # cdef bint is_leaf
        cdef bint first = 1
        # cdef intp_t max_depth_seen = -1
        # cdef int rc = 0

        cdef StackRecord stack_record

        cdef ParentInfo parent_record
        _init_parent_record(&parent_record)

        #####################################################################################

        # Get the number of nodes from the parents array
        cdef intp_t initial_n_nodes = self.parents.shape[0]
        cdef intp_t i

        # Definition of vector of start, split and end positions
        cdef cnp.ndarray start_pos = np.empty(initial_n_nodes, dtype=np.intp)
        cdef cnp.ndarray split_pos = np.empty(initial_n_nodes, dtype=np.intp)
        cdef cnp.ndarray end_pos = np.empty(initial_n_nodes, dtype=np.intp)

        # Vector to store new parents node ids
        cdef cnp.ndarray new_parents = np.array(self.parents, dtype=np.intp, copy=True)

        for i in range(initial_n_nodes):
            if not self.is_leafs[i]:

                # Compute the split position with the threshold and feature used
                threshold = self.thresholds[i]
                feature = self.features[i]

                if first:
                    # Define the start and end positions for this split
                    start = 0
                    end = self.n_node_samples_vec[0]

                    # Not first anymore
                    first = 0

                    # Add root node from initial tree
                    self.splitter.node_reset(start, end, &weighted_n_node_samples)

                    # Compute the split position
                    pos = self.splitter.recompute_node_split(&parent_record, &split, feature, threshold)

                    print("I am first")
                    print("I found the split position p = ", pos)
                    print("-------------")

                    # Define what my childs will see
                    start_pos[0] = start
                    split_pos[0] = pos
                    end_pos[0] = end
                    
                    node_id = tree._add_node(
                        _TREE_UNDEFINED, self.is_lefts[i], self.is_leafs[i],
                        self.features[i], self.thresholds[i], self.impurities[i],
                        self.n_node_samples_vec[i], self.weighted_n_node_samples_vec[i],
                        self.missing_go_to_lefts[i]
                    )

                else:
                    if self.is_lefts[i]:
                        # Define the start and end positions for this split
                        start = start_pos[self.parents[i]]
                        end = split_pos[self.parents[i]]
                        print("I am left node:", i)
                    else:
                        # Define the start and end positions for this split
                        start = split_pos[self.parents[i]]
                        end = end_pos[self.parents[i]]
                        print("I am right node:", i)

                    self.splitter.node_reset(start, end, &weighted_n_node_samples)

                    parent_record.impurity = self.impurities[i]

                    # Compute the split position
                    pos = self.splitter.recompute_node_split(&parent_record, &split, feature, threshold)

                    print("I found the split position p = ", pos)
                    print("-------------")
                    
                    # Define what my childs will see
                    start_pos[i] = start
                    split_pos[i] = pos
                    end_pos[i] = end

                    node_id = tree._add_node(
                        new_parents[i], self.is_lefts[i], self.is_leafs[i],
                        self.features[i], self.thresholds[i], self.impurities[i],
                        self.n_node_samples_vec[i], self.weighted_n_node_samples_vec[i],
                        self.missing_go_to_lefts[i]
                    )

                self.splitter.node_value(tree.value + node_id * tree.value_stride)

                # Replace the parent node id with the new one
                for j, parent in enumerate(self.parents):
                    if parent == i:
                        new_parents[j] = node_id
                # if i + 2 < len(self.parents):
                #     new_parents[i+1] = node_id
                #     new_parents[i+2] = node_id
                    # print('Original:', self.parents)
                    # print('New:', new_parents)      
            else:
                # if self.impurities[i] < 1e-6:
                #     # print("La impureza estaba fea. Soy hoja?:", self.is_leafs[i])

                #     node_id = tree._add_node(
                #         new_parents[i], self.is_lefts[i], self.is_leafs[i],
                #         self.features[i], self.thresholds[i], self.impurities[i],
                #         self.n_node_samples_vec[i], self.weighted_n_node_samples_vec[i],
                #         self.missing_go_to_lefts[i]
                #     )

                #     continue
                if not self.is_lefts[i]:
                    # Push right child on stack
                    print(f"impurity del original node der {i}:", self.impurities[i])
                    self.builder_stack.push({
                        "start": split_pos[self.parents[i]],
                        "end": end_pos[self.parents[i]],
                        "depth": self.depths[i],
                        "parent": new_parents[i],
                        "is_left": 0,
                        "impurity": self.impurities[i],
                        "n_constant_features": parent_record.n_constant_features,
                        "lower_bound": -INFINITY,
                        "upper_bound": INFINITY,
                    })
                    # print("Pushed the right child. Node:", i)
                    # print("His parent is:", new_parents[i])
                else:
                    # Push left child on stack
                    # if i == 5:
                    #     print("Feature del nodo 5:", self.features[i], feature)
                    #     print("Estando en 5:")
                    #     print("* start:", start_pos[self.parents[i]], self.parents[i])
                    #     print("* end:", split_pos[self.parents[i]], self.parents[i])

                    print(f"impurity del original node izq {i}:", self.impurities[i])
                    self.builder_stack.push({
                        "start": start_pos[self.parents[i]],
                        "end": split_pos[self.parents[i]],
                        "depth": self.depths[i],
                        "parent": new_parents[i],
                        "is_left": 1,
                        "impurity": self.impurities[i],
                        "n_constant_features": parent_record.n_constant_features,
                        "lower_bound": -INFINITY,
                        "upper_bound": INFINITY,
                    })
                    # print("Pushed the left child. Node:", i)
                    # print("His parent is:", new_parents[i])
                
                print("About to continue training from node:", i)
                self._continue_training(tree)
        