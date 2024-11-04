from ._tree cimport TreeBuilder, Tree, Node, ParentInfo
from ._splitter cimport Splitter, SplitRecord
from ..utils._typedefs cimport float32_t, float64_t, intp_t, int32_t, uint8_t, uint32_t
from libcpp.stack cimport stack

cimport numpy as cnp
cnp.import_array()

cdef struct StackRecord:
    intp_t start
    intp_t end
    intp_t depth
    intp_t parent
    bint is_left
    float64_t impurity
    intp_t n_constant_features
    float64_t lower_bound
    float64_t upper_bound

cdef class DepthFirstTreeExtensionBuilder(TreeBuilder):
    """
    Extended depth-first tree builder based on initial tree with additional features .
    """
    cdef cnp.ndarray parents
    cdef cnp.ndarray depths
    cdef cnp.ndarray is_lefts
    cdef cnp.ndarray is_leafs
    cdef cnp.ndarray features
    cdef cnp.ndarray thresholds
    cdef cnp.ndarray impurities
    cdef cnp.ndarray n_node_samples_vec
    cdef cnp.ndarray weighted_n_node_samples_vec
    cdef cnp.ndarray missing_go_to_lefts

    cdef stack[StackRecord] builder_stack # Stack to store the leafs of the initial tree
    cdef StackRecord stack_record

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
        const float64_t[:] sample_weight=*,
        const uint8_t[::1] missing_values_in_feature_mask=*,
    )

    cdef _continue_training(self, Tree tree)