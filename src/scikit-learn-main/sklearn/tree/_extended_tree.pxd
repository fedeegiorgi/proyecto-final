from _tree cimport DepthFirstTreeBuilder, StackRecord, Tree
from libcpp.stack cimport stack
from ..utils._typedefs cimport float32_t, float64_t, intp_t, int32_t, uint8_t, uint32_t

cimport numpy as cnp
cnp.import_array()

cdef class DepthFirstTreeExtensionBuilder(DepthFirstTreeBuilder):
    """
    Extended depth-first tree builder based on initial tree with additional features .
    """
    cdef cnp.ndarray parents
    cdef cnp.ndarray is_lefts
    cdef cnp.ndarray is_leafs
    cdef cnp.ndarray features
    cdef cnp.ndarray thresholds
    cdef cnp.ndarray impurities
    cdef cnp.ndarray n_node_samples
    cdef cnp.ndarray weighted_n_node_samples
    cdef cnp.ndarray missing_go_to_lefts

    cdef stack[StackRecord] builder_stack # Stack to store the leafs of the initial tree
    cdef StackRecord stack_record

    cpdef build(
        self,
        Tree tree,
        object X,
        const float64_t[:, ::1] y,
        cnp.ndarray parents,
        cnp.ndarray is_lefts,
        cnp.ndarray is_leafs,
        cnp.ndarray features,
        cnp.ndarray thresholds,
        cnp.ndarray impurities,
        cnp.ndarray n_node_samples,
        cnp.ndarray weighted_n_node_samples,
        cnp.ndarray missing_go_to_lefts,
        const float64_t[:] sample_weight=None,
        const uint8_t[::1] missing_values_in_feature_mask=None,
    )

    cpdef _load_initial_tree(self, Tree tree)  # Acá el tree tiene que ser por referencia??
