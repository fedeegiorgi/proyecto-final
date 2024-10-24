from _tree cimport DepthFirstTreeBuilder, StackRecord, Tree, intp_t, float64_t, uint8_t, stack

cimport numpy as cnp
cnp.import_array()

cdef class DepthFirstTreeExtensionBuilder(DepthFirstTreeBuilder):
    """
    Extended depth-first tree builder based on initial tree with additional features .
    """
    # New parameters
    cdef double[:] parents
    cdef double[:] is_lefts
    cdef double[:] is_leafs
    cdef double[:] features
    cdef double[:] thresholds
    cdef double[:] impurities
    cdef double[:] n_node_samples
    cdef double[:] weighted_n_node_samples
    cdef double[:] missing_go_to_lefts

    cdef stack[StackRecord] initial_builder_stack # Stack to store the leafs of the initial tree
    cdef StackRecord stack_record

    cpdef build(
        self,
        Tree tree,
        object X,
        const float64_t[:, ::1] y,
        const float64_t[:] sample_weight=None,
        const uint8_t[::1] missing_values_in_feature_mask=None,
    )

    cpdef _load_initial_tree(self, Tree tree)
