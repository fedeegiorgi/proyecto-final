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
    cpdef build_extended(
        self,
        Tree tree,
        object X_original,
        object X_peer_prediction,
        intp_t initial_max_depth,
        const float64_t[:, ::1] y,
        cnp.ndarray features,
        const float64_t[:] sample_weight=*,
        const uint8_t[::1] missing_values_in_feature_mask=*,
    )