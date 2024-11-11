from ._tree cimport TreeBuilder, Tree, Node
from ._splitter cimport Splitter
from libcpp.stack cimport stack
from ..utils._typedefs cimport float32_t, float64_t, intp_t, int32_t, uint8_t, uint32_t

cimport numpy as cnp
cnp.import_array()

cdef class DepthFirstTreeCombinerBuilder(TreeBuilder):
    
    cpdef combiner(self, Tree tree, cnp.ndarray features, cnp.ndarray thresholds)
    cpdef recompute_values(self, Tree tree, cnp.ndarray out, cnp.ndarray y)