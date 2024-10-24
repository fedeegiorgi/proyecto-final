from _tree cimport Tree, Node
from libcpp.stack cimport stack
from ..utils._typedefs cimport float32_t, float64_t, intp_t, int32_t, uint8_t, uint32_t

cimport numpy as cnp
cnp.import_array()

cdef class TreeCombiner(Tree):
    cdef double[:] features
    cdef double[:] thresholds
    
    cdef combiner
    cdef recompute_values
    cdef tree_combiner