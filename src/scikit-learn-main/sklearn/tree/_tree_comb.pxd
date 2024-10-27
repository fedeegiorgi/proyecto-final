from ._tree cimport Tree, Node
from libcpp.stack cimport stack
from ..utils._typedefs cimport float32_t, float64_t, intp_t, int32_t, uint8_t, uint32_t

cimport numpy as cnp
cnp.import_array()

cdef class TreeCombiner(Tree):
    cdef cnp.ndarray features
    cdef cnp.ndarray thresholds
    cdef cnp.ndarray impurities
    cdef cnp.ndarray n_node_samples
    cdef cnp.ndarray weighted_n_node_samples
    cdef cnp.ndarray missing_go_to_lefts
    
    cpdef void combiner(self, cnp.ndarray features, cnp.ndarray thresholds, 
                cnp.ndarray impurities, cnp.ndarray n_node_samples, 
                cnp.ndarray weighted_n_node_samples, cnp.ndarray missing_go_to_lefts)
    cpdef recompute_values(self, cnp.ndarray out, cnp.ndarray y)