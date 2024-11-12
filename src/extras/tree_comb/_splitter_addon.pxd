from ._splitter cimport Splitter, SplitRecord
from ._tree cimport ParentInfo
from ..utils._typedefs cimport float32_t, float64_t, intp_t, int32_t, uint8_t, uint32_t
from ._partitioner cimport DensePartitioner

cdef class AddOnBestSplitter(Splitter):
    cdef DensePartitioner partitioner
    cdef int recompute_node_split(
        self,
        ParentInfo* parent_record,
        SplitRecord* split,
        intp_t feature, # feature index of initial_tree
        float64_t threshold, # threshold of initial_tree
    ) except -1 nogil