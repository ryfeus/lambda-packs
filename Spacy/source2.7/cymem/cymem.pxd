cdef class Pool:
    cdef readonly size_t size
    cdef readonly dict addresses
    cdef readonly list refs

    cdef void* alloc(self, size_t number, size_t size) except NULL
    cdef void free(self, void* addr) except *
    cdef void* realloc(self, void* addr, size_t n) except NULL


cdef class Address:
    cdef void* ptr
