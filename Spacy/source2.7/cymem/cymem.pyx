# cython: embedsignature=True

from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
from cpython.ref cimport Py_INCREF, Py_DECREF
from libc.string cimport memset
from libc.string cimport memcpy


cdef class Pool:
    """Track allocated memory addresses, and free them all when the Pool is
    garbage collected.  This provides an easy way to avoid memory leaks, and 
    removes the need for deallocation functions for complicated structs.

    >>> from cymem.cymem cimport Pool
    >>> cdef Pool mem = Pool()
    >>> data1 = <int*>mem.alloc(10, sizeof(int))
    >>> data2 = <float*>mem.alloc(12, sizeof(float))

    Attributes:
        size (size_t): The current size (in bytes) allocated by the pool.
        addresses (dict): The currently allocated addresses and their sizes. Read-only.
    """
    def __cinit__(self):
        self.size = 0
        self.addresses = {}
        self.refs = []

    def __dealloc__(self):
        cdef size_t addr
        if self.addresses is not None:
            for addr in self.addresses:
                if addr != 0:
                    PyMem_Free(<void*>addr)

    cdef void* alloc(self, size_t number, size_t elem_size) except NULL:
        """Allocate a 0-initialized number*elem_size-byte block of memory, and
        remember its address. The block will be freed when the Pool is garbage
        collected.
        """
        cdef void* p = PyMem_Malloc(number * elem_size)
        if p == NULL:
            raise MemoryError("Error assigning %d bytes" % number * elem_size)
        memset(p, 0, number * elem_size)
        self.addresses[<size_t>p] = number * elem_size
        self.size += number * elem_size
        return p

    cdef void* realloc(self, void* p, size_t new_size) except NULL:
        """Resizes the memory block pointed to by p to new_size bytes, returning
        a non-NULL pointer to the new block. new_size must be larger than the
        original.
        
        If p is not in the Pool or new_size is 0, a MemoryError is raised.
        """
        if <size_t>p not in self.addresses:
            raise ValueError("Pointer %d not found in Pool %s" % (<size_t>p, self.addresses))
        if new_size == 0:
            raise ValueError("Realloc requires new_size > 0")
        assert new_size > self.addresses[<size_t>p]         
        cdef void* new_ptr = self.alloc(1, new_size)
        if new_ptr == NULL:
            raise MemoryError("Error reallocating to %d bytes" % new_size)
        memcpy(new_ptr, p, self.addresses[<size_t>p])
        self.free(p)
        self.addresses[<size_t>new_ptr] = new_size
        return new_ptr

    cdef void free(self, void* p) except *:
        """Frees the memory block pointed to by p, which must have been returned
        by a previous call to Pool.alloc.  You don't necessarily need to free
        memory addresses manually --- you can instead let the Pool be garbage
        collected, at which point all the memory will be freed.
        
        If p is not in Pool.addresses, a KeyError is raised.
        """
        self.size -= self.addresses.pop(<size_t>p)
        PyMem_Free(p)

    def own_pyref(self, object py_ref):
        self.refs.append(py_ref)


cdef class Address:
    """A block of number * size-bytes of 0-initialized memory, tied to a Python
    ref-counted object. When the object is garbage collected, the memory is freed.

    >>> from cymem.cymem cimport Address
    >>> cdef Address address = Address(10, sizeof(double))
    >>> d10 = <double*>address.ptr

    Args:
        number (size_t): The number of elements in the memory block.
        elem_size (size_t): The size of each element.

    Attributes:
        ptr (void*): Pointer to the memory block.
        addr (size_t): Read-only size_t cast of the pointer.
    """
    def __cinit__(self, size_t number, size_t elem_size):
        self.ptr = NULL

    def __init__(self, size_t number, size_t elem_size):
        self.ptr = PyMem_Malloc(number * elem_size)
        if self.ptr == NULL:
            raise MemoryError("Error assigning %d bytes" % number * elem_size)
        memset(self.ptr, 0, number * elem_size)

    property addr:
        def __get__(self):
            return <size_t>self.ptr

    def __dealloc__(self):
        if self.ptr != NULL:
            PyMem_Free(self.ptr)
