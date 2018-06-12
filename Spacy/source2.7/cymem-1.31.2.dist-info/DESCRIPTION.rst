.. image:: https://travis-ci.org/spacy-io/cymem.svg?branch=master
    :target: https://travis-ci.org/spacy-io/cymem

Cython Memory Helper
--------------------

cymem provides two small memory-management helpers for Cython. They make it
easy to tie memory to a Python object's life-cycle, so that the memory is freed
when the object is garbage collected.

The most useful is cymem.Pool, which acts as a thin wrapper around the calloc
function:

    >>> from cymem.cymem cimport Pool
    >>> cdef Pool mem = Pool()
    >>> data1 = <int*>mem.alloc(10, sizeof(int))
    >>> data2 = <float*>mem.alloc(12, sizeof(float))

The Pool object saves the memory addresses internally, and frees them when the
object is garbage collected. Typically you'll attach the Pool to some cdef'd
class. This is particularly handy for deeply nested structs, which have
complicated initialization functions. Just pass the pool object into the
initializer, and you don't have to worry about freeing your struct at all ---
all of the calls to Pool.alloc will be automatically freed when the Pool
expires.

Installation
------------

Installation is via pip, and requires Cython.

    pip install cymem

Example Use Case: An array of structs
-------------------------------------

Let's say we want a sequence of sparse matrices. We need fast access, and
a Python list isn't performing well enough. So, we want a C-array or C++
vector, which means we need the sparse matrix to be a C-level struct --- it
can't be a Python class.  We can write this easily enough in Cython:

.. code-block:: cython

    """Example without Cymem

    To use an array of structs, we must carefully walk the data structure when
    we deallocate it.
    """

    from libc.stdlib cimport calloc, free

    cdef struct SparseRow:
        size_t length
        size_t* indices
        double* values

    cdef struct SparseMatrix:
        size_t length
        SparseRow* rows

    cdef class MatrixArray:
        cdef size_t length
        cdef SparseMatrix** matrices

        def __cinit__(self, list py_matrices):
            self.length = 0
            self.matrices = NULL

        def __init__(self, list py_matrices):
            self.length = len(py_matrices)
            self.matrices = <SparseMatrix**>calloc(len(py_matrices), sizeof(SparseMatrix*))

            for i, py_matrix in enumerate(py_matrices):
                self.matrices[i] = sparse_matrix_init(py_matrix)

        def __dealloc__(self):
            for i in range(self.length):
                sparse_matrix_free(self.matrices[i])
            free(self.matrices)


    cdef SparseMatrix* sparse_matrix_init(list py_matrix) except NULL:
        sm = <SparseMatrix*>calloc(1, sizeof(SparseMatrix))
        sm.length = len(py_matrix)
        sm.rows = <SparseRow*>calloc(sm.length, sizeof(SparseRow))
        cdef size_t i, j
        cdef dict py_row
        cdef size_t idx
        cdef double value
        for i, py_row in enumerate(py_matrix):
            sm.rows[i].length = len(py_row)
            sm.rows[i].indices = <size_t*>calloc(sm.rows[i].length, sizeof(size_t))
            sm.rows[i].values = <double*>calloc(sm.rows[i].length, sizeof(double))
            for j, (idx, value) in enumerate(py_row.items()):
                sm.rows[i].indices[j] = idx
                sm.rows[i].values[j] = value
        return sm


    cdef void* sparse_matrix_free(SparseMatrix* sm) except *:
        cdef size_t i
        for i in range(sm.length):
            free(sm.rows[i].indices)
            free(sm.rows[i].values)
        free(sm.rows)
        free(sm)


We wrap the data structure in a Python ref-counted class at as low a level as
we can, given our performance constraints.  This allows us to allocate and free
the memory in the __cinit__ and __dealloc__ Cython special methods.

However, it's very easy to make mistakes when writing the __dealloc__ and
sparse_matrix_free functions, leading to memory leaks. cymem prevents you from
writing these deallocators at all. Instead, you write as follows:

.. code-block:: cython

    """Example with Cymem.

    Memory allocation is hidden behind the Pool class, which remembers the
    addresses it gives out.  When the Pool object is garbage collected, all of
    its addresses are freed.

    We don't need to write MatrixArray.__dealloc__ or sparse_matrix_free,
    eliminating a common class of bugs.
    """
    from cymem.cymem cimport Pool

    cdef struct SparseRow:
        size_t length
        size_t* indices
        double* values

    cdef struct SparseMatrix:
        size_t length
        SparseRow* rows


    cdef class MatrixArray:
        cdef size_t length
        cdef SparseMatrix** matrices
        cdef Pool mem

        def __cinit__(self, list py_matrices):
            self.mem = None
            self.length = 0
            self.matrices = NULL

        def __init__(self, list py_matrices):
            self.mem = Pool()
            self.length = len(py_matrices)
            self.matrices = <SparseMatrix**>self.mem.alloc(self.length, sizeof(SparseMatrix*))
            for i, py_matrix in enumerate(py_matrices):
                self.matrices[i] = sparse_matrix_init(self.mem, py_matrix)

    cdef SparseMatrix* sparse_matrix_init_cymem(Pool mem, list py_matrix) except NULL:
        sm = <SparseMatrix*>mem.alloc(1, sizeof(SparseMatrix))
        sm.length = len(py_matrix)
        sm.rows = <SparseRow*>mem.alloc(sm.length, sizeof(SparseRow))
        cdef size_t i, j
        cdef dict py_row
        cdef size_t idx
        cdef double value
        for i, py_row in enumerate(py_matrix):
            sm.rows[i].length = len(py_row)
            sm.rows[i].indices = <size_t*>mem.alloc(sm.rows[i].length, sizeof(size_t))
            sm.rows[i].values = <double*>mem.alloc(sm.rows[i].length, sizeof(double))
            for j, (idx, value) in enumerate(py_row.items()):
                sm.rows[i].indices[j] = idx
                sm.rows[i].values[j] = value
        return sm


All that the Pool class does is remember the addresses it gives out. When the
MatrixArray object is garbage-collected, the Pool object will also be garbage
collected, which triggers a call to Pool.__dealloc__. The Pool then frees all of
its addresses. This saves you from walking back over your nested data structures
to free them, eliminating a common class of errors.


