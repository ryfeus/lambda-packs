from libc.stdint cimport uint64_t, int64_t, int32_t


cdef extern from "murmurhash/MurmurHash3.h":
    void MurmurHash3_x86_32(void * key, uint64_t len, uint64_t seed, void* out) nogil
    void MurmurHash3_x86_128(void * key, int len, uint32_t seed, void* out) nogil
    void MurmurHash3_x64_128(void * key, int len, uint32_t seed, void* out) nogil

cdef extern from "murmurhash/MurmurHash2.h":
    uint64_t MurmurHash64A(void * key, int length, uint32_t seed) nogil
    uint64_t MurmurHash64B(void * key, int length, uint32_t seed) nogil


cdef uint32_t hash32(void* key, int length, uint32_t seed) nogil:
    cdef int32_t out
    MurmurHash3_x86_32(key, length, seed, &out)
    return out


cdef uint64_t hash64(void* key, int length, uint64_t seed) nogil:
    return MurmurHash64A(key, length, seed)

cdef uint64_t real_hash64(void* key, int length, uint64_t seed) nogil:
    cdef uint64_t[2] out
    MurmurHash3_x86_128(key, length, seed, &out)
    return out[1]


cdef void hash128_x86(const void* key, int length, uint32_t seed, void* out) nogil:
    MurmurHash3_x86_128(key, length, seed, out)


cdef void hash128_x64(const void* key, int length, uint32_t seed, void* out) nogil:
    MurmurHash3_x64_128(key, length, seed, out)
