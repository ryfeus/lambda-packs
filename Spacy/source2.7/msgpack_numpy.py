#!/usr/bin/env python

"""
Support for serialization of numpy data types with msgpack.
"""

# Copyright (c) 2013-2017, Lev Givon
# All rights reserved.
# Distributed under the terms of the BSD license:
# http://www.opensource.org/licenses/bsd-license

import os
import sys

import numpy as np
import msgpack

# Fall back to pure Python
if os.environ.get('MSGPACK_PUREPYTHON'):
    import msgpack.fallback as _packer
    import msgpack.fallback as _unpacker
else:
    try:
        import msgpack._packer as _packer
        import msgpack._unpacker as _unpacker
    except:
        import msgpack.fallback as _packer
        import msgpack.fallback as _unpacker

def encode(obj):
    """
    Data encoder for serializing numpy data types.
    """

    if isinstance(obj, np.ndarray):
        # If the dtype is structured, store the interface description;
        # otherwise, store the corresponding array protocol type string:
        if obj.dtype.kind == 'V':
            kind = b'V'
            descr = obj.dtype.descr
        else:
            kind = b''
            descr = obj.dtype.str
        return {b'nd': True,
                b'type': descr,
                b'kind': kind,
                b'shape': obj.shape,
                b'data': obj.tobytes()}
    elif isinstance(obj, (np.bool_, np.number)):
        return {b'nd': False,
                b'type': obj.dtype.str,
                b'data': obj.tobytes()}
    elif isinstance(obj, complex):
        return {b'complex': True,
                b'data': obj.__repr__()}
    else:
        return obj

def tostr(x):
    if sys.version_info >= (3, 0):
        if isinstance(x, bytes):
            return x.decode()
        else:
            return str(x)
    else:
        return x

def decode(obj):
    """
    Decoder for deserializing numpy data types.
    """
    
    try:
        if b'nd' in obj:
            if obj[b'nd'] is True:
                
                # Check if b'kind' is in obj to enable decoding of data
                # serialized with older versions (#20):
                if b'kind' in obj and obj[b'kind'] == b'V':
                    descr = [tuple(tostr(t) for t in d) for d in obj[b'type']]
                else:
                    descr = obj[b'type']
                return np.fromstring(obj[b'data'],
                            dtype=np.dtype(descr)).reshape(obj[b'shape'])
            else:
                descr = obj[b'type']
                return np.fromstring(obj[b'data'],
                            dtype=np.dtype(descr))[0]
        elif b'complex' in obj:
            return complex(obj[b'data'])
        else:
            return obj
    except KeyError:
        return obj

# Maintain support for msgpack < 0.4.0:
if msgpack.version < (0, 4, 0):
    class Packer(_packer.Packer):
        def __init__(self, default=encode,
                     encoding='utf-8',
                     unicode_errors='strict',
                     use_single_float=False,
                     autoreset=1):
            super(Packer, self).__init__(default=default,
                                         encoding=encoding,
                                         unicode_errors=unicode_errors,
                                         use_single_float=use_single_float,
                                         autoreset=1)
    class Unpacker(_unpacker.Unpacker):
        def __init__(self, file_like=None, read_size=0, use_list=None,
                     object_hook=decode,
                     object_pairs_hook=None, list_hook=None, encoding='utf-8',
                     unicode_errors='strict', max_buffer_size=0):
            super(Unpacker, self).__init__(file_like=file_like,
                                           read_size=read_size,
                                           use_list=use_list,
                                           object_hook=object_hook,
                                           object_pairs_hook=object_pairs_hook,
                                           list_hook=list_hook,
                                           encoding=encoding,
                                           unicode_errors=unicode_errors,
                                           max_buffer_size=max_buffer_size)

else:
    class Packer(_packer.Packer):
        def __init__(self, default=encode,
                     encoding='utf-8',
                     unicode_errors='strict',
                     use_single_float=False,
                     autoreset=1,
                     use_bin_type=1):
            super(Packer, self).__init__(default=default,
                                         encoding=encoding,
                                         unicode_errors=unicode_errors,
                                         use_single_float=use_single_float,
                                         autoreset=1,
                                         use_bin_type=1)

    class Unpacker(_unpacker.Unpacker):
        def __init__(self, file_like=None, read_size=0, use_list=None,
                     object_hook=decode,
                     object_pairs_hook=None, list_hook=None, encoding='utf-8',
                     unicode_errors='strict', max_buffer_size=0,
                     ext_hook=msgpack.ExtType):
            super(Unpacker, self).__init__(file_like=file_like,
                                           read_size=read_size,
                                           use_list=use_list,
                                           object_hook=object_hook,
                                           object_pairs_hook=object_pairs_hook,
                                           list_hook=list_hook,
                                           encoding=encoding,
                                           unicode_errors=unicode_errors,
                                           max_buffer_size=max_buffer_size,
                                           ext_hook=ext_hook)

def pack(o, stream, default=encode, **kwargs):
    """
    Pack an object and write it to a stream.
    """

    kwargs['default'] = default
    packer = Packer(**kwargs)
    stream.write(packer.pack(o))

def packb(o, default=encode, **kwargs):
    """
    Pack an object and return the packed bytes.
    """

    kwargs['default'] = default
    return Packer(**kwargs).pack(o)

def unpack(stream, object_hook=decode, encoding='utf-8', **kwargs):
    """
    Unpack a packed object from a stream.
    """

    kwargs['object_hook'] = object_hook
    return _unpacker.unpack(stream, encoding=encoding, **kwargs)

def unpackb(packed, object_hook=decode, encoding='utf-8', **kwargs):
    """
    Unpack a packed object.
    """

    kwargs['object_hook'] = object_hook
    return _unpacker.unpackb(packed, encoding=encoding, **kwargs)

load = unpack
loads = unpackb
dump = pack
dumps = packb

def patch():
    """
    Monkey patch msgpack module to enable support for serializing numpy types.
    """

    setattr(msgpack, 'Packer', Packer)
    setattr(msgpack, 'Unpacker', Unpacker)
    setattr(msgpack, 'load', unpack)
    setattr(msgpack, 'loads', unpackb)
    setattr(msgpack, 'dump', pack)
    setattr(msgpack, 'dumps', packb)
    setattr(msgpack, 'pack', pack)
    setattr(msgpack, 'packb', packb)
    setattr(msgpack, 'unpack', unpack)
    setattr(msgpack, 'unpackb', unpackb)

if __name__ == '__main__':
    try:
        range = xrange # Python 2
    except NameError:
        pass # Python 3

    from unittest import main, TestCase, TestSuite

    class test_numpy_msgpack(TestCase):
        def setUp(self):
             patch()
        def encode_decode(self, x):
            x_enc = msgpack.packb(x)
            return msgpack.unpackb(x_enc)
        def test_numpy_scalar_bool(self):
            x = np.bool_(True)
            x_rec = self.encode_decode(x)
            assert x == x_rec and type(x) == type(x_rec)
            x = np.bool_(False)
            x_rec = self.encode_decode(x)
            assert x == x_rec and type(x) == type(x_rec)
        def test_numpy_scalar_float(self):
            x = np.float32(np.random.rand())
            x_rec = self.encode_decode(x)
            assert x == x_rec and type(x) == type(x_rec)
        def test_numpy_scalar_complex(self):
            x = np.complex64(np.random.rand()+1j*np.random.rand())
            x_rec = self.encode_decode(x)
            assert x == x_rec and type(x) == type(x_rec)
        def test_scalar_float(self):
            x = np.random.rand()
            x_rec = self.encode_decode(x)
            assert x == x_rec and type(x) == type(x_rec)
        def test_scalar_complex(self):
            x = np.random.rand()+1j*np.random.rand()
            x_rec = self.encode_decode(x)
            assert x == x_rec and type(x) == type(x_rec)
        def test_list_numpy_float(self):
            x = [np.float32(np.random.rand()) for i in range(5)]
            x_rec = self.encode_decode(x)
            assert all(map(lambda x, y: x == y, x, x_rec)) and all(map(lambda x,y: type(x) == type(y), x, x_rec))
        def test_list_numpy_float_complex(self):
            x = [np.float32(np.random.rand()) for i in range(5)] + \
              [np.complex128(np.random.rand()+1j*np.random.rand()) for i in range(5)]
            x_rec = self.encode_decode(x)
            assert all(map(lambda x,y: x == y, x, x_rec)) and all(map(lambda x,y: type(x) == type(y), x, x_rec))
        def test_list_float(self):
            x = [np.random.rand() for i in range(5)]
            x_rec = self.encode_decode(x)
            assert all(map(lambda x,y: x == y, x, x_rec)) and all(map(lambda x,y: type(x) == type(y), x, x_rec))
        def test_list_float_complex(self):
            x = [(np.random.rand()+1j*np.random.rand()) for i in range(5)]
            x_rec = self.encode_decode(x)
            assert all(map(lambda x, y: x == y, x, x_rec)) and all(map(lambda x,y: type(x) == type(y), x, x_rec))
        def test_list_str(self):
            x = ['x'*i for i in range(5)]
            x_rec = self.encode_decode(x)
            assert all(map(lambda x,y: x == y, x, x_rec)) and all(map(lambda x,y: type(x) == type(y), x, x_rec))
        def test_dict_float(self):
            x = {'foo': 1.0, 'bar': 2.0}
            x_rec = self.encode_decode(x)
            assert all(map(lambda x,y: x == y, x.values(), x_rec.values())) and \
                           all(map(lambda x,y: type(x) == type(y), x.values(), x_rec.values()))
        def test_dict_complex(self):
            x = {'foo': 1.0+1.0j, 'bar': 2.0+2.0j}
            x_rec = self.encode_decode(x)
            assert all(map(lambda x,y: x == y, x.values(), x_rec.values())) and \
                           all(map(lambda x,y: type(x) == type(y), x.values(), x_rec.values()))
        def test_dict_str(self):
            x = {'foo': 'xxx', 'bar': 'yyyy'}
            x_rec = self.encode_decode(x)
            assert all(map(lambda x,y: x == y, x.values(), x_rec.values())) and \
                           all(map(lambda x,y: type(x) == type(y), x.values(), x_rec.values()))
        def test_dict_numpy_float(self):
            x = {'foo': np.float32(1.0), 'bar': np.float32(2.0)}
            x_rec = self.encode_decode(x)
            assert all(map(lambda x,y: x == y, x.values(), x_rec.values())) and \
                           all(map(lambda x,y: type(x) == type(y), x.values(), x_rec.values()))
        def test_dict_numpy_complex(self):
            x = {'foo': np.complex128(1.0+1.0j), 'bar': np.complex128(2.0+2.0j)}
            x_rec = self.encode_decode(x)
            assert all(map(lambda x,y: x == y, x.values(), x_rec.values())) and \
                           all(map(lambda x,y: type(x) == type(y), x.values(), x_rec.values()))
        def test_numpy_array_float(self):
            x = np.random.rand(5).astype(np.float32)
            x_rec = self.encode_decode(x)
            assert np.all(x == x_rec) and x.dtype == x_rec.dtype
        def test_numpy_array_complex(self):
            x = (np.random.rand(5)+1j*np.random.rand(5)).astype(np.complex128)
            x_rec = self.encode_decode(x)
            assert np.all(x == x_rec) and x.dtype == x_rec.dtype
        def test_numpy_array_float_2d(self):
            x = np.random.rand(5,5).astype(np.float32)
            x_rec = self.encode_decode(x)
            assert np.all(x == x_rec) and x.dtype == x_rec.dtype
        def test_numpy_array_str(self):
            x = np.array(['aaa', 'bbbb', 'ccccc'])
            x_rec = self.encode_decode(x)
            assert np.all(x == x_rec) and x.dtype == x_rec.dtype
        def test_numpy_array_mixed(self):
            x = np.array([(1, 2, 'a')],
                         np.dtype([('arg0', np.uint32),
                                   ('arg1', np.uint32),
                                   ('arg2', 'S1')]))
            x_rec = self.encode_decode(x)
            assert np.all(x == x_rec) and x.dtype == x_rec.dtype
        def test_list_mixed(self):
            x = [1.0, np.float32(3.5), np.complex128(4.25), 'foo']
            x_rec = self.encode_decode(x)
            assert all(map(lambda x,y: x == y, x, x_rec)) and all(map(lambda x,y: type(x) == type(y), x, x_rec))

    main()

