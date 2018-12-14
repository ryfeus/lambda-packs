"""Python wrappers around TensorFlow ops.

This file is MACHINE GENERATED! Do not edit.
Original C++ source file: string_ops.cc
"""

import collections as _collections

from tensorflow.python.eager import execute as _execute
from tensorflow.python.eager import context as _context
from tensorflow.python.eager import core as _core
from tensorflow.python.framework import dtypes as _dtypes
from tensorflow.python.framework import tensor_shape as _tensor_shape

from tensorflow.core.framework import op_def_pb2 as _op_def_pb2
# Needed to trigger the call to _set_call_cpp_shape_fn.
from tensorflow.python.framework import common_shapes as _common_shapes
from tensorflow.python.framework import op_def_registry as _op_def_registry
from tensorflow.python.framework import ops as _ops
from tensorflow.python.framework import op_def_library as _op_def_library


def as_string(input, precision=-1, scientific=False, shortest=False, width=-1, fill="", name=None):
  r"""Converts each entry in the given tensor to strings.  Supports many numeric

  types and boolean.

  Args:
    input: A `Tensor`. Must be one of the following types: `int32`, `int64`, `complex64`, `float32`, `float64`, `bool`, `int8`.
    precision: An optional `int`. Defaults to `-1`.
      The post-decimal precision to use for floating point numbers.
      Only used if precision > -1.
    scientific: An optional `bool`. Defaults to `False`.
      Use scientific notation for floating point numbers.
    shortest: An optional `bool`. Defaults to `False`.
      Use shortest representation (either scientific or standard) for
      floating point numbers.
    width: An optional `int`. Defaults to `-1`.
      Pad pre-decimal numbers to this width.
      Applies to both floating point and integer numbers.
      Only used if width > -1.
    fill: An optional `string`. Defaults to `""`.
      The value to pad if width > -1.  If empty, pads with spaces.
      Another typical value is '0'.  String cannot be longer than 1 character.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`.
  """
  if precision is None:
    precision = -1
  precision = _execute.make_int(precision, "precision")
  if scientific is None:
    scientific = False
  scientific = _execute.make_bool(scientific, "scientific")
  if shortest is None:
    shortest = False
  shortest = _execute.make_bool(shortest, "shortest")
  if width is None:
    width = -1
  width = _execute.make_int(width, "width")
  if fill is None:
    fill = ""
  fill = _execute.make_str(fill, "fill")
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "AsString", input=input, precision=precision, scientific=scientific,
        shortest=shortest, width=width, fill=fill, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("T", _op.get_attr("T"), "precision", _op.get_attr("precision"),
              "scientific", _op.get_attr("scientific"), "shortest",
              _op.get_attr("shortest"), "width", _op.get_attr("width"),
              "fill", _op.get_attr("fill"))
  else:
    _attr_T, (input,) = _execute.args_to_matching_eager([input], _ctx)
    _attr_T = _attr_T.as_datatype_enum
    _inputs_flat = [input]
    _attrs = ("T", _attr_T, "precision", precision, "scientific", scientific,
              "shortest", shortest, "width", width, "fill", fill)
    _result = _execute.execute(b"AsString", 1, inputs=_inputs_flat,
                               attrs=_attrs, ctx=_ctx, name=name)
  _execute.record_gradient(
      "AsString", _inputs_flat, _attrs, _result, name)
  _result, = _result
  return _result


def decode_base64(input, name=None):
  r"""Decode web-safe base64-encoded strings.

  Input may or may not have padding at the end. See EncodeBase64 for padding.
  Web-safe means that input must use - and _ instead of + and /.

  Args:
    input: A `Tensor` of type `string`. Base64 strings to decode.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`. Decoded strings.
  """
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "DecodeBase64", input=input, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = None
  else:
    input = _ops.convert_to_tensor(input, _dtypes.string)
    _inputs_flat = [input]
    _attrs = None
    _result = _execute.execute(b"DecodeBase64", 1, inputs=_inputs_flat,
                               attrs=_attrs, ctx=_ctx, name=name)
  _execute.record_gradient(
      "DecodeBase64", _inputs_flat, _attrs, _result, name)
  _result, = _result
  return _result


def encode_base64(input, pad=False, name=None):
  r"""Encode strings into web-safe base64 format.

  Refer to the following article for more information on base64 format:
  en.wikipedia.org/wiki/Base64. Base64 strings may have padding with '=' at the
  end so that the encoded has length multiple of 4. See Padding section of the
  link above.

  Web-safe means that the encoder uses - and _ instead of + and /.

  Args:
    input: A `Tensor` of type `string`. Strings to be encoded.
    pad: An optional `bool`. Defaults to `False`.
      Bool whether padding is applied at the ends.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`. Input strings encoded in base64.
  """
  if pad is None:
    pad = False
  pad = _execute.make_bool(pad, "pad")
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "EncodeBase64", input=input, pad=pad, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("pad", _op.get_attr("pad"))
  else:
    input = _ops.convert_to_tensor(input, _dtypes.string)
    _inputs_flat = [input]
    _attrs = ("pad", pad)
    _result = _execute.execute(b"EncodeBase64", 1, inputs=_inputs_flat,
                               attrs=_attrs, ctx=_ctx, name=name)
  _execute.record_gradient(
      "EncodeBase64", _inputs_flat, _attrs, _result, name)
  _result, = _result
  return _result


def reduce_join(inputs, reduction_indices, keep_dims=False, separator="", name=None):
  r"""Joins a string Tensor across the given dimensions.

  Computes the string join across dimensions in the given string Tensor of shape
  `[d_0, d_1, ..., d_n-1]`.  Returns a new Tensor created by joining the input
  strings with the given separator (default: empty string).  Negative indices are
  counted backwards from the end, with `-1` being equivalent to `n - 1`.

  For example:

  ```python
  # tensor `a` is [["a", "b"], ["c", "d"]]
  tf.reduce_join(a, 0) ==> ["ac", "bd"]
  tf.reduce_join(a, 1) ==> ["ab", "cd"]
  tf.reduce_join(a, -2) = tf.reduce_join(a, 0) ==> ["ac", "bd"]
  tf.reduce_join(a, -1) = tf.reduce_join(a, 1) ==> ["ab", "cd"]
  tf.reduce_join(a, 0, keep_dims=True) ==> [["ac", "bd"]]
  tf.reduce_join(a, 1, keep_dims=True) ==> [["ab"], ["cd"]]
  tf.reduce_join(a, 0, separator=".") ==> ["a.c", "b.d"]
  tf.reduce_join(a, [0, 1]) ==> ["acbd"]
  tf.reduce_join(a, [1, 0]) ==> ["abcd"]
  tf.reduce_join(a, []) ==> ["abcd"]
  ```

  Args:
    inputs: A `Tensor` of type `string`.
      The input to be joined.  All reduced indices must have non-zero size.
    reduction_indices: A `Tensor` of type `int32`.
      The dimensions to reduce over.  Dimensions are reduced in the
      order specified.  Omitting `reduction_indices` is equivalent to passing
      `[n-1, n-2, ..., 0]`.  Negative indices from `-n` to `-1` are supported.
    keep_dims: An optional `bool`. Defaults to `False`.
      If `True`, retain reduced dimensions with length `1`.
    separator: An optional `string`. Defaults to `""`.
      The separator to use when joining.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`.
    Has shape equal to that of the input with reduced dimensions removed or
    set to `1` depending on `keep_dims`.
  """
  if keep_dims is None:
    keep_dims = False
  keep_dims = _execute.make_bool(keep_dims, "keep_dims")
  if separator is None:
    separator = ""
  separator = _execute.make_str(separator, "separator")
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "ReduceJoin", inputs=inputs, reduction_indices=reduction_indices,
        keep_dims=keep_dims, separator=separator, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("keep_dims", _op.get_attr("keep_dims"), "separator",
              _op.get_attr("separator"))
  else:
    inputs = _ops.convert_to_tensor(inputs, _dtypes.string)
    reduction_indices = _ops.convert_to_tensor(reduction_indices, _dtypes.int32)
    _inputs_flat = [inputs, reduction_indices]
    _attrs = ("keep_dims", keep_dims, "separator", separator)
    _result = _execute.execute(b"ReduceJoin", 1, inputs=_inputs_flat,
                               attrs=_attrs, ctx=_ctx, name=name)
  _execute.record_gradient(
      "ReduceJoin", _inputs_flat, _attrs, _result, name)
  _result, = _result
  return _result


def string_join(inputs, separator="", name=None):
  r"""Joins the strings in the given list of string tensors into one tensor;

  with the given separator (default is an empty separator).

  Args:
    inputs: A list of at least 1 `Tensor` objects with type `string`.
      A list of string tensors.  The tensors must all have the same shape,
      or be scalars.  Scalars may be mixed in; these will be broadcast to the shape
      of non-scalar inputs.
    separator: An optional `string`. Defaults to `""`.
      string, an optional join separator.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`.
  """
  if not isinstance(inputs, (list, tuple)):
    raise TypeError(
        "Expected list for 'inputs' argument to "
        "'string_join' Op, not %r." % inputs)
  _attr_N = len(inputs)
  if separator is None:
    separator = ""
  separator = _execute.make_str(separator, "separator")
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "StringJoin", inputs=inputs, separator=separator, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("N", _op.get_attr("N"), "separator", _op.get_attr("separator"))
  else:
    inputs = _ops.convert_n_to_tensor(inputs, _dtypes.string)
    _inputs_flat = list(inputs)
    _attrs = ("N", _attr_N, "separator", separator)
    _result = _execute.execute(b"StringJoin", 1, inputs=_inputs_flat,
                               attrs=_attrs, ctx=_ctx, name=name)
  _execute.record_gradient(
      "StringJoin", _inputs_flat, _attrs, _result, name)
  _result, = _result
  return _result


__string_split_outputs = ["indices", "values", "shape"]
_StringSplitOutput = _collections.namedtuple(
    "StringSplit", __string_split_outputs)


def _string_split(input, delimiter, skip_empty=True, name=None):
  r"""Split elements of `input` based on `delimiter` into a `SparseTensor`.

  Let N be the size of source (typically N will be the batch size). Split each
  element of `input` based on `delimiter` and return a `SparseTensor`
  containing the splitted tokens. Empty tokens are ignored.

  `delimiter` can be empty, or a string of split characters. If `delimiter` is an
   empty string, each element of `input` is split into individual single-byte
   character strings, including splitting of UTF-8 multibyte sequences. Otherwise
   every character of `delimiter` is a potential split point.

  For example:
    N = 2, input[0] is 'hello world' and input[1] is 'a b c', then the output
    will be

    indices = [0, 0;
               0, 1;
               1, 0;
               1, 1;
               1, 2]
    shape = [2, 3]
    values = ['hello', 'world', 'a', 'b', 'c']

  Args:
    input: A `Tensor` of type `string`. 1-D. Strings to split.
    delimiter: A `Tensor` of type `string`.
      0-D. Delimiter characters (bytes), or empty string.
    skip_empty: An optional `bool`. Defaults to `True`.
      A `bool`. If `True`, skip the empty strings from the result.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (indices, values, shape).

    indices: A `Tensor` of type `int64`. A dense matrix of int64 representing the indices of the sparse tensor.
    values: A `Tensor` of type `string`. A vector of strings corresponding to the splited values.
    shape: A `Tensor` of type `int64`. a length-2 vector of int64 representing the shape of the sparse
      tensor, where the first value is N and the second value is the maximum number
      of tokens in a single input entry.
  """
  if skip_empty is None:
    skip_empty = True
  skip_empty = _execute.make_bool(skip_empty, "skip_empty")
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "StringSplit", input=input, delimiter=delimiter,
        skip_empty=skip_empty, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("skip_empty", _op.get_attr("skip_empty"))
  else:
    input = _ops.convert_to_tensor(input, _dtypes.string)
    delimiter = _ops.convert_to_tensor(delimiter, _dtypes.string)
    _inputs_flat = [input, delimiter]
    _attrs = ("skip_empty", skip_empty)
    _result = _execute.execute(b"StringSplit", 3, inputs=_inputs_flat,
                               attrs=_attrs, ctx=_ctx, name=name)
  _execute.record_gradient(
      "StringSplit", _inputs_flat, _attrs, _result, name)
  _result = _StringSplitOutput._make(_result)
  return _result


def string_to_hash_bucket(string_tensor, num_buckets, name=None):
  r"""Converts each string in the input Tensor to its hash mod by a number of buckets.

  The hash function is deterministic on the content of the string within the
  process.

  Note that the hash function may change from time to time.
  This functionality will be deprecated and it's recommended to use
  `tf.string_to_hash_bucket_fast()` or `tf.string_to_hash_bucket_strong()`.

  Args:
    string_tensor: A `Tensor` of type `string`.
    num_buckets: An `int` that is `>= 1`. The number of buckets.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int64`.
    A Tensor of the same shape as the input `string_tensor`.
  """
  num_buckets = _execute.make_int(num_buckets, "num_buckets")
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "StringToHashBucket", string_tensor=string_tensor,
        num_buckets=num_buckets, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("num_buckets", _op.get_attr("num_buckets"))
  else:
    string_tensor = _ops.convert_to_tensor(string_tensor, _dtypes.string)
    _inputs_flat = [string_tensor]
    _attrs = ("num_buckets", num_buckets)
    _result = _execute.execute(b"StringToHashBucket", 1, inputs=_inputs_flat,
                               attrs=_attrs, ctx=_ctx, name=name)
  _execute.record_gradient(
      "StringToHashBucket", _inputs_flat, _attrs, _result, name)
  _result, = _result
  return _result


def string_to_hash_bucket_fast(input, num_buckets, name=None):
  r"""Converts each string in the input Tensor to its hash mod by a number of buckets.

  The hash function is deterministic on the content of the string within the
  process and will never change. However, it is not suitable for cryptography.
  This function may be used when CPU time is scarce and inputs are trusted or
  unimportant. There is a risk of adversaries constructing inputs that all hash
  to the same bucket. To prevent this problem, use a strong hash function with
  `tf.string_to_hash_bucket_strong`.

  Args:
    input: A `Tensor` of type `string`. The strings to assign a hash bucket.
    num_buckets: An `int` that is `>= 1`. The number of buckets.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int64`.
    A Tensor of the same shape as the input `string_tensor`.
  """
  num_buckets = _execute.make_int(num_buckets, "num_buckets")
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "StringToHashBucketFast", input=input, num_buckets=num_buckets,
        name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("num_buckets", _op.get_attr("num_buckets"))
  else:
    input = _ops.convert_to_tensor(input, _dtypes.string)
    _inputs_flat = [input]
    _attrs = ("num_buckets", num_buckets)
    _result = _execute.execute(b"StringToHashBucketFast", 1,
                               inputs=_inputs_flat, attrs=_attrs, ctx=_ctx,
                               name=name)
  _execute.record_gradient(
      "StringToHashBucketFast", _inputs_flat, _attrs, _result, name)
  _result, = _result
  return _result


def string_to_hash_bucket_strong(input, num_buckets, key, name=None):
  r"""Converts each string in the input Tensor to its hash mod by a number of buckets.

  The hash function is deterministic on the content of the string within the
  process. The hash function is a keyed hash function, where attribute `key`
  defines the key of the hash function. `key` is an array of 2 elements.

  A strong hash is important when inputs may be malicious, e.g. URLs with
  additional components. Adversaries could try to make their inputs hash to the
  same bucket for a denial-of-service attack or to skew the results. A strong
  hash prevents this by making it difficult, if not infeasible, to compute inputs
  that hash to the same bucket. This comes at a cost of roughly 4x higher compute
  time than `tf.string_to_hash_bucket_fast`.

  Args:
    input: A `Tensor` of type `string`. The strings to assign a hash bucket.
    num_buckets: An `int` that is `>= 1`. The number of buckets.
    key: A list of `ints`.
      The key for the keyed hash function passed as a list of two uint64
      elements.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int64`.
    A Tensor of the same shape as the input `string_tensor`.
  """
  num_buckets = _execute.make_int(num_buckets, "num_buckets")
  if not isinstance(key, (list, tuple)):
    raise TypeError(
        "Expected list for 'key' argument to "
        "'string_to_hash_bucket_strong' Op, not %r." % key)
  key = [_execute.make_int(_i, "key") for _i in key]
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "StringToHashBucketStrong", input=input, num_buckets=num_buckets,
        key=key, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("num_buckets", _op.get_attr("num_buckets"), "key",
              _op.get_attr("key"))
  else:
    input = _ops.convert_to_tensor(input, _dtypes.string)
    _inputs_flat = [input]
    _attrs = ("num_buckets", num_buckets, "key", key)
    _result = _execute.execute(b"StringToHashBucketStrong", 1,
                               inputs=_inputs_flat, attrs=_attrs, ctx=_ctx,
                               name=name)
  _execute.record_gradient(
      "StringToHashBucketStrong", _inputs_flat, _attrs, _result, name)
  _result, = _result
  return _result


def substr(input, pos, len, name=None):
  r"""Return substrings from `Tensor` of strings.

  For each string in the input `Tensor`, creates a substring starting at index
  `pos` with a total length of `len`.

  If `len` defines a substring that would extend beyond the length of the input
  string, then as many characters as possible are used.

  If `pos` is negative or specifies a character index larger than any of the input
  strings, then an `InvalidArgumentError` is thrown.

  `pos` and `len` must have the same shape, otherwise a `ValueError` is thrown on
  Op creation.

  *NOTE*: `Substr` supports broadcasting up to two dimensions. More about
  broadcasting
  [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

  ---

  Examples

  Using scalar `pos` and `len`:

  ```python
  input = [b'Hello', b'World']
  position = 1
  length = 3

  output = [b'ell', b'orl']
  ```

  Using `pos` and `len` with same shape as `input`:

  ```python
  input = [[b'ten', b'eleven', b'twelve'],
           [b'thirteen', b'fourteen', b'fifteen'],
           [b'sixteen', b'seventeen', b'eighteen']]
  position = [[1, 2, 3],
              [1, 2, 3],
              [1, 2, 3]]
  length =   [[2, 3, 4],
              [4, 3, 2],
              [5, 5, 5]]

  output = [[b'en', b'eve', b'lve'],
            [b'hirt', b'urt', b'te'],
            [b'ixtee', b'vente', b'hteen']]
  ```

  Broadcasting `pos` and `len` onto `input`:

  ```
  input = [[b'ten', b'eleven', b'twelve'],
           [b'thirteen', b'fourteen', b'fifteen'],
           [b'sixteen', b'seventeen', b'eighteen'],
           [b'nineteen', b'twenty', b'twentyone']]
  position = [1, 2, 3]
  length =   [1, 2, 3]

  output = [[b'e', b'ev', b'lve'],
            [b'h', b'ur', b'tee'],
            [b'i', b've', b'hte'],
            [b'i', b'en', b'nty']]
  ```

  Broadcasting `input` onto `pos` and `len`:

  ```
  input = b'thirteen'
  position = [1, 5, 7]
  length =   [3, 2, 1]

  output = [b'hir', b'ee', b'n']
  ```

  Args:
    input: A `Tensor` of type `string`. Tensor of strings
    pos: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      Scalar defining the position of first character in each substring
    len: A `Tensor`. Must have the same type as `pos`.
      Scalar defining the number of characters to include in each substring
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`. Tensor of substrings
  """
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "Substr", input=input, pos=pos, len=len, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("T", _op.get_attr("T"))
  else:
    _attr_T, _inputs_T = _execute.args_to_matching_eager([pos, len], _ctx)
    (pos, len) = _inputs_T
    _attr_T = _attr_T.as_datatype_enum
    input = _ops.convert_to_tensor(input, _dtypes.string)
    _inputs_flat = [input, pos, len]
    _attrs = ("T", _attr_T)
    _result = _execute.execute(b"Substr", 1, inputs=_inputs_flat,
                               attrs=_attrs, ctx=_ctx, name=name)
  _execute.record_gradient(
      "Substr", _inputs_flat, _attrs, _result, name)
  _result, = _result
  return _result

def _InitOpDefLibrary(op_list_proto_bytes):
  op_list = _op_def_pb2.OpList()
  op_list.ParseFromString(op_list_proto_bytes)
  _op_def_registry.register_op_list(op_list)
  op_def_lib = _op_def_library.OpDefLibrary()
  op_def_lib.add_op_list(op_list)
  return op_def_lib
# op {
#   name: "AsString"
#   input_arg {
#     name: "input"
#     type_attr: "T"
#   }
#   output_arg {
#     name: "output"
#     type: DT_STRING
#   }
#   attr {
#     name: "T"
#     type: "type"
#     allowed_values {
#       list {
#         type: DT_INT32
#         type: DT_INT64
#         type: DT_COMPLEX64
#         type: DT_FLOAT
#         type: DT_DOUBLE
#         type: DT_BOOL
#         type: DT_INT8
#       }
#     }
#   }
#   attr {
#     name: "precision"
#     type: "int"
#     default_value {
#       i: -1
#     }
#   }
#   attr {
#     name: "scientific"
#     type: "bool"
#     default_value {
#       b: false
#     }
#   }
#   attr {
#     name: "shortest"
#     type: "bool"
#     default_value {
#       b: false
#     }
#   }
#   attr {
#     name: "width"
#     type: "int"
#     default_value {
#       i: -1
#     }
#   }
#   attr {
#     name: "fill"
#     type: "string"
#     default_value {
#       s: ""
#     }
#   }
# }
# op {
#   name: "DecodeBase64"
#   input_arg {
#     name: "input"
#     type: DT_STRING
#   }
#   output_arg {
#     name: "output"
#     type: DT_STRING
#   }
# }
# op {
#   name: "EncodeBase64"
#   input_arg {
#     name: "input"
#     type: DT_STRING
#   }
#   output_arg {
#     name: "output"
#     type: DT_STRING
#   }
#   attr {
#     name: "pad"
#     type: "bool"
#     default_value {
#       b: false
#     }
#   }
# }
# op {
#   name: "ReduceJoin"
#   input_arg {
#     name: "inputs"
#     type: DT_STRING
#   }
#   input_arg {
#     name: "reduction_indices"
#     type: DT_INT32
#   }
#   output_arg {
#     name: "output"
#     type: DT_STRING
#   }
#   attr {
#     name: "keep_dims"
#     type: "bool"
#     default_value {
#       b: false
#     }
#   }
#   attr {
#     name: "separator"
#     type: "string"
#     default_value {
#       s: ""
#     }
#   }
# }
# op {
#   name: "StringJoin"
#   input_arg {
#     name: "inputs"
#     type: DT_STRING
#     number_attr: "N"
#   }
#   output_arg {
#     name: "output"
#     type: DT_STRING
#   }
#   attr {
#     name: "N"
#     type: "int"
#     has_minimum: true
#     minimum: 1
#   }
#   attr {
#     name: "separator"
#     type: "string"
#     default_value {
#       s: ""
#     }
#   }
# }
# op {
#   name: "StringSplit"
#   input_arg {
#     name: "input"
#     type: DT_STRING
#   }
#   input_arg {
#     name: "delimiter"
#     type: DT_STRING
#   }
#   output_arg {
#     name: "indices"
#     type: DT_INT64
#   }
#   output_arg {
#     name: "values"
#     type: DT_STRING
#   }
#   output_arg {
#     name: "shape"
#     type: DT_INT64
#   }
#   attr {
#     name: "skip_empty"
#     type: "bool"
#     default_value {
#       b: true
#     }
#   }
# }
# op {
#   name: "StringToHashBucket"
#   input_arg {
#     name: "string_tensor"
#     type: DT_STRING
#   }
#   output_arg {
#     name: "output"
#     type: DT_INT64
#   }
#   attr {
#     name: "num_buckets"
#     type: "int"
#     has_minimum: true
#     minimum: 1
#   }
# }
# op {
#   name: "StringToHashBucketFast"
#   input_arg {
#     name: "input"
#     type: DT_STRING
#   }
#   output_arg {
#     name: "output"
#     type: DT_INT64
#   }
#   attr {
#     name: "num_buckets"
#     type: "int"
#     has_minimum: true
#     minimum: 1
#   }
# }
# op {
#   name: "StringToHashBucketStrong"
#   input_arg {
#     name: "input"
#     type: DT_STRING
#   }
#   output_arg {
#     name: "output"
#     type: DT_INT64
#   }
#   attr {
#     name: "num_buckets"
#     type: "int"
#     has_minimum: true
#     minimum: 1
#   }
#   attr {
#     name: "key"
#     type: "list(int)"
#   }
# }
# op {
#   name: "Substr"
#   input_arg {
#     name: "input"
#     type: DT_STRING
#   }
#   input_arg {
#     name: "pos"
#     type_attr: "T"
#   }
#   input_arg {
#     name: "len"
#     type_attr: "T"
#   }
#   output_arg {
#     name: "output"
#     type: DT_STRING
#   }
#   attr {
#     name: "T"
#     type: "type"
#     allowed_values {
#       list {
#         type: DT_INT32
#         type: DT_INT64
#       }
#     }
#   }
# }
_op_def_lib = _InitOpDefLibrary(b"\n\266\001\n\010AsString\022\n\n\005input\"\001T\032\n\n\006output\030\007\"\026\n\001T\022\004type:\013\n\t2\007\003\t\010\001\002\n\006\"\035\n\tprecision\022\003int\032\013\030\377\377\377\377\377\377\377\377\377\001\"\026\n\nscientific\022\004bool\032\002(\000\"\024\n\010shortest\022\004bool\032\002(\000\"\031\n\005width\022\003int\032\013\030\377\377\377\377\377\377\377\377\377\001\"\022\n\004fill\022\006string\032\002\022\000\n%\n\014DecodeBase64\022\t\n\005input\030\007\032\n\n\006output\030\007\n6\n\014EncodeBase64\022\t\n\005input\030\007\032\n\n\006output\030\007\"\017\n\003pad\022\004bool\032\002(\000\nk\n\nReduceJoin\022\n\n\006inputs\030\007\022\025\n\021reduction_indices\030\003\032\n\n\006output\030\007\"\025\n\tkeep_dims\022\004bool\032\002(\000\"\027\n\tseparator\022\006string\032\002\022\000\nN\n\nStringJoin\022\r\n\006inputs\030\007*\001N\032\n\n\006output\030\007\"\014\n\001N\022\003int(\0010\001\"\027\n\tseparator\022\006string\032\002\022\000\nc\n\013StringSplit\022\t\n\005input\030\007\022\r\n\tdelimiter\030\007\032\013\n\007indices\030\t\032\n\n\006values\030\007\032\t\n\005shape\030\t\"\026\n\nskip_empty\022\004bool\032\002(\001\nK\n\022StringToHashBucket\022\021\n\rstring_tensor\030\007\032\n\n\006output\030\t\"\026\n\013num_buckets\022\003int(\0010\001\nG\n\026StringToHashBucketFast\022\t\n\005input\030\007\032\n\n\006output\030\t\"\026\n\013num_buckets\022\003int(\0010\001\n[\n\030StringToHashBucketStrong\022\t\n\005input\030\007\032\n\n\006output\030\t\"\026\n\013num_buckets\022\003int(\0010\001\"\020\n\003key\022\tlist(int)\nF\n\006Substr\022\t\n\005input\030\007\022\010\n\003pos\"\001T\022\010\n\003len\"\001T\032\n\n\006output\030\007\"\021\n\001T\022\004type:\006\n\0042\002\003\t")
