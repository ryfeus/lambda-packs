"""Python wrappers around TensorFlow ops.

This file is MACHINE GENERATED! Do not edit.
Original C++ source file: bitwise_ops.cc
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


def bitwise_and(x, y, name=None):
  r"""Elementwise computes the bitwise AND of `x` and `y`.

  The result will have those bits set, that are set in both `x` and `y`. The
  computation is performed on the underlying representations of `x` and `y`.

  Args:
    x: A `Tensor`. Must be one of the following types: `int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`.
    y: A `Tensor`. Must have the same type as `x`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "BitwiseAnd", x=x, y=y, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("T", _op.get_attr("T"))
  else:
    _attr_T, _inputs_T = _execute.args_to_matching_eager([x, y], _ctx)
    (x, y) = _inputs_T
    _attr_T = _attr_T.as_datatype_enum
    _inputs_flat = [x, y]
    _attrs = ("T", _attr_T)
    _result = _execute.execute(b"BitwiseAnd", 1, inputs=_inputs_flat,
                               attrs=_attrs, ctx=_ctx, name=name)
  _execute.record_gradient(
      "BitwiseAnd", _inputs_flat, _attrs, _result, name)
  _result, = _result
  return _result


def bitwise_or(x, y, name=None):
  r"""Elementwise computes the bitwise OR of `x` and `y`.

  The result will have those bits set, that are set in `x`, `y` or both. The
  computation is performed on the underlying representations of `x` and `y`.

  Args:
    x: A `Tensor`. Must be one of the following types: `int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`.
    y: A `Tensor`. Must have the same type as `x`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "BitwiseOr", x=x, y=y, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("T", _op.get_attr("T"))
  else:
    _attr_T, _inputs_T = _execute.args_to_matching_eager([x, y], _ctx)
    (x, y) = _inputs_T
    _attr_T = _attr_T.as_datatype_enum
    _inputs_flat = [x, y]
    _attrs = ("T", _attr_T)
    _result = _execute.execute(b"BitwiseOr", 1, inputs=_inputs_flat,
                               attrs=_attrs, ctx=_ctx, name=name)
  _execute.record_gradient(
      "BitwiseOr", _inputs_flat, _attrs, _result, name)
  _result, = _result
  return _result


def bitwise_xor(x, y, name=None):
  r"""Elementwise computes the bitwise XOR of `x` and `y`.

  The result will have those bits set, that are different in `x` and `y`. The
  computation is performed on the underlying representations of `x` and `y`.

  Args:
    x: A `Tensor`. Must be one of the following types: `int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`.
    y: A `Tensor`. Must have the same type as `x`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "BitwiseXor", x=x, y=y, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("T", _op.get_attr("T"))
  else:
    _attr_T, _inputs_T = _execute.args_to_matching_eager([x, y], _ctx)
    (x, y) = _inputs_T
    _attr_T = _attr_T.as_datatype_enum
    _inputs_flat = [x, y]
    _attrs = ("T", _attr_T)
    _result = _execute.execute(b"BitwiseXor", 1, inputs=_inputs_flat,
                               attrs=_attrs, ctx=_ctx, name=name)
  _execute.record_gradient(
      "BitwiseXor", _inputs_flat, _attrs, _result, name)
  _result, = _result
  return _result


def invert(x, name=None):
  r"""Flips all bits elementwise.

  The result will have exactly those bits set, that are not set in `x`. The
  computation is performed on the underlying representation of x.

  Args:
    x: A `Tensor`. Must be one of the following types: `int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "Invert", x=x, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("T", _op.get_attr("T"))
  else:
    _attr_T, (x,) = _execute.args_to_matching_eager([x], _ctx)
    _attr_T = _attr_T.as_datatype_enum
    _inputs_flat = [x]
    _attrs = ("T", _attr_T)
    _result = _execute.execute(b"Invert", 1, inputs=_inputs_flat,
                               attrs=_attrs, ctx=_ctx, name=name)
  _execute.record_gradient(
      "Invert", _inputs_flat, _attrs, _result, name)
  _result, = _result
  return _result


def population_count(x, name=None):
  r"""Computes element-wise population count (a.k.a. popcount, bitsum, bitcount).

  For each entry in `x`, calculates the number of `1` (on) bits in the binary
  representation of that entry.

  **NOTE**: It is more efficient to first `tf.bitcast` your tensors into
  `int32` or `int64` and perform the bitcount on the result, than to feed in
  8- or 16-bit inputs and then aggregate the resulting counts.

  Args:
    x: A `Tensor`. Must be one of the following types: `int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `uint8`.
  """
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "PopulationCount", x=x, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("T", _op.get_attr("T"))
  else:
    _attr_T, (x,) = _execute.args_to_matching_eager([x], _ctx)
    _attr_T = _attr_T.as_datatype_enum
    _inputs_flat = [x]
    _attrs = ("T", _attr_T)
    _result = _execute.execute(b"PopulationCount", 1, inputs=_inputs_flat,
                               attrs=_attrs, ctx=_ctx, name=name)
  _execute.record_gradient(
      "PopulationCount", _inputs_flat, _attrs, _result, name)
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
#   name: "BitwiseAnd"
#   input_arg {
#     name: "x"
#     type_attr: "T"
#   }
#   input_arg {
#     name: "y"
#     type_attr: "T"
#   }
#   output_arg {
#     name: "z"
#     type_attr: "T"
#   }
#   attr {
#     name: "T"
#     type: "type"
#     allowed_values {
#       list {
#         type: DT_INT8
#         type: DT_INT16
#         type: DT_INT32
#         type: DT_INT64
#         type: DT_UINT8
#         type: DT_UINT16
#       }
#     }
#   }
#   is_commutative: true
# }
# op {
#   name: "BitwiseOr"
#   input_arg {
#     name: "x"
#     type_attr: "T"
#   }
#   input_arg {
#     name: "y"
#     type_attr: "T"
#   }
#   output_arg {
#     name: "z"
#     type_attr: "T"
#   }
#   attr {
#     name: "T"
#     type: "type"
#     allowed_values {
#       list {
#         type: DT_INT8
#         type: DT_INT16
#         type: DT_INT32
#         type: DT_INT64
#         type: DT_UINT8
#         type: DT_UINT16
#       }
#     }
#   }
#   is_commutative: true
# }
# op {
#   name: "BitwiseXor"
#   input_arg {
#     name: "x"
#     type_attr: "T"
#   }
#   input_arg {
#     name: "y"
#     type_attr: "T"
#   }
#   output_arg {
#     name: "z"
#     type_attr: "T"
#   }
#   attr {
#     name: "T"
#     type: "type"
#     allowed_values {
#       list {
#         type: DT_INT8
#         type: DT_INT16
#         type: DT_INT32
#         type: DT_INT64
#         type: DT_UINT8
#         type: DT_UINT16
#       }
#     }
#   }
#   is_commutative: true
# }
# op {
#   name: "Invert"
#   input_arg {
#     name: "x"
#     type_attr: "T"
#   }
#   output_arg {
#     name: "y"
#     type_attr: "T"
#   }
#   attr {
#     name: "T"
#     type: "type"
#     allowed_values {
#       list {
#         type: DT_INT8
#         type: DT_INT16
#         type: DT_INT32
#         type: DT_INT64
#         type: DT_UINT8
#         type: DT_UINT16
#       }
#     }
#   }
# }
# op {
#   name: "PopulationCount"
#   input_arg {
#     name: "x"
#     type_attr: "T"
#   }
#   output_arg {
#     name: "y"
#     type: DT_UINT8
#   }
#   attr {
#     name: "T"
#     type: "type"
#     allowed_values {
#       list {
#         type: DT_INT8
#         type: DT_INT16
#         type: DT_INT32
#         type: DT_INT64
#         type: DT_UINT8
#         type: DT_UINT16
#       }
#     }
#   }
# }
_op_def_lib = _InitOpDefLibrary(b"\n>\n\nBitwiseAnd\022\006\n\001x\"\001T\022\006\n\001y\"\001T\032\006\n\001z\"\001T\"\025\n\001T\022\004type:\n\n\0102\006\006\005\003\t\004\021\220\001\001\n=\n\tBitwiseOr\022\006\n\001x\"\001T\022\006\n\001y\"\001T\032\006\n\001z\"\001T\"\025\n\001T\022\004type:\n\n\0102\006\006\005\003\t\004\021\220\001\001\n>\n\nBitwiseXor\022\006\n\001x\"\001T\022\006\n\001y\"\001T\032\006\n\001z\"\001T\"\025\n\001T\022\004type:\n\n\0102\006\006\005\003\t\004\021\220\001\001\n/\n\006Invert\022\006\n\001x\"\001T\032\006\n\001y\"\001T\"\025\n\001T\022\004type:\n\n\0102\006\006\005\003\t\004\021\n7\n\017PopulationCount\022\006\n\001x\"\001T\032\005\n\001y\030\004\"\025\n\001T\022\004type:\n\n\0102\006\006\005\003\t\004\021")
