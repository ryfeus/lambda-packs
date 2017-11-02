"""Python wrappers around TensorFlow ops.

This file is MACHINE GENERATED! Do not edit.
Original C++ source file: stateless_random_ops.cc
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


def stateless_random_normal(shape, seed, dtype=_dtypes.float32, name=None):
  r"""Outputs deterministic pseudorandom values from a normal distribution.

  The generated values will have mean 0 and standard deviation 1.

  The outputs are a deterministic function of `shape` and `seed`.

  Args:
    shape: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      The shape of the output tensor.
    seed: A `Tensor` of type `int64`. 2 seeds (shape [2]).
    dtype: An optional `tf.DType` from: `tf.half, tf.float32, tf.float64`. Defaults to `tf.float32`.
      The type of the output.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `dtype`. Random values with specified shape.
  """
  if dtype is None:
    dtype = _dtypes.float32
  dtype = _execute.make_type(dtype, "dtype")
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "StatelessRandomNormal", shape=shape, seed=seed, dtype=dtype,
        name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("dtype", _op.get_attr("dtype"), "T", _op.get_attr("T"))
  else:
    _attr_T, (shape,) = _execute.args_to_matching_eager([shape], _ctx, _dtypes.int32)
    _attr_T = _attr_T.as_datatype_enum
    seed = _ops.convert_to_tensor(seed, _dtypes.int64)
    _inputs_flat = [shape, seed]
    _attrs = ("dtype", dtype, "T", _attr_T)
    _result = _execute.execute(b"StatelessRandomNormal", 1,
                               inputs=_inputs_flat, attrs=_attrs, ctx=_ctx,
                               name=name)
  _execute.record_gradient(
      "StatelessRandomNormal", _inputs_flat, _attrs, _result, name)
  _result, = _result
  return _result

_ops.RegisterShape("StatelessRandomNormal")(None)


def stateless_random_uniform(shape, seed, dtype=_dtypes.float32, name=None):
  r"""Outputs deterministic pseudorandom random values from a uniform distribution.

  The generated values follow a uniform distribution in the range `[0, 1)`. The
  lower bound 0 is included in the range, while the upper bound 1 is excluded.

  The outputs are a deterministic function of `shape` and `seed`.

  Args:
    shape: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      The shape of the output tensor.
    seed: A `Tensor` of type `int64`. 2 seeds (shape [2]).
    dtype: An optional `tf.DType` from: `tf.half, tf.float32, tf.float64`. Defaults to `tf.float32`.
      The type of the output.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `dtype`. Random values with specified shape.
  """
  if dtype is None:
    dtype = _dtypes.float32
  dtype = _execute.make_type(dtype, "dtype")
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "StatelessRandomUniform", shape=shape, seed=seed, dtype=dtype,
        name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("dtype", _op.get_attr("dtype"), "T", _op.get_attr("T"))
  else:
    _attr_T, (shape,) = _execute.args_to_matching_eager([shape], _ctx, _dtypes.int32)
    _attr_T = _attr_T.as_datatype_enum
    seed = _ops.convert_to_tensor(seed, _dtypes.int64)
    _inputs_flat = [shape, seed]
    _attrs = ("dtype", dtype, "T", _attr_T)
    _result = _execute.execute(b"StatelessRandomUniform", 1,
                               inputs=_inputs_flat, attrs=_attrs, ctx=_ctx,
                               name=name)
  _execute.record_gradient(
      "StatelessRandomUniform", _inputs_flat, _attrs, _result, name)
  _result, = _result
  return _result

_ops.RegisterShape("StatelessRandomUniform")(None)


def stateless_truncated_normal(shape, seed, dtype=_dtypes.float32, name=None):
  r"""Outputs deterministic pseudorandom values from a truncated normal distribution.

  The generated values follow a normal distribution with mean 0 and standard
  deviation 1, except that values whose magnitude is more than 2 standard
  deviations from the mean are dropped and re-picked.

  The outputs are a deterministic function of `shape` and `seed`.

  Args:
    shape: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      The shape of the output tensor.
    seed: A `Tensor` of type `int64`. 2 seeds (shape [2]).
    dtype: An optional `tf.DType` from: `tf.half, tf.float32, tf.float64`. Defaults to `tf.float32`.
      The type of the output.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `dtype`. Random values with specified shape.
  """
  if dtype is None:
    dtype = _dtypes.float32
  dtype = _execute.make_type(dtype, "dtype")
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "StatelessTruncatedNormal", shape=shape, seed=seed, dtype=dtype,
        name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("dtype", _op.get_attr("dtype"), "T", _op.get_attr("T"))
  else:
    _attr_T, (shape,) = _execute.args_to_matching_eager([shape], _ctx, _dtypes.int32)
    _attr_T = _attr_T.as_datatype_enum
    seed = _ops.convert_to_tensor(seed, _dtypes.int64)
    _inputs_flat = [shape, seed]
    _attrs = ("dtype", dtype, "T", _attr_T)
    _result = _execute.execute(b"StatelessTruncatedNormal", 1,
                               inputs=_inputs_flat, attrs=_attrs, ctx=_ctx,
                               name=name)
  _execute.record_gradient(
      "StatelessTruncatedNormal", _inputs_flat, _attrs, _result, name)
  _result, = _result
  return _result

_ops.RegisterShape("StatelessTruncatedNormal")(None)

def _InitOpDefLibrary(op_list_proto_bytes):
  op_list = _op_def_pb2.OpList()
  op_list.ParseFromString(op_list_proto_bytes)
  _op_def_registry.register_op_list(op_list)
  op_def_lib = _op_def_library.OpDefLibrary()
  op_def_lib.add_op_list(op_list)
  return op_def_lib
# op {
#   name: "StatelessRandomNormal"
#   input_arg {
#     name: "shape"
#     type_attr: "T"
#   }
#   input_arg {
#     name: "seed"
#     type: DT_INT64
#   }
#   output_arg {
#     name: "output"
#     type_attr: "dtype"
#   }
#   attr {
#     name: "dtype"
#     type: "type"
#     default_value {
#       type: DT_FLOAT
#     }
#     allowed_values {
#       list {
#         type: DT_HALF
#         type: DT_FLOAT
#         type: DT_DOUBLE
#       }
#     }
#   }
#   attr {
#     name: "T"
#     type: "type"
#     default_value {
#       type: DT_INT32
#     }
#     allowed_values {
#       list {
#         type: DT_INT32
#         type: DT_INT64
#       }
#     }
#   }
# }
# op {
#   name: "StatelessRandomUniform"
#   input_arg {
#     name: "shape"
#     type_attr: "T"
#   }
#   input_arg {
#     name: "seed"
#     type: DT_INT64
#   }
#   output_arg {
#     name: "output"
#     type_attr: "dtype"
#   }
#   attr {
#     name: "dtype"
#     type: "type"
#     default_value {
#       type: DT_FLOAT
#     }
#     allowed_values {
#       list {
#         type: DT_HALF
#         type: DT_FLOAT
#         type: DT_DOUBLE
#       }
#     }
#   }
#   attr {
#     name: "T"
#     type: "type"
#     default_value {
#       type: DT_INT32
#     }
#     allowed_values {
#       list {
#         type: DT_INT32
#         type: DT_INT64
#       }
#     }
#   }
# }
# op {
#   name: "StatelessTruncatedNormal"
#   input_arg {
#     name: "shape"
#     type_attr: "T"
#   }
#   input_arg {
#     name: "seed"
#     type: DT_INT64
#   }
#   output_arg {
#     name: "output"
#     type_attr: "dtype"
#   }
#   attr {
#     name: "dtype"
#     type: "type"
#     default_value {
#       type: DT_FLOAT
#     }
#     allowed_values {
#       list {
#         type: DT_HALF
#         type: DT_FLOAT
#         type: DT_DOUBLE
#       }
#     }
#   }
#   attr {
#     name: "T"
#     type: "type"
#     default_value {
#       type: DT_INT32
#     }
#     allowed_values {
#       list {
#         type: DT_INT32
#         type: DT_INT64
#       }
#     }
#   }
# }
_op_def_lib = _InitOpDefLibrary(b"\nq\n\025StatelessRandomNormal\022\n\n\005shape\"\001T\022\010\n\004seed\030\t\032\017\n\006output\"\005dtype\"\032\n\005dtype\022\004type\032\0020\001:\007\n\0052\003\023\001\002\"\025\n\001T\022\004type\032\0020\003:\006\n\0042\002\003\t\nr\n\026StatelessRandomUniform\022\n\n\005shape\"\001T\022\010\n\004seed\030\t\032\017\n\006output\"\005dtype\"\032\n\005dtype\022\004type\032\0020\001:\007\n\0052\003\023\001\002\"\025\n\001T\022\004type\032\0020\003:\006\n\0042\002\003\t\nt\n\030StatelessTruncatedNormal\022\n\n\005shape\"\001T\022\010\n\004seed\030\t\032\017\n\006output\"\005dtype\"\032\n\005dtype\022\004type\032\0020\001:\007\n\0052\003\023\001\002\"\025\n\001T\022\004type\032\0020\003:\006\n\0042\002\003\t")
