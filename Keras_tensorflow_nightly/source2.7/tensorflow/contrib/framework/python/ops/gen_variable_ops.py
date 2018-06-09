"""Python wrappers around TensorFlow ops.

This file is MACHINE GENERATED! Do not edit.
Original C++ source file: gen_variable_ops.cc
"""

import collections as _collections
import six as _six

from tensorflow.python import pywrap_tensorflow as _pywrap_tensorflow
from tensorflow.python.eager import context as _context
from tensorflow.python.eager import core as _core
from tensorflow.python.eager import execute as _execute
from tensorflow.python.framework import dtypes as _dtypes
from tensorflow.python.framework import errors as _errors
from tensorflow.python.framework import tensor_shape as _tensor_shape

from tensorflow.core.framework import op_def_pb2 as _op_def_pb2
# Needed to trigger the call to _set_call_cpp_shape_fn.
from tensorflow.python.framework import common_shapes as _common_shapes
from tensorflow.python.framework import op_def_registry as _op_def_registry
from tensorflow.python.framework import ops as _ops
from tensorflow.python.framework import op_def_library as _op_def_library
from tensorflow.python.util.tf_export import tf_export


@tf_export('zero_initializer')
def zero_initializer(ref, name=None):
  r"""Initialize 'ref' with all zeros. This op requires that the tensor is not

  initialized. The tensor will first be allocated memory, then be filled with all
  zeros. This op is intended to save memory during initialization,
  if you use this op, you should not run initializer of the 'ref' tensor.

  Args:
    ref: A mutable `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `int64`, `bfloat16`, `uint16`, `half`, `uint32`, `uint64`.
      Should be from a `Variable` node.
    name: A name for the operation (optional).

  Returns:
    Same as "ref".
  """
  _ctx = _context._context
  if _ctx is None or not _ctx._eager_context.is_eager:
    _, _, _op = _op_def_lib._apply_op_helper(
        "ZeroInitializer", ref=ref, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("T", _op.get_attr("T"))
    _execute.record_gradient(
      "ZeroInitializer", _inputs_flat, _attrs, _result, name)
    _result, = _result
    return _result

  else:
    raise RuntimeError("zero_initializer op does not support eager execution. Arg 'output_ref' is a ref.")


  raise RuntimeError("zero_initializer op does not support eager execution. Arg 'output_ref' is a ref.")
_ops.RegisterShape("ZeroInitializer")(None)

def _InitOpDefLibrary(op_list_proto_bytes):
  op_list = _op_def_pb2.OpList()
  op_list.ParseFromString(op_list_proto_bytes)
  _op_def_registry.register_op_list(op_list)
  op_def_lib = _op_def_library.OpDefLibrary()
  op_def_lib.add_op_list(op_list)
  return op_def_lib
# op {
#   name: "ZeroInitializer"
#   input_arg {
#     name: "ref"
#     type_attr: "T"
#     is_ref: true
#   }
#   output_arg {
#     name: "output_ref"
#     type_attr: "T"
#     is_ref: true
#   }
#   attr {
#     name: "T"
#     type: "type"
#     allowed_values {
#       list {
#         type: DT_FLOAT
#         type: DT_DOUBLE
#         type: DT_INT32
#         type: DT_UINT8
#         type: DT_INT16
#         type: DT_INT8
#         type: DT_INT64
#         type: DT_BFLOAT16
#         type: DT_UINT16
#         type: DT_HALF
#         type: DT_UINT32
#         type: DT_UINT64
#       }
#     }
#   }
#   allows_uninitialized_input: true
# }
_op_def_lib = _InitOpDefLibrary(b"\nR\n\017ZeroInitializer\022\013\n\003ref\"\001T\200\001\001\032\022\n\noutput_ref\"\001T\200\001\001\"\033\n\001T\022\004type:\020\n\0162\014\001\002\003\004\005\006\t\016\021\023\026\027\230\001\001")
