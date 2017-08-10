"""Python wrappers around Brain.

This file is MACHINE GENERATED! Do not edit.
"""

import collections as _collections

from google.protobuf import text_format as _text_format

from tensorflow.core.framework import op_def_pb2 as _op_def_pb2

# Needed to trigger the call to _set_call_cpp_shape_fn.
from tensorflow.python.framework import common_shapes as _common_shapes

from tensorflow.python.framework import op_def_registry as _op_def_registry
from tensorflow.python.framework import ops as _ops
from tensorflow.python.framework import op_def_library as _op_def_library
_zero_initializer_outputs = ["output_ref"]


def zero_initializer(ref, name=None):
  r"""Initialize 'ref' with all zeros. This op requires that the tensor is not

  initialized. The tensor will first be allocated memory, then be filled with all
  zeros. This op is intended to save memory during initialization,
  if you use this op, you should not run initializer of the 'ref' tensor.

  Args:
    ref: A mutable `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`, `uint8`, `int16`, `int8`, `uint16`, `half`.
      Should be from a `Variable` node.
    name: A name for the operation (optional).

  Returns:
    Same as "ref".
  """
  result = _op_def_lib.apply_op("ZeroInitializer", ref=ref, name=name)
  return result


_ops.RegisterShape("ZeroInitializer")(None)
def _InitOpDefLibrary():
  op_list = _op_def_pb2.OpList()
  _text_format.Merge(_InitOpDefLibrary.op_list_ascii, op_list)
  _op_def_registry.register_op_list(op_list)
  op_def_lib = _op_def_library.OpDefLibrary()
  op_def_lib.add_op_list(op_list)
  return op_def_lib


_InitOpDefLibrary.op_list_ascii = """op {
  name: "ZeroInitializer"
  input_arg {
    name: "ref"
    type_attr: "T"
    is_ref: true
  }
  output_arg {
    name: "output_ref"
    type_attr: "T"
    is_ref: true
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT32
        type: DT_INT64
        type: DT_UINT8
        type: DT_INT16
        type: DT_INT8
        type: DT_UINT16
        type: DT_HALF
      }
    }
  }
  allows_uninitialized_input: true
}
"""


_op_def_lib = _InitOpDefLibrary()
