"""Python wrappers around TensorFlow ops.

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

def obtain_next(list, counter, name=None):
  r"""Takes a list and returns the next based on a counter in a round-robin fashion.

  Returns the element in the list at the new position of the counter, so if you
  want to circle the list around start by setting the counter value = -1.

  Args:
    list: A `Tensor` of type `string`. A list of strings
    counter: A `Tensor` of type mutable `int64`.
      A reference to an int64 variable
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`.
  """
  result = _op_def_lib.apply_op("ObtainNext", list=list, counter=counter,
                                name=name)
  return result


_ops.RegisterShape("ObtainNext")(None)
def _InitOpDefLibrary():
  op_list = _op_def_pb2.OpList()
  _text_format.Merge(_InitOpDefLibrary.op_list_ascii, op_list)
  _op_def_registry.register_op_list(op_list)
  op_def_lib = _op_def_library.OpDefLibrary()
  op_def_lib.add_op_list(op_list)
  return op_def_lib


_InitOpDefLibrary.op_list_ascii = """op {
  name: "ObtainNext"
  input_arg {
    name: "list"
    type: DT_STRING
  }
  input_arg {
    name: "counter"
    type: DT_INT64
    is_ref: true
  }
  output_arg {
    name: "out_element"
    type: DT_STRING
  }
}
"""


_op_def_lib = _InitOpDefLibrary()
