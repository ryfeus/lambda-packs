"""Python wrappers around Brain.

This file is MACHINE GENERATED! Do not edit.
"""

import collections

from google.protobuf import text_format

from tensorflow.core.framework import op_def_pb2
from tensorflow.python.framework import op_def_registry
from tensorflow.python.framework import ops
from tensorflow.python.ops import op_def_library


__py_func_outputs = ["output"]


def _py_func(input, token, Tout, name=None):
  r"""Invokes a python function to compute func(input)->output.

  Args:
    input: A list of `Tensor` objects.
      List of Tensors that will provide input to the Op.
    token: A `string`.
      A token representing a registered python function in this address space.
    Tout: A list of `tf.DTypes` that has length `>= 1`.
      Data types of the outputs from the op.
      The length of the list specifies the number of outputs.
    name: A name for the operation (optional).

  Returns:
    A list of `Tensor` objects of type `Tout`. The outputs from the Op.
  """
  result = _op_def_lib.apply_op("PyFunc", input=input, token=token, Tout=Tout,
                                name=name)
  return result


def _InitOpDefLibrary():
  op_list = op_def_pb2.OpList()
  text_format.Merge(_InitOpDefLibrary.op_list_ascii, op_list)
  op_def_registry.register_op_list(op_list)
  op_def_lib = op_def_library.OpDefLibrary()
  op_def_lib.add_op_list(op_list)
  return op_def_lib


_InitOpDefLibrary.op_list_ascii = """op {
  name: "PyFunc"
  input_arg {
    name: "input"
    type_list_attr: "Tin"
  }
  output_arg {
    name: "output"
    type_list_attr: "Tout"
  }
  attr {
    name: "token"
    type: "string"
  }
  attr {
    name: "Tin"
    type: "list(type)"
    has_minimum: true
  }
  attr {
    name: "Tout"
    type: "list(type)"
    has_minimum: true
    minimum: 1
  }
  is_stateful: true
}
"""


_op_def_lib = _InitOpDefLibrary()
