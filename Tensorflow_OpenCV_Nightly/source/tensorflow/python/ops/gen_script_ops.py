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

def _py_func(input, token, Tout, name=None):
  r"""Invokes a python function to compute func(input)->output.

  This operation is considered stateful. For a stateless version, see
  PyFuncStateless.

  Args:
    input: A list of `Tensor` objects.
      List of Tensors that will provide input to the Op.
    token: A `string`.
      A token representing a registered python function in this address space.
    Tout: A list of `tf.DTypes`. Data types of the outputs from the op.
      The length of the list specifies the number of outputs.
    name: A name for the operation (optional).

  Returns:
    A list of `Tensor` objects of type `Tout`. The outputs from the Op.
  """
  result = _op_def_lib.apply_op("PyFunc", input=input, token=token, Tout=Tout,
                                name=name)
  return result



def _py_func_stateless(input, token, Tout, name=None):
  r"""A stateless version of PyFunc.

  Args:
    input: A list of `Tensor` objects.
    token: A `string`.
    Tout: A list of `tf.DTypes`.
    name: A name for the operation (optional).

  Returns:
    A list of `Tensor` objects of type `Tout`.
  """
  result = _op_def_lib.apply_op("PyFuncStateless", input=input, token=token,
                                Tout=Tout, name=name)
  return result


def _InitOpDefLibrary():
  op_list = _op_def_pb2.OpList()
  _text_format.Merge(_InitOpDefLibrary.op_list_ascii, op_list)
  _op_def_registry.register_op_list(op_list)
  op_def_lib = _op_def_library.OpDefLibrary()
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
  }
  is_stateful: true
}
op {
  name: "PyFuncStateless"
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
  }
}
"""


_op_def_lib = _InitOpDefLibrary()
