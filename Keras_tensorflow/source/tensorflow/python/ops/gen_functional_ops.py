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
__symbolic_gradient_outputs = ["output"]


def _symbolic_gradient(input, Tout, f, name=None):
  r"""Computes the gradient function for function f via backpropagation.

  Args:
    input: A list of `Tensor` objects. a list of input tensors of size N + M;
    Tout: A list of `tf.DTypes` that has length `>= 1`.
      the type list for the input list.
    f: . The function we want to compute the gradient for.

      The function 'f' must be a numerical function which takes N inputs and
      produces M outputs. Its gradient function 'g', which is computed by
      this SymbolicGradient op is a function taking N + M inputs and
      produces N outputs.

      I.e. if we have
         (y1, y2, ..., y_M) = f(x1, x2, ..., x_N),
      then, g is
         (dL/dx1, dL/dx2, ..., dL/dx_N) = g(x1, x2, ..., x_N,
                                           dL/dy1, dL/dy2, ..., dL/dy_M),

      where L is a scalar-value function of (x1, x2, ..., xN) (e.g., the
      loss function). dL/dx_i is the partial derivative of L with respect
      to x_i.

      (Needs some math expert to say the comment above better.)
    name: A name for the operation (optional).

  Returns:
    A list of `Tensor` objects of type `Tout`.
    a list of output tensors of size N;
  """
  result = _op_def_lib.apply_op("SymbolicGradient", input=input, Tout=Tout,
                                f=f, name=name)
  return result


_ops.RegisterShape("SymbolicGradient")(None)
def _InitOpDefLibrary():
  op_list = _op_def_pb2.OpList()
  _text_format.Merge(_InitOpDefLibrary.op_list_ascii, op_list)
  _op_def_registry.register_op_list(op_list)
  op_def_lib = _op_def_library.OpDefLibrary()
  op_def_lib.add_op_list(op_list)
  return op_def_lib


_InitOpDefLibrary.op_list_ascii = """op {
  name: "SymbolicGradient"
  input_arg {
    name: "input"
    type_list_attr: "Tin"
  }
  output_arg {
    name: "output"
    type_list_attr: "Tout"
  }
  attr {
    name: "Tin"
    type: "list(type)"
    has_minimum: true
    minimum: 1
  }
  attr {
    name: "Tout"
    type: "list(type)"
    has_minimum: true
    minimum: 1
  }
  attr {
    name: "f"
    type: "func"
  }
}
"""


_op_def_lib = _InitOpDefLibrary()
