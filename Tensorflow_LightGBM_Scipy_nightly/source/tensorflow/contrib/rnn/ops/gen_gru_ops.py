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

_gru_block_cell_outputs = ["r", "u", "c", "h"]
_GRUBlockCellOutput = _collections.namedtuple(
    "GRUBlockCell", _gru_block_cell_outputs)


def gru_block_cell(x, h_prev, w_ru, w_c, b_ru, b_c, name=None):
  r"""Computes the GRU cell forward propagation for 1 time step.

  Args
      x: Input to the GRU cell.
      h_prev: State input from the previous GRU cell.
      w_ru: Weight matrix for the reset and update gate.
      w_c: Weight matrix for the cell connection gate.
      b_ru: Bias vector for the reset and update gate.
      b_c: Bias vector for the cell connection gate.

  Returns
      r: Output of the reset gate.
      u: Output of the update gate.
      c: Output of the cell connection gate.
      h: Current state of the GRU cell.

  Note on notation of the variables:

  Concatenation of a and b is represented by a_b
  Element-wise dot product of a and b is represented by ab
  Element-wise dot product is represented by \circ
  Matrix multiplication is represented by *

  Baises are initialized with :
  `b_ru` - constant_initializer(1.0)
  `b_c` - constant_initializer(0.0)

  This kernel op implements the following mathematical equations:

  ```
  x_h_prev = [x, h_prev]

  [r_bar u_bar] = x_h_prev * w_ru + b_ru

  r = sigmoid(r_bar)
  u = sigmoid(u_bar)

  h_prevr = h_prev \circ r

  x_h_prevr = [x h_prevr]

  c_bar = x_h_prevr * w_c + b_c
  c = tanh(c_bar)

  h = (1-u) \circ c + u \circ h_prev
  ```

  Args:
    x: A `Tensor`. Must be one of the following types: `float32`.
    h_prev: A `Tensor`. Must have the same type as `x`.
    w_ru: A `Tensor`. Must have the same type as `x`.
    w_c: A `Tensor`. Must have the same type as `x`.
    b_ru: A `Tensor`. Must have the same type as `x`.
    b_c: A `Tensor`. Must have the same type as `x`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (r, u, c, h).

    r: A `Tensor`. Has the same type as `x`.
    u: A `Tensor`. Has the same type as `x`.
    c: A `Tensor`. Has the same type as `x`.
    h: A `Tensor`. Has the same type as `x`.
  """
  result = _op_def_lib.apply_op("GRUBlockCell", x=x, h_prev=h_prev, w_ru=w_ru,
                                w_c=w_c, b_ru=b_ru, b_c=b_c, name=name)
  return _GRUBlockCellOutput._make(result)


_ops.RegisterShape("GRUBlockCell")(None)

_gru_block_cell_grad_outputs = ["d_x", "d_h_prev", "d_c_bar", "d_r_bar_u_bar"]
_GRUBlockCellGradOutput = _collections.namedtuple(
    "GRUBlockCellGrad", _gru_block_cell_grad_outputs)


def gru_block_cell_grad(x, h_prev, w_ru, w_c, b_ru, b_c, r, u, c, d_h,
                        name=None):
  r"""Computes the GRU cell back-propagation for 1 time step.

  Args
      x: Input to the GRU cell.
      h_prev: State input from the previous GRU cell.
      w_ru: Weight matrix for the reset and update gate.
      w_c: Weight matrix for the cell connection gate.
      b_ru: Bias vector for the reset and update gate.
      b_c: Bias vector for the cell connection gate.
      r: Output of the reset gate.
      u: Output of the update gate.
      c: Output of the cell connection gate.
      d_h: Gradients of the h_new wrt to objective function.

  Returns
      d_x: Gradients of the x wrt to objective function.
      d_h_prev: Gradients of the h wrt to objective function.
      d_c_bar Gradients of the c_bar wrt to objective function.
      d_r_bar_u_bar Gradients of the r_bar & u_bar wrt to objective function.

  This kernel op implements the following mathematical equations:

  Note on notation of the variables:

  Concatenation of a and b is represented by a_b
  Element-wise dot product of a and b is represented by ab
  Element-wise dot product is represented by \circ
  Matrix multiplication is represented by *

  Additional notes for clarity:

  `w_ru` can be segmented into 4 different matrices.
  ```
  w_ru = [w_r_x w_u_x
          w_r_h_prev w_u_h_prev]
  ```
  Similarly, `w_c` can be segmented into 2 different matrices.
  ```
  w_c = [w_c_x w_c_h_prevr]
  ```
  Same goes for biases.
  ```
  b_ru = [b_ru_x b_ru_h]
  b_c = [b_c_x b_c_h]
  ```
  Another note on notation:
  ```
  d_x = d_x_component_1 + d_x_component_2

  where d_x_component_1 = d_r_bar * w_r_x^T + d_u_bar * w_r_x^T
  and d_x_component_2 = d_c_bar * w_c_x^T

  d_h_prev = d_h_prev_component_1 + d_h_prevr \circ r + d_h \circ u
  where d_h_prev_componenet_1 = d_r_bar * w_r_h_prev^T + d_u_bar * w_r_h_prev^T
  ```

  Mathematics behind the Gradients below:
  ```
  d_c_bar = d_h \circ (1-u) \circ (1-c \circ c)
  d_u_bar = d_h \circ (h-c) \circ u \circ (1-u)

  d_r_bar_u_bar = [d_r_bar d_u_bar]

  [d_x_component_1 d_h_prev_component_1] = d_r_bar_u_bar * w_ru^T

  [d_x_component_2 d_h_prevr] = d_c_bar * w_c^T

  d_x = d_x_component_1 + d_x_component_2

  d_h_prev = d_h_prev_component_1 + d_h_prevr \circ r + u
  ```
  Below calculation is performed in the python wrapper for the Gradients
  (not in the gradient kernel.)
  ```
  d_w_ru = x_h_prevr^T * d_c_bar

  d_w_c = x_h_prev^T * d_r_bar_u_bar

  d_b_ru = sum of d_r_bar_u_bar along axis = 0

  d_b_c = sum of d_c_bar along axis = 0
  ```

  Args:
    x: A `Tensor`. Must be one of the following types: `float32`.
    h_prev: A `Tensor`. Must have the same type as `x`.
    w_ru: A `Tensor`. Must have the same type as `x`.
    w_c: A `Tensor`. Must have the same type as `x`.
    b_ru: A `Tensor`. Must have the same type as `x`.
    b_c: A `Tensor`. Must have the same type as `x`.
    r: A `Tensor`. Must have the same type as `x`.
    u: A `Tensor`. Must have the same type as `x`.
    c: A `Tensor`. Must have the same type as `x`.
    d_h: A `Tensor`. Must have the same type as `x`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (d_x, d_h_prev, d_c_bar, d_r_bar_u_bar).

    d_x: A `Tensor`. Has the same type as `x`.
    d_h_prev: A `Tensor`. Has the same type as `x`.
    d_c_bar: A `Tensor`. Has the same type as `x`.
    d_r_bar_u_bar: A `Tensor`. Has the same type as `x`.
  """
  result = _op_def_lib.apply_op("GRUBlockCellGrad", x=x, h_prev=h_prev,
                                w_ru=w_ru, w_c=w_c, b_ru=b_ru, b_c=b_c, r=r,
                                u=u, c=c, d_h=d_h, name=name)
  return _GRUBlockCellGradOutput._make(result)


_ops.RegisterShape("GRUBlockCellGrad")(None)
def _InitOpDefLibrary():
  op_list = _op_def_pb2.OpList()
  _text_format.Merge(_InitOpDefLibrary.op_list_ascii, op_list)
  _op_def_registry.register_op_list(op_list)
  op_def_lib = _op_def_library.OpDefLibrary()
  op_def_lib.add_op_list(op_list)
  return op_def_lib


_InitOpDefLibrary.op_list_ascii = """op {
  name: "GRUBlockCell"
  input_arg {
    name: "x"
    type_attr: "T"
  }
  input_arg {
    name: "h_prev"
    type_attr: "T"
  }
  input_arg {
    name: "w_ru"
    type_attr: "T"
  }
  input_arg {
    name: "w_c"
    type_attr: "T"
  }
  input_arg {
    name: "b_ru"
    type_attr: "T"
  }
  input_arg {
    name: "b_c"
    type_attr: "T"
  }
  output_arg {
    name: "r"
    type_attr: "T"
  }
  output_arg {
    name: "u"
    type_attr: "T"
  }
  output_arg {
    name: "c"
    type_attr: "T"
  }
  output_arg {
    name: "h"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
      }
    }
  }
}
op {
  name: "GRUBlockCellGrad"
  input_arg {
    name: "x"
    type_attr: "T"
  }
  input_arg {
    name: "h_prev"
    type_attr: "T"
  }
  input_arg {
    name: "w_ru"
    type_attr: "T"
  }
  input_arg {
    name: "w_c"
    type_attr: "T"
  }
  input_arg {
    name: "b_ru"
    type_attr: "T"
  }
  input_arg {
    name: "b_c"
    type_attr: "T"
  }
  input_arg {
    name: "r"
    type_attr: "T"
  }
  input_arg {
    name: "u"
    type_attr: "T"
  }
  input_arg {
    name: "c"
    type_attr: "T"
  }
  input_arg {
    name: "d_h"
    type_attr: "T"
  }
  output_arg {
    name: "d_x"
    type_attr: "T"
  }
  output_arg {
    name: "d_h_prev"
    type_attr: "T"
  }
  output_arg {
    name: "d_c_bar"
    type_attr: "T"
  }
  output_arg {
    name: "d_r_bar_u_bar"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
      }
    }
  }
}
"""


_op_def_lib = _InitOpDefLibrary()
