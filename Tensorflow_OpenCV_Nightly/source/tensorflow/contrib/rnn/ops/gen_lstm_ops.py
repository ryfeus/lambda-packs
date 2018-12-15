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

_block_lstm_outputs = ["i", "cs", "f", "o", "ci", "co", "h"]
_BlockLSTMOutput = _collections.namedtuple(
    "BlockLSTM", _block_lstm_outputs)


def block_lstm(seq_len_max, x, cs_prev, h_prev, w, wci, wcf, wco, b,
               forget_bias=None, cell_clip=None, use_peephole=None,
               name=None):
  r"""Computes the LSTM cell forward propagation for all the time steps.

  This is equivalent to applying LSTMBlockCell in a loop, like so:

  ```python
  for x1 in unpack(x):
    i1, cs1, f1, o1, ci1, co1, h1 = LSTMBlock(
      x1, cs_prev, h_prev, w, wci, wcf, wco, b)
    cs_prev = cs1
    h_prev = h1
    i.append(i1)
    cs.append(cs1)
    f.append(f1)
    o.append(o1)
    ci.append(ci1)
    co.append(co1)
    h.append(h1)
  return pack(i), pack(cs), pack(f), pack(o), pack(ci), pack(ch), pack(h)
  ```

  Args:
    seq_len_max: A `Tensor` of type `int64`.
      Maximum time length actually used by this input. Outputs are padded
      with zeros beyond this length.
    x: A `Tensor`. Must be one of the following types: `float32`.
      The sequence input to the LSTM, shape (timelen, batch_size, num_inputs).
    cs_prev: A `Tensor`. Must have the same type as `x`.
      Value of the initial cell state.
    h_prev: A `Tensor`. Must have the same type as `x`.
      Initial output of cell (to be used for peephole).
    w: A `Tensor`. Must have the same type as `x`. The weight matrix.
    wci: A `Tensor`. Must have the same type as `x`.
      The weight matrix for input gate peephole connection.
    wcf: A `Tensor`. Must have the same type as `x`.
      The weight matrix for forget gate peephole connection.
    wco: A `Tensor`. Must have the same type as `x`.
      The weight matrix for output gate peephole connection.
    b: A `Tensor`. Must have the same type as `x`. The bias vector.
    forget_bias: An optional `float`. Defaults to `1`. The forget gate bias.
    cell_clip: An optional `float`. Defaults to `3`.
      Value to clip the 'cs' value to.
    use_peephole: An optional `bool`. Defaults to `False`.
      Whether to use peephole weights.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (i, cs, f, o, ci, co, h).

    i: A `Tensor`. Has the same type as `x`. The input gate over the whole time sequence.
    cs: A `Tensor`. Has the same type as `x`. The cell state before the tanh over the whole time sequence.
    f: A `Tensor`. Has the same type as `x`. The forget gate over the whole time sequence.
    o: A `Tensor`. Has the same type as `x`. The output gate over the whole time sequence.
    ci: A `Tensor`. Has the same type as `x`. The cell input over the whole time sequence.
    co: A `Tensor`. Has the same type as `x`. The cell after the tanh over the whole time sequence.
    h: A `Tensor`. Has the same type as `x`. The output h vector over the whole time sequence.
  """
  result = _op_def_lib.apply_op("BlockLSTM", seq_len_max=seq_len_max, x=x,
                                cs_prev=cs_prev, h_prev=h_prev, w=w, wci=wci,
                                wcf=wcf, wco=wco, b=b,
                                forget_bias=forget_bias, cell_clip=cell_clip,
                                use_peephole=use_peephole, name=name)
  return _BlockLSTMOutput._make(result)


_ops.RegisterShape("BlockLSTM")(None)

_block_lstm_grad_outputs = ["x_grad", "cs_prev_grad", "h_prev_grad", "w_grad",
                           "wci_grad", "wcf_grad", "wco_grad", "b_grad"]
_BlockLSTMGradOutput = _collections.namedtuple(
    "BlockLSTMGrad", _block_lstm_grad_outputs)


def block_lstm_grad(seq_len_max, x, cs_prev, h_prev, w, wci, wcf, wco, b, i,
                    cs, f, o, ci, co, h, cs_grad, h_grad, use_peephole,
                    name=None):
  r"""Computes the LSTM cell backward propagation for the entire time sequence.

  This implementation is to be used in conjunction of LSTMBlock.

  Args:
    seq_len_max: A `Tensor` of type `int64`.
      Maximum time length actually used by this input. Outputs are padded
      with zeros beyond this length.
    x: A `Tensor`. Must be one of the following types: `float32`.
      The sequence input to the LSTM, shape (timelen, batch_size, num_inputs).
    cs_prev: A `Tensor`. Must have the same type as `x`.
      Value of the initial cell state.
    h_prev: A `Tensor`. Must have the same type as `x`.
      Initial output of cell (to be used for peephole).
    w: A `Tensor`. Must have the same type as `x`. The weight matrix.
    wci: A `Tensor`. Must have the same type as `x`.
      The weight matrix for input gate peephole connection.
    wcf: A `Tensor`. Must have the same type as `x`.
      The weight matrix for forget gate peephole connection.
    wco: A `Tensor`. Must have the same type as `x`.
      The weight matrix for output gate peephole connection.
    b: A `Tensor`. Must have the same type as `x`. The bias vector.
    i: A `Tensor`. Must have the same type as `x`.
      The input gate over the whole time sequence.
    cs: A `Tensor`. Must have the same type as `x`.
      The cell state before the tanh over the whole time sequence.
    f: A `Tensor`. Must have the same type as `x`.
      The forget gate over the whole time sequence.
    o: A `Tensor`. Must have the same type as `x`.
      The output gate over the whole time sequence.
    ci: A `Tensor`. Must have the same type as `x`.
      The cell input over the whole time sequence.
    co: A `Tensor`. Must have the same type as `x`.
      The cell after the tanh over the whole time sequence.
    h: A `Tensor`. Must have the same type as `x`.
      The output h vector over the whole time sequence.
    cs_grad: A `Tensor`. Must have the same type as `x`.
      The current gradient of cs.
    h_grad: A `Tensor`. Must have the same type as `x`.
      The gradient of h vector.
    use_peephole: A `bool`. Whether to use peephole weights.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (x_grad, cs_prev_grad, h_prev_grad, w_grad, wci_grad, wcf_grad, wco_grad, b_grad).

    x_grad: A `Tensor`. Has the same type as `x`. The gradient of x to be back-propped.
    cs_prev_grad: A `Tensor`. Has the same type as `x`. The gradient of cs_prev to be back-propped.
    h_prev_grad: A `Tensor`. Has the same type as `x`. The gradient of h_prev to be back-propped.
    w_grad: A `Tensor`. Has the same type as `x`. The gradient for w to be back-propped.
    wci_grad: A `Tensor`. Has the same type as `x`. The gradient for wci to be back-propped.
    wcf_grad: A `Tensor`. Has the same type as `x`. The gradient for wcf to be back-propped.
    wco_grad: A `Tensor`. Has the same type as `x`. The gradient for wco to be back-propped.
    b_grad: A `Tensor`. Has the same type as `x`. The gradient for w to be back-propped.
  """
  result = _op_def_lib.apply_op("BlockLSTMGrad", seq_len_max=seq_len_max, x=x,
                                cs_prev=cs_prev, h_prev=h_prev, w=w, wci=wci,
                                wcf=wcf, wco=wco, b=b, i=i, cs=cs, f=f, o=o,
                                ci=ci, co=co, h=h, cs_grad=cs_grad,
                                h_grad=h_grad, use_peephole=use_peephole,
                                name=name)
  return _BlockLSTMGradOutput._make(result)


_ops.RegisterShape("BlockLSTMGrad")(None)

_lstm_block_cell_outputs = ["i", "cs", "f", "o", "ci", "co", "h"]
_LSTMBlockCellOutput = _collections.namedtuple(
    "LSTMBlockCell", _lstm_block_cell_outputs)


def lstm_block_cell(x, cs_prev, h_prev, w, wci, wcf, wco, b, forget_bias=None,
                    cell_clip=None, use_peephole=None, name=None):
  r"""Computes the LSTM cell forward propagation for 1 time step.

  This implementation uses 1 weight matrix and 1 bias vector, and there's an
  optional peephole connection.

  This kernel op implements the following mathematical equations:

  ```python
  xh = [x, h_prev]
  [i, f, ci, o] = xh * w + b
  f = f + forget_bias

  if not use_peephole:
    wci = wcf = wco = 0

  i = sigmoid(cs_prev * wci + i)
  f = sigmoid(cs_prev * wcf + f)
  ci = tanh(ci)

  cs = ci .* i + cs_prev .* f
  cs = clip(cs, cell_clip)

  o = sigmoid(cs * wco + o)
  co = tanh(cs)
  h = co .* o
  ```

  Args:
    x: A `Tensor`. Must be one of the following types: `float32`.
      The input to the LSTM cell, shape (batch_size, num_inputs).
    cs_prev: A `Tensor`. Must have the same type as `x`.
      Value of the cell state at previous time step.
    h_prev: A `Tensor`. Must have the same type as `x`.
      Output of the previous cell at previous time step.
    w: A `Tensor`. Must have the same type as `x`. The weight matrix.
    wci: A `Tensor`. Must have the same type as `x`.
      The weight matrix for input gate peephole connection.
    wcf: A `Tensor`. Must have the same type as `x`.
      The weight matrix for forget gate peephole connection.
    wco: A `Tensor`. Must have the same type as `x`.
      The weight matrix for output gate peephole connection.
    b: A `Tensor`. Must have the same type as `x`. The bias vector.
    forget_bias: An optional `float`. Defaults to `1`. The forget gate bias.
    cell_clip: An optional `float`. Defaults to `3`.
      Value to clip the 'cs' value to.
    use_peephole: An optional `bool`. Defaults to `False`.
      Whether to use peephole weights.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (i, cs, f, o, ci, co, h).

    i: A `Tensor`. Has the same type as `x`. The input gate.
    cs: A `Tensor`. Has the same type as `x`. The cell state before the tanh.
    f: A `Tensor`. Has the same type as `x`. The forget gate.
    o: A `Tensor`. Has the same type as `x`. The output gate.
    ci: A `Tensor`. Has the same type as `x`. The cell input.
    co: A `Tensor`. Has the same type as `x`. The cell after the tanh.
    h: A `Tensor`. Has the same type as `x`. The output h vector.
  """
  result = _op_def_lib.apply_op("LSTMBlockCell", x=x, cs_prev=cs_prev,
                                h_prev=h_prev, w=w, wci=wci, wcf=wcf, wco=wco,
                                b=b, forget_bias=forget_bias,
                                cell_clip=cell_clip,
                                use_peephole=use_peephole, name=name)
  return _LSTMBlockCellOutput._make(result)


_ops.RegisterShape("LSTMBlockCell")(None)

_lstm_block_cell_grad_outputs = ["cs_prev_grad", "dicfo", "wci_grad",
                                "wcf_grad", "wco_grad"]
_LSTMBlockCellGradOutput = _collections.namedtuple(
    "LSTMBlockCellGrad", _lstm_block_cell_grad_outputs)


def lstm_block_cell_grad(x, cs_prev, h_prev, w, wci, wcf, wco, b, i, cs, f, o,
                         ci, co, cs_grad, h_grad, use_peephole, name=None):
  r"""Computes the LSTM cell backward propagation for 1 timestep.

  This implementation is to be used in conjunction of LSTMBlockCell.

  Args:
    x: A `Tensor`. Must be one of the following types: `float32`.
      The input to the LSTM cell, shape (batch_size, num_inputs).
    cs_prev: A `Tensor`. Must have the same type as `x`.
      The previous cell state.
    h_prev: A `Tensor`. Must have the same type as `x`. The previous h state.
    w: A `Tensor`. Must have the same type as `x`. The weight matrix.
    wci: A `Tensor`. Must have the same type as `x`.
      The weight matrix for input gate peephole connection.
    wcf: A `Tensor`. Must have the same type as `x`.
      The weight matrix for forget gate peephole connection.
    wco: A `Tensor`. Must have the same type as `x`.
      The weight matrix for output gate peephole connection.
    b: A `Tensor`. Must have the same type as `x`. The bias vector.
    i: A `Tensor`. Must have the same type as `x`. The input gate.
    cs: A `Tensor`. Must have the same type as `x`.
      The cell state before the tanh.
    f: A `Tensor`. Must have the same type as `x`. The forget gate.
    o: A `Tensor`. Must have the same type as `x`. The output gate.
    ci: A `Tensor`. Must have the same type as `x`. The cell input.
    co: A `Tensor`. Must have the same type as `x`. The cell after the tanh.
    cs_grad: A `Tensor`. Must have the same type as `x`.
      The current gradient of cs.
    h_grad: A `Tensor`. Must have the same type as `x`.
      The gradient of h vector.
    use_peephole: A `bool`. Whether the cell uses peephole connections.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (cs_prev_grad, dicfo, wci_grad, wcf_grad, wco_grad).

    cs_prev_grad: A `Tensor`. Has the same type as `x`. The gradient of cs to be back-propped.
    dicfo: A `Tensor`. Has the same type as `x`. The derivative wrt to [i, cs, f, o].
    wci_grad: A `Tensor`. Has the same type as `x`. The gradient for wci to be back-propped.
    wcf_grad: A `Tensor`. Has the same type as `x`. The gradient for wcf to be back-propped.
    wco_grad: A `Tensor`. Has the same type as `x`. The gradient for wco to be back-propped.
  """
  result = _op_def_lib.apply_op("LSTMBlockCellGrad", x=x, cs_prev=cs_prev,
                                h_prev=h_prev, w=w, wci=wci, wcf=wcf, wco=wco,
                                b=b, i=i, cs=cs, f=f, o=o, ci=ci, co=co,
                                cs_grad=cs_grad, h_grad=h_grad,
                                use_peephole=use_peephole, name=name)
  return _LSTMBlockCellGradOutput._make(result)


_ops.RegisterShape("LSTMBlockCellGrad")(None)
def _InitOpDefLibrary():
  op_list = _op_def_pb2.OpList()
  _text_format.Merge(_InitOpDefLibrary.op_list_ascii, op_list)
  _op_def_registry.register_op_list(op_list)
  op_def_lib = _op_def_library.OpDefLibrary()
  op_def_lib.add_op_list(op_list)
  return op_def_lib


_InitOpDefLibrary.op_list_ascii = """op {
  name: "BlockLSTM"
  input_arg {
    name: "seq_len_max"
    type: DT_INT64
  }
  input_arg {
    name: "x"
    type_attr: "T"
  }
  input_arg {
    name: "cs_prev"
    type_attr: "T"
  }
  input_arg {
    name: "h_prev"
    type_attr: "T"
  }
  input_arg {
    name: "w"
    type_attr: "T"
  }
  input_arg {
    name: "wci"
    type_attr: "T"
  }
  input_arg {
    name: "wcf"
    type_attr: "T"
  }
  input_arg {
    name: "wco"
    type_attr: "T"
  }
  input_arg {
    name: "b"
    type_attr: "T"
  }
  output_arg {
    name: "i"
    type_attr: "T"
  }
  output_arg {
    name: "cs"
    type_attr: "T"
  }
  output_arg {
    name: "f"
    type_attr: "T"
  }
  output_arg {
    name: "o"
    type_attr: "T"
  }
  output_arg {
    name: "ci"
    type_attr: "T"
  }
  output_arg {
    name: "co"
    type_attr: "T"
  }
  output_arg {
    name: "h"
    type_attr: "T"
  }
  attr {
    name: "forget_bias"
    type: "float"
    default_value {
      f: 1
    }
  }
  attr {
    name: "cell_clip"
    type: "float"
    default_value {
      f: 3
    }
  }
  attr {
    name: "use_peephole"
    type: "bool"
    default_value {
      b: false
    }
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
  name: "BlockLSTMGrad"
  input_arg {
    name: "seq_len_max"
    type: DT_INT64
  }
  input_arg {
    name: "x"
    type_attr: "T"
  }
  input_arg {
    name: "cs_prev"
    type_attr: "T"
  }
  input_arg {
    name: "h_prev"
    type_attr: "T"
  }
  input_arg {
    name: "w"
    type_attr: "T"
  }
  input_arg {
    name: "wci"
    type_attr: "T"
  }
  input_arg {
    name: "wcf"
    type_attr: "T"
  }
  input_arg {
    name: "wco"
    type_attr: "T"
  }
  input_arg {
    name: "b"
    type_attr: "T"
  }
  input_arg {
    name: "i"
    type_attr: "T"
  }
  input_arg {
    name: "cs"
    type_attr: "T"
  }
  input_arg {
    name: "f"
    type_attr: "T"
  }
  input_arg {
    name: "o"
    type_attr: "T"
  }
  input_arg {
    name: "ci"
    type_attr: "T"
  }
  input_arg {
    name: "co"
    type_attr: "T"
  }
  input_arg {
    name: "h"
    type_attr: "T"
  }
  input_arg {
    name: "cs_grad"
    type_attr: "T"
  }
  input_arg {
    name: "h_grad"
    type_attr: "T"
  }
  output_arg {
    name: "x_grad"
    type_attr: "T"
  }
  output_arg {
    name: "cs_prev_grad"
    type_attr: "T"
  }
  output_arg {
    name: "h_prev_grad"
    type_attr: "T"
  }
  output_arg {
    name: "w_grad"
    type_attr: "T"
  }
  output_arg {
    name: "wci_grad"
    type_attr: "T"
  }
  output_arg {
    name: "wcf_grad"
    type_attr: "T"
  }
  output_arg {
    name: "wco_grad"
    type_attr: "T"
  }
  output_arg {
    name: "b_grad"
    type_attr: "T"
  }
  attr {
    name: "use_peephole"
    type: "bool"
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
  name: "LSTMBlockCell"
  input_arg {
    name: "x"
    type_attr: "T"
  }
  input_arg {
    name: "cs_prev"
    type_attr: "T"
  }
  input_arg {
    name: "h_prev"
    type_attr: "T"
  }
  input_arg {
    name: "w"
    type_attr: "T"
  }
  input_arg {
    name: "wci"
    type_attr: "T"
  }
  input_arg {
    name: "wcf"
    type_attr: "T"
  }
  input_arg {
    name: "wco"
    type_attr: "T"
  }
  input_arg {
    name: "b"
    type_attr: "T"
  }
  output_arg {
    name: "i"
    type_attr: "T"
  }
  output_arg {
    name: "cs"
    type_attr: "T"
  }
  output_arg {
    name: "f"
    type_attr: "T"
  }
  output_arg {
    name: "o"
    type_attr: "T"
  }
  output_arg {
    name: "ci"
    type_attr: "T"
  }
  output_arg {
    name: "co"
    type_attr: "T"
  }
  output_arg {
    name: "h"
    type_attr: "T"
  }
  attr {
    name: "forget_bias"
    type: "float"
    default_value {
      f: 1
    }
  }
  attr {
    name: "cell_clip"
    type: "float"
    default_value {
      f: 3
    }
  }
  attr {
    name: "use_peephole"
    type: "bool"
    default_value {
      b: false
    }
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
  name: "LSTMBlockCellGrad"
  input_arg {
    name: "x"
    type_attr: "T"
  }
  input_arg {
    name: "cs_prev"
    type_attr: "T"
  }
  input_arg {
    name: "h_prev"
    type_attr: "T"
  }
  input_arg {
    name: "w"
    type_attr: "T"
  }
  input_arg {
    name: "wci"
    type_attr: "T"
  }
  input_arg {
    name: "wcf"
    type_attr: "T"
  }
  input_arg {
    name: "wco"
    type_attr: "T"
  }
  input_arg {
    name: "b"
    type_attr: "T"
  }
  input_arg {
    name: "i"
    type_attr: "T"
  }
  input_arg {
    name: "cs"
    type_attr: "T"
  }
  input_arg {
    name: "f"
    type_attr: "T"
  }
  input_arg {
    name: "o"
    type_attr: "T"
  }
  input_arg {
    name: "ci"
    type_attr: "T"
  }
  input_arg {
    name: "co"
    type_attr: "T"
  }
  input_arg {
    name: "cs_grad"
    type_attr: "T"
  }
  input_arg {
    name: "h_grad"
    type_attr: "T"
  }
  output_arg {
    name: "cs_prev_grad"
    type_attr: "T"
  }
  output_arg {
    name: "dicfo"
    type_attr: "T"
  }
  output_arg {
    name: "wci_grad"
    type_attr: "T"
  }
  output_arg {
    name: "wcf_grad"
    type_attr: "T"
  }
  output_arg {
    name: "wco_grad"
    type_attr: "T"
  }
  attr {
    name: "use_peephole"
    type: "bool"
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
