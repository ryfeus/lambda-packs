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

_cudnn_rnn_outputs = ["output", "output_h", "output_c", "reserve_space"]
_CudnnRNNOutput = _collections.namedtuple(
    "CudnnRNN", _cudnn_rnn_outputs)


def cudnn_rnn(input, input_h, input_c, params, rnn_mode=None, input_mode=None,
              direction=None, dropout=None, seed=None, seed2=None,
              is_training=None, name=None):
  r"""Computes the RNN from the input and initial states, with respect to the params

  buffer.

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`.
      a 3-D tensor with the shape of [seq_length, batch_size, input_size].
    input_h: A `Tensor`. Must have the same type as `input`.
      a 3-D tensor with the shape of [num_layer * dir, batch_size,
      num_units].
    input_c: A `Tensor`. Must have the same type as `input`.
      For LSTM, a 3-D tensor with the shape of
      [num_layer * dir, batch, num_units]. For other models, it is ignored.
    params: A `Tensor`. Must have the same type as `input`.
      a 1-D tensor that contains the weights and biases in an opaque layout.
      The size must be created through CudnnRNNParamsSize, and initialized
      separately. Note that they might not be compatible across different
      generations. So it is a good idea to save and restore
    rnn_mode: An optional `string` from: `"rnn_relu", "rnn_tanh", "lstm", "gru"`. Defaults to `"lstm"`.
      Indicates the type of the RNN model.
    input_mode: An optional `string` from: `"linear_input", "skip_input", "auto_select"`. Defaults to `"auto_select"`.
      Indicate whether there is a linear projection between the input and
      The actual computation before the first layer. 'skip_input' is only allowed
      when input_size == num_units; 'auto_select' implies 'skip_input' when
      input_size == num_units; otherwise, it implies 'linear_input'.
    direction: An optional `string` from: `"unidirectional", "bidirectional"`. Defaults to `"unidirectional"`.
      Indicates whether a bidirectional model will be used.
      dir = (direction == bidirectional) ? 2 : 1
    dropout: An optional `float`. Defaults to `0`.
      dropout probability. When set to 0., dropout is disabled.
    seed: An optional `int`. Defaults to `0`.
      the 1st part of a seed to initialize dropout.
    seed2: An optional `int`. Defaults to `0`.
      the 2nd part of a seed to initialize dropout.
    is_training: An optional `bool`. Defaults to `True`.
      Indicates whether this operation is used for inferenece or
      training.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output, output_h, output_c, reserve_space).

    output: A `Tensor`. Has the same type as `input`. a 3-D tensor with the shape of [seq_length, batch_size,
      dir * num_units].
    output_h: A `Tensor`. Has the same type as `input`. the same shape has input_h.
    output_c: A `Tensor`. Has the same type as `input`. the same shape as input_c for LSTM. An empty tensor for other models.
    reserve_space: A `Tensor`. Has the same type as `input`. an opaque tensor that can be used in backprop calculation. It
      is only produced if is_training is false.
  """
  result = _op_def_lib.apply_op("CudnnRNN", input=input, input_h=input_h,
                                input_c=input_c, params=params,
                                rnn_mode=rnn_mode, input_mode=input_mode,
                                direction=direction, dropout=dropout,
                                seed=seed, seed2=seed2,
                                is_training=is_training, name=name)
  return _CudnnRNNOutput._make(result)


_ops.RegisterShape("CudnnRNN")(None)

_cudnn_rnn_backprop_outputs = ["input_backprop", "input_h_backprop",
                              "input_c_backprop", "params_backprop"]
_CudnnRNNBackpropOutput = _collections.namedtuple(
    "CudnnRNNBackprop", _cudnn_rnn_backprop_outputs)


def cudnn_rnn_backprop(input, input_h, input_c, params, output, output_h,
                       output_c, output_backprop, output_h_backprop,
                       output_c_backprop, reserve_space, rnn_mode=None,
                       input_mode=None, direction=None, dropout=None,
                       seed=None, seed2=None, name=None):
  r"""Compute the backprop of both data and weights in a RNN.

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`.
      a 3-D tensor with the shape of [seq_length, batch_size, input_size].
    input_h: A `Tensor`. Must have the same type as `input`.
      a 3-D tensor with the shape of [num_layer * dir, batch_size,
      num_units].
    input_c: A `Tensor`. Must have the same type as `input`.
      For LSTM, a 3-D tensor with the shape of
      [num_layer * dir, batch, num_units]. For other models, it is ignored.
    params: A `Tensor`. Must have the same type as `input`.
      a 1-D tensor that contains the weights and biases in an opaque layout.
      The size must be created through CudnnRNNParamsSize, and initialized
      separately. Note that they might not be compatible across different
      generations. So it is a good idea to save and restore
    output: A `Tensor`. Must have the same type as `input`.
      a 3-D tensor with the shape of [seq_length, batch_size,
      dir * num_units].
    output_h: A `Tensor`. Must have the same type as `input`.
      the same shape has input_h.
    output_c: A `Tensor`. Must have the same type as `input`.
      the same shape as input_c for LSTM. An empty tensor for other models.
    output_backprop: A `Tensor`. Must have the same type as `input`.
      A 3-D tensor with the same shape as output in the forward pass.
    output_h_backprop: A `Tensor`. Must have the same type as `input`.
      A 3-D tensor with the same shape as output_h in the forward
      pass.
    output_c_backprop: A `Tensor`. Must have the same type as `input`.
      A 3-D tensor with the same shape as output_c in the forward
      pass.
    reserve_space: A `Tensor`. Must have the same type as `input`.
      The same reserve_space produced in for forward operation.
    rnn_mode: An optional `string` from: `"rnn_relu", "rnn_tanh", "lstm", "gru"`. Defaults to `"lstm"`.
      Indicates the type of the RNN model.
    input_mode: An optional `string` from: `"linear_input", "skip_input", "auto_select"`. Defaults to `"auto_select"`.
      Indicate whether there is a linear projection between the input and
      The actual computation before the first layer. 'skip_input' is only allowed
      when input_size == num_units; 'auto_select' implies 'skip_input' when
      input_size == num_units; otherwise, it implies 'linear_input'.
    direction: An optional `string` from: `"unidirectional", "bidirectional"`. Defaults to `"unidirectional"`.
      Indicates whether a bidirectional model will be used.
      dir = (direction == bidirectional) ? 2 : 1
    dropout: An optional `float`. Defaults to `0`.
      dropout probability. When set to 0., dropout is disabled.
    seed: An optional `int`. Defaults to `0`.
      the 1st part of a seed to initialize dropout.
    seed2: An optional `int`. Defaults to `0`.
      the 2nd part of a seed to initialize dropout.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (input_backprop, input_h_backprop, input_c_backprop, params_backprop).

    input_backprop: A `Tensor`. Has the same type as `input`. The backprop to input in the forward pass. Has the same shape
      as input.
    input_h_backprop: A `Tensor`. Has the same type as `input`. The backprop to input_h in the forward pass. Has the same
      shape as input_h.
    input_c_backprop: A `Tensor`. Has the same type as `input`. The backprop to input_c in the forward pass. Has the same
      shape as input_c.
    params_backprop: A `Tensor`. Has the same type as `input`. The backprop to the params buffer in the forward pass. Has the
      same shape as params.
  """
  result = _op_def_lib.apply_op("CudnnRNNBackprop", input=input,
                                input_h=input_h, input_c=input_c,
                                params=params, output=output,
                                output_h=output_h, output_c=output_c,
                                output_backprop=output_backprop,
                                output_h_backprop=output_h_backprop,
                                output_c_backprop=output_c_backprop,
                                reserve_space=reserve_space,
                                rnn_mode=rnn_mode, input_mode=input_mode,
                                direction=direction, dropout=dropout,
                                seed=seed, seed2=seed2, name=name)
  return _CudnnRNNBackpropOutput._make(result)


_ops.RegisterShape("CudnnRNNBackprop")(None)

def cudnn_rnn_canonical_to_params(num_layers, num_units, input_size, weights,
                                  biases, rnn_mode=None, input_mode=None,
                                  direction=None, dropout=None, seed=None,
                                  seed2=None, name=None):
  r"""Writes a set of weights into the opaque params buffer so they can be used in

  upcoming training or inferences.

  Args:
    num_layers: A `Tensor` of type `int32`.
      Specifies the number of layers in the RNN model.
    num_units: A `Tensor` of type `int32`.
      Specifies the size of the hidden state.
    input_size: A `Tensor` of type `int32`.
      Specifies the size of the input state.
    weights: A list of at least 1 `Tensor` objects with the same type in: `float32`.
      the canonical form of weights that can be used for saving
      and restoration. They are more likely to be compatible across different
      generations.
    biases: A list with the same length as `weights` of `Tensor` objects with the same type as `weights`.
      the canonical form of biases that can be used for saving
          and restoration. They are more likely to be compatible across different
          generations.

      Note that the params buffer may not be compatible across different GPUs. So any
      save and restoration should be converted to and from the canonical weights and
      biases.
    rnn_mode: An optional `string` from: `"rnn_relu", "rnn_tanh", "lstm", "gru"`. Defaults to `"lstm"`.
      Indicates the type of the RNN model.
    input_mode: An optional `string` from: `"linear_input", "skip_input", "auto_select"`. Defaults to `"auto_select"`.
      Indicate whether there is a linear projection between the input and
      The actual computation before the first layer. 'skip_input' is only allowed
      when input_size == num_units; 'auto_select' implies 'skip_input' when
      input_size == num_units; otherwise, it implies 'linear_input'.
    direction: An optional `string` from: `"unidirectional", "bidirectional"`. Defaults to `"unidirectional"`.
      Indicates whether a bidirectional model will be used.
      dir = (direction == bidirectional) ? 2 : 1
    dropout: An optional `float`. Defaults to `0`.
      dropout probability. When set to 0., dropout is disabled.
    seed: An optional `int`. Defaults to `0`.
      the 1st part of a seed to initialize dropout.
    seed2: An optional `int`. Defaults to `0`.
      the 2nd part of a seed to initialize dropout.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `weights`.
  """
  result = _op_def_lib.apply_op("CudnnRNNCanonicalToParams",
                                num_layers=num_layers, num_units=num_units,
                                input_size=input_size, weights=weights,
                                biases=biases, rnn_mode=rnn_mode,
                                input_mode=input_mode, direction=direction,
                                dropout=dropout, seed=seed, seed2=seed2,
                                name=name)
  return result


_ops.RegisterShape("CudnnRNNCanonicalToParams")(None)

def cudnn_rnn_params_size(num_layers, num_units, input_size, T, S,
                          rnn_mode=None, input_mode=None, direction=None,
                          dropout=None, seed=None, seed2=None, name=None):
  r"""Return the params size that can be used by the Cudnn RNN model. Subsequent

  weight allocation and initialization should use this size.

  Args:
    num_layers: A `Tensor` of type `int32`.
      Specifies the number of layers in the RNN model.
    num_units: A `Tensor` of type `int32`.
      Specifies the size of the hidden state.
    input_size: A `Tensor` of type `int32`.
      Specifies the size of the input state.
    T: A `tf.DType` from: `tf.float32`.
    S: A `tf.DType` from: `tf.int32, tf.int64`.
    rnn_mode: An optional `string` from: `"rnn_relu", "rnn_tanh", "lstm", "gru"`. Defaults to `"lstm"`.
      Indicates the type of the RNN model.
    input_mode: An optional `string` from: `"linear_input", "skip_input", "auto_select"`. Defaults to `"auto_select"`.
      Indicate whether there is a linear projection between the input and
      The actual computation before the first layer. 'skip_input' is only allowed
      when input_size == num_units; 'auto_select' implies 'skip_input' when
      input_size == num_units; otherwise, it implies 'linear_input'.
    direction: An optional `string` from: `"unidirectional", "bidirectional"`. Defaults to `"unidirectional"`.
      Indicates whether a bidirectional model will be used.
      dir = (direction == bidirectional) ? 2 : 1
    dropout: An optional `float`. Defaults to `0`.
      dropout probability. When set to 0., dropout is disabled.
    seed: An optional `int`. Defaults to `0`.
      the 1st part of a seed to initialize dropout.
    seed2: An optional `int`. Defaults to `0`.
      the 2nd part of a seed to initialize dropout.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `S`.
    The size of the params buffer that should be allocated and
        initialized for this RNN model. Note that this params buffer may not be
        compatible across GPUs. Please use CudnnRNNParamsWeights and
        CudnnRNNParamsBiases to save and restore them in a way that is compatible
        across different runs.

    Note that the params buffer may not be compatible across different GPUs. So any
    save and restoration should be converted to and from the canonical weights and
    biases.
  """
  result = _op_def_lib.apply_op("CudnnRNNParamsSize", num_layers=num_layers,
                                num_units=num_units, input_size=input_size,
                                T=T, S=S, rnn_mode=rnn_mode,
                                input_mode=input_mode, direction=direction,
                                dropout=dropout, seed=seed, seed2=seed2,
                                name=name)
  return result


_ops.RegisterShape("CudnnRNNParamsSize")(None)

_cudnn_rnn_params_to_canonical_outputs = ["weights", "biases"]
_CudnnRNNParamsToCanonicalOutput = _collections.namedtuple(
    "CudnnRNNParamsToCanonical", _cudnn_rnn_params_to_canonical_outputs)


def cudnn_rnn_params_to_canonical(num_layers, num_units, input_size, params,
                                  num_params, rnn_mode=None, input_mode=None,
                                  direction=None, dropout=None, seed=None,
                                  seed2=None, name=None):
  r"""Retrieves a set of weights from the opaque params buffer that can be saved and

  restored in a way compatible with future runs.

  Args:
    num_layers: A `Tensor` of type `int32`.
      Specifies the number of layers in the RNN model.
    num_units: A `Tensor` of type `int32`.
      Specifies the size of the hidden state.
    input_size: A `Tensor` of type `int32`.
      Specifies the size of the input state.

      Note that the params buffer may not be compatible across different GPUs. So any
      save and restoration should be converted to and from the canonical weights and
      biases.
    params: A `Tensor`. Must be one of the following types: `float32`.
    num_params: An `int` that is `>= 1`.
      number of parameter sets for all layers.
      Each layer may contain multiple parameter sets, with each set consisting of
      a weight matrix and a bias vector.
    rnn_mode: An optional `string` from: `"rnn_relu", "rnn_tanh", "lstm", "gru"`. Defaults to `"lstm"`.
      Indicates the type of the RNN model.
    input_mode: An optional `string` from: `"linear_input", "skip_input", "auto_select"`. Defaults to `"auto_select"`.
      Indicate whether there is a linear projection between the input and
      The actual computation before the first layer. 'skip_input' is only allowed
      when input_size == num_units; 'auto_select' implies 'skip_input' when
      input_size == num_units; otherwise, it implies 'linear_input'.
    direction: An optional `string` from: `"unidirectional", "bidirectional"`. Defaults to `"unidirectional"`.
      Indicates whether a bidirectional model will be used.
      dir = (direction == bidirectional) ? 2 : 1
    dropout: An optional `float`. Defaults to `0`.
      dropout probability. When set to 0., dropout is disabled.
    seed: An optional `int`. Defaults to `0`.
      the 1st part of a seed to initialize dropout.
    seed2: An optional `int`. Defaults to `0`.
      the 2nd part of a seed to initialize dropout.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (weights, biases).

    weights: A list of `num_params` `Tensor` objects with the same type as `params`. the canonical form of weights that can be used for saving
      and restoration. They are more likely to be compatible across different
      generations.
    biases: A list of `num_params` `Tensor` objects with the same type as `params`. the canonical form of biases that can be used for saving
      and restoration. They are more likely to be compatible across different
      generations.
  """
  result = _op_def_lib.apply_op("CudnnRNNParamsToCanonical",
                                num_layers=num_layers, num_units=num_units,
                                input_size=input_size, params=params,
                                num_params=num_params, rnn_mode=rnn_mode,
                                input_mode=input_mode, direction=direction,
                                dropout=dropout, seed=seed, seed2=seed2,
                                name=name)
  return _CudnnRNNParamsToCanonicalOutput._make(result)


_ops.RegisterShape("CudnnRNNParamsToCanonical")(None)
def _InitOpDefLibrary():
  op_list = _op_def_pb2.OpList()
  _text_format.Merge(_InitOpDefLibrary.op_list_ascii, op_list)
  _op_def_registry.register_op_list(op_list)
  op_def_lib = _op_def_library.OpDefLibrary()
  op_def_lib.add_op_list(op_list)
  return op_def_lib


_InitOpDefLibrary.op_list_ascii = """op {
  name: "CudnnRNN"
  input_arg {
    name: "input"
    type_attr: "T"
  }
  input_arg {
    name: "input_h"
    type_attr: "T"
  }
  input_arg {
    name: "input_c"
    type_attr: "T"
  }
  input_arg {
    name: "params"
    type_attr: "T"
  }
  output_arg {
    name: "output"
    type_attr: "T"
  }
  output_arg {
    name: "output_h"
    type_attr: "T"
  }
  output_arg {
    name: "output_c"
    type_attr: "T"
  }
  output_arg {
    name: "reserve_space"
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
  attr {
    name: "rnn_mode"
    type: "string"
    default_value {
      s: "lstm"
    }
    allowed_values {
      list {
        s: "rnn_relu"
        s: "rnn_tanh"
        s: "lstm"
        s: "gru"
      }
    }
  }
  attr {
    name: "input_mode"
    type: "string"
    default_value {
      s: "auto_select"
    }
    allowed_values {
      list {
        s: "linear_input"
        s: "skip_input"
        s: "auto_select"
      }
    }
  }
  attr {
    name: "direction"
    type: "string"
    default_value {
      s: "unidirectional"
    }
    allowed_values {
      list {
        s: "unidirectional"
        s: "bidirectional"
      }
    }
  }
  attr {
    name: "dropout"
    type: "float"
    default_value {
      f: 0
    }
  }
  attr {
    name: "seed"
    type: "int"
    default_value {
      i: 0
    }
  }
  attr {
    name: "seed2"
    type: "int"
    default_value {
      i: 0
    }
  }
  attr {
    name: "is_training"
    type: "bool"
    default_value {
      b: true
    }
  }
  is_stateful: true
}
op {
  name: "CudnnRNNBackprop"
  input_arg {
    name: "input"
    type_attr: "T"
  }
  input_arg {
    name: "input_h"
    type_attr: "T"
  }
  input_arg {
    name: "input_c"
    type_attr: "T"
  }
  input_arg {
    name: "params"
    type_attr: "T"
  }
  input_arg {
    name: "output"
    type_attr: "T"
  }
  input_arg {
    name: "output_h"
    type_attr: "T"
  }
  input_arg {
    name: "output_c"
    type_attr: "T"
  }
  input_arg {
    name: "output_backprop"
    type_attr: "T"
  }
  input_arg {
    name: "output_h_backprop"
    type_attr: "T"
  }
  input_arg {
    name: "output_c_backprop"
    type_attr: "T"
  }
  input_arg {
    name: "reserve_space"
    type_attr: "T"
  }
  output_arg {
    name: "input_backprop"
    type_attr: "T"
  }
  output_arg {
    name: "input_h_backprop"
    type_attr: "T"
  }
  output_arg {
    name: "input_c_backprop"
    type_attr: "T"
  }
  output_arg {
    name: "params_backprop"
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
  attr {
    name: "rnn_mode"
    type: "string"
    default_value {
      s: "lstm"
    }
    allowed_values {
      list {
        s: "rnn_relu"
        s: "rnn_tanh"
        s: "lstm"
        s: "gru"
      }
    }
  }
  attr {
    name: "input_mode"
    type: "string"
    default_value {
      s: "auto_select"
    }
    allowed_values {
      list {
        s: "linear_input"
        s: "skip_input"
        s: "auto_select"
      }
    }
  }
  attr {
    name: "direction"
    type: "string"
    default_value {
      s: "unidirectional"
    }
    allowed_values {
      list {
        s: "unidirectional"
        s: "bidirectional"
      }
    }
  }
  attr {
    name: "dropout"
    type: "float"
    default_value {
      f: 0
    }
  }
  attr {
    name: "seed"
    type: "int"
    default_value {
      i: 0
    }
  }
  attr {
    name: "seed2"
    type: "int"
    default_value {
      i: 0
    }
  }
  is_stateful: true
}
op {
  name: "CudnnRNNCanonicalToParams"
  input_arg {
    name: "num_layers"
    type: DT_INT32
  }
  input_arg {
    name: "num_units"
    type: DT_INT32
  }
  input_arg {
    name: "input_size"
    type: DT_INT32
  }
  input_arg {
    name: "weights"
    type_attr: "T"
    number_attr: "num_params"
  }
  input_arg {
    name: "biases"
    type_attr: "T"
    number_attr: "num_params"
  }
  output_arg {
    name: "params"
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
  attr {
    name: "num_params"
    type: "int"
    has_minimum: true
    minimum: 1
  }
  attr {
    name: "rnn_mode"
    type: "string"
    default_value {
      s: "lstm"
    }
    allowed_values {
      list {
        s: "rnn_relu"
        s: "rnn_tanh"
        s: "lstm"
        s: "gru"
      }
    }
  }
  attr {
    name: "input_mode"
    type: "string"
    default_value {
      s: "auto_select"
    }
    allowed_values {
      list {
        s: "linear_input"
        s: "skip_input"
        s: "auto_select"
      }
    }
  }
  attr {
    name: "direction"
    type: "string"
    default_value {
      s: "unidirectional"
    }
    allowed_values {
      list {
        s: "unidirectional"
        s: "bidirectional"
      }
    }
  }
  attr {
    name: "dropout"
    type: "float"
    default_value {
      f: 0
    }
  }
  attr {
    name: "seed"
    type: "int"
    default_value {
      i: 0
    }
  }
  attr {
    name: "seed2"
    type: "int"
    default_value {
      i: 0
    }
  }
}
op {
  name: "CudnnRNNParamsSize"
  input_arg {
    name: "num_layers"
    type: DT_INT32
  }
  input_arg {
    name: "num_units"
    type: DT_INT32
  }
  input_arg {
    name: "input_size"
    type: DT_INT32
  }
  output_arg {
    name: "params_size"
    type_attr: "S"
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
  attr {
    name: "S"
    type: "type"
    allowed_values {
      list {
        type: DT_INT32
        type: DT_INT64
      }
    }
  }
  attr {
    name: "rnn_mode"
    type: "string"
    default_value {
      s: "lstm"
    }
    allowed_values {
      list {
        s: "rnn_relu"
        s: "rnn_tanh"
        s: "lstm"
        s: "gru"
      }
    }
  }
  attr {
    name: "input_mode"
    type: "string"
    default_value {
      s: "auto_select"
    }
    allowed_values {
      list {
        s: "linear_input"
        s: "skip_input"
        s: "auto_select"
      }
    }
  }
  attr {
    name: "direction"
    type: "string"
    default_value {
      s: "unidirectional"
    }
    allowed_values {
      list {
        s: "unidirectional"
        s: "bidirectional"
      }
    }
  }
  attr {
    name: "dropout"
    type: "float"
    default_value {
      f: 0
    }
  }
  attr {
    name: "seed"
    type: "int"
    default_value {
      i: 0
    }
  }
  attr {
    name: "seed2"
    type: "int"
    default_value {
      i: 0
    }
  }
}
op {
  name: "CudnnRNNParamsToCanonical"
  input_arg {
    name: "num_layers"
    type: DT_INT32
  }
  input_arg {
    name: "num_units"
    type: DT_INT32
  }
  input_arg {
    name: "input_size"
    type: DT_INT32
  }
  input_arg {
    name: "params"
    type_attr: "T"
  }
  output_arg {
    name: "weights"
    type_attr: "T"
    number_attr: "num_params"
  }
  output_arg {
    name: "biases"
    type_attr: "T"
    number_attr: "num_params"
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
  attr {
    name: "num_params"
    type: "int"
    has_minimum: true
    minimum: 1
  }
  attr {
    name: "rnn_mode"
    type: "string"
    default_value {
      s: "lstm"
    }
    allowed_values {
      list {
        s: "rnn_relu"
        s: "rnn_tanh"
        s: "lstm"
        s: "gru"
      }
    }
  }
  attr {
    name: "input_mode"
    type: "string"
    default_value {
      s: "auto_select"
    }
    allowed_values {
      list {
        s: "linear_input"
        s: "skip_input"
        s: "auto_select"
      }
    }
  }
  attr {
    name: "direction"
    type: "string"
    default_value {
      s: "unidirectional"
    }
    allowed_values {
      list {
        s: "unidirectional"
        s: "bidirectional"
      }
    }
  }
  attr {
    name: "dropout"
    type: "float"
    default_value {
      f: 0
    }
  }
  attr {
    name: "seed"
    type: "int"
    default_value {
      i: 0
    }
  }
  attr {
    name: "seed2"
    type: "int"
    default_value {
      i: 0
    }
  }
}
"""


_op_def_lib = _InitOpDefLibrary()
