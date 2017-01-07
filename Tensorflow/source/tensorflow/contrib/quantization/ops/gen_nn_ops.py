"""Python wrappers around Brain.

This file is MACHINE GENERATED! Do not edit.
"""

import collections

from google.protobuf import text_format

from tensorflow.core.framework import op_def_pb2
from tensorflow.python.framework import op_def_registry
from tensorflow.python.framework import ops
from tensorflow.python.ops import op_def_library


_quantized_avg_pool_outputs = ["output", "min_output", "max_output"]


_QuantizedAvgPoolOutput = collections.namedtuple("QuantizedAvgPool",
                                                 _quantized_avg_pool_outputs)


def quantized_avg_pool(input, min_input, max_input, ksize, strides, padding,
                       name=None):
  r"""Produces the average pool of the input tensor for quantized types.

  Args:
    input: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint16`, `quint16`, `qint32`.
      4-D with shape `[batch, height, width, channels]`.
    min_input: A `Tensor` of type `float32`.
      The float value that the lowest quantized input value represents.
    max_input: A `Tensor` of type `float32`.
      The float value that the highest quantized input value represents.
    ksize: A list of `ints`.
      The size of the window for each dimension of the input tensor.
      The length must be 4 to match the number of dimensions of the input.
    strides: A list of `ints`.
      The stride of the sliding window for each dimension of the input
      tensor.  The length must be 4 to match the number of dimensions of the input.
    padding: A `string` from: `"SAME", "VALID"`.
      The type of padding algorithm to use.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output, min_output, max_output).
    output: A `Tensor`. Has the same type as `input`.
    min_output: A `Tensor` of type `float32`. The float value that the lowest quantized output value represents.
    max_output: A `Tensor` of type `float32`. The float value that the highest quantized output value represents.
  """
  result = _op_def_lib.apply_op("QuantizedAvgPool", input=input,
                                min_input=min_input, max_input=max_input,
                                ksize=ksize, strides=strides, padding=padding,
                                name=name)
  return _QuantizedAvgPoolOutput._make(result)


ops.RegisterShape("QuantizedAvgPool")(None)
_quantized_batch_norm_with_global_normalization_outputs = ["result",
                                                          "result_min",
                                                          "result_max"]


_QuantizedBatchNormWithGlobalNormalizationOutput = collections.namedtuple("QuantizedBatchNormWithGlobalNormalization",
                                                                          _quantized_batch_norm_with_global_normalization_outputs)


def quantized_batch_norm_with_global_normalization(t, t_min, t_max, m, m_min,
                                                   m_max, v, v_min, v_max,
                                                   beta, beta_min, beta_max,
                                                   gamma, gamma_min,
                                                   gamma_max, out_type,
                                                   variance_epsilon,
                                                   scale_after_normalization,
                                                   name=None):
  r"""Quantized Batch normalization.

  This op is deprecated and will be removed in the future. Prefer
  `tf.nn.batch_normalization`.

  Args:
    t: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint16`, `quint16`, `qint32`.
      A 4D input Tensor.
    t_min: A `Tensor` of type `float32`.
      The value represented by the lowest quantized input.
    t_max: A `Tensor` of type `float32`.
      The value represented by the highest quantized input.
    m: A `Tensor`. Must have the same type as `t`.
      A 1D mean Tensor with size matching the last dimension of t.
      This is the first output from tf.nn.moments,
      or a saved moving average thereof.
    m_min: A `Tensor` of type `float32`.
      The value represented by the lowest quantized mean.
    m_max: A `Tensor` of type `float32`.
      The value represented by the highest quantized mean.
    v: A `Tensor`. Must have the same type as `t`.
      A 1D variance Tensor with size matching the last dimension of t.
      This is the second output from tf.nn.moments,
      or a saved moving average thereof.
    v_min: A `Tensor` of type `float32`.
      The value represented by the lowest quantized variance.
    v_max: A `Tensor` of type `float32`.
      The value represented by the highest quantized variance.
    beta: A `Tensor`. Must have the same type as `t`.
      A 1D beta Tensor with size matching the last dimension of t.
      An offset to be added to the normalized tensor.
    beta_min: A `Tensor` of type `float32`.
      The value represented by the lowest quantized offset.
    beta_max: A `Tensor` of type `float32`.
      The value represented by the highest quantized offset.
    gamma: A `Tensor`. Must have the same type as `t`.
      A 1D gamma Tensor with size matching the last dimension of t.
      If "scale_after_normalization" is true, this tensor will be multiplied
      with the normalized tensor.
    gamma_min: A `Tensor` of type `float32`.
      The value represented by the lowest quantized gamma.
    gamma_max: A `Tensor` of type `float32`.
      The value represented by the highest quantized gamma.
    out_type: A `tf.DType` from: `tf.qint8, tf.quint8, tf.qint16, tf.quint16, tf.qint32`.
    variance_epsilon: A `float`. A small float number to avoid dividing by 0.
    scale_after_normalization: A `bool`.
      A bool indicating whether the resulted tensor
      needs to be multiplied with gamma.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (result, result_min, result_max).
    result: A `Tensor` of type `out_type`.
    result_min: A `Tensor` of type `float32`.
    result_max: A `Tensor` of type `float32`.
  """
  result = _op_def_lib.apply_op("QuantizedBatchNormWithGlobalNormalization",
                                t=t, t_min=t_min, t_max=t_max, m=m,
                                m_min=m_min, m_max=m_max, v=v, v_min=v_min,
                                v_max=v_max, beta=beta, beta_min=beta_min,
                                beta_max=beta_max, gamma=gamma,
                                gamma_min=gamma_min, gamma_max=gamma_max,
                                out_type=out_type,
                                variance_epsilon=variance_epsilon,
                                scale_after_normalization=scale_after_normalization,
                                name=name)
  return _QuantizedBatchNormWithGlobalNormalizationOutput._make(result)


ops.RegisterShape("QuantizedBatchNormWithGlobalNormalization")(None)
_quantized_bias_add_outputs = ["output", "min_out", "max_out"]


_QuantizedBiasAddOutput = collections.namedtuple("QuantizedBiasAdd",
                                                 _quantized_bias_add_outputs)


def quantized_bias_add(input, bias, min_input, max_input, min_bias, max_bias,
                       out_type, name=None):
  r"""Adds Tensor 'bias' to Tensor 'input' for Quantized types.

  Broadcasts the values of bias on dimensions 0..N-2 of 'input'.

  Args:
    input: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint16`, `quint16`, `qint32`.
    bias: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint16`, `quint16`, `qint32`.
      A 1D bias Tensor with size matching the last dimension of 'input'.
    min_input: A `Tensor` of type `float32`.
      The float value that the lowest quantized input value represents.
    max_input: A `Tensor` of type `float32`.
      The float value that the highest quantized input value represents.
    min_bias: A `Tensor` of type `float32`.
      The float value that the lowest quantized bias value represents.
    max_bias: A `Tensor` of type `float32`.
      The float value that the highest quantized bias value represents.
    out_type: A `tf.DType` from: `tf.qint8, tf.quint8, tf.qint16, tf.quint16, tf.qint32`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output, min_out, max_out).
    output: A `Tensor` of type `out_type`.
    min_out: A `Tensor` of type `float32`. The float value that the lowest quantized output value represents.
    max_out: A `Tensor` of type `float32`. The float value that the highest quantized output value represents.
  """
  result = _op_def_lib.apply_op("QuantizedBiasAdd", input=input, bias=bias,
                                min_input=min_input, max_input=max_input,
                                min_bias=min_bias, max_bias=max_bias,
                                out_type=out_type, name=name)
  return _QuantizedBiasAddOutput._make(result)


ops.RegisterShape("QuantizedBiasAdd")(None)
_quantized_conv2d_outputs = ["output", "min_output", "max_output"]


_QuantizedConv2DOutput = collections.namedtuple("QuantizedConv2D",
                                                _quantized_conv2d_outputs)


def quantized_conv2d(input, filter, min_input, max_input, min_filter,
                     max_filter, strides, padding, out_type=None, name=None):
  r"""Computes a 2D convolution given quantized 4D input and filter tensors.

  The inputs are quantized tensors where the lowest value represents the real
  number of the associated minimum, and the highest represents the maximum.
  This means that you can only interpret the quantized output in the same way, by
  taking the returned minimum and maximum values into account.

  Args:
    input: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint16`, `quint16`, `qint32`.
    filter: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint16`, `quint16`, `qint32`.
      filter's input_depth dimension must match input's depth dimensions.
    min_input: A `Tensor` of type `float32`.
      The float value that the lowest quantized input value represents.
    max_input: A `Tensor` of type `float32`.
      The float value that the highest quantized input value represents.
    min_filter: A `Tensor` of type `float32`.
      The float value that the lowest quantized filter value represents.
    max_filter: A `Tensor` of type `float32`.
      The float value that the highest quantized filter value represents.
    strides: A list of `ints`.
      The stride of the sliding window for each dimension of the input
      tensor.
    padding: A `string` from: `"SAME", "VALID"`.
      The type of padding algorithm to use.
    out_type: An optional `tf.DType` from: `tf.qint8, tf.quint8, tf.qint16, tf.quint16, tf.qint32`. Defaults to `tf.qint32`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output, min_output, max_output).
    output: A `Tensor` of type `out_type`.
    min_output: A `Tensor` of type `float32`. The float value that the lowest quantized output value represents.
    max_output: A `Tensor` of type `float32`. The float value that the highest quantized output value represents.
  """
  result = _op_def_lib.apply_op("QuantizedConv2D", input=input, filter=filter,
                                min_input=min_input, max_input=max_input,
                                min_filter=min_filter, max_filter=max_filter,
                                strides=strides, padding=padding,
                                out_type=out_type, name=name)
  return _QuantizedConv2DOutput._make(result)


ops.RegisterShape("QuantizedConv2D")(None)
_quantized_max_pool_outputs = ["output", "min_output", "max_output"]


_QuantizedMaxPoolOutput = collections.namedtuple("QuantizedMaxPool",
                                                 _quantized_max_pool_outputs)


def quantized_max_pool(input, min_input, max_input, ksize, strides, padding,
                       name=None):
  r"""Produces the max pool of the input tensor for quantized types.

  Args:
    input: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint16`, `quint16`, `qint32`.
      The 4D (batch x rows x cols x depth) Tensor to MaxReduce over.
    min_input: A `Tensor` of type `float32`.
      The float value that the lowest quantized input value represents.
    max_input: A `Tensor` of type `float32`.
      The float value that the highest quantized input value represents.
    ksize: A list of `ints`.
      The size of the window for each dimension of the input tensor.
      The length must be 4 to match the number of dimensions of the input.
    strides: A list of `ints`.
      The stride of the sliding window for each dimension of the input
      tensor. The length must be 4 to match the number of dimensions of the input.
    padding: A `string` from: `"SAME", "VALID"`.
      The type of padding algorithm to use.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output, min_output, max_output).
    output: A `Tensor`. Has the same type as `input`.
    min_output: A `Tensor` of type `float32`. The float value that the lowest quantized output value represents.
    max_output: A `Tensor` of type `float32`. The float value that the highest quantized output value represents.
  """
  result = _op_def_lib.apply_op("QuantizedMaxPool", input=input,
                                min_input=min_input, max_input=max_input,
                                ksize=ksize, strides=strides, padding=padding,
                                name=name)
  return _QuantizedMaxPoolOutput._make(result)


ops.RegisterShape("QuantizedMaxPool")(None)
_quantized_relu_outputs = ["activations", "min_activations",
                          "max_activations"]


_QuantizedReluOutput = collections.namedtuple("QuantizedRelu",
                                              _quantized_relu_outputs)


def quantized_relu(features, min_features, max_features, out_type=None,
                   name=None):
  r"""Computes Quantized Rectified Linear: `max(features, 0)`

  Args:
    features: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint16`, `quint16`, `qint32`.
    min_features: A `Tensor` of type `float32`.
      The float value that the lowest quantized value represents.
    max_features: A `Tensor` of type `float32`.
      The float value that the highest quantized value represents.
    out_type: An optional `tf.DType` from: `tf.qint8, tf.quint8, tf.qint16, tf.quint16, tf.qint32`. Defaults to `tf.quint8`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (activations, min_activations, max_activations).
    activations: A `Tensor` of type `out_type`. Has the same output shape as "features".
    min_activations: A `Tensor` of type `float32`. The float value that the lowest quantized value represents.
    max_activations: A `Tensor` of type `float32`. The float value that the highest quantized value represents.
  """
  result = _op_def_lib.apply_op("QuantizedRelu", features=features,
                                min_features=min_features,
                                max_features=max_features, out_type=out_type,
                                name=name)
  return _QuantizedReluOutput._make(result)


ops.RegisterShape("QuantizedRelu")(None)
_quantized_relu6_outputs = ["activations", "min_activations",
                           "max_activations"]


_QuantizedRelu6Output = collections.namedtuple("QuantizedRelu6",
                                               _quantized_relu6_outputs)


def quantized_relu6(features, min_features, max_features, out_type=None,
                    name=None):
  r"""Computes Quantized Rectified Linear 6: `min(max(features, 0), 6)`

  Args:
    features: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint16`, `quint16`, `qint32`.
    min_features: A `Tensor` of type `float32`.
      The float value that the lowest quantized value represents.
    max_features: A `Tensor` of type `float32`.
      The float value that the highest quantized value represents.
    out_type: An optional `tf.DType` from: `tf.qint8, tf.quint8, tf.qint16, tf.quint16, tf.qint32`. Defaults to `tf.quint8`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (activations, min_activations, max_activations).
    activations: A `Tensor` of type `out_type`. Has the same output shape as "features".
    min_activations: A `Tensor` of type `float32`. The float value that the lowest quantized value represents.
    max_activations: A `Tensor` of type `float32`. The float value that the highest quantized value represents.
  """
  result = _op_def_lib.apply_op("QuantizedRelu6", features=features,
                                min_features=min_features,
                                max_features=max_features, out_type=out_type,
                                name=name)
  return _QuantizedRelu6Output._make(result)


ops.RegisterShape("QuantizedRelu6")(None)
_quantized_relu_x_outputs = ["activations", "min_activations",
                            "max_activations"]


_QuantizedReluXOutput = collections.namedtuple("QuantizedReluX",
                                               _quantized_relu_x_outputs)


def quantized_relu_x(features, max_value, min_features, max_features,
                     out_type=None, name=None):
  r"""Computes Quantized Rectified Linear X: `min(max(features, 0), max_value)`

  Args:
    features: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint16`, `quint16`, `qint32`.
    max_value: A `Tensor` of type `float32`.
    min_features: A `Tensor` of type `float32`.
      The float value that the lowest quantized value represents.
    max_features: A `Tensor` of type `float32`.
      The float value that the highest quantized value represents.
    out_type: An optional `tf.DType` from: `tf.qint8, tf.quint8, tf.qint16, tf.quint16, tf.qint32`. Defaults to `tf.quint8`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (activations, min_activations, max_activations).
    activations: A `Tensor` of type `out_type`. Has the same output shape as "features".
    min_activations: A `Tensor` of type `float32`. The float value that the lowest quantized value represents.
    max_activations: A `Tensor` of type `float32`. The float value that the highest quantized value represents.
  """
  result = _op_def_lib.apply_op("QuantizedReluX", features=features,
                                max_value=max_value,
                                min_features=min_features,
                                max_features=max_features, out_type=out_type,
                                name=name)
  return _QuantizedReluXOutput._make(result)


ops.RegisterShape("QuantizedReluX")(None)
def _InitOpDefLibrary():
  op_list = op_def_pb2.OpList()
  text_format.Merge(_InitOpDefLibrary.op_list_ascii, op_list)
  op_def_registry.register_op_list(op_list)
  op_def_lib = op_def_library.OpDefLibrary()
  op_def_lib.add_op_list(op_list)
  return op_def_lib


_InitOpDefLibrary.op_list_ascii = """op {
  name: "QuantizedAvgPool"
  input_arg {
    name: "input"
    type_attr: "T"
  }
  input_arg {
    name: "min_input"
    type: DT_FLOAT
  }
  input_arg {
    name: "max_input"
    type: DT_FLOAT
  }
  output_arg {
    name: "output"
    type_attr: "T"
  }
  output_arg {
    name: "min_output"
    type: DT_FLOAT
  }
  output_arg {
    name: "max_output"
    type: DT_FLOAT
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_QINT8
        type: DT_QUINT8
        type: DT_QINT16
        type: DT_QUINT16
        type: DT_QINT32
      }
    }
  }
  attr {
    name: "ksize"
    type: "list(int)"
  }
  attr {
    name: "strides"
    type: "list(int)"
  }
  attr {
    name: "padding"
    type: "string"
    allowed_values {
      list {
        s: "SAME"
        s: "VALID"
      }
    }
  }
}
op {
  name: "QuantizedBatchNormWithGlobalNormalization"
  input_arg {
    name: "t"
    type_attr: "Tinput"
  }
  input_arg {
    name: "t_min"
    type: DT_FLOAT
  }
  input_arg {
    name: "t_max"
    type: DT_FLOAT
  }
  input_arg {
    name: "m"
    type_attr: "Tinput"
  }
  input_arg {
    name: "m_min"
    type: DT_FLOAT
  }
  input_arg {
    name: "m_max"
    type: DT_FLOAT
  }
  input_arg {
    name: "v"
    type_attr: "Tinput"
  }
  input_arg {
    name: "v_min"
    type: DT_FLOAT
  }
  input_arg {
    name: "v_max"
    type: DT_FLOAT
  }
  input_arg {
    name: "beta"
    type_attr: "Tinput"
  }
  input_arg {
    name: "beta_min"
    type: DT_FLOAT
  }
  input_arg {
    name: "beta_max"
    type: DT_FLOAT
  }
  input_arg {
    name: "gamma"
    type_attr: "Tinput"
  }
  input_arg {
    name: "gamma_min"
    type: DT_FLOAT
  }
  input_arg {
    name: "gamma_max"
    type: DT_FLOAT
  }
  output_arg {
    name: "result"
    type_attr: "out_type"
  }
  output_arg {
    name: "result_min"
    type: DT_FLOAT
  }
  output_arg {
    name: "result_max"
    type: DT_FLOAT
  }
  attr {
    name: "Tinput"
    type: "type"
    allowed_values {
      list {
        type: DT_QINT8
        type: DT_QUINT8
        type: DT_QINT16
        type: DT_QUINT16
        type: DT_QINT32
      }
    }
  }
  attr {
    name: "out_type"
    type: "type"
    allowed_values {
      list {
        type: DT_QINT8
        type: DT_QUINT8
        type: DT_QINT16
        type: DT_QUINT16
        type: DT_QINT32
      }
    }
  }
  attr {
    name: "variance_epsilon"
    type: "float"
  }
  attr {
    name: "scale_after_normalization"
    type: "bool"
  }
}
op {
  name: "QuantizedBiasAdd"
  input_arg {
    name: "input"
    type_attr: "T1"
  }
  input_arg {
    name: "bias"
    type_attr: "T2"
  }
  input_arg {
    name: "min_input"
    type: DT_FLOAT
  }
  input_arg {
    name: "max_input"
    type: DT_FLOAT
  }
  input_arg {
    name: "min_bias"
    type: DT_FLOAT
  }
  input_arg {
    name: "max_bias"
    type: DT_FLOAT
  }
  output_arg {
    name: "output"
    type_attr: "out_type"
  }
  output_arg {
    name: "min_out"
    type: DT_FLOAT
  }
  output_arg {
    name: "max_out"
    type: DT_FLOAT
  }
  attr {
    name: "T1"
    type: "type"
    allowed_values {
      list {
        type: DT_QINT8
        type: DT_QUINT8
        type: DT_QINT16
        type: DT_QUINT16
        type: DT_QINT32
      }
    }
  }
  attr {
    name: "T2"
    type: "type"
    allowed_values {
      list {
        type: DT_QINT8
        type: DT_QUINT8
        type: DT_QINT16
        type: DT_QUINT16
        type: DT_QINT32
      }
    }
  }
  attr {
    name: "out_type"
    type: "type"
    allowed_values {
      list {
        type: DT_QINT8
        type: DT_QUINT8
        type: DT_QINT16
        type: DT_QUINT16
        type: DT_QINT32
      }
    }
  }
}
op {
  name: "QuantizedConv2D"
  input_arg {
    name: "input"
    type_attr: "Tinput"
  }
  input_arg {
    name: "filter"
    type_attr: "Tfilter"
  }
  input_arg {
    name: "min_input"
    type: DT_FLOAT
  }
  input_arg {
    name: "max_input"
    type: DT_FLOAT
  }
  input_arg {
    name: "min_filter"
    type: DT_FLOAT
  }
  input_arg {
    name: "max_filter"
    type: DT_FLOAT
  }
  output_arg {
    name: "output"
    type_attr: "out_type"
  }
  output_arg {
    name: "min_output"
    type: DT_FLOAT
  }
  output_arg {
    name: "max_output"
    type: DT_FLOAT
  }
  attr {
    name: "Tinput"
    type: "type"
    allowed_values {
      list {
        type: DT_QINT8
        type: DT_QUINT8
        type: DT_QINT16
        type: DT_QUINT16
        type: DT_QINT32
      }
    }
  }
  attr {
    name: "Tfilter"
    type: "type"
    allowed_values {
      list {
        type: DT_QINT8
        type: DT_QUINT8
        type: DT_QINT16
        type: DT_QUINT16
        type: DT_QINT32
      }
    }
  }
  attr {
    name: "out_type"
    type: "type"
    default_value {
      type: DT_QINT32
    }
    allowed_values {
      list {
        type: DT_QINT8
        type: DT_QUINT8
        type: DT_QINT16
        type: DT_QUINT16
        type: DT_QINT32
      }
    }
  }
  attr {
    name: "strides"
    type: "list(int)"
  }
  attr {
    name: "padding"
    type: "string"
    allowed_values {
      list {
        s: "SAME"
        s: "VALID"
      }
    }
  }
}
op {
  name: "QuantizedMaxPool"
  input_arg {
    name: "input"
    type_attr: "T"
  }
  input_arg {
    name: "min_input"
    type: DT_FLOAT
  }
  input_arg {
    name: "max_input"
    type: DT_FLOAT
  }
  output_arg {
    name: "output"
    type_attr: "T"
  }
  output_arg {
    name: "min_output"
    type: DT_FLOAT
  }
  output_arg {
    name: "max_output"
    type: DT_FLOAT
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_QINT8
        type: DT_QUINT8
        type: DT_QINT16
        type: DT_QUINT16
        type: DT_QINT32
      }
    }
  }
  attr {
    name: "ksize"
    type: "list(int)"
  }
  attr {
    name: "strides"
    type: "list(int)"
  }
  attr {
    name: "padding"
    type: "string"
    allowed_values {
      list {
        s: "SAME"
        s: "VALID"
      }
    }
  }
}
op {
  name: "QuantizedRelu"
  input_arg {
    name: "features"
    type_attr: "Tinput"
  }
  input_arg {
    name: "min_features"
    type: DT_FLOAT
  }
  input_arg {
    name: "max_features"
    type: DT_FLOAT
  }
  output_arg {
    name: "activations"
    type_attr: "out_type"
  }
  output_arg {
    name: "min_activations"
    type: DT_FLOAT
  }
  output_arg {
    name: "max_activations"
    type: DT_FLOAT
  }
  attr {
    name: "Tinput"
    type: "type"
    allowed_values {
      list {
        type: DT_QINT8
        type: DT_QUINT8
        type: DT_QINT16
        type: DT_QUINT16
        type: DT_QINT32
      }
    }
  }
  attr {
    name: "out_type"
    type: "type"
    default_value {
      type: DT_QUINT8
    }
    allowed_values {
      list {
        type: DT_QINT8
        type: DT_QUINT8
        type: DT_QINT16
        type: DT_QUINT16
        type: DT_QINT32
      }
    }
  }
}
op {
  name: "QuantizedRelu6"
  input_arg {
    name: "features"
    type_attr: "Tinput"
  }
  input_arg {
    name: "min_features"
    type: DT_FLOAT
  }
  input_arg {
    name: "max_features"
    type: DT_FLOAT
  }
  output_arg {
    name: "activations"
    type_attr: "out_type"
  }
  output_arg {
    name: "min_activations"
    type: DT_FLOAT
  }
  output_arg {
    name: "max_activations"
    type: DT_FLOAT
  }
  attr {
    name: "Tinput"
    type: "type"
    allowed_values {
      list {
        type: DT_QINT8
        type: DT_QUINT8
        type: DT_QINT16
        type: DT_QUINT16
        type: DT_QINT32
      }
    }
  }
  attr {
    name: "out_type"
    type: "type"
    default_value {
      type: DT_QUINT8
    }
    allowed_values {
      list {
        type: DT_QINT8
        type: DT_QUINT8
        type: DT_QINT16
        type: DT_QUINT16
        type: DT_QINT32
      }
    }
  }
}
op {
  name: "QuantizedReluX"
  input_arg {
    name: "features"
    type_attr: "Tinput"
  }
  input_arg {
    name: "max_value"
    type: DT_FLOAT
  }
  input_arg {
    name: "min_features"
    type: DT_FLOAT
  }
  input_arg {
    name: "max_features"
    type: DT_FLOAT
  }
  output_arg {
    name: "activations"
    type_attr: "out_type"
  }
  output_arg {
    name: "min_activations"
    type: DT_FLOAT
  }
  output_arg {
    name: "max_activations"
    type: DT_FLOAT
  }
  attr {
    name: "Tinput"
    type: "type"
    allowed_values {
      list {
        type: DT_QINT8
        type: DT_QUINT8
        type: DT_QINT16
        type: DT_QUINT16
        type: DT_QINT32
      }
    }
  }
  attr {
    name: "out_type"
    type: "type"
    default_value {
      type: DT_QUINT8
    }
    allowed_values {
      list {
        type: DT_QINT8
        type: DT_QUINT8
        type: DT_QINT16
        type: DT_QUINT16
        type: DT_QINT32
      }
    }
  }
}
"""


_op_def_lib = _InitOpDefLibrary()
