"""Python wrappers around Brain.

This file is MACHINE GENERATED! Do not edit.
"""

import collections

from google.protobuf import text_format

from tensorflow.core.framework import op_def_pb2
from tensorflow.python.framework import op_def_registry
from tensorflow.python.framework import ops
from tensorflow.python.ops import op_def_library


_dequantize_outputs = ["output"]


def dequantize(input, min_range, max_range, mode=None, name=None):
  r"""Dequantize the 'input' tensor into a float Tensor.

  [min_range, max_range] are scalar floats that specify the range for
  the 'input' data. The 'mode' attribute controls exactly which calculations are
  used to convert the float values to their quantized equivalents.

  In 'MIN_COMBINED' mode, each value of the tensor will undergo the following:

  ```
  if T == qint8, in[i] += (range(T) + 1)/ 2.0
  out[i] = min_range + (in[i]* (max_range - min_range) / range(T))
  ```
  here `range(T) = numeric_limits<T>::max() - numeric_limits<T>::min()`

  *MIN_COMBINED Mode Example*

  If the input comes from a QuantizedRelu6, the output type is
  quint8 (range of 0-255) but the possible range of QuantizedRelu6 is
  0-6.  The min_range and max_range values are therefore 0.0 and 6.0.
  Dequantize on quint8 will take each value, cast to float, and multiply
  by 6 / 255.
  Note that if quantizedtype is qint8, the operation will additionally add
  each value by 128 prior to casting.

  If the mode is 'MIN_FIRST', then this approach is used:

  ```
  number_of_steps = 1 << (# of bits in T)
  range_adjust = number_of_steps / (number_of_steps - 1)
  range = (range_max - range_min) * range_adjust
  range_scale = number_of_steps / range
  const double offset_input = static_cast<double>(input) - lowest_quantized;
  result = range_min + ((input - numeric_limits<T>::min()) * range_scale)
  ```

  Args:
    input: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint16`, `quint16`, `qint32`.
    min_range: A `Tensor` of type `float32`.
      The minimum scalar value possibly produced for the input.
    max_range: A `Tensor` of type `float32`.
      The maximum scalar value possibly produced for the input.
    mode: An optional `string` from: `"MIN_COMBINED", "MIN_FIRST"`. Defaults to `"MIN_COMBINED"`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32`.
  """
  result = _op_def_lib.apply_op("Dequantize", input=input,
                                min_range=min_range, max_range=max_range,
                                mode=mode, name=name)
  return result


ops.RegisterShape("Dequantize")(None)
_quantize_v2_outputs = ["output", "output_min", "output_max"]


_QuantizeV2Output = collections.namedtuple("QuantizeV2", _quantize_v2_outputs)


def quantize_v2(input, min_range, max_range, T, mode=None, name=None):
  r"""Quantize the 'input' tensor of type float to 'output' tensor of type 'T'.

  [min_range, max_range] are scalar floats that specify the range for
  the 'input' data. The 'mode' attribute controls exactly which calculations are
  used to convert the float values to their quantized equivalents.

  In 'MIN_COMBINED' mode, each value of the tensor will undergo the following:

  ```
  out[i] = (in[i] - min_range) * range(T) / (max_range - min_range)
  if T == qint8, out[i] -= (range(T) + 1) / 2.0
  ```
  here `range(T) = numeric_limits<T>::max() - numeric_limits<T>::min()`

  *MIN_COMBINED Mode Example*

  Assume the input is type float and has a possible range of [0.0, 6.0] and the
  output type is quint8 ([0, 255]). The min_range and max_range values should be
  specified as 0.0 and 6.0. Quantizing from float to quint8 will multiply each
  value of the input by 255/6 and cast to quint8.

  If the output type was qint8 ([-128, 127]), the operation will additionally
  subtract each value by 128 prior to casting, so that the range of values aligns
  with the range of qint8.

  If the mode is 'MIN_FIRST', then this approach is used:

  ```
  number_of_steps = 1 << (# of bits in T)
  range_adjust = number_of_steps / (number_of_steps - 1)
  range = (range_max - range_min) * range_adjust
  range_scale = number_of_steps / range
  quantized = round(input * range_scale) - round(range_min * range_scale) +
    numeric_limits<T>::min()
  quantized = max(quantized, numeric_limits<T>::min())
  quantized = min(quantized, numeric_limits<T>::max())
  ```

  The biggest difference between this and MIN_COMBINED is that the minimum range
  is rounded first, before it's subtracted from the rounded value. With
  MIN_COMBINED, a small bias is introduced where repeated iterations of quantizing
  and dequantizing will introduce a larger and larger error.

  One thing to watch out for is that the operator may choose to adjust the
  requested minimum and maximum values slightly during the quantization process,
  so you should always use the output ports as the range for further calculations.
  For example, if the requested minimum and maximum values are close to equal,
  they will be separated by a small epsilon value to prevent ill-formed quantized
  buffers from being created. Otherwise, you can end up with buffers where all the
  quantized values map to the same float value, which causes problems for
  operations that have to perform further calculations on them.

  Args:
    input: A `Tensor` of type `float32`.
    min_range: A `Tensor` of type `float32`.
      The minimum scalar value possibly produced for the input.
    max_range: A `Tensor` of type `float32`.
      The maximum scalar value possibly produced for the input.
    T: A `tf.DType` from: `tf.qint8, tf.quint8, tf.qint16, tf.quint16, tf.qint32`.
    mode: An optional `string` from: `"MIN_COMBINED", "MIN_FIRST"`. Defaults to `"MIN_COMBINED"`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output, output_min, output_max).
    output: A `Tensor` of type `T`. The quantized data produced from the float input.
    output_min: A `Tensor` of type `float32`. The actual minimum scalar value used for the output.
    output_max: A `Tensor` of type `float32`. The actual maximum scalar value used for the output.
  """
  result = _op_def_lib.apply_op("QuantizeV2", input=input,
                                min_range=min_range, max_range=max_range, T=T,
                                mode=mode, name=name)
  return _QuantizeV2Output._make(result)


ops.RegisterShape("QuantizeV2")(None)
_quantized_concat_outputs = ["output", "output_min", "output_max"]


_QuantizedConcatOutput = collections.namedtuple("QuantizedConcat",
                                                _quantized_concat_outputs)


def quantized_concat(concat_dim, values, input_mins, input_maxes, name=None):
  r"""Concatenates quantized tensors along one dimension.

  Args:
    concat_dim: A `Tensor` of type `int32`.
      0-D.  The dimension along which to concatenate.  Must be in the
      range [0, rank(values)).
    values: A list of at least 2 `Tensor` objects of the same type.
      The `N` Tensors to concatenate. Their ranks and types must match,
      and their sizes must match in all dimensions except `concat_dim`.
    input_mins: A list with the same number of `Tensor` objects as `values` of `Tensor` objects of type `float32`.
      The minimum scalar values for each of the input tensors.
    input_maxes: A list with the same number of `Tensor` objects as `values` of `Tensor` objects of type `float32`.
      The maximum scalar values for each of the input tensors.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output, output_min, output_max).
    output: A `Tensor`. Has the same type as `values`. A `Tensor` with the concatenation of values stacked along the
      `concat_dim` dimension.  This tensor's shape matches that of `values` except
      in `concat_dim` where it has the sum of the sizes.
    output_min: A `Tensor` of type `float32`. The float value that the minimum quantized output value represents.
    output_max: A `Tensor` of type `float32`. The float value that the maximum quantized output value represents.
  """
  result = _op_def_lib.apply_op("QuantizedConcat", concat_dim=concat_dim,
                                values=values, input_mins=input_mins,
                                input_maxes=input_maxes, name=name)
  return _QuantizedConcatOutput._make(result)


ops.RegisterShape("QuantizedConcat")(None)
def _InitOpDefLibrary():
  op_list = op_def_pb2.OpList()
  text_format.Merge(_InitOpDefLibrary.op_list_ascii, op_list)
  op_def_registry.register_op_list(op_list)
  op_def_lib = op_def_library.OpDefLibrary()
  op_def_lib.add_op_list(op_list)
  return op_def_lib


_InitOpDefLibrary.op_list_ascii = """op {
  name: "Dequantize"
  input_arg {
    name: "input"
    type_attr: "T"
  }
  input_arg {
    name: "min_range"
    type: DT_FLOAT
  }
  input_arg {
    name: "max_range"
    type: DT_FLOAT
  }
  output_arg {
    name: "output"
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
    name: "mode"
    type: "string"
    default_value {
      s: "MIN_COMBINED"
    }
    allowed_values {
      list {
        s: "MIN_COMBINED"
        s: "MIN_FIRST"
      }
    }
  }
}
op {
  name: "QuantizeV2"
  input_arg {
    name: "input"
    type: DT_FLOAT
  }
  input_arg {
    name: "min_range"
    type: DT_FLOAT
  }
  input_arg {
    name: "max_range"
    type: DT_FLOAT
  }
  output_arg {
    name: "output"
    type_attr: "T"
  }
  output_arg {
    name: "output_min"
    type: DT_FLOAT
  }
  output_arg {
    name: "output_max"
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
    name: "mode"
    type: "string"
    default_value {
      s: "MIN_COMBINED"
    }
    allowed_values {
      list {
        s: "MIN_COMBINED"
        s: "MIN_FIRST"
      }
    }
  }
}
op {
  name: "QuantizedConcat"
  input_arg {
    name: "concat_dim"
    type: DT_INT32
  }
  input_arg {
    name: "values"
    type_attr: "T"
    number_attr: "N"
  }
  input_arg {
    name: "input_mins"
    type: DT_FLOAT
    number_attr: "N"
  }
  input_arg {
    name: "input_maxes"
    type: DT_FLOAT
    number_attr: "N"
  }
  output_arg {
    name: "output"
    type_attr: "T"
  }
  output_arg {
    name: "output_min"
    type: DT_FLOAT
  }
  output_arg {
    name: "output_max"
    type: DT_FLOAT
  }
  attr {
    name: "N"
    type: "int"
    has_minimum: true
    minimum: 2
  }
  attr {
    name: "T"
    type: "type"
  }
}
"""


_op_def_lib = _InitOpDefLibrary()
