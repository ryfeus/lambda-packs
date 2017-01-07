"""Python wrappers around Brain.

This file is MACHINE GENERATED! Do not edit.
"""

import collections

from google.protobuf import text_format

from tensorflow.core.framework import op_def_pb2
from tensorflow.python.framework import op_def_registry
from tensorflow.python.framework import ops
from tensorflow.python.ops import op_def_library


_quantize_down_and_shrink_range_outputs = ["output", "output_min",
                                          "output_max"]


_QuantizeDownAndShrinkRangeOutput = collections.namedtuple("QuantizeDownAndShrinkRange",
                                                           _quantize_down_and_shrink_range_outputs)


def quantize_down_and_shrink_range(input, input_min, input_max, out_type,
                                   name=None):
  r"""Convert the quantized 'input' tensor into a lower-precision 'output', using the

  actual distribution of the values to maximize the usage of the lower bit depth
  and adjusting the output min and max ranges accordingly.

  [input_min, input_max] are scalar floats that specify the range for the float
  interpretation of the 'input' data. For example, if input_min is -1.0f and
  input_max is 1.0f, and we are dealing with quint16 quantized data, then a 0
  value in the 16-bit data should be interpreted as -1.0f, and a 65535 means 1.0f.

  This operator tries to squeeze as much precision as possible into an output with
  a lower bit depth by calculating the actual min and max values found in the
  data. For example, maybe that quint16 input has no values lower than 16,384 and
  none higher than 49,152. That means only half the range is actually needed, all
  the float interpretations are between -0.5f and 0.5f, so if we want to compress
  the data into a quint8 output, we can use that range rather than the theoretical
  -1.0f to 1.0f that is suggested by the input min and max.

  In practice, this is most useful for taking output from operations like
  QuantizedMatMul that can produce higher bit-depth outputs than their inputs and
  may have large potential output ranges, but in practice have a distribution of
  input values that only uses a small fraction of the possible range. By feeding
  that output into this operator, we can reduce it from 32 bits down to 8 with
  minimal loss of accuracy.

  Args:
    input: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint16`, `quint16`, `qint32`.
    input_min: A `Tensor` of type `float32`.
      The float value that the minimum quantized input value represents.
    input_max: A `Tensor` of type `float32`.
      The float value that the maximum quantized input value represents.
    out_type: A `tf.DType` from: `tf.qint8, tf.quint8, tf.qint16, tf.quint16, tf.qint32`.
      The type of the output. Should be a lower bit depth than Tinput.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output, output_min, output_max).
    output: A `Tensor` of type `out_type`.
    output_min: A `Tensor` of type `float32`. The float value that the minimum quantized output value represents.
    output_max: A `Tensor` of type `float32`. The float value that the maximum quantized output value represents.
  """
  result = _op_def_lib.apply_op("QuantizeDownAndShrinkRange", input=input,
                                input_min=input_min, input_max=input_max,
                                out_type=out_type, name=name)
  return _QuantizeDownAndShrinkRangeOutput._make(result)


ops.RegisterShape("QuantizeDownAndShrinkRange")(None)
_quantized_mat_mul_outputs = ["out", "min_out", "max_out"]


_QuantizedMatMulOutput = collections.namedtuple("QuantizedMatMul",
                                                _quantized_mat_mul_outputs)


def quantized_mat_mul(a, b, min_a, max_a, min_b, max_b, Toutput=None,
                      transpose_a=None, transpose_b=None, name=None):
  r"""Perform a quantized matrix multiplication of  `a` by the matrix `b`.

  The inputs must be two-dimensional matrices and the inner dimension of
  `a` (after being transposed if `transpose_a` is non-zero) must match the
  outer dimension of `b` (after being transposed if `transposed_b` is
  non-zero).

  Args:
    a: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint16`, `quint16`, `qint32`.
      Must be a two-dimensional tensor.
    b: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint16`, `quint16`, `qint32`.
      Must be a two-dimensional tensor.
    min_a: A `Tensor` of type `float32`.
      The float value that the lowest quantized `a` value represents.
    max_a: A `Tensor` of type `float32`.
      The float value that the highest quantized `a` value represents.
    min_b: A `Tensor` of type `float32`.
      The float value that the lowest quantized `b` value represents.
    max_b: A `Tensor` of type `float32`.
      The float value that the highest quantized `b` value represents.
    Toutput: An optional `tf.DType` from: `tf.qint8, tf.quint8, tf.qint16, tf.quint16, tf.qint32`. Defaults to `tf.qint32`.
    transpose_a: An optional `bool`. Defaults to `False`.
      If true, `a` is transposed before multiplication.
    transpose_b: An optional `bool`. Defaults to `False`.
      If true, `b` is transposed before multiplication.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (out, min_out, max_out).
    out: A `Tensor` of type `Toutput`.
    min_out: A `Tensor` of type `float32`. The float value that the lowest quantized output value represents.
    max_out: A `Tensor` of type `float32`. The float value that the highest quantized output value represents.
  """
  result = _op_def_lib.apply_op("QuantizedMatMul", a=a, b=b, min_a=min_a,
                                max_a=max_a, min_b=min_b, max_b=max_b,
                                Toutput=Toutput, transpose_a=transpose_a,
                                transpose_b=transpose_b, name=name)
  return _QuantizedMatMulOutput._make(result)


ops.RegisterShape("QuantizedMatMul")(None)
def _InitOpDefLibrary():
  op_list = op_def_pb2.OpList()
  text_format.Merge(_InitOpDefLibrary.op_list_ascii, op_list)
  op_def_registry.register_op_list(op_list)
  op_def_lib = op_def_library.OpDefLibrary()
  op_def_lib.add_op_list(op_list)
  return op_def_lib


_InitOpDefLibrary.op_list_ascii = """op {
  name: "QuantizeDownAndShrinkRange"
  input_arg {
    name: "input"
    type_attr: "Tinput"
  }
  input_arg {
    name: "input_min"
    type: DT_FLOAT
  }
  input_arg {
    name: "input_max"
    type: DT_FLOAT
  }
  output_arg {
    name: "output"
    type_attr: "out_type"
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
}
op {
  name: "QuantizedMatMul"
  input_arg {
    name: "a"
    type_attr: "T1"
  }
  input_arg {
    name: "b"
    type_attr: "T2"
  }
  input_arg {
    name: "min_a"
    type: DT_FLOAT
  }
  input_arg {
    name: "max_a"
    type: DT_FLOAT
  }
  input_arg {
    name: "min_b"
    type: DT_FLOAT
  }
  input_arg {
    name: "max_b"
    type: DT_FLOAT
  }
  output_arg {
    name: "out"
    type_attr: "Toutput"
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
    name: "Toutput"
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
    name: "transpose_a"
    type: "bool"
    default_value {
      b: false
    }
  }
  attr {
    name: "transpose_b"
    type: "bool"
    default_value {
      b: false
    }
  }
}
"""


_op_def_lib = _InitOpDefLibrary()
