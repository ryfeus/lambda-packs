"""Python wrappers around Brain.

This file is MACHINE GENERATED! Do not edit.
"""

import collections

from google.protobuf import text_format

from tensorflow.core.framework import op_def_pb2
from tensorflow.python.framework import op_def_registry
from tensorflow.python.framework import ops
from tensorflow.python.ops import op_def_library


__avg_pool_outputs = ["output"]


def _avg_pool(value, ksize, strides, padding, data_format=None, name=None):
  r"""Performs average pooling on the input.

  Each entry in `output` is the mean of the corresponding size `ksize`
  window in `value`.

  Args:
    value: A `Tensor`. Must be one of the following types: `float32`, `float64`.
      4-D with shape `[batch, height, width, channels]`.
    ksize: A list of `ints` that has length `>= 4`.
      The size of the sliding window for each dimension of `value`.
    strides: A list of `ints` that has length `>= 4`.
      The stride of the sliding window for each dimension of `value`.
    padding: A `string` from: `"SAME", "VALID"`.
      The type of padding algorithm to use.
    data_format: An optional `string` from: `"NHWC", "NCHW"`. Defaults to `"NHWC"`.
      Specify the data format of the input and output data. With the
      default format "NHWC", the data is stored in the order of:
          [batch, in_height, in_width, in_channels].
      Alternatively, the format could be "NCHW", the data storage order of:
          [batch, in_channels, in_height, in_width].
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `value`.
    The average pooled output tensor.
  """
  result = _op_def_lib.apply_op("AvgPool", value=value, ksize=ksize,
                                strides=strides, padding=padding,
                                data_format=data_format, name=name)
  return result


_avg_pool3d_outputs = ["output"]


def avg_pool3d(input, ksize, strides, padding, name=None):
  r"""Performs 3D average pooling on the input.

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`, `complex128`, `qint8`, `quint8`, `qint32`, `half`.
      Shape `[batch, depth, rows, cols, channels]` tensor to pool over.
    ksize: A list of `ints` that has length `>= 5`.
      1-D tensor of length 5. The size of the window for each dimension of
      the input tensor. Must have `ksize[0] = ksize[1] = 1`.
    strides: A list of `ints` that has length `>= 5`.
      1-D tensor of length 5. The stride of the sliding window for each
      dimension of `input`. Must have `strides[0] = strides[4] = 1`.
    padding: A `string` from: `"SAME", "VALID"`.
      The type of padding algorithm to use.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
    The average pooled output tensor.
  """
  result = _op_def_lib.apply_op("AvgPool3D", input=input, ksize=ksize,
                                strides=strides, padding=padding, name=name)
  return result


_avg_pool3d_grad_outputs = ["output"]


def avg_pool3d_grad(orig_input_shape, grad, ksize, strides, padding,
                    name=None):
  r"""Computes gradients of average pooling function.

  Args:
    orig_input_shape: A `Tensor` of type `int32`.
      The original input dimensions.
    grad: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`, `complex128`, `qint8`, `quint8`, `qint32`, `half`.
      Output backprop of shape `[batch, depth, rows, cols, channels]`.
    ksize: A list of `ints` that has length `>= 5`.
      1-D tensor of length 5. The size of the window for each dimension of
      the input tensor. Must have `ksize[0] = ksize[1] = 1`.
    strides: A list of `ints` that has length `>= 5`.
      1-D tensor of length 5. The stride of the sliding window for each
      dimension of `input`. Must have `strides[0] = strides[4] = 1`.
    padding: A `string` from: `"SAME", "VALID"`.
      The type of padding algorithm to use.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `grad`. The backprop for input.
  """
  result = _op_def_lib.apply_op("AvgPool3DGrad",
                                orig_input_shape=orig_input_shape, grad=grad,
                                ksize=ksize, strides=strides, padding=padding,
                                name=name)
  return result


__avg_pool_grad_outputs = ["output"]


def _avg_pool_grad(orig_input_shape, grad, ksize, strides, padding,
                   data_format=None, name=None):
  r"""Computes gradients of the average pooling function.

  Args:
    orig_input_shape: A `Tensor` of type `int32`.
      1-D.  Shape of the original input to `avg_pool`.
    grad: A `Tensor`. Must be one of the following types: `float32`, `float64`.
      4-D with shape `[batch, height, width, channels]`.  Gradients w.r.t.
      the output of `avg_pool`.
    ksize: A list of `ints` that has length `>= 4`.
      The size of the sliding window for each dimension of the input.
    strides: A list of `ints` that has length `>= 4`.
      The stride of the sliding window for each dimension of the input.
    padding: A `string` from: `"SAME", "VALID"`.
      The type of padding algorithm to use.
    data_format: An optional `string` from: `"NHWC", "NCHW"`. Defaults to `"NHWC"`.
      Specify the data format of the input and output data. With the
      default format "NHWC", the data is stored in the order of:
          [batch, in_height, in_width, in_channels].
      Alternatively, the format could be "NCHW", the data storage order of:
          [batch, in_channels, in_height, in_width].
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `grad`.
    4-D.  Gradients w.r.t. the input of `avg_pool`.
  """
  result = _op_def_lib.apply_op("AvgPoolGrad",
                                orig_input_shape=orig_input_shape, grad=grad,
                                ksize=ksize, strides=strides, padding=padding,
                                data_format=data_format, name=name)
  return result


__batch_norm_with_global_normalization_outputs = ["result"]


def _batch_norm_with_global_normalization(t, m, v, beta, gamma,
                                          variance_epsilon,
                                          scale_after_normalization,
                                          name=None):
  r"""Batch normalization.

  This op is deprecated. Prefer `tf.nn.batch_normalization`.

  Args:
    t: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`, `complex128`, `qint8`, `quint8`, `qint32`, `half`.
      A 4D input Tensor.
    m: A `Tensor`. Must have the same type as `t`.
      A 1D mean Tensor with size matching the last dimension of t.
      This is the first output from tf.nn.moments,
      or a saved moving average thereof.
    v: A `Tensor`. Must have the same type as `t`.
      A 1D variance Tensor with size matching the last dimension of t.
      This is the second output from tf.nn.moments,
      or a saved moving average thereof.
    beta: A `Tensor`. Must have the same type as `t`.
      A 1D beta Tensor with size matching the last dimension of t.
      An offset to be added to the normalized tensor.
    gamma: A `Tensor`. Must have the same type as `t`.
      A 1D gamma Tensor with size matching the last dimension of t.
      If "scale_after_normalization" is true, this tensor will be multiplied
      with the normalized tensor.
    variance_epsilon: A `float`. A small float number to avoid dividing by 0.
    scale_after_normalization: A `bool`.
      A bool indicating whether the resulted tensor
      needs to be multiplied with gamma.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `t`.
  """
  result = _op_def_lib.apply_op("BatchNormWithGlobalNormalization", t=t, m=m,
                                v=v, beta=beta, gamma=gamma,
                                variance_epsilon=variance_epsilon,
                                scale_after_normalization=scale_after_normalization,
                                name=name)
  return result


__batch_norm_with_global_normalization_grad_outputs = ["dx", "dm", "dv", "db",
                                                      "dg"]


_BatchNormWithGlobalNormalizationGradOutput = collections.namedtuple("BatchNormWithGlobalNormalizationGrad",
                                                                     __batch_norm_with_global_normalization_grad_outputs)


def _batch_norm_with_global_normalization_grad(t, m, v, gamma, backprop,
                                               variance_epsilon,
                                               scale_after_normalization,
                                               name=None):
  r"""Gradients for batch normalization.

  This op is deprecated. See `tf.nn.batch_normalization`.

  Args:
    t: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`, `complex128`, `qint8`, `quint8`, `qint32`, `half`.
      A 4D input Tensor.
    m: A `Tensor`. Must have the same type as `t`.
      A 1D mean Tensor with size matching the last dimension of t.
      This is the first output from tf.nn.moments,
      or a saved moving average thereof.
    v: A `Tensor`. Must have the same type as `t`.
      A 1D variance Tensor with size matching the last dimension of t.
      This is the second output from tf.nn.moments,
      or a saved moving average thereof.
    gamma: A `Tensor`. Must have the same type as `t`.
      A 1D gamma Tensor with size matching the last dimension of t.
      If "scale_after_normalization" is true, this Tensor will be multiplied
      with the normalized Tensor.
    backprop: A `Tensor`. Must have the same type as `t`. 4D backprop Tensor.
    variance_epsilon: A `float`. A small float number to avoid dividing by 0.
    scale_after_normalization: A `bool`.
      A bool indicating whether the resulted tensor
      needs to be multiplied with gamma.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (dx, dm, dv, db, dg).
    dx: A `Tensor`. Has the same type as `t`. 4D backprop tensor for input.
    dm: A `Tensor`. Has the same type as `t`. 1D backprop tensor for mean.
    dv: A `Tensor`. Has the same type as `t`. 1D backprop tensor for variance.
    db: A `Tensor`. Has the same type as `t`. 1D backprop tensor for beta.
    dg: A `Tensor`. Has the same type as `t`. 1D backprop tensor for gamma.
  """
  result = _op_def_lib.apply_op("BatchNormWithGlobalNormalizationGrad", t=t,
                                m=m, v=v, gamma=gamma, backprop=backprop,
                                variance_epsilon=variance_epsilon,
                                scale_after_normalization=scale_after_normalization,
                                name=name)
  return _BatchNormWithGlobalNormalizationGradOutput._make(result)


__bias_add_outputs = ["output"]


def _bias_add(value, bias, data_format=None, name=None):
  r"""Adds `bias` to `value`.

  This is a special case of `tf.add` where `bias` is restricted to be 1-D.
  Broadcasting is supported, so `value` may have any number of dimensions.

  Args:
    value: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`, `complex128`, `qint8`, `quint8`, `qint32`, `half`.
      Any number of dimensions.
    bias: A `Tensor`. Must have the same type as `value`.
      1-D with size the last dimension of `value`.
    data_format: An optional `string` from: `"NHWC", "NCHW"`. Defaults to `"NHWC"`.
      Specify the data format of the input and output data. With the
      default format "NHWC", the bias tensor will be added to the last dimension
      of the value tensor.
      Alternatively, the format could be "NCHW", the data storage order of:
          [batch, in_channels, in_height, in_width].
      The tensor will be added to "in_channels", the third-to-the-last
          dimension.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `value`.
    Broadcasted sum of `value` and `bias`.
  """
  result = _op_def_lib.apply_op("BiasAdd", value=value, bias=bias,
                                data_format=data_format, name=name)
  return result


_bias_add_grad_outputs = ["output"]


def bias_add_grad(out_backprop, data_format=None, name=None):
  r"""The backward operation for "BiasAdd" on the "bias" tensor.

  It accumulates all the values from out_backprop into the feature dimension.
  For NHWC data format, the feature dimension is the last. For NCHW data format,
  the feature dimension is the third-to-last.

  Args:
    out_backprop: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`, `complex128`, `qint8`, `quint8`, `qint32`, `half`.
      Any number of dimensions.
    data_format: An optional `string` from: `"NHWC", "NCHW"`. Defaults to `"NHWC"`.
      Specify the data format of the input and output data. With the
      default format "NHWC", the bias tensor will be added to the last dimension
      of the value tensor.
      Alternatively, the format could be "NCHW", the data storage order of:
          [batch, in_channels, in_height, in_width].
      The tensor will be added to "in_channels", the third-to-the-last
          dimension.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `out_backprop`.
    1-D with size the feature dimension of `out_backprop`.
  """
  result = _op_def_lib.apply_op("BiasAddGrad", out_backprop=out_backprop,
                                data_format=data_format, name=name)
  return result


__bias_add_v1_outputs = ["output"]


def _bias_add_v1(value, bias, name=None):
  r"""Adds `bias` to `value`.

  This is a deprecated version of BiasAdd and will be soon removed.

  This is a special case of `tf.add` where `bias` is restricted to be 1-D.
  Broadcasting is supported, so `value` may have any number of dimensions.

  Args:
    value: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`, `complex128`, `qint8`, `quint8`, `qint32`, `half`.
      Any number of dimensions.
    bias: A `Tensor`. Must have the same type as `value`.
      1-D with size the last dimension of `value`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `value`.
    Broadcasted sum of `value` and `bias`.
  """
  result = _op_def_lib.apply_op("BiasAddV1", value=value, bias=bias,
                                name=name)
  return result


_conv2d_outputs = ["output"]


def conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None,
           data_format=None, name=None):
  r"""Computes a 2-D convolution given 4-D `input` and `filter` tensors.

  Given an input tensor of shape `[batch, in_height, in_width, in_channels]`
  and a filter / kernel tensor of shape
  `[filter_height, filter_width, in_channels, out_channels]`, this op
  performs the following:

  1. Flattens the filter to a 2-D matrix with shape
     `[filter_height * filter_width * in_channels, output_channels]`.
  2. Extracts image patches from the input tensor to form a *virtual*
     tensor of shape `[batch, out_height, out_width,
     filter_height * filter_width * in_channels]`.
  3. For each patch, right-multiplies the filter matrix and the image patch
     vector.

  In detail, with the default NHWC format,

      output[b, i, j, k] =
          sum_{di, dj, q} input[b, strides[1] * i + di, strides[2] * j + dj, q] *
                          filter[di, dj, q, k]

  Must have `strides[0] = strides[3] = 1`.  For the most common case of the same
  horizontal and vertices strides, `strides = [1, stride, stride, 1]`.

  Args:
    input: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`.
    filter: A `Tensor`. Must have the same type as `input`.
    strides: A list of `ints`.
      1-D of length 4.  The stride of the sliding window for each dimension
      of `input`. Must be in the same order as the dimension specified with format.
    padding: A `string` from: `"SAME", "VALID"`.
      The type of padding algorithm to use.
    use_cudnn_on_gpu: An optional `bool`. Defaults to `True`.
    data_format: An optional `string` from: `"NHWC", "NCHW"`. Defaults to `"NHWC"`.
      Specify the data format of the input and output data. With the
      default format "NHWC", the data is stored in the order of:
          [batch, in_height, in_width, in_channels].
      Alternatively, the format could be "NCHW", the data storage order of:
          [batch, in_channels, in_height, in_width].
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  result = _op_def_lib.apply_op("Conv2D", input=input, filter=filter,
                                strides=strides, padding=padding,
                                use_cudnn_on_gpu=use_cudnn_on_gpu,
                                data_format=data_format, name=name)
  return result


_conv2d_backprop_filter_outputs = ["output"]


def conv2d_backprop_filter(input, filter_sizes, out_backprop, strides,
                           padding, use_cudnn_on_gpu=None, data_format=None,
                           name=None):
  r"""Computes the gradients of convolution with respect to the filter.

  Args:
    input: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`.
      4-D with shape `[batch, in_height, in_width, in_channels]`.
    filter_sizes: A `Tensor` of type `int32`.
      An integer vector representing the tensor shape of `filter`,
      where `filter` is a 4-D
      `[filter_height, filter_width, in_channels, out_channels]` tensor.
    out_backprop: A `Tensor`. Must have the same type as `input`.
      4-D with shape `[batch, out_height, out_width, out_channels]`.
      Gradients w.r.t. the output of the convolution.
    strides: A list of `ints`.
      The stride of the sliding window for each dimension of the input
      of the convolution. Must be in the same order as the dimension specified with
      format.
    padding: A `string` from: `"SAME", "VALID"`.
      The type of padding algorithm to use.
    use_cudnn_on_gpu: An optional `bool`. Defaults to `True`.
    data_format: An optional `string` from: `"NHWC", "NCHW"`. Defaults to `"NHWC"`.
      Specify the data format of the input and output data. With the
      default format "NHWC", the data is stored in the order of:
          [batch, in_height, in_width, in_channels].
      Alternatively, the format could be "NCHW", the data storage order of:
          [batch, in_channels, in_height, in_width].
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`. 4-D with shape
    `[filter_height, filter_width, in_channels, out_channels]`.  Gradient w.r.t.
    the `filter` input of the convolution.
  """
  result = _op_def_lib.apply_op("Conv2DBackpropFilter", input=input,
                                filter_sizes=filter_sizes,
                                out_backprop=out_backprop, strides=strides,
                                padding=padding,
                                use_cudnn_on_gpu=use_cudnn_on_gpu,
                                data_format=data_format, name=name)
  return result


_conv2d_backprop_input_outputs = ["output"]


def conv2d_backprop_input(input_sizes, filter, out_backprop, strides, padding,
                          use_cudnn_on_gpu=None, data_format=None, name=None):
  r"""Computes the gradients of convolution with respect to the input.

  Args:
    input_sizes: A `Tensor` of type `int32`.
      An integer vector representing the shape of `input`,
      where `input` is a 4-D `[batch, height, width, channels]` tensor.
    filter: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`.
      4-D with shape
      `[filter_height, filter_width, in_channels, out_channels]`.
    out_backprop: A `Tensor`. Must have the same type as `filter`.
      4-D with shape `[batch, out_height, out_width, out_channels]`.
      Gradients w.r.t. the output of the convolution.
    strides: A list of `ints`.
      The stride of the sliding window for each dimension of the input
      of the convolution. Must be in the same order as the dimension specified with
      format.
    padding: A `string` from: `"SAME", "VALID"`.
      The type of padding algorithm to use.
    use_cudnn_on_gpu: An optional `bool`. Defaults to `True`.
    data_format: An optional `string` from: `"NHWC", "NCHW"`. Defaults to `"NHWC"`.
      Specify the data format of the input and output data. With the
      default format "NHWC", the data is stored in the order of:
          [batch, in_height, in_width, in_channels].
      Alternatively, the format could be "NCHW", the data storage order of:
          [batch, in_channels, in_height, in_width].
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `filter`.
    4-D with shape `[batch, in_height, in_width, in_channels]`.  Gradient
    w.r.t. the input of the convolution.
  """
  result = _op_def_lib.apply_op("Conv2DBackpropInput",
                                input_sizes=input_sizes, filter=filter,
                                out_backprop=out_backprop, strides=strides,
                                padding=padding,
                                use_cudnn_on_gpu=use_cudnn_on_gpu,
                                data_format=data_format, name=name)
  return result


_conv3d_outputs = ["output"]


def conv3d(input, filter, strides, padding, name=None):
  r"""Computes a 3-D convolution given 5-D `input` and `filter` tensors.

  In signal processing, cross-correlation is a measure of similarity of
  two waveforms as a function of a time-lag applied to one of them. This
  is also known as a sliding dot product or sliding inner-product.

  Our Conv3D implements a form of cross-correlation.

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`, `complex128`, `qint8`, `quint8`, `qint32`, `half`.
      Shape `[batch, in_depth, in_height, in_width, in_channels]`.
    filter: A `Tensor`. Must have the same type as `input`.
      Shape `[filter_depth, filter_height, filter_width, in_channels, out_channels]`.
      `in_channels` must match between `input` and `filter`.
    strides: A list of `ints` that has length `>= 5`.
      1-D tensor of length 5. The stride of the sliding window for each
      dimension of `input`. Must have `strides[0] = strides[4] = 1`.
    padding: A `string` from: `"SAME", "VALID"`.
      The type of padding algorithm to use.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  result = _op_def_lib.apply_op("Conv3D", input=input, filter=filter,
                                strides=strides, padding=padding, name=name)
  return result


_conv3d_backprop_filter_outputs = ["output"]


def conv3d_backprop_filter(input, filter, out_backprop, strides, padding,
                           name=None):
  r"""Computes the gradients of 3D convolution with respect to the filter.

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`, `complex128`, `qint8`, `quint8`, `qint32`, `half`.
      Shape `[batch, depth, rows, cols, in_channels]`.
    filter: A `Tensor`. Must have the same type as `input`.
      Shape `[depth, rows, cols, in_channels, out_channels]`.
      `in_channels` must match between `input` and `filter`.
    out_backprop: A `Tensor`. Must have the same type as `input`.
      Backprop signal of shape `[batch, out_depth, out_rows, out_cols, out_channels]`.
    strides: A list of `ints` that has length `>= 5`.
      1-D tensor of length 5. The stride of the sliding window for each
      dimension of `input`. Must have `strides[0] = strides[4] = 1`.
    padding: A `string` from: `"SAME", "VALID"`.
      The type of padding algorithm to use.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  result = _op_def_lib.apply_op("Conv3DBackpropFilter", input=input,
                                filter=filter, out_backprop=out_backprop,
                                strides=strides, padding=padding, name=name)
  return result


_conv3d_backprop_input_outputs = ["output"]


def conv3d_backprop_input(input, filter, out_backprop, strides, padding,
                          name=None):
  r"""Computes the gradients of 3D convolution with respect to the input.

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`, `complex128`, `qint8`, `quint8`, `qint32`, `half`.
      Shape `[batch, depth, rows, cols, in_channels]`.
    filter: A `Tensor`. Must have the same type as `input`.
      Shape `[depth, rows, cols, in_channels, out_channels]`.
      `in_channels` must match between `input` and `filter`.
    out_backprop: A `Tensor`. Must have the same type as `input`.
      Backprop signal of shape `[batch, out_depth, out_rows, out_cols, out_channels]`.
    strides: A list of `ints` that has length `>= 5`.
      1-D tensor of length 5. The stride of the sliding window for each
      dimension of `input`. Must have `strides[0] = strides[4] = 1`.
    padding: A `string` from: `"SAME", "VALID"`.
      The type of padding algorithm to use.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  result = _op_def_lib.apply_op("Conv3DBackpropInput", input=input,
                                filter=filter, out_backprop=out_backprop,
                                strides=strides, padding=padding, name=name)
  return result


_depthwise_conv2d_native_outputs = ["output"]


def depthwise_conv2d_native(input, filter, strides, padding, name=None):
  r"""Computes a 2-D depthwise convolution given 4-D `input` and `filter` tensors.

  Given an input tensor of shape `[batch, in_height, in_width, in_channels]`
  and a filter / kernel tensor of shape
  `[filter_height, filter_width, in_channels, channel_multiplier]`, containing
  `in_channels` convolutional filters of depth 1, `depthwise_conv2d` applies
  a different filter to each input channel (expanding from 1 channel to
  `channel_multiplier` channels for each), then concatenates the results
  together. Thus, the output has `in_channels * channel_multiplier` channels.

  for k in 0..in_channels-1
    for q in 0..channel_multiplier-1
      output[b, i, j, k * channel_multiplier + q] =
        sum_{di, dj} input[b, strides[1] * i + di, strides[2] * j + dj, k] *
                          filter[di, dj, k, q]

  Must have `strides[0] = strides[3] = 1`.  For the most common case of the same
  horizontal and vertices strides, `strides = [1, stride, stride, 1]`.

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`.
    filter: A `Tensor`. Must have the same type as `input`.
    strides: A list of `ints`.
      1-D of length 4.  The stride of the sliding window for each dimension
      of `input`.
    padding: A `string` from: `"SAME", "VALID"`.
      The type of padding algorithm to use.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  result = _op_def_lib.apply_op("DepthwiseConv2dNative", input=input,
                                filter=filter, strides=strides,
                                padding=padding, name=name)
  return result


_depthwise_conv2d_native_backprop_filter_outputs = ["output"]


def depthwise_conv2d_native_backprop_filter(input, filter_sizes, out_backprop,
                                            strides, padding, name=None):
  r"""Computes the gradients of depthwise convolution with respect to the filter.

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`.
      4-D with shape `[batch, in_height, in_width, in_channels]`.
    filter_sizes: A `Tensor` of type `int32`.
      An integer vector representing the tensor shape of `filter`,
      where `filter` is a 4-D
      `[filter_height, filter_width, in_channels, depthwise_multiplier]` tensor.
    out_backprop: A `Tensor`. Must have the same type as `input`.
      4-D with shape `[batch, out_height, out_width, out_channels]`.
      Gradients w.r.t. the output of the convolution.
    strides: A list of `ints`.
      The stride of the sliding window for each dimension of the input
      of the convolution.
    padding: A `string` from: `"SAME", "VALID"`.
      The type of padding algorithm to use.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`. 4-D with shape
    `[filter_height, filter_width, in_channels, out_channels]`.  Gradient w.r.t.
    the `filter` input of the convolution.
  """
  result = _op_def_lib.apply_op("DepthwiseConv2dNativeBackpropFilter",
                                input=input, filter_sizes=filter_sizes,
                                out_backprop=out_backprop, strides=strides,
                                padding=padding, name=name)
  return result


_depthwise_conv2d_native_backprop_input_outputs = ["output"]


def depthwise_conv2d_native_backprop_input(input_sizes, filter, out_backprop,
                                           strides, padding, name=None):
  r"""Computes the gradients of depthwise convolution with respect to the input.

  Args:
    input_sizes: A `Tensor` of type `int32`.
      An integer vector representing the shape of `input`,
      where `input` is a 4-D `[batch, height, width, channels]` tensor.
    filter: A `Tensor`. Must be one of the following types: `float32`, `float64`.
      4-D with shape
      `[filter_height, filter_width, in_channels, depthwise_multiplier]`.
    out_backprop: A `Tensor`. Must have the same type as `filter`.
      4-D with shape `[batch, out_height, out_width, out_channels]`.
      Gradients w.r.t. the output of the convolution.
    strides: A list of `ints`.
      The stride of the sliding window for each dimension of the input
      of the convolution.
    padding: A `string` from: `"SAME", "VALID"`.
      The type of padding algorithm to use.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `filter`.
    4-D with shape `[batch, in_height, in_width, in_channels]`.  Gradient
    w.r.t. the input of the convolution.
  """
  result = _op_def_lib.apply_op("DepthwiseConv2dNativeBackpropInput",
                                input_sizes=input_sizes, filter=filter,
                                out_backprop=out_backprop, strides=strides,
                                padding=padding, name=name)
  return result


_dilation2d_outputs = ["output"]


def dilation2d(input, filter, strides, rates, padding, name=None):
  r"""Computes the grayscale dilation of 4-D `input` and 3-D `filter` tensors.

  The `input` tensor has shape `[batch, in_height, in_width, depth]` and the
  `filter` tensor has shape `[filter_height, filter_width, depth]`, i.e., each
  input channel is processed independently of the others with its own structuring
  function. The `output` tensor has shape
  `[batch, out_height, out_width, depth]`. The spatial dimensions of the output
  tensor depend on the `padding` algorithm. We currently only support the default
  "NHWC" `data_format`.

  In detail, the grayscale morphological 2-D dilation is the max-sum correlation
  (for consistency with `conv2d`, we use unmirrored filters):

      output[b, y, x, c] =
         max_{dy, dx} input[b,
                            strides[1] * y + rates[1] * dy,
                            strides[2] * x + rates[2] * dx,
                            c] +
                      filter[dy, dx, c]

  Max-pooling is a special case when the filter has size equal to the pooling
  kernel size and contains all zeros.

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`, `uint8`, `int16`, `int8`, `uint16`, `half`.
    filter: A `Tensor`. Must have the same type as `input`.
    strides: A list of `ints` that has length `>= 4`.
    rates: A list of `ints` that has length `>= 4`.
    padding: A `string` from: `"SAME", "VALID"`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  result = _op_def_lib.apply_op("Dilation2D", input=input, filter=filter,
                                strides=strides, rates=rates, padding=padding,
                                name=name)
  return result


_dilation2d_backprop_filter_outputs = ["filter_backprop"]


def dilation2d_backprop_filter(input, filter, out_backprop, strides, rates,
                               padding, name=None):
  r"""Computes the gradient of morphological 2-D dilation with respect to the filter.

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`, `uint8`, `int16`, `int8`, `uint16`, `half`.
      4-D with shape `[batch, in_height, in_width, depth]`.
    filter: A `Tensor`. Must have the same type as `input`.
      3-D with shape `[filter_height, filter_width, depth]`.
    out_backprop: A `Tensor`. Must have the same type as `input`.
      4-D with shape `[batch, out_height, out_width, depth]`.
    strides: A list of `ints` that has length `>= 4`.
      1-D of length 4. The stride of the sliding window for each dimension of
      the input tensor. Must be: `[1, stride_height, stride_width, 1]`.
    rates: A list of `ints` that has length `>= 4`.
      1-D of length 4. The input stride for atrous morphological dilation.
      Must be: `[1, rate_height, rate_width, 1]`.
    padding: A `string` from: `"SAME", "VALID"`.
      The type of padding algorithm to use.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
    3-D with shape `[filter_height, filter_width, depth]`.
  """
  result = _op_def_lib.apply_op("Dilation2DBackpropFilter", input=input,
                                filter=filter, out_backprop=out_backprop,
                                strides=strides, rates=rates, padding=padding,
                                name=name)
  return result


_dilation2d_backprop_input_outputs = ["in_backprop"]


def dilation2d_backprop_input(input, filter, out_backprop, strides, rates,
                              padding, name=None):
  r"""Computes the gradient of morphological 2-D dilation with respect to the input.

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`, `uint8`, `int16`, `int8`, `uint16`, `half`.
      4-D with shape `[batch, in_height, in_width, depth]`.
    filter: A `Tensor`. Must have the same type as `input`.
      3-D with shape `[filter_height, filter_width, depth]`.
    out_backprop: A `Tensor`. Must have the same type as `input`.
      4-D with shape `[batch, out_height, out_width, depth]`.
    strides: A list of `ints` that has length `>= 4`.
      1-D of length 4. The stride of the sliding window for each dimension of
      the input tensor. Must be: `[1, stride_height, stride_width, 1]`.
    rates: A list of `ints` that has length `>= 4`.
      1-D of length 4. The input stride for atrous morphological dilation.
      Must be: `[1, rate_height, rate_width, 1]`.
    padding: A `string` from: `"SAME", "VALID"`.
      The type of padding algorithm to use.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
    4-D with shape `[batch, in_height, in_width, depth]`.
  """
  result = _op_def_lib.apply_op("Dilation2DBackpropInput", input=input,
                                filter=filter, out_backprop=out_backprop,
                                strides=strides, rates=rates, padding=padding,
                                name=name)
  return result


_elu_outputs = ["activations"]


def elu(features, name=None):
  r"""Computes exponential linear: `exp(features) - 1` if < 0, `features` otherwise.

  See [Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)
  ](http://arxiv.org/abs/1511.07289)

  Args:
    features: A `Tensor`. Must be one of the following types: `float32`, `float64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `features`.
  """
  result = _op_def_lib.apply_op("Elu", features=features, name=name)
  return result


__elu_grad_outputs = ["backprops"]


def _elu_grad(gradients, outputs, name=None):
  r"""Computes gradients for the exponential linear (Elu) operation.

  Args:
    gradients: A `Tensor`. Must be one of the following types: `float32`, `float64`.
      The backpropagated gradients to the corresponding Elu operation.
    outputs: A `Tensor`. Must have the same type as `gradients`.
      The outputs of the corresponding Elu operation.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `gradients`.
    The gradients: `gradients * (outputs + 1)` if outputs < 0,
    `gradients` otherwise.
  """
  result = _op_def_lib.apply_op("EluGrad", gradients=gradients,
                                outputs=outputs, name=name)
  return result


_in_top_k_outputs = ["precision"]


def in_top_k(predictions, targets, k, name=None):
  r"""Says whether the targets are in the top `K` predictions.

  This outputs a `batch_size` bool array, an entry `out[i]` is `true` if the
  prediction for the target class is among the top `k` predictions among
  all predictions for example `i`. Note that the behavior of `InTopK` differs
  from the `TopK` op in its handling of ties; if multiple classes have the
  same prediction value and straddle the top-`k` boundary, all of those
  classes are considered to be in the top `k`.

  More formally, let

    \\(predictions_i\\) be the predictions for all classes for example `i`,
    \\(targets_i\\) be the target class for example `i`,
    \\(out_i\\) be the output for example `i`,

  $$out_i = predictions_{i, targets_i} \in TopKIncludingTies(predictions_i)$$

  Args:
    predictions: A `Tensor` of type `float32`.
      A `batch_size` x `classes` tensor.
    targets: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A `batch_size` vector of class ids.
    k: An `int`. Number of top elements to look at for computing precision.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `bool`. Computed Precision at `k` as a `bool Tensor`.
  """
  result = _op_def_lib.apply_op("InTopK", predictions=predictions,
                                targets=targets, k=k, name=name)
  return result


_l2_loss_outputs = ["output"]


def l2_loss(t, name=None):
  r"""L2 Loss.

  Computes half the L2 norm of a tensor without the `sqrt`:

      output = sum(t ** 2) / 2

  Args:
    t: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`, `complex128`, `qint8`, `quint8`, `qint32`, `half`.
      Typically 2-D, but may have any dimensions.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `t`. 0-D.
  """
  result = _op_def_lib.apply_op("L2Loss", t=t, name=name)
  return result


_lrn_outputs = ["output"]


def lrn(input, depth_radius=None, bias=None, alpha=None, beta=None,
        name=None):
  r"""Local Response Normalization.

  The 4-D `input` tensor is treated as a 3-D array of 1-D vectors (along the last
  dimension), and each vector is normalized independently.  Within a given vector,
  each component is divided by the weighted, squared sum of inputs within
  `depth_radius`.  In detail,

      sqr_sum[a, b, c, d] =
          sum(input[a, b, c, d - depth_radius : d + depth_radius + 1] ** 2)
      output = input / (bias + alpha * sqr_sum) ** beta

  For details, see [Krizhevsky et al., ImageNet classification with deep
  convolutional neural networks (NIPS 2012)]
  (http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks).

  Args:
    input: A `Tensor` of type `float32`. 4-D.
    depth_radius: An optional `int`. Defaults to `5`.
      0-D.  Half-width of the 1-D normalization window.
    bias: An optional `float`. Defaults to `1`.
      An offset (usually positive to avoid dividing by 0).
    alpha: An optional `float`. Defaults to `1`.
      A scale factor, usually positive.
    beta: An optional `float`. Defaults to `0.5`. An exponent.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32`.
  """
  result = _op_def_lib.apply_op("LRN", input=input, depth_radius=depth_radius,
                                bias=bias, alpha=alpha, beta=beta, name=name)
  return result


__lrn_grad_outputs = ["output"]


def _lrn_grad(input_grads, input_image, output_image, depth_radius=None,
              bias=None, alpha=None, beta=None, name=None):
  r"""Gradients for Local Response Normalization.

  Args:
    input_grads: A `Tensor` of type `float32`.
      4-D with shape `[batch, height, width, channels]`.
    input_image: A `Tensor` of type `float32`.
      4-D with shape `[batch, height, width, channels]`.
    output_image: A `Tensor` of type `float32`.
      4-D with shape `[batch, height, width, channels]`.
    depth_radius: An optional `int`. Defaults to `5`. A depth radius.
    bias: An optional `float`. Defaults to `1`.
      An offset (usually > 0 to avoid dividing by 0).
    alpha: An optional `float`. Defaults to `1`.
      A scale factor, usually positive.
    beta: An optional `float`. Defaults to `0.5`. An exponent.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32`. The gradients for LRN.
  """
  result = _op_def_lib.apply_op("LRNGrad", input_grads=input_grads,
                                input_image=input_image,
                                output_image=output_image,
                                depth_radius=depth_radius, bias=bias,
                                alpha=alpha, beta=beta, name=name)
  return result


_log_softmax_outputs = ["logsoftmax"]


def log_softmax(logits, name=None):
  r"""Computes log softmax activations.

  For each batch `i` and class `j` we have

      logsoftmax[i, j] = logits[i, j] - log(sum(exp(logits[i])))

  Args:
    logits: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`.
      2-D with shape `[batch_size, num_classes]`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `logits`. Same shape as `logits`.
  """
  result = _op_def_lib.apply_op("LogSoftmax", logits=logits, name=name)
  return result


__max_pool_outputs = ["output"]


def _max_pool(input, ksize, strides, padding, data_format=None, name=None):
  r"""Performs max pooling on the input.

  Args:
    input: A `Tensor` of type `float32`. 4-D input to pool over.
    ksize: A list of `ints` that has length `>= 4`.
      The size of the window for each dimension of the input tensor.
    strides: A list of `ints` that has length `>= 4`.
      The stride of the sliding window for each dimension of the
      input tensor.
    padding: A `string` from: `"SAME", "VALID"`.
      The type of padding algorithm to use.
    data_format: An optional `string` from: `"NHWC", "NCHW"`. Defaults to `"NHWC"`.
      Specify the data format of the input and output data. With the
      default format "NHWC", the data is stored in the order of:
          [batch, in_height, in_width, in_channels].
      Alternatively, the format could be "NCHW", the data storage order of:
          [batch, in_channels, in_height, in_width].
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32`. The max pooled output tensor.
  """
  result = _op_def_lib.apply_op("MaxPool", input=input, ksize=ksize,
                                strides=strides, padding=padding,
                                data_format=data_format, name=name)
  return result


_max_pool3d_outputs = ["output"]


def max_pool3d(input, ksize, strides, padding, name=None):
  r"""Performs 3D max pooling on the input.

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`, `complex128`, `qint8`, `quint8`, `qint32`, `half`.
      Shape `[batch, depth, rows, cols, channels]` tensor to pool over.
    ksize: A list of `ints` that has length `>= 5`.
      1-D tensor of length 5. The size of the window for each dimension of
      the input tensor. Must have `ksize[0] = ksize[1] = 1`.
    strides: A list of `ints` that has length `>= 5`.
      1-D tensor of length 5. The stride of the sliding window for each
      dimension of `input`. Must have `strides[0] = strides[4] = 1`.
    padding: A `string` from: `"SAME", "VALID"`.
      The type of padding algorithm to use.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`. The max pooled output tensor.
  """
  result = _op_def_lib.apply_op("MaxPool3D", input=input, ksize=ksize,
                                strides=strides, padding=padding, name=name)
  return result


_max_pool3d_grad_outputs = ["output"]


def max_pool3d_grad(orig_input, orig_output, grad, ksize, strides, padding,
                    name=None):
  r"""Computes gradients of max pooling function.

  Args:
    orig_input: A `Tensor` of type `float32`. The original input tensor.
    orig_output: A `Tensor` of type `float32`. The original output tensor.
    grad: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`, `complex128`, `qint8`, `quint8`, `qint32`, `half`.
      Output backprop of shape `[batch, depth, rows, cols, channels]`.
    ksize: A list of `ints` that has length `>= 5`.
      1-D tensor of length 5. The size of the window for each dimension of
      the input tensor. Must have `ksize[0] = ksize[1] = 1`.
    strides: A list of `ints` that has length `>= 5`.
      1-D tensor of length 5. The stride of the sliding window for each
      dimension of `input`. Must have `strides[0] = strides[4] = 1`.
    padding: A `string` from: `"SAME", "VALID"`.
      The type of padding algorithm to use.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `grad`.
  """
  result = _op_def_lib.apply_op("MaxPool3DGrad", orig_input=orig_input,
                                orig_output=orig_output, grad=grad,
                                ksize=ksize, strides=strides, padding=padding,
                                name=name)
  return result


__max_pool_grad_outputs = ["output"]


def _max_pool_grad(orig_input, orig_output, grad, ksize, strides, padding,
                   data_format=None, name=None):
  r"""Computes gradients of the maxpooling function.

  Args:
    orig_input: A `Tensor` of type `float32`. The original input tensor.
    orig_output: A `Tensor` of type `float32`. The original output tensor.
    grad: A `Tensor` of type `float32`.
      4-D.  Gradients w.r.t. the output of `max_pool`.
    ksize: A list of `ints` that has length `>= 4`.
      The size of the window for each dimension of the input tensor.
    strides: A list of `ints` that has length `>= 4`.
      The stride of the sliding window for each dimension of the
      input tensor.
    padding: A `string` from: `"SAME", "VALID"`.
      The type of padding algorithm to use.
    data_format: An optional `string` from: `"NHWC", "NCHW"`. Defaults to `"NHWC"`.
      Specify the data format of the input and output data. With the
      default format "NHWC", the data is stored in the order of:
          [batch, in_height, in_width, in_channels].
      Alternatively, the format could be "NCHW", the data storage order of:
          [batch, in_channels, in_height, in_width].
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32`. Gradients w.r.t. the input to `max_pool`.
  """
  result = _op_def_lib.apply_op("MaxPoolGrad", orig_input=orig_input,
                                orig_output=orig_output, grad=grad,
                                ksize=ksize, strides=strides, padding=padding,
                                data_format=data_format, name=name)
  return result


__max_pool_grad_with_argmax_outputs = ["output"]


def _max_pool_grad_with_argmax(input, grad, argmax, ksize, strides, padding,
                               name=None):
  r"""Computes gradients of the maxpooling function.

  Args:
    input: A `Tensor` of type `float32`. The original input.
    grad: A `Tensor` of type `float32`.
      4-D with shape `[batch, height, width, channels]`.  Gradients w.r.t. the
      output of `max_pool`.
    argmax: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      The indices of the maximum values chosen for each output of `max_pool`.
    ksize: A list of `ints` that has length `>= 4`.
      The size of the window for each dimension of the input tensor.
    strides: A list of `ints` that has length `>= 4`.
      The stride of the sliding window for each dimension of the
      input tensor.
    padding: A `string` from: `"SAME", "VALID"`.
      The type of padding algorithm to use.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32`. Gradients w.r.t. the input of `max_pool`.
  """
  result = _op_def_lib.apply_op("MaxPoolGradWithArgmax", input=input,
                                grad=grad, argmax=argmax, ksize=ksize,
                                strides=strides, padding=padding, name=name)
  return result


_max_pool_with_argmax_outputs = ["output", "argmax"]


_MaxPoolWithArgmaxOutput = collections.namedtuple("MaxPoolWithArgmax",
                                                  _max_pool_with_argmax_outputs)


def max_pool_with_argmax(input, ksize, strides, padding, Targmax=None,
                         name=None):
  r"""Performs max pooling on the input and outputs both max values and indices.

  The indices in `argmax` are flattened, so that a maximum value at position
  `[b, y, x, c]` becomes flattened index
  `((b * height + y) * width + x) * channels + c`.

  Args:
    input: A `Tensor` of type `float32`.
      4-D with shape `[batch, height, width, channels]`.  Input to pool over.
    ksize: A list of `ints` that has length `>= 4`.
      The size of the window for each dimension of the input tensor.
    strides: A list of `ints` that has length `>= 4`.
      The stride of the sliding window for each dimension of the
      input tensor.
    padding: A `string` from: `"SAME", "VALID"`.
      The type of padding algorithm to use.
    Targmax: An optional `tf.DType` from: `tf.int32, tf.int64`. Defaults to `tf.int64`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output, argmax).
    output: A `Tensor` of type `float32`. The max pooled output tensor.
    argmax: A `Tensor` of type `Targmax`. 4-D.  The flattened indices of the max values chosen for each output.
  """
  result = _op_def_lib.apply_op("MaxPoolWithArgmax", input=input, ksize=ksize,
                                strides=strides, padding=padding,
                                Targmax=Targmax, name=name)
  return _MaxPoolWithArgmaxOutput._make(result)


_relu_outputs = ["activations"]


def relu(features, name=None):
  r"""Computes rectified linear: `max(features, 0)`.

  Args:
    features: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`, `uint8`, `int16`, `int8`, `uint16`, `half`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `features`.
  """
  result = _op_def_lib.apply_op("Relu", features=features, name=name)
  return result


__relu6_outputs = ["activations"]


def _relu6(features, name=None):
  r"""Computes rectified linear 6: `min(max(features, 0), 6)`.

  Args:
    features: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`, `uint8`, `int16`, `int8`, `uint16`, `half`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `features`.
  """
  result = _op_def_lib.apply_op("Relu6", features=features, name=name)
  return result


__relu6_grad_outputs = ["backprops"]


def _relu6_grad(gradients, features, name=None):
  r"""Computes rectified linear 6 gradients for a Relu6 operation.

  Args:
    gradients: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`, `uint8`, `int16`, `int8`, `uint16`, `half`.
      The backpropagated gradients to the corresponding Relu6 operation.
    features: A `Tensor`. Must have the same type as `gradients`.
      The features passed as input to the corresponding Relu6 operation.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `gradients`. The gradients:
    `gradients * features * (features > 0) * (features < 6)`.
  """
  result = _op_def_lib.apply_op("Relu6Grad", gradients=gradients,
                                features=features, name=name)
  return result


__relu_grad_outputs = ["backprops"]


def _relu_grad(gradients, features, name=None):
  r"""Computes rectified linear gradients for a Relu operation.

  Args:
    gradients: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`, `uint8`, `int16`, `int8`, `uint16`, `half`.
      The backpropagated gradients to the corresponding Relu operation.
    features: A `Tensor`. Must have the same type as `gradients`.
      The features passed as input to the corresponding Relu operation, OR
      the outputs of that operation (both work equivalently).
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `gradients`.
    `gradients * (features > 0)`.
  """
  result = _op_def_lib.apply_op("ReluGrad", gradients=gradients,
                                features=features, name=name)
  return result


_softmax_outputs = ["softmax"]


def softmax(logits, name=None):
  r"""Computes softmax activations.

  For each batch `i` and class `j` we have

      softmax[i, j] = exp(logits[i, j]) / sum(exp(logits[i]))

  Args:
    logits: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`.
      2-D with shape `[batch_size, num_classes]`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `logits`. Same shape as `logits`.
  """
  result = _op_def_lib.apply_op("Softmax", logits=logits, name=name)
  return result


__softmax_cross_entropy_with_logits_outputs = ["loss", "backprop"]


_SoftmaxCrossEntropyWithLogitsOutput = collections.namedtuple("SoftmaxCrossEntropyWithLogits",
                                                              __softmax_cross_entropy_with_logits_outputs)


def _softmax_cross_entropy_with_logits(features, labels, name=None):
  r"""Computes softmax cross entropy cost and gradients to backpropagate.

  Inputs are the logits, not probabilities.

  Args:
    features: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`.
      batch_size x num_classes matrix
    labels: A `Tensor`. Must have the same type as `features`.
      batch_size x num_classes matrix
      The caller must ensure that each batch of labels represents a valid
      probability distribution.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (loss, backprop).
    loss: A `Tensor`. Has the same type as `features`. Per example loss (batch_size vector).
    backprop: A `Tensor`. Has the same type as `features`. backpropagated gradients (batch_size x num_classes matrix).
  """
  result = _op_def_lib.apply_op("SoftmaxCrossEntropyWithLogits",
                                features=features, labels=labels, name=name)
  return _SoftmaxCrossEntropyWithLogitsOutput._make(result)


_softplus_outputs = ["activations"]


def softplus(features, name=None):
  r"""Computes softplus: `log(exp(features) + 1)`.

  Args:
    features: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`, `uint8`, `int16`, `int8`, `uint16`, `half`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `features`.
  """
  result = _op_def_lib.apply_op("Softplus", features=features, name=name)
  return result


__softplus_grad_outputs = ["backprops"]


def _softplus_grad(gradients, features, name=None):
  r"""Computes softplus gradients for a softplus operation.

  Args:
    gradients: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`, `uint8`, `int16`, `int8`, `uint16`, `half`.
      The backpropagated gradients to the corresponding softplus operation.
    features: A `Tensor`. Must have the same type as `gradients`.
      The features passed as input to the corresponding softplus operation.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `gradients`.
    The gradients: `gradients / (1 + exp(-features))`.
  """
  result = _op_def_lib.apply_op("SoftplusGrad", gradients=gradients,
                                features=features, name=name)
  return result


_softsign_outputs = ["activations"]


def softsign(features, name=None):
  r"""Computes softsign: `features / (abs(features) + 1)`.

  Args:
    features: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`, `uint8`, `int16`, `int8`, `uint16`, `half`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `features`.
  """
  result = _op_def_lib.apply_op("Softsign", features=features, name=name)
  return result


__softsign_grad_outputs = ["backprops"]


def _softsign_grad(gradients, features, name=None):
  r"""Computes softsign gradients for a softsign operation.

  Args:
    gradients: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`, `uint8`, `int16`, `int8`, `uint16`, `half`.
      The backpropagated gradients to the corresponding softsign operation.
    features: A `Tensor`. Must have the same type as `gradients`.
      The features passed as input to the corresponding softsign operation.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `gradients`.
    The gradients: `gradients / (1 + abs(-features)) ** 2`.
  """
  result = _op_def_lib.apply_op("SoftsignGrad", gradients=gradients,
                                features=features, name=name)
  return result


__sparse_softmax_cross_entropy_with_logits_outputs = ["loss", "backprop"]


_SparseSoftmaxCrossEntropyWithLogitsOutput = collections.namedtuple("SparseSoftmaxCrossEntropyWithLogits",
                                                                    __sparse_softmax_cross_entropy_with_logits_outputs)


def _sparse_softmax_cross_entropy_with_logits(features, labels, name=None):
  r"""Computes softmax cross entropy cost and gradients to backpropagate.

  Unlike `SoftmaxCrossEntropyWithLogits`, this operation does not accept
  a matrix of label probabilities, but rather a single label per row
  of features.  This label is considered to have probability 1.0 for the
  given row.

  Inputs are the logits, not probabilities.

  Args:
    features: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`.
      batch_size x num_classes matrix
    labels: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      batch_size vector with values in [0, num_classes).
      This is the label for the given minibatch entry.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (loss, backprop).
    loss: A `Tensor`. Has the same type as `features`. Per example loss (batch_size vector).
    backprop: A `Tensor`. Has the same type as `features`. backpropagated gradients (batch_size x num_classes matrix).
  """
  result = _op_def_lib.apply_op("SparseSoftmaxCrossEntropyWithLogits",
                                features=features, labels=labels, name=name)
  return _SparseSoftmaxCrossEntropyWithLogitsOutput._make(result)


__top_k_outputs = ["values", "indices"]


_TopKOutput = collections.namedtuple("TopK", __top_k_outputs)


def _top_k(input, k, sorted=None, name=None):
  r"""Finds values and indices of the `k` largest elements for the last dimension.

  If the input is a vector (rank-1), finds the `k` largest entries in the vector
  and outputs their values and indices as vectors.  Thus `values[j]` is the
  `j`-th largest entry in `input`, and its index is `indices[j]`.

  For matrices (resp. higher rank input), computes the top `k` entries in each
  row (resp. vector along the last dimension).  Thus,

      values.shape = indices.shape = input.shape[:-1] + [k]

  If two elements are equal, the lower-index element appears first.

  If `k` varies dynamically, use `TopKV2` below.

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`, `uint8`, `int16`, `int8`, `uint16`, `half`.
      1-D or higher with last dimension at least `k`.
    k: An `int` that is `>= 0`.
      Number of top elements to look for along the last dimension (along each
      row for matrices).
    sorted: An optional `bool`. Defaults to `True`.
      If true the resulting `k` elements will be sorted by the values in
      descending order.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (values, indices).
    values: A `Tensor`. Has the same type as `input`. The `k` largest elements along each last dimensional slice.
    indices: A `Tensor` of type `int32`. The indices of `values` within the last dimension of `input`.
  """
  result = _op_def_lib.apply_op("TopK", input=input, k=k, sorted=sorted,
                                name=name)
  return _TopKOutput._make(result)


__top_kv2_outputs = ["values", "indices"]


_TopKV2Output = collections.namedtuple("TopKV2", __top_kv2_outputs)


def _top_kv2(input, k, sorted=None, name=None):
  r"""Finds values and indices of the `k` largest elements for the last dimension.

  If the input is a vector (rank-1), finds the `k` largest entries in the vector
  and outputs their values and indices as vectors.  Thus `values[j]` is the
  `j`-th largest entry in `input`, and its index is `indices[j]`.

  For matrices (resp. higher rank input), computes the top `k` entries in each
  row (resp. vector along the last dimension).  Thus,

      values.shape = indices.shape = input.shape[:-1] + [k]

  If two elements are equal, the lower-index element appears first.

  This is the same as `TopK`, but takes `k` as in input rather than an attr.

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`, `uint8`, `int16`, `int8`, `uint16`, `half`.
      1-D or higher with last dimension at least `k`.
    k: A `Tensor` of type `int32`.
      0-D.  Number of top elements to look for along the last dimension (along each
      row for matrices).
    sorted: An optional `bool`. Defaults to `True`.
      If true the resulting `k` elements will be sorted by the values in
      descending order.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (values, indices).
    values: A `Tensor`. Has the same type as `input`. The `k` largest elements along each last dimensional slice.
    indices: A `Tensor` of type `int32`. The indices of `values` within the last dimension of `input`.
  """
  result = _op_def_lib.apply_op("TopKV2", input=input, k=k, sorted=sorted,
                                name=name)
  return _TopKV2Output._make(result)


def _InitOpDefLibrary():
  op_list = op_def_pb2.OpList()
  text_format.Merge(_InitOpDefLibrary.op_list_ascii, op_list)
  op_def_registry.register_op_list(op_list)
  op_def_lib = op_def_library.OpDefLibrary()
  op_def_lib.add_op_list(op_list)
  return op_def_lib


_InitOpDefLibrary.op_list_ascii = """op {
  name: "AvgPool"
  input_arg {
    name: "value"
    type_attr: "T"
  }
  output_arg {
    name: "output"
    type_attr: "T"
  }
  attr {
    name: "ksize"
    type: "list(int)"
    has_minimum: true
    minimum: 4
  }
  attr {
    name: "strides"
    type: "list(int)"
    has_minimum: true
    minimum: 4
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
  attr {
    name: "data_format"
    type: "string"
    default_value {
      s: "NHWC"
    }
    allowed_values {
      list {
        s: "NHWC"
        s: "NCHW"
      }
    }
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
      }
    }
  }
}
op {
  name: "AvgPool3D"
  input_arg {
    name: "input"
    type_attr: "T"
  }
  output_arg {
    name: "output"
    type_attr: "T"
  }
  attr {
    name: "ksize"
    type: "list(int)"
    has_minimum: true
    minimum: 5
  }
  attr {
    name: "strides"
    type: "list(int)"
    has_minimum: true
    minimum: 5
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
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT64
        type: DT_INT32
        type: DT_UINT8
        type: DT_UINT16
        type: DT_INT16
        type: DT_INT8
        type: DT_COMPLEX64
        type: DT_COMPLEX128
        type: DT_QINT8
        type: DT_QUINT8
        type: DT_QINT32
        type: DT_HALF
      }
    }
  }
}
op {
  name: "AvgPool3DGrad"
  input_arg {
    name: "orig_input_shape"
    type: DT_INT32
  }
  input_arg {
    name: "grad"
    type_attr: "T"
  }
  output_arg {
    name: "output"
    type_attr: "T"
  }
  attr {
    name: "ksize"
    type: "list(int)"
    has_minimum: true
    minimum: 5
  }
  attr {
    name: "strides"
    type: "list(int)"
    has_minimum: true
    minimum: 5
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
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT64
        type: DT_INT32
        type: DT_UINT8
        type: DT_UINT16
        type: DT_INT16
        type: DT_INT8
        type: DT_COMPLEX64
        type: DT_COMPLEX128
        type: DT_QINT8
        type: DT_QUINT8
        type: DT_QINT32
        type: DT_HALF
      }
    }
  }
}
op {
  name: "AvgPoolGrad"
  input_arg {
    name: "orig_input_shape"
    type: DT_INT32
  }
  input_arg {
    name: "grad"
    type_attr: "T"
  }
  output_arg {
    name: "output"
    type_attr: "T"
  }
  attr {
    name: "ksize"
    type: "list(int)"
    has_minimum: true
    minimum: 4
  }
  attr {
    name: "strides"
    type: "list(int)"
    has_minimum: true
    minimum: 4
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
  attr {
    name: "data_format"
    type: "string"
    default_value {
      s: "NHWC"
    }
    allowed_values {
      list {
        s: "NHWC"
        s: "NCHW"
      }
    }
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
      }
    }
  }
}
op {
  name: "BatchNormWithGlobalNormalization"
  input_arg {
    name: "t"
    type_attr: "T"
  }
  input_arg {
    name: "m"
    type_attr: "T"
  }
  input_arg {
    name: "v"
    type_attr: "T"
  }
  input_arg {
    name: "beta"
    type_attr: "T"
  }
  input_arg {
    name: "gamma"
    type_attr: "T"
  }
  output_arg {
    name: "result"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT64
        type: DT_INT32
        type: DT_UINT8
        type: DT_UINT16
        type: DT_INT16
        type: DT_INT8
        type: DT_COMPLEX64
        type: DT_COMPLEX128
        type: DT_QINT8
        type: DT_QUINT8
        type: DT_QINT32
        type: DT_HALF
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
  deprecation {
    version: 9
    explanation: "Use tf.nn.batch_normalization()"
  }
}
op {
  name: "BatchNormWithGlobalNormalizationGrad"
  input_arg {
    name: "t"
    type_attr: "T"
  }
  input_arg {
    name: "m"
    type_attr: "T"
  }
  input_arg {
    name: "v"
    type_attr: "T"
  }
  input_arg {
    name: "gamma"
    type_attr: "T"
  }
  input_arg {
    name: "backprop"
    type_attr: "T"
  }
  output_arg {
    name: "dx"
    type_attr: "T"
  }
  output_arg {
    name: "dm"
    type_attr: "T"
  }
  output_arg {
    name: "dv"
    type_attr: "T"
  }
  output_arg {
    name: "db"
    type_attr: "T"
  }
  output_arg {
    name: "dg"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT64
        type: DT_INT32
        type: DT_UINT8
        type: DT_UINT16
        type: DT_INT16
        type: DT_INT8
        type: DT_COMPLEX64
        type: DT_COMPLEX128
        type: DT_QINT8
        type: DT_QUINT8
        type: DT_QINT32
        type: DT_HALF
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
  deprecation {
    version: 9
    explanation: "Use tf.nn.batch_normalization()"
  }
}
op {
  name: "BiasAdd"
  input_arg {
    name: "value"
    type_attr: "T"
  }
  input_arg {
    name: "bias"
    type_attr: "T"
  }
  output_arg {
    name: "output"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT64
        type: DT_INT32
        type: DT_UINT8
        type: DT_UINT16
        type: DT_INT16
        type: DT_INT8
        type: DT_COMPLEX64
        type: DT_COMPLEX128
        type: DT_QINT8
        type: DT_QUINT8
        type: DT_QINT32
        type: DT_HALF
      }
    }
  }
  attr {
    name: "data_format"
    type: "string"
    default_value {
      s: "NHWC"
    }
    allowed_values {
      list {
        s: "NHWC"
        s: "NCHW"
      }
    }
  }
}
op {
  name: "BiasAddGrad"
  input_arg {
    name: "out_backprop"
    type_attr: "T"
  }
  output_arg {
    name: "output"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT64
        type: DT_INT32
        type: DT_UINT8
        type: DT_UINT16
        type: DT_INT16
        type: DT_INT8
        type: DT_COMPLEX64
        type: DT_COMPLEX128
        type: DT_QINT8
        type: DT_QUINT8
        type: DT_QINT32
        type: DT_HALF
      }
    }
  }
  attr {
    name: "data_format"
    type: "string"
    default_value {
      s: "NHWC"
    }
    allowed_values {
      list {
        s: "NHWC"
        s: "NCHW"
      }
    }
  }
}
op {
  name: "BiasAddV1"
  input_arg {
    name: "value"
    type_attr: "T"
  }
  input_arg {
    name: "bias"
    type_attr: "T"
  }
  output_arg {
    name: "output"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT64
        type: DT_INT32
        type: DT_UINT8
        type: DT_UINT16
        type: DT_INT16
        type: DT_INT8
        type: DT_COMPLEX64
        type: DT_COMPLEX128
        type: DT_QINT8
        type: DT_QUINT8
        type: DT_QINT32
        type: DT_HALF
      }
    }
  }
}
op {
  name: "Conv2D"
  input_arg {
    name: "input"
    type_attr: "T"
  }
  input_arg {
    name: "filter"
    type_attr: "T"
  }
  output_arg {
    name: "output"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_HALF
        type: DT_FLOAT
        type: DT_DOUBLE
      }
    }
  }
  attr {
    name: "strides"
    type: "list(int)"
  }
  attr {
    name: "use_cudnn_on_gpu"
    type: "bool"
    default_value {
      b: true
    }
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
  attr {
    name: "data_format"
    type: "string"
    default_value {
      s: "NHWC"
    }
    allowed_values {
      list {
        s: "NHWC"
        s: "NCHW"
      }
    }
  }
}
op {
  name: "Conv2DBackpropFilter"
  input_arg {
    name: "input"
    type_attr: "T"
  }
  input_arg {
    name: "filter_sizes"
    type: DT_INT32
  }
  input_arg {
    name: "out_backprop"
    type_attr: "T"
  }
  output_arg {
    name: "output"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_HALF
        type: DT_FLOAT
        type: DT_DOUBLE
      }
    }
  }
  attr {
    name: "strides"
    type: "list(int)"
  }
  attr {
    name: "use_cudnn_on_gpu"
    type: "bool"
    default_value {
      b: true
    }
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
  attr {
    name: "data_format"
    type: "string"
    default_value {
      s: "NHWC"
    }
    allowed_values {
      list {
        s: "NHWC"
        s: "NCHW"
      }
    }
  }
}
op {
  name: "Conv2DBackpropInput"
  input_arg {
    name: "input_sizes"
    type: DT_INT32
  }
  input_arg {
    name: "filter"
    type_attr: "T"
  }
  input_arg {
    name: "out_backprop"
    type_attr: "T"
  }
  output_arg {
    name: "output"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_HALF
        type: DT_FLOAT
        type: DT_DOUBLE
      }
    }
  }
  attr {
    name: "strides"
    type: "list(int)"
  }
  attr {
    name: "use_cudnn_on_gpu"
    type: "bool"
    default_value {
      b: true
    }
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
  attr {
    name: "data_format"
    type: "string"
    default_value {
      s: "NHWC"
    }
    allowed_values {
      list {
        s: "NHWC"
        s: "NCHW"
      }
    }
  }
}
op {
  name: "Conv3D"
  input_arg {
    name: "input"
    type_attr: "T"
  }
  input_arg {
    name: "filter"
    type_attr: "T"
  }
  output_arg {
    name: "output"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT64
        type: DT_INT32
        type: DT_UINT8
        type: DT_UINT16
        type: DT_INT16
        type: DT_INT8
        type: DT_COMPLEX64
        type: DT_COMPLEX128
        type: DT_QINT8
        type: DT_QUINT8
        type: DT_QINT32
        type: DT_HALF
      }
    }
  }
  attr {
    name: "strides"
    type: "list(int)"
    has_minimum: true
    minimum: 5
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
  name: "Conv3DBackpropFilter"
  input_arg {
    name: "input"
    type_attr: "T"
  }
  input_arg {
    name: "filter"
    type_attr: "T"
  }
  input_arg {
    name: "out_backprop"
    type_attr: "T"
  }
  output_arg {
    name: "output"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT64
        type: DT_INT32
        type: DT_UINT8
        type: DT_UINT16
        type: DT_INT16
        type: DT_INT8
        type: DT_COMPLEX64
        type: DT_COMPLEX128
        type: DT_QINT8
        type: DT_QUINT8
        type: DT_QINT32
        type: DT_HALF
      }
    }
  }
  attr {
    name: "strides"
    type: "list(int)"
    has_minimum: true
    minimum: 5
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
  name: "Conv3DBackpropInput"
  input_arg {
    name: "input"
    type_attr: "T"
  }
  input_arg {
    name: "filter"
    type_attr: "T"
  }
  input_arg {
    name: "out_backprop"
    type_attr: "T"
  }
  output_arg {
    name: "output"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT64
        type: DT_INT32
        type: DT_UINT8
        type: DT_UINT16
        type: DT_INT16
        type: DT_INT8
        type: DT_COMPLEX64
        type: DT_COMPLEX128
        type: DT_QINT8
        type: DT_QUINT8
        type: DT_QINT32
        type: DT_HALF
      }
    }
  }
  attr {
    name: "strides"
    type: "list(int)"
    has_minimum: true
    minimum: 5
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
  name: "DepthwiseConv2dNative"
  input_arg {
    name: "input"
    type_attr: "T"
  }
  input_arg {
    name: "filter"
    type_attr: "T"
  }
  output_arg {
    name: "output"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
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
  name: "DepthwiseConv2dNativeBackpropFilter"
  input_arg {
    name: "input"
    type_attr: "T"
  }
  input_arg {
    name: "filter_sizes"
    type: DT_INT32
  }
  input_arg {
    name: "out_backprop"
    type_attr: "T"
  }
  output_arg {
    name: "output"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
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
  name: "DepthwiseConv2dNativeBackpropInput"
  input_arg {
    name: "input_sizes"
    type: DT_INT32
  }
  input_arg {
    name: "filter"
    type_attr: "T"
  }
  input_arg {
    name: "out_backprop"
    type_attr: "T"
  }
  output_arg {
    name: "output"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
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
  name: "Dilation2D"
  input_arg {
    name: "input"
    type_attr: "T"
  }
  input_arg {
    name: "filter"
    type_attr: "T"
  }
  output_arg {
    name: "output"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT32
        type: DT_INT64
        type: DT_UINT8
        type: DT_INT16
        type: DT_INT8
        type: DT_UINT16
        type: DT_HALF
      }
    }
  }
  attr {
    name: "strides"
    type: "list(int)"
    has_minimum: true
    minimum: 4
  }
  attr {
    name: "rates"
    type: "list(int)"
    has_minimum: true
    minimum: 4
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
  name: "Dilation2DBackpropFilter"
  input_arg {
    name: "input"
    type_attr: "T"
  }
  input_arg {
    name: "filter"
    type_attr: "T"
  }
  input_arg {
    name: "out_backprop"
    type_attr: "T"
  }
  output_arg {
    name: "filter_backprop"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT32
        type: DT_INT64
        type: DT_UINT8
        type: DT_INT16
        type: DT_INT8
        type: DT_UINT16
        type: DT_HALF
      }
    }
  }
  attr {
    name: "strides"
    type: "list(int)"
    has_minimum: true
    minimum: 4
  }
  attr {
    name: "rates"
    type: "list(int)"
    has_minimum: true
    minimum: 4
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
  name: "Dilation2DBackpropInput"
  input_arg {
    name: "input"
    type_attr: "T"
  }
  input_arg {
    name: "filter"
    type_attr: "T"
  }
  input_arg {
    name: "out_backprop"
    type_attr: "T"
  }
  output_arg {
    name: "in_backprop"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT32
        type: DT_INT64
        type: DT_UINT8
        type: DT_INT16
        type: DT_INT8
        type: DT_UINT16
        type: DT_HALF
      }
    }
  }
  attr {
    name: "strides"
    type: "list(int)"
    has_minimum: true
    minimum: 4
  }
  attr {
    name: "rates"
    type: "list(int)"
    has_minimum: true
    minimum: 4
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
  name: "Elu"
  input_arg {
    name: "features"
    type_attr: "T"
  }
  output_arg {
    name: "activations"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
      }
    }
  }
}
op {
  name: "EluGrad"
  input_arg {
    name: "gradients"
    type_attr: "T"
  }
  input_arg {
    name: "outputs"
    type_attr: "T"
  }
  output_arg {
    name: "backprops"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
      }
    }
  }
}
op {
  name: "InTopK"
  input_arg {
    name: "predictions"
    type: DT_FLOAT
  }
  input_arg {
    name: "targets"
    type_attr: "T"
  }
  output_arg {
    name: "precision"
    type: DT_BOOL
  }
  attr {
    name: "k"
    type: "int"
  }
  attr {
    name: "T"
    type: "type"
    default_value {
      type: DT_INT32
    }
    allowed_values {
      list {
        type: DT_INT32
        type: DT_INT64
      }
    }
  }
}
op {
  name: "L2Loss"
  input_arg {
    name: "t"
    type_attr: "T"
  }
  output_arg {
    name: "output"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT64
        type: DT_INT32
        type: DT_UINT8
        type: DT_UINT16
        type: DT_INT16
        type: DT_INT8
        type: DT_COMPLEX64
        type: DT_COMPLEX128
        type: DT_QINT8
        type: DT_QUINT8
        type: DT_QINT32
        type: DT_HALF
      }
    }
  }
}
op {
  name: "LRN"
  input_arg {
    name: "input"
    type: DT_FLOAT
  }
  output_arg {
    name: "output"
    type: DT_FLOAT
  }
  attr {
    name: "depth_radius"
    type: "int"
    default_value {
      i: 5
    }
  }
  attr {
    name: "bias"
    type: "float"
    default_value {
      f: 1
    }
  }
  attr {
    name: "alpha"
    type: "float"
    default_value {
      f: 1
    }
  }
  attr {
    name: "beta"
    type: "float"
    default_value {
      f: 0.5
    }
  }
}
op {
  name: "LRNGrad"
  input_arg {
    name: "input_grads"
    type: DT_FLOAT
  }
  input_arg {
    name: "input_image"
    type: DT_FLOAT
  }
  input_arg {
    name: "output_image"
    type: DT_FLOAT
  }
  output_arg {
    name: "output"
    type: DT_FLOAT
  }
  attr {
    name: "depth_radius"
    type: "int"
    default_value {
      i: 5
    }
  }
  attr {
    name: "bias"
    type: "float"
    default_value {
      f: 1
    }
  }
  attr {
    name: "alpha"
    type: "float"
    default_value {
      f: 1
    }
  }
  attr {
    name: "beta"
    type: "float"
    default_value {
      f: 0.5
    }
  }
}
op {
  name: "LogSoftmax"
  input_arg {
    name: "logits"
    type_attr: "T"
  }
  output_arg {
    name: "logsoftmax"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_HALF
        type: DT_FLOAT
        type: DT_DOUBLE
      }
    }
  }
}
op {
  name: "MaxPool"
  input_arg {
    name: "input"
    type: DT_FLOAT
  }
  output_arg {
    name: "output"
    type: DT_FLOAT
  }
  attr {
    name: "ksize"
    type: "list(int)"
    has_minimum: true
    minimum: 4
  }
  attr {
    name: "strides"
    type: "list(int)"
    has_minimum: true
    minimum: 4
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
  attr {
    name: "data_format"
    type: "string"
    default_value {
      s: "NHWC"
    }
    allowed_values {
      list {
        s: "NHWC"
        s: "NCHW"
      }
    }
  }
}
op {
  name: "MaxPool3D"
  input_arg {
    name: "input"
    type_attr: "T"
  }
  output_arg {
    name: "output"
    type_attr: "T"
  }
  attr {
    name: "ksize"
    type: "list(int)"
    has_minimum: true
    minimum: 5
  }
  attr {
    name: "strides"
    type: "list(int)"
    has_minimum: true
    minimum: 5
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
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT64
        type: DT_INT32
        type: DT_UINT8
        type: DT_UINT16
        type: DT_INT16
        type: DT_INT8
        type: DT_COMPLEX64
        type: DT_COMPLEX128
        type: DT_QINT8
        type: DT_QUINT8
        type: DT_QINT32
        type: DT_HALF
      }
    }
  }
}
op {
  name: "MaxPool3DGrad"
  input_arg {
    name: "orig_input"
    type: DT_FLOAT
  }
  input_arg {
    name: "orig_output"
    type: DT_FLOAT
  }
  input_arg {
    name: "grad"
    type_attr: "T"
  }
  output_arg {
    name: "output"
    type_attr: "T"
  }
  attr {
    name: "ksize"
    type: "list(int)"
    has_minimum: true
    minimum: 5
  }
  attr {
    name: "strides"
    type: "list(int)"
    has_minimum: true
    minimum: 5
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
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT64
        type: DT_INT32
        type: DT_UINT8
        type: DT_UINT16
        type: DT_INT16
        type: DT_INT8
        type: DT_COMPLEX64
        type: DT_COMPLEX128
        type: DT_QINT8
        type: DT_QUINT8
        type: DT_QINT32
        type: DT_HALF
      }
    }
  }
}
op {
  name: "MaxPoolGrad"
  input_arg {
    name: "orig_input"
    type: DT_FLOAT
  }
  input_arg {
    name: "orig_output"
    type: DT_FLOAT
  }
  input_arg {
    name: "grad"
    type: DT_FLOAT
  }
  output_arg {
    name: "output"
    type: DT_FLOAT
  }
  attr {
    name: "ksize"
    type: "list(int)"
    has_minimum: true
    minimum: 4
  }
  attr {
    name: "strides"
    type: "list(int)"
    has_minimum: true
    minimum: 4
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
  attr {
    name: "data_format"
    type: "string"
    default_value {
      s: "NHWC"
    }
    allowed_values {
      list {
        s: "NHWC"
        s: "NCHW"
      }
    }
  }
}
op {
  name: "MaxPoolGradWithArgmax"
  input_arg {
    name: "input"
    type: DT_FLOAT
  }
  input_arg {
    name: "grad"
    type: DT_FLOAT
  }
  input_arg {
    name: "argmax"
    type_attr: "Targmax"
  }
  output_arg {
    name: "output"
    type: DT_FLOAT
  }
  attr {
    name: "ksize"
    type: "list(int)"
    has_minimum: true
    minimum: 4
  }
  attr {
    name: "strides"
    type: "list(int)"
    has_minimum: true
    minimum: 4
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
  attr {
    name: "Targmax"
    type: "type"
    allowed_values {
      list {
        type: DT_INT32
        type: DT_INT64
      }
    }
  }
}
op {
  name: "MaxPoolWithArgmax"
  input_arg {
    name: "input"
    type: DT_FLOAT
  }
  output_arg {
    name: "output"
    type: DT_FLOAT
  }
  output_arg {
    name: "argmax"
    type_attr: "Targmax"
  }
  attr {
    name: "ksize"
    type: "list(int)"
    has_minimum: true
    minimum: 4
  }
  attr {
    name: "strides"
    type: "list(int)"
    has_minimum: true
    minimum: 4
  }
  attr {
    name: "Targmax"
    type: "type"
    default_value {
      type: DT_INT64
    }
    allowed_values {
      list {
        type: DT_INT32
        type: DT_INT64
      }
    }
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
  name: "Relu"
  input_arg {
    name: "features"
    type_attr: "T"
  }
  output_arg {
    name: "activations"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT32
        type: DT_INT64
        type: DT_UINT8
        type: DT_INT16
        type: DT_INT8
        type: DT_UINT16
        type: DT_HALF
      }
    }
  }
}
op {
  name: "Relu6"
  input_arg {
    name: "features"
    type_attr: "T"
  }
  output_arg {
    name: "activations"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT32
        type: DT_INT64
        type: DT_UINT8
        type: DT_INT16
        type: DT_INT8
        type: DT_UINT16
        type: DT_HALF
      }
    }
  }
}
op {
  name: "Relu6Grad"
  input_arg {
    name: "gradients"
    type_attr: "T"
  }
  input_arg {
    name: "features"
    type_attr: "T"
  }
  output_arg {
    name: "backprops"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT32
        type: DT_INT64
        type: DT_UINT8
        type: DT_INT16
        type: DT_INT8
        type: DT_UINT16
        type: DT_HALF
      }
    }
  }
}
op {
  name: "ReluGrad"
  input_arg {
    name: "gradients"
    type_attr: "T"
  }
  input_arg {
    name: "features"
    type_attr: "T"
  }
  output_arg {
    name: "backprops"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT32
        type: DT_INT64
        type: DT_UINT8
        type: DT_INT16
        type: DT_INT8
        type: DT_UINT16
        type: DT_HALF
      }
    }
  }
}
op {
  name: "Softmax"
  input_arg {
    name: "logits"
    type_attr: "T"
  }
  output_arg {
    name: "softmax"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_HALF
        type: DT_FLOAT
        type: DT_DOUBLE
      }
    }
  }
}
op {
  name: "SoftmaxCrossEntropyWithLogits"
  input_arg {
    name: "features"
    type_attr: "T"
  }
  input_arg {
    name: "labels"
    type_attr: "T"
  }
  output_arg {
    name: "loss"
    type_attr: "T"
  }
  output_arg {
    name: "backprop"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_HALF
        type: DT_FLOAT
        type: DT_DOUBLE
      }
    }
  }
}
op {
  name: "Softplus"
  input_arg {
    name: "features"
    type_attr: "T"
  }
  output_arg {
    name: "activations"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT32
        type: DT_INT64
        type: DT_UINT8
        type: DT_INT16
        type: DT_INT8
        type: DT_UINT16
        type: DT_HALF
      }
    }
  }
}
op {
  name: "SoftplusGrad"
  input_arg {
    name: "gradients"
    type_attr: "T"
  }
  input_arg {
    name: "features"
    type_attr: "T"
  }
  output_arg {
    name: "backprops"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT32
        type: DT_INT64
        type: DT_UINT8
        type: DT_INT16
        type: DT_INT8
        type: DT_UINT16
        type: DT_HALF
      }
    }
  }
}
op {
  name: "Softsign"
  input_arg {
    name: "features"
    type_attr: "T"
  }
  output_arg {
    name: "activations"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT32
        type: DT_INT64
        type: DT_UINT8
        type: DT_INT16
        type: DT_INT8
        type: DT_UINT16
        type: DT_HALF
      }
    }
  }
}
op {
  name: "SoftsignGrad"
  input_arg {
    name: "gradients"
    type_attr: "T"
  }
  input_arg {
    name: "features"
    type_attr: "T"
  }
  output_arg {
    name: "backprops"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT32
        type: DT_INT64
        type: DT_UINT8
        type: DT_INT16
        type: DT_INT8
        type: DT_UINT16
        type: DT_HALF
      }
    }
  }
}
op {
  name: "SparseSoftmaxCrossEntropyWithLogits"
  input_arg {
    name: "features"
    type_attr: "T"
  }
  input_arg {
    name: "labels"
    type_attr: "Tlabels"
  }
  output_arg {
    name: "loss"
    type_attr: "T"
  }
  output_arg {
    name: "backprop"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_HALF
        type: DT_FLOAT
        type: DT_DOUBLE
      }
    }
  }
  attr {
    name: "Tlabels"
    type: "type"
    default_value {
      type: DT_INT64
    }
    allowed_values {
      list {
        type: DT_INT32
        type: DT_INT64
      }
    }
  }
}
op {
  name: "TopK"
  input_arg {
    name: "input"
    type_attr: "T"
  }
  output_arg {
    name: "values"
    type_attr: "T"
  }
  output_arg {
    name: "indices"
    type: DT_INT32
  }
  attr {
    name: "k"
    type: "int"
    has_minimum: true
  }
  attr {
    name: "sorted"
    type: "bool"
    default_value {
      b: true
    }
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT32
        type: DT_INT64
        type: DT_UINT8
        type: DT_INT16
        type: DT_INT8
        type: DT_UINT16
        type: DT_HALF
      }
    }
  }
  deprecation {
    version: 7
    explanation: "Use TopKV2 instead"
  }
}
op {
  name: "TopKV2"
  input_arg {
    name: "input"
    type_attr: "T"
  }
  input_arg {
    name: "k"
    type: DT_INT32
  }
  output_arg {
    name: "values"
    type_attr: "T"
  }
  output_arg {
    name: "indices"
    type: DT_INT32
  }
  attr {
    name: "sorted"
    type: "bool"
    default_value {
      b: true
    }
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT32
        type: DT_INT64
        type: DT_UINT8
        type: DT_INT16
        type: DT_INT8
        type: DT_UINT16
        type: DT_HALF
      }
    }
  }
}
"""


_op_def_lib = _InitOpDefLibrary()
