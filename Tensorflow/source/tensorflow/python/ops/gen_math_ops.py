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
__abs_outputs = ["y"]


def _abs(x, name=None):
  r"""Computes the absolute value of a tensor.

  Given a tensor `x`, this operation returns a tensor containing the absolute
  value of each element in `x`. For example, if x is an input element and y is
  an output element, this operation computes \\(y = |x|\\).

  Args:
    x: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`, `int32`, `int64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  result = _op_def_lib.apply_op("Abs", x=x, name=name)
  return result


_acos_outputs = ["y"]


def acos(x, name=None):
  r"""Computes acos of x element-wise.

  Args:
    x: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`, `int32`, `int64`, `complex64`, `complex128`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  result = _op_def_lib.apply_op("Acos", x=x, name=name)
  return result


_add_outputs = ["z"]


def add(x, y, name=None):
  r"""Returns x + y element-wise.

  *NOTE*: `Add` supports broadcasting. `AddN` does not. More about broadcasting
  [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

  Args:
    x: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`, `uint8`, `int8`, `int16`, `int32`, `int64`, `complex64`, `complex128`, `string`.
    y: A `Tensor`. Must have the same type as `x`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  result = _op_def_lib.apply_op("Add", x=x, y=y, name=name)
  return result


__add_n_outputs = ["sum"]


def _add_n(inputs, name=None):
  r"""Add all input tensors element wise.

  Args:
    inputs: A list of at least 1 `Tensor` objects of the same type in: `float32`, `float64`, `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`, `complex128`, `qint8`, `quint8`, `qint32`, `half`.
      Must all be the same size and shape.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `inputs`.
  """
  result = _op_def_lib.apply_op("AddN", inputs=inputs, name=name)
  return result


__all_outputs = ["output"]


def _all(input, reduction_indices, keep_dims=None, name=None):
  r"""Computes the "logical and" of elements across dimensions of a tensor.

  Reduces `input` along the dimensions given in `reduction_indices`. Unless
  `keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
  `reduction_indices`. If `keep_dims` is true, the reduced dimensions are
  retained with length 1.

  Args:
    input: A `Tensor` of type `bool`. The tensor to reduce.
    reduction_indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      The dimensions to reduce.
    keep_dims: An optional `bool`. Defaults to `False`.
      If true, retain reduced dimensions with length 1.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `bool`. The reduced tensor.
  """
  result = _op_def_lib.apply_op("All", input=input,
                                reduction_indices=reduction_indices,
                                keep_dims=keep_dims, name=name)
  return result


__any_outputs = ["output"]


def _any(input, reduction_indices, keep_dims=None, name=None):
  r"""Computes the "logical or" of elements across dimensions of a tensor.

  Reduces `input` along the dimensions given in `reduction_indices`. Unless
  `keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
  `reduction_indices`. If `keep_dims` is true, the reduced dimensions are
  retained with length 1.

  Args:
    input: A `Tensor` of type `bool`. The tensor to reduce.
    reduction_indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      The dimensions to reduce.
    keep_dims: An optional `bool`. Defaults to `False`.
      If true, retain reduced dimensions with length 1.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `bool`. The reduced tensor.
  """
  result = _op_def_lib.apply_op("Any", input=input,
                                reduction_indices=reduction_indices,
                                keep_dims=keep_dims, name=name)
  return result


_arg_max_outputs = ["output"]


def arg_max(input, dimension, name=None):
  r"""Returns the index with the largest value across dimensions of a tensor.

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`, `complex128`, `qint8`, `quint8`, `qint32`, `half`.
    dimension: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      int32, 0 <= dimension < rank(input).  Describes which dimension
      of the input Tensor to reduce across. For vectors, use dimension = 0.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int64`.
  """
  result = _op_def_lib.apply_op("ArgMax", input=input, dimension=dimension,
                                name=name)
  return result


_arg_min_outputs = ["output"]


def arg_min(input, dimension, name=None):
  r"""Returns the index with the smallest value across dimensions of a tensor.

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`, `complex128`, `qint8`, `quint8`, `qint32`, `half`.
    dimension: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      int32, 0 <= dimension < rank(input).  Describes which dimension
      of the input Tensor to reduce across. For vectors, use dimension = 0.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int64`.
  """
  result = _op_def_lib.apply_op("ArgMin", input=input, dimension=dimension,
                                name=name)
  return result


_asin_outputs = ["y"]


def asin(x, name=None):
  r"""Computes asin of x element-wise.

  Args:
    x: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`, `int32`, `int64`, `complex64`, `complex128`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  result = _op_def_lib.apply_op("Asin", x=x, name=name)
  return result


_atan_outputs = ["y"]


def atan(x, name=None):
  r"""Computes atan of x element-wise.

  Args:
    x: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`, `int32`, `int64`, `complex64`, `complex128`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  result = _op_def_lib.apply_op("Atan", x=x, name=name)
  return result


__batch_fft_outputs = ["output"]


def _batch_fft(input, name=None):
  r"""TODO: add doc.

  Args:
    input: A `Tensor` of type `complex64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `complex64`.
  """
  result = _op_def_lib.apply_op("BatchFFT", input=input, name=name)
  return result


__batch_fft2d_outputs = ["output"]


def _batch_fft2d(input, name=None):
  r"""TODO: add doc.

  Args:
    input: A `Tensor` of type `complex64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `complex64`.
  """
  result = _op_def_lib.apply_op("BatchFFT2D", input=input, name=name)
  return result


__batch_fft3d_outputs = ["output"]


def _batch_fft3d(input, name=None):
  r"""TODO: add doc.

  Args:
    input: A `Tensor` of type `complex64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `complex64`.
  """
  result = _op_def_lib.apply_op("BatchFFT3D", input=input, name=name)
  return result


__batch_ifft_outputs = ["output"]


def _batch_ifft(input, name=None):
  r"""TODO: add doc.

  Args:
    input: A `Tensor` of type `complex64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `complex64`.
  """
  result = _op_def_lib.apply_op("BatchIFFT", input=input, name=name)
  return result


__batch_ifft2d_outputs = ["output"]


def _batch_ifft2d(input, name=None):
  r"""TODO: add doc.

  Args:
    input: A `Tensor` of type `complex64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `complex64`.
  """
  result = _op_def_lib.apply_op("BatchIFFT2D", input=input, name=name)
  return result


__batch_ifft3d_outputs = ["output"]


def _batch_ifft3d(input, name=None):
  r"""TODO: add doc.

  Args:
    input: A `Tensor` of type `complex64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `complex64`.
  """
  result = _op_def_lib.apply_op("BatchIFFT3D", input=input, name=name)
  return result


__batch_mat_mul_outputs = ["output"]


def _batch_mat_mul(x, y, adj_x=None, adj_y=None, name=None):
  r"""Multiplies slices of two tensors in batches.

  Multiplies all slices of `Tensor` `x` and `y` (each slice can be
  viewed as an element of a batch), and arranges the individual results
  in a single output tensor of the same batch size. Each of the
  individual slices can optionally be adjointed (to adjoint a matrix
  means to transpose and conjugate it) before multiplication by setting
  the `adj_x` or `adj_y` flag to `True`, which are by default `False`.

  The input tensors `x` and `y` are 3-D or higher with shape `[..., r_x, c_x]`
  and `[..., r_y, c_y]`.

  The output tensor is 3-D or higher with shape `[..., r_o, c_o]`, where:

      r_o = c_x if adj_x else r_x
      c_o = r_y if adj_y else c_y

  It is computed as:

      output[..., :, :] = matrix(x[..., :, :]) * matrix(y[..., :, :])

  Args:
    x: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`, `int32`, `complex64`, `complex128`.
      3-D or higher with shape `[..., r_x, c_x]`.
    y: A `Tensor`. Must have the same type as `x`.
      3-D or higher with shape `[..., r_y, c_y]`.
    adj_x: An optional `bool`. Defaults to `False`.
      If `True`, adjoint the slices of `x`. Defaults to `False`.
    adj_y: An optional `bool`. Defaults to `False`.
      If `True`, adjoint the slices of `y`. Defaults to `False`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
    3-D or higher with shape `[..., r_o, c_o]`
  """
  result = _op_def_lib.apply_op("BatchMatMul", x=x, y=y, adj_x=adj_x,
                                adj_y=adj_y, name=name)
  return result


_betainc_outputs = ["z"]


def betainc(a, b, x, name=None):
  r"""Compute the regularized incomplete beta integral \\(I_x(a, b)\\).

  The regularized incomplete beta integral is defined as:

  ```
  I_x(a, b) = \frac{B(x; a, b)}{B(a, b)}
  ```
  where

  ```
  B(x; a, b) = \int_0^x t^{a-1} (1 - t)^{b-1} dt
  ```

  is the incomplete beta function and \\(B(a, b)\\) is the *complete*
  beta function.

  Args:
    a: A `Tensor`. Must be one of the following types: `float32`, `float64`.
    b: A `Tensor`. Must have the same type as `a`.
    x: A `Tensor`. Must have the same type as `a`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `a`.
  """
  result = _op_def_lib.apply_op("Betainc", a=a, b=b, x=x, name=name)
  return result


_cast_outputs = ["y"]


def cast(x, DstT, name=None):
  r"""Cast x of type SrcT to y of DstT.

  Args:
    x: A `Tensor`.
    DstT: A `tf.DType`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `DstT`.
  """
  result = _op_def_lib.apply_op("Cast", x=x, DstT=DstT, name=name)
  return result


_ceil_outputs = ["y"]


def ceil(x, name=None):
  r"""Returns element-wise smallest integer in not less than x.

  Args:
    x: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  result = _op_def_lib.apply_op("Ceil", x=x, name=name)
  return result


__complex_outputs = ["out"]


def _complex(real, imag, Tout=None, name=None):
  r"""Converts two real numbers to a complex number.

  Given a tensor `real` representing the real part of a complex number, and a
  tensor `imag` representing the imaginary part of a complex number, this
  operation returns complex numbers elementwise of the form \\(a + bj\\), where
  *a* represents the `real` part and *b* represents the `imag` part.

  The input tensors `real` and `imag` must have the same shape.

  For example:

  ```
  # tensor 'real' is [2.25, 3.25]
  # tensor `imag` is [4.75, 5.75]
  tf.complex(real, imag) ==> [[2.25 + 4.75j], [3.25 + 5.75j]]
  ```

  Args:
    real: A `Tensor`. Must be one of the following types: `float32`, `float64`.
    imag: A `Tensor`. Must have the same type as `real`.
    Tout: An optional `tf.DType` from: `tf.complex64, tf.complex128`. Defaults to `tf.complex64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `Tout`.
  """
  result = _op_def_lib.apply_op("Complex", real=real, imag=imag, Tout=Tout,
                                name=name)
  return result


__complex_abs_outputs = ["y"]


def _complex_abs(x, Tout=None, name=None):
  r"""Computes the complex absolute value of a tensor.

  Given a tensor `x` of complex numbers, this operation returns a tensor of type
  `float` or `double` that is the absolute value of each element in `x`. All
  elements in `x` must be complex numbers of the form \\(a + bj\\). The absolute
  value is computed as \\( \sqrt{a^2 + b^2}\\).

  Args:
    x: A `Tensor`. Must be one of the following types: `complex64`, `complex128`.
    Tout: An optional `tf.DType` from: `tf.float32, tf.float64`. Defaults to `tf.float32`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `Tout`.
  """
  result = _op_def_lib.apply_op("ComplexAbs", x=x, Tout=Tout, name=name)
  return result


__conj_outputs = ["output"]


def _conj(input, name=None):
  r"""Returns the complex conjugate of a complex number.

  Given a tensor `input` of complex numbers, this operation returns a tensor of
  complex numbers that are the complex conjugate of each element in `input`. The
  complex numbers in `input` must be of the form \\(a + bj\\), where *a* is the
  real part and *b* is the imaginary part.

  The complex conjugate returned by this operation is of the form \\(a - bj\\).

  For example:

  ```
  # tensor 'input' is [-2.25 + 4.75j, 3.25 + 5.75j]
  tf.conj(input) ==> [-2.25 - 4.75j, 3.25 - 5.75j]
  ```

  Args:
    input: A `Tensor`. Must be one of the following types: `complex64`, `complex128`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  result = _op_def_lib.apply_op("Conj", input=input, name=name)
  return result


_cos_outputs = ["y"]


def cos(x, name=None):
  r"""Computes cos of x element-wise.

  Args:
    x: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`, `complex64`, `complex128`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  result = _op_def_lib.apply_op("Cos", x=x, name=name)
  return result


_cross_outputs = ["product"]


def cross(a, b, name=None):
  r"""Compute the pairwise cross product.

  `a` and `b` must be the same shape; they can either be simple 3-element vectors,
  or any shape where the innermost dimension is 3. In the latter case, each pair
  of corresponding 3-element vectors is cross-multiplied independently.

  Args:
    a: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`, `uint8`, `int16`, `int8`, `uint16`, `half`.
      A tensor containing 3-element vectors.
    b: A `Tensor`. Must have the same type as `a`.
      Another tensor, of same type and shape as `a`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `a`.
    Pairwise cross product of the vectors in `a` and `b`.
  """
  result = _op_def_lib.apply_op("Cross", a=a, b=b, name=name)
  return result


_cumprod_outputs = ["out"]


def cumprod(x, axis, exclusive=None, reverse=None, name=None):
  r"""Compute the cumulative product of the tensor `x` along `axis`.

  By default, this op performs an inclusive cumprod, which means that the first
  element of the input is identical to the first element of the output:
  ```prettyprint
  tf.cumprod([a, b, c]) ==> [a, a * b, a * b * c]
  ```

  By setting the `exclusive` kwarg to `True`, an exclusive cumprod is
  performed instead:
  ```prettyprint
  tf.cumprod([a, b, c], exclusive=True) ==> [0, a, a * b]
  ```

  By setting the `reverse` kwarg to `True`, the cumprod is performed in the
  opposite direction:
  ```prettyprint
  tf.cumprod([a, b, c], reverse=True) ==> [a * b * c, b * c, c]
  ```
  This is more efficient than using separate `tf.reverse` ops.

  The `reverse` and `exclusive` kwargs can also be combined:
  ```prettyprint
  tf.cumprod([a, b, c], exclusive=True, reverse=True) ==> [b * c, c, 0]
  ```

  Args:
    x: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`, `complex128`, `qint8`, `quint8`, `qint32`, `half`.
    axis: A `Tensor`. Must be one of the following types: `int32`, `int64`.
    exclusive: An optional `bool`. Defaults to `False`.
    reverse: An optional `bool`. Defaults to `False`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  result = _op_def_lib.apply_op("Cumprod", x=x, axis=axis,
                                exclusive=exclusive, reverse=reverse,
                                name=name)
  return result


_cumsum_outputs = ["out"]


def cumsum(x, axis, exclusive=None, reverse=None, name=None):
  r"""Compute the cumulative sum of the tensor `x` along `axis`.

  By default, this op performs an inclusive cumsum, which means that the first
  element of the input is identical to the first element of the output:
  ```prettyprint
  tf.cumsum([a, b, c]) ==> [a, a + b, a + b + c]
  ```

  By setting the `exclusive` kwarg to `True`, an exclusive cumsum is
  performed instead:
  ```prettyprint
  tf.cumsum([a, b, c], exclusive=True) ==> [0, a, a + b]
  ```

  By setting the `reverse` kwarg to `True`, the cumsum is performed in the
  opposite direction:
  ```prettyprint
  tf.cumsum([a, b, c], reverse=True) ==> [a + b + c, b + c, c]
  ```
  This is more efficient than using separate `tf.reverse` ops.

  The `reverse` and `exclusive` kwargs can also be combined:
  ```prettyprint
  tf.cumsum([a, b, c], exclusive=True, reverse=True) ==> [b + c, c, 0]
  ```

  Args:
    x: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`, `complex128`, `qint8`, `quint8`, `qint32`, `half`.
    axis: A `Tensor`. Must be one of the following types: `int32`, `int64`.
    exclusive: An optional `bool`. Defaults to `False`.
    reverse: An optional `bool`. Defaults to `False`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  result = _op_def_lib.apply_op("Cumsum", x=x, axis=axis, exclusive=exclusive,
                                reverse=reverse, name=name)
  return result


_digamma_outputs = ["y"]


def digamma(x, name=None):
  r"""Computes Psi, the derivative of Lgamma (the log of the absolute value of

  `Gamma(x)`), element-wise.

  Args:
    x: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  result = _op_def_lib.apply_op("Digamma", x=x, name=name)
  return result


_div_outputs = ["z"]


def div(x, y, name=None):
  r"""Returns x / y element-wise.

  *NOTE*: `Div` supports broadcasting. More about broadcasting
  [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

  Args:
    x: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`, `uint8`, `int8`, `uint16`, `int16`, `int32`, `int64`, `complex64`, `complex128`.
    y: A `Tensor`. Must have the same type as `x`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  result = _op_def_lib.apply_op("Div", x=x, y=y, name=name)
  return result


_equal_outputs = ["z"]


def equal(x, y, name=None):
  r"""Returns the truth value of (x == y) element-wise.

  *NOTE*: `Equal` supports broadcasting. More about broadcasting
  [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

  Args:
    x: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`, `uint8`, `int8`, `int16`, `int32`, `int64`, `complex64`, `quint8`, `qint8`, `qint32`, `string`, `bool`, `complex128`.
    y: A `Tensor`. Must have the same type as `x`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `bool`.
  """
  result = _op_def_lib.apply_op("Equal", x=x, y=y, name=name)
  return result


_erf_outputs = ["y"]


def erf(x, name=None):
  r"""Computes the Gauss error function of `x` element-wise.

  Args:
    x: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  result = _op_def_lib.apply_op("Erf", x=x, name=name)
  return result


_erfc_outputs = ["y"]


def erfc(x, name=None):
  r"""Computes the complementary error function of `x` element-wise.

  Args:
    x: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  result = _op_def_lib.apply_op("Erfc", x=x, name=name)
  return result


_exp_outputs = ["y"]


def exp(x, name=None):
  r"""Computes exponential of x element-wise.  \\(y = e^x\\).

  Args:
    x: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`, `complex64`, `complex128`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  result = _op_def_lib.apply_op("Exp", x=x, name=name)
  return result


_expm1_outputs = ["y"]


def expm1(x, name=None):
  r"""Computes exponential of x - 1 element-wise.

  I.e., \\(y = (\exp x) - 1\\).

  Args:
    x: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`, `complex64`, `complex128`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  result = _op_def_lib.apply_op("Expm1", x=x, name=name)
  return result


_fft_outputs = ["output"]


def fft(input, name=None):
  r"""Compute the 1-dimensional discrete Fourier Transform over the inner-most

  dimension of `input`.

  Args:
    input: A `Tensor` of type `complex64`. A complex64 tensor.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `complex64`.
    A complex64 tensor of the same shape as `input`. The inner-most
    dimension of `input` is replaced with its 1D Fourier Transform.
  """
  result = _op_def_lib.apply_op("FFT", input=input, name=name)
  return result


_fft2d_outputs = ["output"]


def fft2d(input, name=None):
  r"""Compute the 2-dimensional discrete Fourier Transform over the inner-most

  2 dimensions of `input`.

  Args:
    input: A `Tensor` of type `complex64`. A complex64 tensor.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `complex64`.
    A complex64 tensor of the same shape as `input`. The inner-most 2
      dimensions of `input` are replaced with their 2D Fourier Transform.

    @compatibility(numpy)
    Equivalent to np.fft2
    @end_compatibility
  """
  result = _op_def_lib.apply_op("FFT2D", input=input, name=name)
  return result


_fft3d_outputs = ["output"]


def fft3d(input, name=None):
  r"""Compute the 3-dimensional discrete Fourier Transform over the inner-most 3

  dimensions of `input`.

  Args:
    input: A `Tensor` of type `complex64`. A complex64 tensor.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `complex64`.
    A complex64 tensor of the same shape as `input`. The inner-most 3
      dimensions of `input` are replaced with their 3D Fourier Transform.

    @compatibility(numpy)
    Equivalent to np.fft3
    @end_compatibility
  """
  result = _op_def_lib.apply_op("FFT3D", input=input, name=name)
  return result


_floor_outputs = ["y"]


def floor(x, name=None):
  r"""Returns element-wise largest integer not greater than x.

  Args:
    x: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  result = _op_def_lib.apply_op("Floor", x=x, name=name)
  return result


__floor_div_outputs = ["z"]


def _floor_div(x, y, name=None):
  r"""Returns x // y element-wise.

  *NOTE*: `FloorDiv` supports broadcasting. More about broadcasting
  [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

  Args:
    x: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`, `uint8`, `int8`, `uint16`, `int16`, `int32`, `int64`, `complex64`, `complex128`.
    y: A `Tensor`. Must have the same type as `x`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  result = _op_def_lib.apply_op("FloorDiv", x=x, y=y, name=name)
  return result


__floor_mod_outputs = ["z"]


def _floor_mod(x, y, name=None):
  r"""Returns element-wise remainder of division. When `x < 0` xor `y < 0` is

  true, this follows Python semantics in that the result here is consistent
  with a flooring divide. E.g. `floor(x / y) * y + mod(x, y) = x`.

  *NOTE*: `FloorMod` supports broadcasting. More about broadcasting
  [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

  Args:
    x: A `Tensor`. Must be one of the following types: `int32`, `int64`, `float32`, `float64`.
    y: A `Tensor`. Must have the same type as `x`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  result = _op_def_lib.apply_op("FloorMod", x=x, y=y, name=name)
  return result


_greater_outputs = ["z"]


def greater(x, y, name=None):
  r"""Returns the truth value of (x > y) element-wise.

  *NOTE*: `Greater` supports broadcasting. More about broadcasting
  [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

  Args:
    x: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`, `uint8`, `int16`, `int8`, `uint16`, `half`.
    y: A `Tensor`. Must have the same type as `x`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `bool`.
  """
  result = _op_def_lib.apply_op("Greater", x=x, y=y, name=name)
  return result


_greater_equal_outputs = ["z"]


def greater_equal(x, y, name=None):
  r"""Returns the truth value of (x >= y) element-wise.

  *NOTE*: `GreaterEqual` supports broadcasting. More about broadcasting
  [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

  Args:
    x: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`, `uint8`, `int16`, `int8`, `uint16`, `half`.
    y: A `Tensor`. Must have the same type as `x`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `bool`.
  """
  result = _op_def_lib.apply_op("GreaterEqual", x=x, y=y, name=name)
  return result


_ifft_outputs = ["output"]


def ifft(input, name=None):
  r"""Compute the inverse 1-dimensional discrete Fourier Transform over the inner-most

  dimension of `input`.

  Args:
    input: A `Tensor` of type `complex64`. A complex64 tensor.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `complex64`.
    A complex64 tensor of the same shape as `input`. The inner-most
    dimension of `input` is replaced with its inverse 1D Fourier Transform.
  """
  result = _op_def_lib.apply_op("IFFT", input=input, name=name)
  return result


_ifft2d_outputs = ["output"]


def ifft2d(input, name=None):
  r"""Compute the inverse 2-dimensional discrete Fourier Transform over the inner-most

  2 dimensions of `input`.

  Args:
    input: A `Tensor` of type `complex64`. A complex64 tensor.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `complex64`.
    A complex64 tensor of the same shape as `input`. The inner-most 2
      dimensions of `input` are replaced with their inverse 2D Fourier Transform.

    @compatibility(numpy)
    Equivalent to np.ifft2
    @end_compatibility
  """
  result = _op_def_lib.apply_op("IFFT2D", input=input, name=name)
  return result


_ifft3d_outputs = ["output"]


def ifft3d(input, name=None):
  r"""Compute the inverse 3-dimensional discrete Fourier Transform over the inner-most

  3 dimensions of `input`.

  Args:
    input: A `Tensor` of type `complex64`. A complex64 tensor.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `complex64`.
    A complex64 tensor of the same shape as `input`. The inner-most 3
      dimensions of `input` are replaced with their inverse 3D Fourier Transform.

    @compatibility(numpy)
    Equivalent to np.fft3
    @end_compatibility
  """
  result = _op_def_lib.apply_op("IFFT3D", input=input, name=name)
  return result


_igamma_outputs = ["z"]


def igamma(a, x, name=None):
  r"""Compute the lower regularized incomplete Gamma function `Q(a, x)`.

  The lower regularized incomplete Gamma function is defined as:

  ```
  P(a, x) = gamma(a, x) / Gamma(a) = 1 - Q(a, x)
  ```
  where
  ```
  gamma(a, x) = int_{0}^{x} t^{a-1} exp(-t) dt
  ```
  is the lower incomplete Gamma function.

  Note, above `Q(a, x)` (`Igammac`) is the upper regularized complete
  Gamma function.

  Args:
    a: A `Tensor`. Must be one of the following types: `float32`, `float64`.
    x: A `Tensor`. Must have the same type as `a`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `a`.
  """
  result = _op_def_lib.apply_op("Igamma", a=a, x=x, name=name)
  return result


_igammac_outputs = ["z"]


def igammac(a, x, name=None):
  r"""Compute the upper regularized incomplete Gamma function `Q(a, x)`.

  The upper regularized incomplete Gamma function is defined as:

  ```
  Q(a, x) = Gamma(a, x) / Gamma(a) = 1 - P(a, x)
  ```
  where
  ```
  Gamma(a, x) = int_{x}^{\infty} t^{a-1} exp(-t) dt
  ```
  is the upper incomplete Gama function.

  Note, above `P(a, x)` (`Igamma`) is the lower regularized complete
  Gamma function.

  Args:
    a: A `Tensor`. Must be one of the following types: `float32`, `float64`.
    x: A `Tensor`. Must have the same type as `a`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `a`.
  """
  result = _op_def_lib.apply_op("Igammac", a=a, x=x, name=name)
  return result


_imag_outputs = ["output"]


def imag(input, Tout=None, name=None):
  r"""Returns the imaginary part of a complex number.

  Given a tensor `input` of complex numbers, this operation returns a tensor of
  type `float` that is the imaginary part of each element in `input`. All
  elements in `input` must be complex numbers of the form \\(a + bj\\), where *a*
  is the real part and *b* is the imaginary part returned by this operation.

  For example:

  ```
  # tensor 'input' is [-2.25 + 4.75j, 3.25 + 5.75j]
  tf.imag(input) ==> [4.75, 5.75]
  ```

  Args:
    input: A `Tensor`. Must be one of the following types: `complex64`, `complex128`.
    Tout: An optional `tf.DType` from: `tf.float32, tf.float64`. Defaults to `tf.float32`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `Tout`.
  """
  result = _op_def_lib.apply_op("Imag", input=input, Tout=Tout, name=name)
  return result


_inv_outputs = ["y"]


def inv(x, name=None):
  r"""Computes the reciprocal of x element-wise.

  I.e., \\(y = 1 / x\\).

  Args:
    x: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`, `int32`, `int64`, `complex64`, `complex128`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  result = _op_def_lib.apply_op("Inv", x=x, name=name)
  return result


__inv_grad_outputs = ["z"]


def _inv_grad(x, y, name=None):
  r"""Computes the gradient for the inverse of `x` wrt its input.

  Specifically, `grad = -dy * y*y`, where `y = 1/x`, and `dy`
  is the corresponding input gradient.

  Args:
    x: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`, `complex64`, `complex128`.
    y: A `Tensor`. Must have the same type as `x`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  result = _op_def_lib.apply_op("InvGrad", x=x, y=y, name=name)
  return result


_is_finite_outputs = ["y"]


def is_finite(x, name=None):
  r"""Returns which elements of x are finite.

  @compatibility(numpy)
  Equivalent to np.isfinite
  @end_compatibility

  Args:
    x: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `bool`.
  """
  result = _op_def_lib.apply_op("IsFinite", x=x, name=name)
  return result


_is_inf_outputs = ["y"]


def is_inf(x, name=None):
  r"""Returns which elements of x are Inf.

  @compatibility(numpy)
  Equivalent to np.isinf
  @end_compatibility

  Args:
    x: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `bool`.
  """
  result = _op_def_lib.apply_op("IsInf", x=x, name=name)
  return result


_is_nan_outputs = ["y"]


def is_nan(x, name=None):
  r"""Returns which elements of x are NaN.

  @compatibility(numpy)
  Equivalent to np.isnan
  @end_compatibility

  Args:
    x: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `bool`.
  """
  result = _op_def_lib.apply_op("IsNan", x=x, name=name)
  return result


_less_outputs = ["z"]


def less(x, y, name=None):
  r"""Returns the truth value of (x < y) element-wise.

  *NOTE*: `Less` supports broadcasting. More about broadcasting
  [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

  Args:
    x: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`, `uint8`, `int16`, `int8`, `uint16`, `half`.
    y: A `Tensor`. Must have the same type as `x`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `bool`.
  """
  result = _op_def_lib.apply_op("Less", x=x, y=y, name=name)
  return result


_less_equal_outputs = ["z"]


def less_equal(x, y, name=None):
  r"""Returns the truth value of (x <= y) element-wise.

  *NOTE*: `LessEqual` supports broadcasting. More about broadcasting
  [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

  Args:
    x: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`, `uint8`, `int16`, `int8`, `uint16`, `half`.
    y: A `Tensor`. Must have the same type as `x`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `bool`.
  """
  result = _op_def_lib.apply_op("LessEqual", x=x, y=y, name=name)
  return result


_lgamma_outputs = ["y"]


def lgamma(x, name=None):
  r"""Computes the log of the absolute value of `Gamma(x)` element-wise.

  Args:
    x: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  result = _op_def_lib.apply_op("Lgamma", x=x, name=name)
  return result


_lin_space_outputs = ["output"]


def lin_space(start, stop, num, name=None):
  r"""Generates values in an interval.

  A sequence of `num` evenly-spaced values are generated beginning at `start`.
  If `num > 1`, the values in the sequence increase by `stop - start / num - 1`,
  so that the last one is exactly `stop`.

  For example:

  ```
  tf.linspace(10.0, 12.0, 3, name="linspace") => [ 10.0  11.0  12.0]
  ```

  Args:
    start: A `Tensor`. Must be one of the following types: `float32`, `float64`.
      First entry in the range.
    stop: A `Tensor`. Must have the same type as `start`.
      Last entry in the range.
    num: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      Number of values to generate.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `start`. 1-D. The generated values.
  """
  result = _op_def_lib.apply_op("LinSpace", start=start, stop=stop, num=num,
                                name=name)
  return result


_log_outputs = ["y"]


def log(x, name=None):
  r"""Computes natural logarithm of x element-wise.

  I.e., \\(y = \log_e x\\).

  Args:
    x: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`, `complex64`, `complex128`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  result = _op_def_lib.apply_op("Log", x=x, name=name)
  return result


_log1p_outputs = ["y"]


def log1p(x, name=None):
  r"""Computes natural logarithm of (1 + x) element-wise.

  I.e., \\(y = \log_e (1 + x)\\).

  Args:
    x: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`, `complex64`, `complex128`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  result = _op_def_lib.apply_op("Log1p", x=x, name=name)
  return result


_logical_and_outputs = ["z"]


def logical_and(x, y, name=None):
  r"""Returns the truth value of x AND y element-wise.

  *NOTE*: `LogicalAnd` supports broadcasting. More about broadcasting
  [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

  Args:
    x: A `Tensor` of type `bool`.
    y: A `Tensor` of type `bool`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `bool`.
  """
  result = _op_def_lib.apply_op("LogicalAnd", x=x, y=y, name=name)
  return result


_logical_not_outputs = ["y"]


def logical_not(x, name=None):
  r"""Returns the truth value of NOT x element-wise.

  Args:
    x: A `Tensor` of type `bool`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `bool`.
  """
  result = _op_def_lib.apply_op("LogicalNot", x=x, name=name)
  return result


_logical_or_outputs = ["z"]


def logical_or(x, y, name=None):
  r"""Returns the truth value of x OR y element-wise.

  *NOTE*: `LogicalOr` supports broadcasting. More about broadcasting
  [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

  Args:
    x: A `Tensor` of type `bool`.
    y: A `Tensor` of type `bool`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `bool`.
  """
  result = _op_def_lib.apply_op("LogicalOr", x=x, y=y, name=name)
  return result


__mat_mul_outputs = ["product"]


def _mat_mul(a, b, transpose_a=None, transpose_b=None, name=None):
  r"""Multiply the matrix "a" by the matrix "b".

  The inputs must be two-dimensional matrices and the inner dimension of
  "a" (after being transposed if transpose_a is true) must match the
  outer dimension of "b" (after being transposed if transposed_b is
  true).

  *Note*: The default kernel implementation for MatMul on GPUs uses
  cublas.

  Args:
    a: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`, `int32`, `complex64`, `complex128`.
    b: A `Tensor`. Must have the same type as `a`.
    transpose_a: An optional `bool`. Defaults to `False`.
      If true, "a" is transposed before multiplication.
    transpose_b: An optional `bool`. Defaults to `False`.
      If true, "b" is transposed before multiplication.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `a`.
  """
  result = _op_def_lib.apply_op("MatMul", a=a, b=b, transpose_a=transpose_a,
                                transpose_b=transpose_b, name=name)
  return result


__max_outputs = ["output"]


def _max(input, reduction_indices, keep_dims=None, name=None):
  r"""Computes the maximum of elements across dimensions of a tensor.

  Reduces `input` along the dimensions given in `reduction_indices`. Unless
  `keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
  `reduction_indices`. If `keep_dims` is true, the reduced dimensions are
  retained with length 1.

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`, `complex128`, `qint8`, `quint8`, `qint32`, `half`.
      The tensor to reduce.
    reduction_indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      The dimensions to reduce.
    keep_dims: An optional `bool`. Defaults to `False`.
      If true, retain reduced dimensions with length 1.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`. The reduced tensor.
  """
  result = _op_def_lib.apply_op("Max", input=input,
                                reduction_indices=reduction_indices,
                                keep_dims=keep_dims, name=name)
  return result


_maximum_outputs = ["z"]


def maximum(x, y, name=None):
  r"""Returns the max of x and y (i.e. x > y ? x : y) element-wise.

  *NOTE*: `Maximum` supports broadcasting. More about broadcasting
  [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

  Args:
    x: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`, `int32`, `int64`.
    y: A `Tensor`. Must have the same type as `x`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  result = _op_def_lib.apply_op("Maximum", x=x, y=y, name=name)
  return result


__mean_outputs = ["output"]


def _mean(input, reduction_indices, keep_dims=None, name=None):
  r"""Computes the mean of elements across dimensions of a tensor.

  Reduces `input` along the dimensions given in `reduction_indices`. Unless
  `keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
  `reduction_indices`. If `keep_dims` is true, the reduced dimensions are
  retained with length 1.

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`, `complex128`, `qint8`, `quint8`, `qint32`, `half`.
      The tensor to reduce.
    reduction_indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      The dimensions to reduce.
    keep_dims: An optional `bool`. Defaults to `False`.
      If true, retain reduced dimensions with length 1.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`. The reduced tensor.
  """
  result = _op_def_lib.apply_op("Mean", input=input,
                                reduction_indices=reduction_indices,
                                keep_dims=keep_dims, name=name)
  return result


__min_outputs = ["output"]


def _min(input, reduction_indices, keep_dims=None, name=None):
  r"""Computes the minimum of elements across dimensions of a tensor.

  Reduces `input` along the dimensions given in `reduction_indices`. Unless
  `keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
  `reduction_indices`. If `keep_dims` is true, the reduced dimensions are
  retained with length 1.

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`, `complex128`, `qint8`, `quint8`, `qint32`, `half`.
      The tensor to reduce.
    reduction_indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      The dimensions to reduce.
    keep_dims: An optional `bool`. Defaults to `False`.
      If true, retain reduced dimensions with length 1.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`. The reduced tensor.
  """
  result = _op_def_lib.apply_op("Min", input=input,
                                reduction_indices=reduction_indices,
                                keep_dims=keep_dims, name=name)
  return result


_minimum_outputs = ["z"]


def minimum(x, y, name=None):
  r"""Returns the min of x and y (i.e. x < y ? x : y) element-wise.

  *NOTE*: `Minimum` supports broadcasting. More about broadcasting
  [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

  Args:
    x: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`, `int32`, `int64`.
    y: A `Tensor`. Must have the same type as `x`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  result = _op_def_lib.apply_op("Minimum", x=x, y=y, name=name)
  return result


_mod_outputs = ["z"]


def mod(x, y, name=None):
  r"""Returns element-wise remainder of division.

  *NOTE*: `Mod` supports broadcasting. More about broadcasting
  [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

  Args:
    x: A `Tensor`. Must be one of the following types: `int32`, `int64`, `float32`, `float64`.
    y: A `Tensor`. Must have the same type as `x`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  result = _op_def_lib.apply_op("Mod", x=x, y=y, name=name)
  return result


__mul_outputs = ["z"]


def _mul(x, y, name=None):
  r"""Returns x * y element-wise.

  *NOTE*: `Mul` supports broadcasting. More about broadcasting
  [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

  Args:
    x: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`, `uint8`, `int8`, `uint16`, `int16`, `int32`, `int64`, `complex64`, `complex128`.
    y: A `Tensor`. Must have the same type as `x`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  result = _op_def_lib.apply_op("Mul", x=x, y=y, name=name)
  return result


__neg_outputs = ["y"]


def _neg(x, name=None):
  r"""Computes numerical negative value element-wise.

  I.e., \\(y = -x\\).

  Args:
    x: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`, `int32`, `int64`, `complex64`, `complex128`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  result = _op_def_lib.apply_op("Neg", x=x, name=name)
  return result


_not_equal_outputs = ["z"]


def not_equal(x, y, name=None):
  r"""Returns the truth value of (x != y) element-wise.

  *NOTE*: `NotEqual` supports broadcasting. More about broadcasting
  [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

  Args:
    x: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`, `uint8`, `int8`, `int16`, `int32`, `int64`, `complex64`, `quint8`, `qint8`, `qint32`, `string`, `bool`, `complex128`.
    y: A `Tensor`. Must have the same type as `x`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `bool`.
  """
  result = _op_def_lib.apply_op("NotEqual", x=x, y=y, name=name)
  return result


_polygamma_outputs = ["z"]


def polygamma(a, x, name=None):
  r"""Compute the polygamma function \\(\psi^{(n)}(x)\\).

  The polygamma function is defined as:

  ```
  \psi^{(n)}(x) = \frac{d^n}{dx^n} \psi(x)
  ```
  where \\(\psi(x)\\) is the digamma function.

  Args:
    a: A `Tensor`. Must be one of the following types: `float32`, `float64`.
    x: A `Tensor`. Must have the same type as `a`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `a`.
  """
  result = _op_def_lib.apply_op("Polygamma", a=a, x=x, name=name)
  return result


__pow_outputs = ["z"]


def _pow(x, y, name=None):
  r"""Computes the power of one value to another.

  Given a tensor `x` and a tensor `y`, this operation computes \\(x^y\\) for
  corresponding elements in `x` and `y`. For example:

  ```
  # tensor 'x' is [[2, 2]], [3, 3]]
  # tensor 'y' is [[8, 16], [2, 3]]
  tf.pow(x, y) ==> [[256, 65536], [9, 27]]
  ```

  Args:
    x: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`, `int32`, `int64`, `complex64`, `complex128`.
    y: A `Tensor`. Must have the same type as `x`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  result = _op_def_lib.apply_op("Pow", x=x, y=y, name=name)
  return result


__prod_outputs = ["output"]


def _prod(input, reduction_indices, keep_dims=None, name=None):
  r"""Computes the product of elements across dimensions of a tensor.

  Reduces `input` along the dimensions given in `reduction_indices`. Unless
  `keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
  `reduction_indices`. If `keep_dims` is true, the reduced dimensions are
  retained with length 1.

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`, `complex128`, `qint8`, `quint8`, `qint32`, `half`.
      The tensor to reduce.
    reduction_indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      The dimensions to reduce.
    keep_dims: An optional `bool`. Defaults to `False`.
      If true, retain reduced dimensions with length 1.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`. The reduced tensor.
  """
  result = _op_def_lib.apply_op("Prod", input=input,
                                reduction_indices=reduction_indices,
                                keep_dims=keep_dims, name=name)
  return result


_quantize_down_and_shrink_range_outputs = ["output", "output_min",
                                          "output_max"]


_QuantizeDownAndShrinkRangeOutput = _collections.namedtuple("QuantizeDownAndShrinkRange",
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


_quantized_mat_mul_outputs = ["out", "min_out", "max_out"]


_QuantizedMatMulOutput = _collections.namedtuple("QuantizedMatMul",
                                                 _quantized_mat_mul_outputs)


def quantized_mat_mul(a, b, min_a, max_a, min_b, max_b, Toutput=None,
                      transpose_a=None, transpose_b=None, Tactivation=None,
                      name=None):
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
    Tactivation: An optional `tf.DType` from: `tf.qint8, tf.quint8, tf.qint16, tf.quint16, tf.qint32`. Defaults to `tf.quint8`.
      The type of output produced by activation function
      following this operation.
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
                                transpose_b=transpose_b,
                                Tactivation=Tactivation, name=name)
  return _QuantizedMatMulOutput._make(result)


__range_outputs = ["output"]


def _range(start, limit, delta, name=None):
  r"""Creates a sequence of numbers.

  This operation creates a sequence of numbers that begins at `start` and
  extends by increments of `delta` up to but not including `limit`.

  For example:

  ```
  # 'start' is 3
  # 'limit' is 18
  # 'delta' is 3
  tf.range(start, limit, delta) ==> [3, 6, 9, 12, 15]
  ```

  Args:
    start: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`.
      0-D (scalar). First entry in the sequence.
    limit: A `Tensor`. Must have the same type as `start`.
      0-D (scalar). Upper limit of sequence, exclusive.
    delta: A `Tensor`. Must have the same type as `start`.
      0-D (scalar). Optional. Default is 1. Number that increments `start`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `start`. 1-D.
  """
  result = _op_def_lib.apply_op("Range", start=start, limit=limit,
                                delta=delta, name=name)
  return result


_real_outputs = ["output"]


def real(input, Tout=None, name=None):
  r"""Returns the real part of a complex number.

  Given a tensor `input` of complex numbers, this operation returns a tensor of
  type `float` that is the real part of each element in `input`. All elements in
  `input` must be complex numbers of the form \\(a + bj\\), where *a* is the real
   part returned by this operation and *b* is the imaginary part.

  For example:

  ```
  # tensor 'input' is [-2.25 + 4.75j, 3.25 + 5.75j]
  tf.real(input) ==> [-2.25, 3.25]
  ```

  Args:
    input: A `Tensor`. Must be one of the following types: `complex64`, `complex128`.
    Tout: An optional `tf.DType` from: `tf.float32, tf.float64`. Defaults to `tf.float32`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `Tout`.
  """
  result = _op_def_lib.apply_op("Real", input=input, Tout=Tout, name=name)
  return result


__real_div_outputs = ["z"]


def _real_div(x, y, name=None):
  r"""Returns x / y element-wise for real types.

  If `x` and `y` are reals, this will return the floating-point division.

  *NOTE*: `Div` supports broadcasting. More about broadcasting
  [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

  Args:
    x: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`, `uint8`, `int8`, `uint16`, `int16`, `int32`, `int64`, `complex64`, `complex128`.
    y: A `Tensor`. Must have the same type as `x`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  result = _op_def_lib.apply_op("RealDiv", x=x, y=y, name=name)
  return result


_reciprocal_outputs = ["y"]


def reciprocal(x, name=None):
  r"""Computes the reciprocal of x element-wise.

  I.e., \\(y = 1 / x\\).

  Args:
    x: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`, `int32`, `int64`, `complex64`, `complex128`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  result = _op_def_lib.apply_op("Reciprocal", x=x, name=name)
  return result


__reciprocal_grad_outputs = ["z"]


def _reciprocal_grad(x, y, name=None):
  r"""Computes the gradient for the inverse of `x` wrt its input.

  Specifically, `grad = -dy * y*y`, where `y = 1/x`, and `dy`
  is the corresponding input gradient.

  Args:
    x: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`, `complex64`, `complex128`.
    y: A `Tensor`. Must have the same type as `x`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  result = _op_def_lib.apply_op("ReciprocalGrad", x=x, y=y, name=name)
  return result


_requantization_range_outputs = ["output_min", "output_max"]


_RequantizationRangeOutput = _collections.namedtuple("RequantizationRange",
                                                     _requantization_range_outputs)


def requantization_range(input, input_min, input_max, name=None):
  r"""Given a quantized tensor described by (input, input_min, input_max), outputs a

  range that covers the actual values present in that tensor.  This op is
  typically used to produce the requested_output_min and requested_output_max for
  Requantize.

  Args:
    input: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint16`, `quint16`, `qint32`.
    input_min: A `Tensor` of type `float32`.
      The float value that the minimum quantized input value represents.
    input_max: A `Tensor` of type `float32`.
      The float value that the maximum quantized input value represents.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output_min, output_max).
    output_min: A `Tensor` of type `float32`. The computed min output.
    output_max: A `Tensor` of type `float32`. the computed max output.
  """
  result = _op_def_lib.apply_op("RequantizationRange", input=input,
                                input_min=input_min, input_max=input_max,
                                name=name)
  return _RequantizationRangeOutput._make(result)


_requantize_outputs = ["output", "output_min", "output_max"]


_RequantizeOutput = _collections.namedtuple("Requantize", _requantize_outputs)


def requantize(input, input_min, input_max, requested_output_min,
               requested_output_max, out_type, name=None):
  r"""Convert the quantized 'input' tensor into a lower-precision 'output', using the

  output range specified with 'requested_output_min' and 'requested_output_max'.

  [input_min, input_max] are scalar floats that specify the range for the float
  interpretation of the 'input' data. For example, if input_min is -1.0f and
  input_max is 1.0f, and we are dealing with quint16 quantized data, then a 0
  value in the 16-bit data should be interpreted as -1.0f, and a 65535 means 1.0f.

  Args:
    input: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint16`, `quint16`, `qint32`.
    input_min: A `Tensor` of type `float32`.
      The float value that the minimum quantized input value represents.
    input_max: A `Tensor` of type `float32`.
      The float value that the maximum quantized input value represents.
    requested_output_min: A `Tensor` of type `float32`.
      The float value that the minimum quantized output value represents.
    requested_output_max: A `Tensor` of type `float32`.
      The float value that the maximum quantized output value represents.
    out_type: A `tf.DType` from: `tf.qint8, tf.quint8, tf.qint16, tf.quint16, tf.qint32`.
      The type of the output. Should be a lower bit depth than Tinput.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output, output_min, output_max).
    output: A `Tensor` of type `out_type`.
    output_min: A `Tensor` of type `float32`. The requested_output_min value is copied into this output.
    output_max: A `Tensor` of type `float32`. The requested_output_max value is copied into this output.
  """
  result = _op_def_lib.apply_op("Requantize", input=input,
                                input_min=input_min, input_max=input_max,
                                requested_output_min=requested_output_min,
                                requested_output_max=requested_output_max,
                                out_type=out_type, name=name)
  return _RequantizeOutput._make(result)


_rint_outputs = ["y"]


def rint(x, name=None):
  r"""Returns element-wise integer closest to x.

  If the result is midway between two representable values,
  the even representable is chosen.
  For example:

  ```
  rint(-1.5) ==> -2.0
  rint(0.5000001) ==> 1.0
  rint([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0]) ==> [-2., -2., -0., 0., 2., 2., 2.]
  ```

  Args:
    x: A `Tensor`. Must be one of the following types: `float32`, `float64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  result = _op_def_lib.apply_op("Rint", x=x, name=name)
  return result


_round_outputs = ["y"]


def round(x, name=None):
  r"""Rounds the values of a tensor to the nearest integer, element-wise.

  Rounds half to even.  Also known as bankers rounding. If you want to round
  according to the current system rounding mode use std::cint.

  Args:
    x: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`, `int32`, `int64`, `complex64`, `complex128`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  result = _op_def_lib.apply_op("Round", x=x, name=name)
  return result


_rsqrt_outputs = ["y"]


def rsqrt(x, name=None):
  r"""Computes reciprocal of square root of x element-wise.

  I.e., \\(y = 1 / \sqrt{x}\\).

  Args:
    x: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`, `complex64`, `complex128`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  result = _op_def_lib.apply_op("Rsqrt", x=x, name=name)
  return result


__rsqrt_grad_outputs = ["z"]


def _rsqrt_grad(x, y, name=None):
  r"""Computes the gradient for the rsqrt of `x` wrt its input.

  Specifically, `grad = dy * -0.5 * y^3`, where `y = rsqrt(x)`, and `dy`
  is the corresponding input gradient.

  Args:
    x: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`, `complex64`, `complex128`.
    y: A `Tensor`. Must have the same type as `x`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  result = _op_def_lib.apply_op("RsqrtGrad", x=x, y=y, name=name)
  return result


_segment_max_outputs = ["output"]


def segment_max(data, segment_ids, name=None):
  r"""Computes the maximum along segments of a tensor.

  Read [the section on Segmentation](../../api_docs/python/math_ops.md#segmentation)
  for an explanation of segments.

  Computes a tensor such that
  \\(output_i = \max_j(data_j)\\) where `max` is over `j` such
  that `segment_ids[j] == i`.

  <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
  <img style="width:100%" src="../../images/SegmentMax.png" alt>
  </div>

  Args:
    data: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`, `uint8`, `int16`, `int8`, `uint16`, `half`.
    segment_ids: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A 1-D tensor whose rank is equal to the rank of `data`'s
      first dimension.  Values should be sorted and can be repeated.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `data`.
    Has same shape as data, except for dimension 0 which
    has size `k`, the number of segments.
  """
  result = _op_def_lib.apply_op("SegmentMax", data=data,
                                segment_ids=segment_ids, name=name)
  return result


_segment_mean_outputs = ["output"]


def segment_mean(data, segment_ids, name=None):
  r"""Computes the mean along segments of a tensor.

  Read [the section on
  Segmentation](../../api_docs/python/math_ops.md#segmentation) for an explanation
  of segments.

  Computes a tensor such that
  \\(output_i = \frac{\sum_j data_j}{N}\\) where `mean` is
  over `j` such that `segment_ids[j] == i` and `N` is the total number of
  values summed.

  <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
  <img style="width:100%" src="../../images/SegmentMean.png" alt>
  </div>

  Args:
    data: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`, `uint8`, `int16`, `int8`, `uint16`, `half`.
    segment_ids: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A 1-D tensor whose rank is equal to the rank of `data`'s
      first dimension.  Values should be sorted and can be repeated.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `data`.
    Has same shape as data, except for dimension 0 which
    has size `k`, the number of segments.
  """
  result = _op_def_lib.apply_op("SegmentMean", data=data,
                                segment_ids=segment_ids, name=name)
  return result


_segment_min_outputs = ["output"]


def segment_min(data, segment_ids, name=None):
  r"""Computes the minimum along segments of a tensor.

  Read [the section on
  Segmentation](../../api_docs/python/math_ops.md#segmentation) for an explanation
  of segments.

  Computes a tensor such that
  \\(output_i = \min_j(data_j)\\) where `min` is over `j` such
  that `segment_ids[j] == i`.

  <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
  <img style="width:100%" src="../../images/SegmentMin.png" alt>
  </div>

  Args:
    data: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`, `uint8`, `int16`, `int8`, `uint16`, `half`.
    segment_ids: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A 1-D tensor whose rank is equal to the rank of `data`'s
      first dimension.  Values should be sorted and can be repeated.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `data`.
    Has same shape as data, except for dimension 0 which
    has size `k`, the number of segments.
  """
  result = _op_def_lib.apply_op("SegmentMin", data=data,
                                segment_ids=segment_ids, name=name)
  return result


_segment_prod_outputs = ["output"]


def segment_prod(data, segment_ids, name=None):
  r"""Computes the product along segments of a tensor.

  Read [the section on
  Segmentation](../../api_docs/python/math_ops.md#segmentation) for an explanation
  of segments.

  Computes a tensor such that
  \\(output_i = \prod_j data_j\\) where the product is over `j` such
  that `segment_ids[j] == i`.

  <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
  <img style="width:100%" src="../../images/SegmentProd.png" alt>
  </div>

  Args:
    data: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`, `complex128`, `qint8`, `quint8`, `qint32`, `half`.
    segment_ids: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A 1-D tensor whose rank is equal to the rank of `data`'s
      first dimension.  Values should be sorted and can be repeated.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `data`.
    Has same shape as data, except for dimension 0 which
    has size `k`, the number of segments.
  """
  result = _op_def_lib.apply_op("SegmentProd", data=data,
                                segment_ids=segment_ids, name=name)
  return result


_segment_sum_outputs = ["output"]


def segment_sum(data, segment_ids, name=None):
  r"""Computes the sum along segments of a tensor.

  Read [the section on Segmentation](../../api_docs/python/math_ops.md#segmentation)
  for an explanation of segments.

  Computes a tensor such that
  \\(output_i = \sum_j data_j\\) where sum is over `j` such
  that `segment_ids[j] == i`.

  <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
  <img style="width:100%" src="../../images/SegmentSum.png" alt>
  </div>

  Args:
    data: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`, `complex128`, `qint8`, `quint8`, `qint32`, `half`.
    segment_ids: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A 1-D tensor whose rank is equal to the rank of `data`'s
      first dimension.  Values should be sorted and can be repeated.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `data`.
    Has same shape as data, except for dimension 0 which
    has size `k`, the number of segments.
  """
  result = _op_def_lib.apply_op("SegmentSum", data=data,
                                segment_ids=segment_ids, name=name)
  return result


__select_outputs = ["output"]


def _select(condition, t, e, name=None):
  r"""Selects elements from `t` or `e`, depending on `condition`.

  The `t`, and `e` tensors must all have the same shape, and the
  output will also have that shape.

  The `condition` tensor must be a scalar if `t` and `e` are scalars.
  If `t` and `e` are vectors or higher rank, then `condition` must be either a
  scalar, a vector with size matching the first dimension of `t`, or must have
  the same shape as `t`.

  The `condition` tensor acts as a mask that chooses, based on the value at each
  element, whether the corresponding element / row in the output should be
  taken from `t` (if true) or `e` (if false).

  If `condition` is a vector and `t` and `e` are higher rank matrices, then
  it chooses which row (outer dimension) to copy from `t` and `e`.
  If `condition` has the same shape as `t` and `e`, then it chooses which
  element to copy from `t` and `e`.

  For example:

  ```prettyprint
  # 'condition' tensor is [[True,  False]
  #                        [False, True]]
  # 't' is [[1, 2],
  #         [3, 4]]
  # 'e' is [[5, 6],
  #         [7, 8]]
  select(condition, t, e) ==> [[1, 6],
                               [7, 4]]


  # 'condition' tensor is [True, False]
  # 't' is [[1, 2],
  #         [3, 4]]
  # 'e' is [[5, 6],
  #         [7, 8]]
  select(condition, t, e) ==> [[1, 2],
                               [7, 8]]

  ```

  Args:
    condition: A `Tensor` of type `bool`.
    t:  A `Tensor` which may have the same shape as `condition`.
      If `condition` is rank 1, `t` may have higher rank,
      but its first dimension must match the size of `condition`.
    e:  A `Tensor` with the same type and shape as `t`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` with the same type and shape as `t` and `e`.
  """
  result = _op_def_lib.apply_op("Select", condition=condition, t=t, e=e,
                                name=name)
  return result


__sigmoid_outputs = ["y"]


def _sigmoid(x, name=None):
  r"""Computes sigmoid of `x` element-wise.

  Specifically, `y = 1 / (1 + exp(-x))`.

  Args:
    x: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`, `complex64`, `complex128`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  result = _op_def_lib.apply_op("Sigmoid", x=x, name=name)
  return result


__sigmoid_grad_outputs = ["z"]


def _sigmoid_grad(x, y, name=None):
  r"""Computes the gradient of the sigmoid of `x` wrt its input.

  Specifically, `grad = dy * y * (1 - y)`, where `y = sigmoid(x)`, and
  `dy` is the corresponding input gradient.

  Args:
    x: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`, `complex64`, `complex128`.
    y: A `Tensor`. Must have the same type as `x`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  result = _op_def_lib.apply_op("SigmoidGrad", x=x, y=y, name=name)
  return result


_sign_outputs = ["y"]


def sign(x, name=None):
  r"""Returns an element-wise indication of the sign of a number.

  `y = sign(x) = -1` if `x < 0`; 0 if `x == 0`; 1 if `x > 0`.

  For complex numbers, `y = sign(x) = x / |x|` if `x != 0`, otherwise `y = 0`.

  Args:
    x: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`, `int32`, `int64`, `complex64`, `complex128`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  result = _op_def_lib.apply_op("Sign", x=x, name=name)
  return result


_sin_outputs = ["y"]


def sin(x, name=None):
  r"""Computes sin of x element-wise.

  Args:
    x: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`, `complex64`, `complex128`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  result = _op_def_lib.apply_op("Sin", x=x, name=name)
  return result


__sparse_mat_mul_outputs = ["product"]


def _sparse_mat_mul(a, b, transpose_a=None, transpose_b=None,
                    a_is_sparse=None, b_is_sparse=None, name=None):
  r"""Multiply matrix "a" by matrix "b".

  The inputs must be two-dimensional matrices and the inner dimension of "a" must
  match the outer dimension of "b". This op is optimized for the case where at
  least one of "a" or "b" is sparse. The breakeven for using this versus a dense
  matrix multiply on one platform was 30% zero values in the sparse matrix.

  Args:
    a: A `Tensor`. Must be one of the following types: `float32`, `bfloat16`.
    b: A `Tensor`. Must be one of the following types: `float32`, `bfloat16`.
    transpose_a: An optional `bool`. Defaults to `False`.
    transpose_b: An optional `bool`. Defaults to `False`.
    a_is_sparse: An optional `bool`. Defaults to `False`.
    b_is_sparse: An optional `bool`. Defaults to `False`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32`.
  """
  result = _op_def_lib.apply_op("SparseMatMul", a=a, b=b,
                                transpose_a=transpose_a,
                                transpose_b=transpose_b,
                                a_is_sparse=a_is_sparse,
                                b_is_sparse=b_is_sparse, name=name)
  return result


_sparse_segment_mean_outputs = ["output"]


def sparse_segment_mean(data, indices, segment_ids, name=None):
  r"""Computes the mean along sparse segments of a tensor.

  Read [the section on
  Segmentation](../../api_docs/python/math_ops.md#segmentation) for an explanation
  of segments.

  Like `SegmentMean`, but `segment_ids` can have rank less than `data`'s first
  dimension, selecting a subset of dimension 0, specified by `indices`.

  Args:
    data: A `Tensor`. Must be one of the following types: `float32`, `float64`.
    indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A 1-D tensor. Has same rank as `segment_ids`.
    segment_ids: A `Tensor` of type `int32`.
      A 1-D tensor. Values should be sorted and can be repeated.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `data`.
    Has same shape as data, except for dimension 0 which
    has size `k`, the number of segments.
  """
  result = _op_def_lib.apply_op("SparseSegmentMean", data=data,
                                indices=indices, segment_ids=segment_ids,
                                name=name)
  return result


_sparse_segment_mean_grad_outputs = ["output"]


def sparse_segment_mean_grad(grad, indices, segment_ids, output_dim0,
                             name=None):
  r"""Computes gradients for SparseSegmentMean.

  Returns tensor "output" with same shape as grad, except for dimension 0 whose
  value is output_dim0.

  Args:
    grad: A `Tensor`. Must be one of the following types: `float32`, `float64`.
      gradient propagated to the SparseSegmentMean op.
    indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      indices passed to the corresponding SparseSegmentMean op.
    segment_ids: A `Tensor` of type `int32`.
      segment_ids passed to the corresponding SparseSegmentMean op.
    output_dim0: A `Tensor` of type `int32`.
      dimension 0 of "data" passed to SparseSegmentMean op.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `grad`.
  """
  result = _op_def_lib.apply_op("SparseSegmentMeanGrad", grad=grad,
                                indices=indices, segment_ids=segment_ids,
                                output_dim0=output_dim0, name=name)
  return result


_sparse_segment_sqrt_n_outputs = ["output"]


def sparse_segment_sqrt_n(data, indices, segment_ids, name=None):
  r"""Computes the sum along sparse segments of a tensor divided by the sqrt of N.

  N is the size of the segment being reduced.

  Read [the section on
  Segmentation](../../api_docs/python/math_ops.md#segmentation) for an explanation
  of segments.

  Args:
    data: A `Tensor`. Must be one of the following types: `float32`, `float64`.
    indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A 1-D tensor. Has same rank as `segment_ids`.
    segment_ids: A `Tensor` of type `int32`.
      A 1-D tensor. Values should be sorted and can be repeated.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `data`.
    Has same shape as data, except for dimension 0 which
    has size `k`, the number of segments.
  """
  result = _op_def_lib.apply_op("SparseSegmentSqrtN", data=data,
                                indices=indices, segment_ids=segment_ids,
                                name=name)
  return result


_sparse_segment_sqrt_n_grad_outputs = ["output"]


def sparse_segment_sqrt_n_grad(grad, indices, segment_ids, output_dim0,
                               name=None):
  r"""Computes gradients for SparseSegmentSqrtN.

  Returns tensor "output" with same shape as grad, except for dimension 0 whose
  value is output_dim0.

  Args:
    grad: A `Tensor`. Must be one of the following types: `float32`, `float64`.
      gradient propagated to the SparseSegmentSqrtN op.
    indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      indices passed to the corresponding SparseSegmentSqrtN op.
    segment_ids: A `Tensor` of type `int32`.
      segment_ids passed to the corresponding SparseSegmentSqrtN op.
    output_dim0: A `Tensor` of type `int32`.
      dimension 0 of "data" passed to SparseSegmentSqrtN op.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `grad`.
  """
  result = _op_def_lib.apply_op("SparseSegmentSqrtNGrad", grad=grad,
                                indices=indices, segment_ids=segment_ids,
                                output_dim0=output_dim0, name=name)
  return result


_sparse_segment_sum_outputs = ["output"]


def sparse_segment_sum(data, indices, segment_ids, name=None):
  r"""Computes the sum along sparse segments of a tensor.

  Read [the section on
  Segmentation](../../api_docs/python/math_ops.md#segmentation) for an explanation
  of segments.

  Like `SegmentSum`, but `segment_ids` can have rank less than `data`'s first
  dimension, selecting a subset of dimension 0, specified by `indices`.

  For example:

  ```prettyprint
  c = tf.constant([[1,2,3,4], [-1,-2,-3,-4], [5,6,7,8]])

  # Select two rows, one segment.
  tf.sparse_segment_sum(c, tf.constant([0, 1]), tf.constant([0, 0]))
    ==> [[0 0 0 0]]

  # Select two rows, two segment.
  tf.sparse_segment_sum(c, tf.constant([0, 1]), tf.constant([0, 1]))
    ==> [[ 1  2  3  4]
         [-1 -2 -3 -4]]

  # Select all rows, two segments.
  tf.sparse_segment_sum(c, tf.constant([0, 1, 2]), tf.constant([0, 0, 1]))
    ==> [[0 0 0 0]
         [5 6 7 8]]

  # Which is equivalent to:
  tf.segment_sum(c, tf.constant([0, 0, 1]))
  ```

  Args:
    data: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`, `uint8`, `int16`, `int8`, `uint16`, `half`.
    indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A 1-D tensor. Has same rank as `segment_ids`.
    segment_ids: A `Tensor` of type `int32`.
      A 1-D tensor. Values should be sorted and can be repeated.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `data`.
    Has same shape as data, except for dimension 0 which
    has size `k`, the number of segments.
  """
  result = _op_def_lib.apply_op("SparseSegmentSum", data=data,
                                indices=indices, segment_ids=segment_ids,
                                name=name)
  return result


_sqrt_outputs = ["y"]


def sqrt(x, name=None):
  r"""Computes square root of x element-wise.

  I.e., \\(y = \sqrt{x} = x^{1/2}\\).

  Args:
    x: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`, `complex64`, `complex128`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  result = _op_def_lib.apply_op("Sqrt", x=x, name=name)
  return result


__sqrt_grad_outputs = ["z"]


def _sqrt_grad(x, y, name=None):
  r"""Computes the gradient for the sqrt of `x` wrt its input.

  Specifically, `grad = dy * 0.5 / y`, where `y = sqrt(x)`, and `dy`
  is the corresponding input gradient.

  Args:
    x: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`, `complex64`, `complex128`.
    y: A `Tensor`. Must have the same type as `x`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  result = _op_def_lib.apply_op("SqrtGrad", x=x, y=y, name=name)
  return result


_square_outputs = ["y"]


def square(x, name=None):
  r"""Computes square of x element-wise.

  I.e., \\(y = x * x = x^2\\).

  Args:
    x: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`, `int32`, `int64`, `complex64`, `complex128`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  result = _op_def_lib.apply_op("Square", x=x, name=name)
  return result


_squared_difference_outputs = ["z"]


def squared_difference(x, y, name=None):
  r"""Returns (x - y)(x - y) element-wise.

  *NOTE*: `SquaredDifference` supports broadcasting. More about broadcasting
  [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

  Args:
    x: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`, `int32`, `int64`, `complex64`, `complex128`.
    y: A `Tensor`. Must have the same type as `x`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  result = _op_def_lib.apply_op("SquaredDifference", x=x, y=y, name=name)
  return result


__sub_outputs = ["z"]


def _sub(x, y, name=None):
  r"""Returns x - y element-wise.

  *NOTE*: `Sub` supports broadcasting. More about broadcasting
  [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

  Args:
    x: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`, `int32`, `int64`, `complex64`, `complex128`.
    y: A `Tensor`. Must have the same type as `x`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  result = _op_def_lib.apply_op("Sub", x=x, y=y, name=name)
  return result


__sum_outputs = ["output"]


def _sum(input, reduction_indices, keep_dims=None, name=None):
  r"""Computes the sum of elements across dimensions of a tensor.

  Reduces `input` along the dimensions given in `reduction_indices`. Unless
  `keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
  `reduction_indices`. If `keep_dims` is true, the reduced dimensions are
  retained with length 1.

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`, `complex128`, `qint8`, `quint8`, `qint32`, `half`.
      The tensor to reduce.
    reduction_indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      The dimensions to reduce.
    keep_dims: An optional `bool`. Defaults to `False`.
      If true, retain reduced dimensions with length 1.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`. The reduced tensor.
  """
  result = _op_def_lib.apply_op("Sum", input=input,
                                reduction_indices=reduction_indices,
                                keep_dims=keep_dims, name=name)
  return result


_tan_outputs = ["y"]


def tan(x, name=None):
  r"""Computes tan of x element-wise.

  Args:
    x: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`, `int32`, `int64`, `complex64`, `complex128`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  result = _op_def_lib.apply_op("Tan", x=x, name=name)
  return result


__tanh_outputs = ["y"]


def _tanh(x, name=None):
  r"""Computes hyperbolic tangent of `x` element-wise.

  Args:
    x: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`, `complex64`, `complex128`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  result = _op_def_lib.apply_op("Tanh", x=x, name=name)
  return result


__tanh_grad_outputs = ["z"]


def _tanh_grad(x, y, name=None):
  r"""Computes the gradient for the tanh of `x` wrt its input.

  Specifically, `grad = dy * (1 - y*y)`, where `y = tanh(x)`, and `dy`
  is the corresponding input gradient.

  Args:
    x: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`, `complex64`, `complex128`.
    y: A `Tensor`. Must have the same type as `x`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  result = _op_def_lib.apply_op("TanhGrad", x=x, y=y, name=name)
  return result


__truncate_div_outputs = ["z"]


def _truncate_div(x, y, name=None):
  r"""Returns x / y element-wise for integer types.

  Truncation designates that negative numbers will round fractional quantities
  toward zero. I.e. -7 / 5 = 1. This matches C semantics but it is different
  than Python semantics. See `FloorDiv` for a division function that matches
  Python Semantics.

  *NOTE*: `TruncateDiv` supports broadcasting. More about broadcasting
  [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

  Args:
    x: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`, `uint8`, `int8`, `uint16`, `int16`, `int32`, `int64`, `complex64`, `complex128`.
    y: A `Tensor`. Must have the same type as `x`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  result = _op_def_lib.apply_op("TruncateDiv", x=x, y=y, name=name)
  return result


__truncate_mod_outputs = ["z"]


def _truncate_mod(x, y, name=None):
  r"""Returns element-wise remainder of division. This emulates C semantics where

  true, this follows C semantics in that the result here is consistent
  with a flooring divide. E.g. `floor(x / y) * y + mod(x, y) = x`.

  *NOTE*: `Mod` supports broadcasting. More about broadcasting
  [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

  Args:
    x: A `Tensor`. Must be one of the following types: `int32`, `int64`, `float32`, `float64`.
    y: A `Tensor`. Must have the same type as `x`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  result = _op_def_lib.apply_op("TruncateMod", x=x, y=y, name=name)
  return result


_unsorted_segment_sum_outputs = ["output"]


def unsorted_segment_sum(data, segment_ids, num_segments, name=None):
  r"""Computes the sum along segments of a tensor.

  Read [the section on
  Segmentation](../../api_docs/python/math_ops.md#segmentation) for an explanation
  of segments.

  Computes a tensor such that
  `(output[i] = sum_{j...} data[j...]` where the sum is over tuples `j...` such
  that `segment_ids[j...] == i`.  Unlike `SegmentSum`, `segment_ids`
  need not be sorted and need not cover all values in the full
  range of valid values.

  If the sum is empty for a given segment ID `i`, `output[i] = 0`.

  `num_segments` should equal the number of distinct segment IDs.

  <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
  <img style="width:100%" src="../../images/UnsortedSegmentSum.png" alt>
  </div>

  Args:
    data: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`, `complex128`, `qint8`, `quint8`, `qint32`, `half`.
    segment_ids: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A tensor whose shape is a prefix of `data.shape`.
    num_segments: A `Tensor` of type `int32`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `data`.
    Has same shape as data, except for the first `segment_ids.rank`
    dimensions, which are replaced with a single dimension which has size
    `num_segments`.
  """
  result = _op_def_lib.apply_op("UnsortedSegmentSum", data=data,
                                segment_ids=segment_ids,
                                num_segments=num_segments, name=name)
  return result


_zeta_outputs = ["z"]


def zeta(x, q, name=None):
  r"""Compute the Hurwitz zeta function \\(\zeta(x, q)\\).

  The Hurwitz zeta function is defined as:

  ```
  \zeta(x, q) = \sum_{n=0}^{\infty} (q + n)^{-x}
  ```

  Args:
    x: A `Tensor`. Must be one of the following types: `float32`, `float64`.
    q: A `Tensor`. Must have the same type as `x`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  result = _op_def_lib.apply_op("Zeta", x=x, q=q, name=name)
  return result


def _InitOpDefLibrary():
  op_list = _op_def_pb2.OpList()
  _text_format.Merge(_InitOpDefLibrary.op_list_ascii, op_list)
  _op_def_registry.register_op_list(op_list)
  op_def_lib = _op_def_library.OpDefLibrary()
  op_def_lib.add_op_list(op_list)
  return op_def_lib


_InitOpDefLibrary.op_list_ascii = """op {
  name: "Abs"
  input_arg {
    name: "x"
    type_attr: "T"
  }
  output_arg {
    name: "y"
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
        type: DT_INT32
        type: DT_INT64
      }
    }
  }
}
op {
  name: "Acos"
  input_arg {
    name: "x"
    type_attr: "T"
  }
  output_arg {
    name: "y"
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
        type: DT_INT32
        type: DT_INT64
        type: DT_COMPLEX64
        type: DT_COMPLEX128
      }
    }
  }
}
op {
  name: "Add"
  input_arg {
    name: "x"
    type_attr: "T"
  }
  input_arg {
    name: "y"
    type_attr: "T"
  }
  output_arg {
    name: "z"
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
        type: DT_UINT8
        type: DT_INT8
        type: DT_INT16
        type: DT_INT32
        type: DT_INT64
        type: DT_COMPLEX64
        type: DT_COMPLEX128
        type: DT_STRING
      }
    }
  }
}
op {
  name: "AddN"
  input_arg {
    name: "inputs"
    type_attr: "T"
    number_attr: "N"
  }
  output_arg {
    name: "sum"
    type_attr: "T"
  }
  attr {
    name: "N"
    type: "int"
    has_minimum: true
    minimum: 1
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
  is_aggregate: true
  is_commutative: true
}
op {
  name: "All"
  input_arg {
    name: "input"
    type: DT_BOOL
  }
  input_arg {
    name: "reduction_indices"
    type_attr: "Tidx"
  }
  output_arg {
    name: "output"
    type: DT_BOOL
  }
  attr {
    name: "keep_dims"
    type: "bool"
    default_value {
      b: false
    }
  }
  attr {
    name: "Tidx"
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
  name: "Any"
  input_arg {
    name: "input"
    type: DT_BOOL
  }
  input_arg {
    name: "reduction_indices"
    type_attr: "Tidx"
  }
  output_arg {
    name: "output"
    type: DT_BOOL
  }
  attr {
    name: "keep_dims"
    type: "bool"
    default_value {
      b: false
    }
  }
  attr {
    name: "Tidx"
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
  name: "ArgMax"
  input_arg {
    name: "input"
    type_attr: "T"
  }
  input_arg {
    name: "dimension"
    type_attr: "Tidx"
  }
  output_arg {
    name: "output"
    type: DT_INT64
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
    name: "Tidx"
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
  name: "ArgMin"
  input_arg {
    name: "input"
    type_attr: "T"
  }
  input_arg {
    name: "dimension"
    type_attr: "Tidx"
  }
  output_arg {
    name: "output"
    type: DT_INT64
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
    name: "Tidx"
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
  name: "Asin"
  input_arg {
    name: "x"
    type_attr: "T"
  }
  output_arg {
    name: "y"
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
        type: DT_INT32
        type: DT_INT64
        type: DT_COMPLEX64
        type: DT_COMPLEX128
      }
    }
  }
}
op {
  name: "Atan"
  input_arg {
    name: "x"
    type_attr: "T"
  }
  output_arg {
    name: "y"
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
        type: DT_INT32
        type: DT_INT64
        type: DT_COMPLEX64
        type: DT_COMPLEX128
      }
    }
  }
}
op {
  name: "BatchFFT"
  input_arg {
    name: "input"
    type: DT_COMPLEX64
  }
  output_arg {
    name: "output"
    type: DT_COMPLEX64
  }
  deprecation {
    version: 15
    explanation: "Use FFT"
  }
}
op {
  name: "BatchFFT2D"
  input_arg {
    name: "input"
    type: DT_COMPLEX64
  }
  output_arg {
    name: "output"
    type: DT_COMPLEX64
  }
  deprecation {
    version: 15
    explanation: "Use FFT2D"
  }
}
op {
  name: "BatchFFT3D"
  input_arg {
    name: "input"
    type: DT_COMPLEX64
  }
  output_arg {
    name: "output"
    type: DT_COMPLEX64
  }
  deprecation {
    version: 15
    explanation: "Use FFT3D"
  }
}
op {
  name: "BatchIFFT"
  input_arg {
    name: "input"
    type: DT_COMPLEX64
  }
  output_arg {
    name: "output"
    type: DT_COMPLEX64
  }
  deprecation {
    version: 15
    explanation: "Use IFFT"
  }
}
op {
  name: "BatchIFFT2D"
  input_arg {
    name: "input"
    type: DT_COMPLEX64
  }
  output_arg {
    name: "output"
    type: DT_COMPLEX64
  }
  deprecation {
    version: 15
    explanation: "Use IFFT2D"
  }
}
op {
  name: "BatchIFFT3D"
  input_arg {
    name: "input"
    type: DT_COMPLEX64
  }
  output_arg {
    name: "output"
    type: DT_COMPLEX64
  }
  deprecation {
    version: 15
    explanation: "Use IFFT3D"
  }
}
op {
  name: "BatchMatMul"
  input_arg {
    name: "x"
    type_attr: "T"
  }
  input_arg {
    name: "y"
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
        type: DT_INT32
        type: DT_COMPLEX64
        type: DT_COMPLEX128
      }
    }
  }
  attr {
    name: "adj_x"
    type: "bool"
    default_value {
      b: false
    }
  }
  attr {
    name: "adj_y"
    type: "bool"
    default_value {
      b: false
    }
  }
}
op {
  name: "Betainc"
  input_arg {
    name: "a"
    type_attr: "T"
  }
  input_arg {
    name: "b"
    type_attr: "T"
  }
  input_arg {
    name: "x"
    type_attr: "T"
  }
  output_arg {
    name: "z"
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
  name: "Cast"
  input_arg {
    name: "x"
    type_attr: "SrcT"
  }
  output_arg {
    name: "y"
    type_attr: "DstT"
  }
  attr {
    name: "SrcT"
    type: "type"
  }
  attr {
    name: "DstT"
    type: "type"
  }
}
op {
  name: "Ceil"
  input_arg {
    name: "x"
    type_attr: "T"
  }
  output_arg {
    name: "y"
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
  name: "Complex"
  input_arg {
    name: "real"
    type_attr: "T"
  }
  input_arg {
    name: "imag"
    type_attr: "T"
  }
  output_arg {
    name: "out"
    type_attr: "Tout"
  }
  attr {
    name: "T"
    type: "type"
    default_value {
      type: DT_FLOAT
    }
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
      }
    }
  }
  attr {
    name: "Tout"
    type: "type"
    default_value {
      type: DT_COMPLEX64
    }
    allowed_values {
      list {
        type: DT_COMPLEX64
        type: DT_COMPLEX128
      }
    }
  }
}
op {
  name: "ComplexAbs"
  input_arg {
    name: "x"
    type_attr: "T"
  }
  output_arg {
    name: "y"
    type_attr: "Tout"
  }
  attr {
    name: "T"
    type: "type"
    default_value {
      type: DT_COMPLEX64
    }
    allowed_values {
      list {
        type: DT_COMPLEX64
        type: DT_COMPLEX128
      }
    }
  }
  attr {
    name: "Tout"
    type: "type"
    default_value {
      type: DT_FLOAT
    }
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
      }
    }
  }
}
op {
  name: "Conj"
  input_arg {
    name: "input"
    type_attr: "T"
  }
  output_arg {
    name: "output"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
    default_value {
      type: DT_COMPLEX64
    }
    allowed_values {
      list {
        type: DT_COMPLEX64
        type: DT_COMPLEX128
      }
    }
  }
}
op {
  name: "Cos"
  input_arg {
    name: "x"
    type_attr: "T"
  }
  output_arg {
    name: "y"
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
        type: DT_COMPLEX64
        type: DT_COMPLEX128
      }
    }
  }
}
op {
  name: "Cross"
  input_arg {
    name: "a"
    type_attr: "T"
  }
  input_arg {
    name: "b"
    type_attr: "T"
  }
  output_arg {
    name: "product"
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
  name: "Cumprod"
  input_arg {
    name: "x"
    type_attr: "T"
  }
  input_arg {
    name: "axis"
    type_attr: "Tidx"
  }
  output_arg {
    name: "out"
    type_attr: "T"
  }
  attr {
    name: "exclusive"
    type: "bool"
    default_value {
      b: false
    }
  }
  attr {
    name: "reverse"
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
    name: "Tidx"
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
  name: "Cumsum"
  input_arg {
    name: "x"
    type_attr: "T"
  }
  input_arg {
    name: "axis"
    type_attr: "Tidx"
  }
  output_arg {
    name: "out"
    type_attr: "T"
  }
  attr {
    name: "exclusive"
    type: "bool"
    default_value {
      b: false
    }
  }
  attr {
    name: "reverse"
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
    name: "Tidx"
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
  name: "Digamma"
  input_arg {
    name: "x"
    type_attr: "T"
  }
  output_arg {
    name: "y"
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
  name: "Div"
  input_arg {
    name: "x"
    type_attr: "T"
  }
  input_arg {
    name: "y"
    type_attr: "T"
  }
  output_arg {
    name: "z"
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
        type: DT_UINT8
        type: DT_INT8
        type: DT_UINT16
        type: DT_INT16
        type: DT_INT32
        type: DT_INT64
        type: DT_COMPLEX64
        type: DT_COMPLEX128
      }
    }
  }
}
op {
  name: "Equal"
  input_arg {
    name: "x"
    type_attr: "T"
  }
  input_arg {
    name: "y"
    type_attr: "T"
  }
  output_arg {
    name: "z"
    type: DT_BOOL
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_HALF
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_UINT8
        type: DT_INT8
        type: DT_INT16
        type: DT_INT32
        type: DT_INT64
        type: DT_COMPLEX64
        type: DT_QUINT8
        type: DT_QINT8
        type: DT_QINT32
        type: DT_STRING
        type: DT_BOOL
        type: DT_COMPLEX128
      }
    }
  }
  is_commutative: true
}
op {
  name: "Erf"
  input_arg {
    name: "x"
    type_attr: "T"
  }
  output_arg {
    name: "y"
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
  name: "Erfc"
  input_arg {
    name: "x"
    type_attr: "T"
  }
  output_arg {
    name: "y"
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
  name: "Exp"
  input_arg {
    name: "x"
    type_attr: "T"
  }
  output_arg {
    name: "y"
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
        type: DT_COMPLEX64
        type: DT_COMPLEX128
      }
    }
  }
}
op {
  name: "Expm1"
  input_arg {
    name: "x"
    type_attr: "T"
  }
  output_arg {
    name: "y"
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
        type: DT_COMPLEX64
        type: DT_COMPLEX128
      }
    }
  }
}
op {
  name: "FFT"
  input_arg {
    name: "input"
    type: DT_COMPLEX64
  }
  output_arg {
    name: "output"
    type: DT_COMPLEX64
  }
}
op {
  name: "FFT2D"
  input_arg {
    name: "input"
    type: DT_COMPLEX64
  }
  output_arg {
    name: "output"
    type: DT_COMPLEX64
  }
}
op {
  name: "FFT3D"
  input_arg {
    name: "input"
    type: DT_COMPLEX64
  }
  output_arg {
    name: "output"
    type: DT_COMPLEX64
  }
}
op {
  name: "Floor"
  input_arg {
    name: "x"
    type_attr: "T"
  }
  output_arg {
    name: "y"
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
  name: "FloorDiv"
  input_arg {
    name: "x"
    type_attr: "T"
  }
  input_arg {
    name: "y"
    type_attr: "T"
  }
  output_arg {
    name: "z"
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
        type: DT_UINT8
        type: DT_INT8
        type: DT_UINT16
        type: DT_INT16
        type: DT_INT32
        type: DT_INT64
        type: DT_COMPLEX64
        type: DT_COMPLEX128
      }
    }
  }
}
op {
  name: "FloorMod"
  input_arg {
    name: "x"
    type_attr: "T"
  }
  input_arg {
    name: "y"
    type_attr: "T"
  }
  output_arg {
    name: "z"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_INT32
        type: DT_INT64
        type: DT_FLOAT
        type: DT_DOUBLE
      }
    }
  }
}
op {
  name: "Greater"
  input_arg {
    name: "x"
    type_attr: "T"
  }
  input_arg {
    name: "y"
    type_attr: "T"
  }
  output_arg {
    name: "z"
    type: DT_BOOL
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
  name: "GreaterEqual"
  input_arg {
    name: "x"
    type_attr: "T"
  }
  input_arg {
    name: "y"
    type_attr: "T"
  }
  output_arg {
    name: "z"
    type: DT_BOOL
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
  name: "IFFT"
  input_arg {
    name: "input"
    type: DT_COMPLEX64
  }
  output_arg {
    name: "output"
    type: DT_COMPLEX64
  }
}
op {
  name: "IFFT2D"
  input_arg {
    name: "input"
    type: DT_COMPLEX64
  }
  output_arg {
    name: "output"
    type: DT_COMPLEX64
  }
}
op {
  name: "IFFT3D"
  input_arg {
    name: "input"
    type: DT_COMPLEX64
  }
  output_arg {
    name: "output"
    type: DT_COMPLEX64
  }
}
op {
  name: "Igamma"
  input_arg {
    name: "a"
    type_attr: "T"
  }
  input_arg {
    name: "x"
    type_attr: "T"
  }
  output_arg {
    name: "z"
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
  name: "Igammac"
  input_arg {
    name: "a"
    type_attr: "T"
  }
  input_arg {
    name: "x"
    type_attr: "T"
  }
  output_arg {
    name: "z"
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
  name: "Imag"
  input_arg {
    name: "input"
    type_attr: "T"
  }
  output_arg {
    name: "output"
    type_attr: "Tout"
  }
  attr {
    name: "T"
    type: "type"
    default_value {
      type: DT_COMPLEX64
    }
    allowed_values {
      list {
        type: DT_COMPLEX64
        type: DT_COMPLEX128
      }
    }
  }
  attr {
    name: "Tout"
    type: "type"
    default_value {
      type: DT_FLOAT
    }
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
      }
    }
  }
}
op {
  name: "Inv"
  input_arg {
    name: "x"
    type_attr: "T"
  }
  output_arg {
    name: "y"
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
        type: DT_INT32
        type: DT_INT64
        type: DT_COMPLEX64
        type: DT_COMPLEX128
      }
    }
  }
  deprecation {
    version: 17
    explanation: "Use Reciprocal"
  }
}
op {
  name: "InvGrad"
  input_arg {
    name: "x"
    type_attr: "T"
  }
  input_arg {
    name: "y"
    type_attr: "T"
  }
  output_arg {
    name: "z"
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
        type: DT_COMPLEX64
        type: DT_COMPLEX128
      }
    }
  }
  deprecation {
    version: 17
    explanation: "Use ReciprocalGrad"
  }
}
op {
  name: "IsFinite"
  input_arg {
    name: "x"
    type_attr: "T"
  }
  output_arg {
    name: "y"
    type: DT_BOOL
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
  name: "IsInf"
  input_arg {
    name: "x"
    type_attr: "T"
  }
  output_arg {
    name: "y"
    type: DT_BOOL
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
  name: "IsNan"
  input_arg {
    name: "x"
    type_attr: "T"
  }
  output_arg {
    name: "y"
    type: DT_BOOL
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
  name: "Less"
  input_arg {
    name: "x"
    type_attr: "T"
  }
  input_arg {
    name: "y"
    type_attr: "T"
  }
  output_arg {
    name: "z"
    type: DT_BOOL
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
  name: "LessEqual"
  input_arg {
    name: "x"
    type_attr: "T"
  }
  input_arg {
    name: "y"
    type_attr: "T"
  }
  output_arg {
    name: "z"
    type: DT_BOOL
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
  name: "Lgamma"
  input_arg {
    name: "x"
    type_attr: "T"
  }
  output_arg {
    name: "y"
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
  name: "LinSpace"
  input_arg {
    name: "start"
    type_attr: "T"
  }
  input_arg {
    name: "stop"
    type_attr: "T"
  }
  input_arg {
    name: "num"
    type_attr: "Tidx"
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
    name: "Tidx"
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
  name: "Log"
  input_arg {
    name: "x"
    type_attr: "T"
  }
  output_arg {
    name: "y"
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
        type: DT_COMPLEX64
        type: DT_COMPLEX128
      }
    }
  }
}
op {
  name: "Log1p"
  input_arg {
    name: "x"
    type_attr: "T"
  }
  output_arg {
    name: "y"
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
        type: DT_COMPLEX64
        type: DT_COMPLEX128
      }
    }
  }
}
op {
  name: "LogicalAnd"
  input_arg {
    name: "x"
    type: DT_BOOL
  }
  input_arg {
    name: "y"
    type: DT_BOOL
  }
  output_arg {
    name: "z"
    type: DT_BOOL
  }
  is_commutative: true
}
op {
  name: "LogicalNot"
  input_arg {
    name: "x"
    type: DT_BOOL
  }
  output_arg {
    name: "y"
    type: DT_BOOL
  }
}
op {
  name: "LogicalOr"
  input_arg {
    name: "x"
    type: DT_BOOL
  }
  input_arg {
    name: "y"
    type: DT_BOOL
  }
  output_arg {
    name: "z"
    type: DT_BOOL
  }
  is_commutative: true
}
op {
  name: "MatMul"
  input_arg {
    name: "a"
    type_attr: "T"
  }
  input_arg {
    name: "b"
    type_attr: "T"
  }
  output_arg {
    name: "product"
    type_attr: "T"
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
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_HALF
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT32
        type: DT_COMPLEX64
        type: DT_COMPLEX128
      }
    }
  }
}
op {
  name: "Max"
  input_arg {
    name: "input"
    type_attr: "T"
  }
  input_arg {
    name: "reduction_indices"
    type_attr: "Tidx"
  }
  output_arg {
    name: "output"
    type_attr: "T"
  }
  attr {
    name: "keep_dims"
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
    name: "Tidx"
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
  name: "Maximum"
  input_arg {
    name: "x"
    type_attr: "T"
  }
  input_arg {
    name: "y"
    type_attr: "T"
  }
  output_arg {
    name: "z"
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
        type: DT_INT32
        type: DT_INT64
      }
    }
  }
  is_commutative: true
}
op {
  name: "Mean"
  input_arg {
    name: "input"
    type_attr: "T"
  }
  input_arg {
    name: "reduction_indices"
    type_attr: "Tidx"
  }
  output_arg {
    name: "output"
    type_attr: "T"
  }
  attr {
    name: "keep_dims"
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
    name: "Tidx"
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
  name: "Min"
  input_arg {
    name: "input"
    type_attr: "T"
  }
  input_arg {
    name: "reduction_indices"
    type_attr: "Tidx"
  }
  output_arg {
    name: "output"
    type_attr: "T"
  }
  attr {
    name: "keep_dims"
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
    name: "Tidx"
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
  name: "Minimum"
  input_arg {
    name: "x"
    type_attr: "T"
  }
  input_arg {
    name: "y"
    type_attr: "T"
  }
  output_arg {
    name: "z"
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
        type: DT_INT32
        type: DT_INT64
      }
    }
  }
  is_commutative: true
}
op {
  name: "Mod"
  input_arg {
    name: "x"
    type_attr: "T"
  }
  input_arg {
    name: "y"
    type_attr: "T"
  }
  output_arg {
    name: "z"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_INT32
        type: DT_INT64
        type: DT_FLOAT
        type: DT_DOUBLE
      }
    }
  }
}
op {
  name: "Mul"
  input_arg {
    name: "x"
    type_attr: "T"
  }
  input_arg {
    name: "y"
    type_attr: "T"
  }
  output_arg {
    name: "z"
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
        type: DT_UINT8
        type: DT_INT8
        type: DT_UINT16
        type: DT_INT16
        type: DT_INT32
        type: DT_INT64
        type: DT_COMPLEX64
        type: DT_COMPLEX128
      }
    }
  }
  is_commutative: true
}
op {
  name: "Neg"
  input_arg {
    name: "x"
    type_attr: "T"
  }
  output_arg {
    name: "y"
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
        type: DT_INT32
        type: DT_INT64
        type: DT_COMPLEX64
        type: DT_COMPLEX128
      }
    }
  }
}
op {
  name: "NotEqual"
  input_arg {
    name: "x"
    type_attr: "T"
  }
  input_arg {
    name: "y"
    type_attr: "T"
  }
  output_arg {
    name: "z"
    type: DT_BOOL
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_HALF
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_UINT8
        type: DT_INT8
        type: DT_INT16
        type: DT_INT32
        type: DT_INT64
        type: DT_COMPLEX64
        type: DT_QUINT8
        type: DT_QINT8
        type: DT_QINT32
        type: DT_STRING
        type: DT_BOOL
        type: DT_COMPLEX128
      }
    }
  }
  is_commutative: true
}
op {
  name: "Polygamma"
  input_arg {
    name: "a"
    type_attr: "T"
  }
  input_arg {
    name: "x"
    type_attr: "T"
  }
  output_arg {
    name: "z"
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
  name: "Pow"
  input_arg {
    name: "x"
    type_attr: "T"
  }
  input_arg {
    name: "y"
    type_attr: "T"
  }
  output_arg {
    name: "z"
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
        type: DT_INT32
        type: DT_INT64
        type: DT_COMPLEX64
        type: DT_COMPLEX128
      }
    }
  }
}
op {
  name: "Prod"
  input_arg {
    name: "input"
    type_attr: "T"
  }
  input_arg {
    name: "reduction_indices"
    type_attr: "Tidx"
  }
  output_arg {
    name: "output"
    type_attr: "T"
  }
  attr {
    name: "keep_dims"
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
    name: "Tidx"
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
  attr {
    name: "Tactivation"
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
  name: "Range"
  input_arg {
    name: "start"
    type_attr: "Tidx"
  }
  input_arg {
    name: "limit"
    type_attr: "Tidx"
  }
  input_arg {
    name: "delta"
    type_attr: "Tidx"
  }
  output_arg {
    name: "output"
    type_attr: "Tidx"
  }
  attr {
    name: "Tidx"
    type: "type"
    default_value {
      type: DT_INT32
    }
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT32
        type: DT_INT64
      }
    }
  }
}
op {
  name: "Real"
  input_arg {
    name: "input"
    type_attr: "T"
  }
  output_arg {
    name: "output"
    type_attr: "Tout"
  }
  attr {
    name: "T"
    type: "type"
    default_value {
      type: DT_COMPLEX64
    }
    allowed_values {
      list {
        type: DT_COMPLEX64
        type: DT_COMPLEX128
      }
    }
  }
  attr {
    name: "Tout"
    type: "type"
    default_value {
      type: DT_FLOAT
    }
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
      }
    }
  }
}
op {
  name: "RealDiv"
  input_arg {
    name: "x"
    type_attr: "T"
  }
  input_arg {
    name: "y"
    type_attr: "T"
  }
  output_arg {
    name: "z"
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
        type: DT_UINT8
        type: DT_INT8
        type: DT_UINT16
        type: DT_INT16
        type: DT_INT32
        type: DT_INT64
        type: DT_COMPLEX64
        type: DT_COMPLEX128
      }
    }
  }
}
op {
  name: "Reciprocal"
  input_arg {
    name: "x"
    type_attr: "T"
  }
  output_arg {
    name: "y"
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
        type: DT_INT32
        type: DT_INT64
        type: DT_COMPLEX64
        type: DT_COMPLEX128
      }
    }
  }
}
op {
  name: "ReciprocalGrad"
  input_arg {
    name: "x"
    type_attr: "T"
  }
  input_arg {
    name: "y"
    type_attr: "T"
  }
  output_arg {
    name: "z"
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
        type: DT_COMPLEX64
        type: DT_COMPLEX128
      }
    }
  }
}
op {
  name: "RequantizationRange"
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
}
op {
  name: "Requantize"
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
  input_arg {
    name: "requested_output_min"
    type: DT_FLOAT
  }
  input_arg {
    name: "requested_output_max"
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
  name: "Rint"
  input_arg {
    name: "x"
    type_attr: "T"
  }
  output_arg {
    name: "y"
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
  name: "Round"
  input_arg {
    name: "x"
    type_attr: "T"
  }
  output_arg {
    name: "y"
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
        type: DT_INT32
        type: DT_INT64
        type: DT_COMPLEX64
        type: DT_COMPLEX128
      }
    }
  }
}
op {
  name: "Rsqrt"
  input_arg {
    name: "x"
    type_attr: "T"
  }
  output_arg {
    name: "y"
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
        type: DT_COMPLEX64
        type: DT_COMPLEX128
      }
    }
  }
}
op {
  name: "RsqrtGrad"
  input_arg {
    name: "x"
    type_attr: "T"
  }
  input_arg {
    name: "y"
    type_attr: "T"
  }
  output_arg {
    name: "z"
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
        type: DT_COMPLEX64
        type: DT_COMPLEX128
      }
    }
  }
}
op {
  name: "SegmentMax"
  input_arg {
    name: "data"
    type_attr: "T"
  }
  input_arg {
    name: "segment_ids"
    type_attr: "Tindices"
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
    name: "Tindices"
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
  name: "SegmentMean"
  input_arg {
    name: "data"
    type_attr: "T"
  }
  input_arg {
    name: "segment_ids"
    type_attr: "Tindices"
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
    name: "Tindices"
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
  name: "SegmentMin"
  input_arg {
    name: "data"
    type_attr: "T"
  }
  input_arg {
    name: "segment_ids"
    type_attr: "Tindices"
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
    name: "Tindices"
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
  name: "SegmentProd"
  input_arg {
    name: "data"
    type_attr: "T"
  }
  input_arg {
    name: "segment_ids"
    type_attr: "Tindices"
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
    name: "Tindices"
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
  name: "SegmentSum"
  input_arg {
    name: "data"
    type_attr: "T"
  }
  input_arg {
    name: "segment_ids"
    type_attr: "Tindices"
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
    name: "Tindices"
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
  name: "Select"
  input_arg {
    name: "condition"
    type: DT_BOOL
  }
  input_arg {
    name: "t"
    type_attr: "T"
  }
  input_arg {
    name: "e"
    type_attr: "T"
  }
  output_arg {
    name: "output"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
  }
}
op {
  name: "Sigmoid"
  input_arg {
    name: "x"
    type_attr: "T"
  }
  output_arg {
    name: "y"
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
        type: DT_COMPLEX64
        type: DT_COMPLEX128
      }
    }
  }
}
op {
  name: "SigmoidGrad"
  input_arg {
    name: "x"
    type_attr: "T"
  }
  input_arg {
    name: "y"
    type_attr: "T"
  }
  output_arg {
    name: "z"
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
        type: DT_COMPLEX64
        type: DT_COMPLEX128
      }
    }
  }
}
op {
  name: "Sign"
  input_arg {
    name: "x"
    type_attr: "T"
  }
  output_arg {
    name: "y"
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
        type: DT_INT32
        type: DT_INT64
        type: DT_COMPLEX64
        type: DT_COMPLEX128
      }
    }
  }
}
op {
  name: "Sin"
  input_arg {
    name: "x"
    type_attr: "T"
  }
  output_arg {
    name: "y"
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
        type: DT_COMPLEX64
        type: DT_COMPLEX128
      }
    }
  }
}
op {
  name: "SparseMatMul"
  input_arg {
    name: "a"
    type_attr: "Ta"
  }
  input_arg {
    name: "b"
    type_attr: "Tb"
  }
  output_arg {
    name: "product"
    type: DT_FLOAT
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
  attr {
    name: "a_is_sparse"
    type: "bool"
    default_value {
      b: false
    }
  }
  attr {
    name: "b_is_sparse"
    type: "bool"
    default_value {
      b: false
    }
  }
  attr {
    name: "Ta"
    type: "type"
    default_value {
      type: DT_FLOAT
    }
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_BFLOAT16
      }
    }
  }
  attr {
    name: "Tb"
    type: "type"
    default_value {
      type: DT_FLOAT
    }
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_BFLOAT16
      }
    }
  }
}
op {
  name: "SparseSegmentMean"
  input_arg {
    name: "data"
    type_attr: "T"
  }
  input_arg {
    name: "indices"
    type_attr: "Tidx"
  }
  input_arg {
    name: "segment_ids"
    type: DT_INT32
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
    name: "Tidx"
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
  name: "SparseSegmentMeanGrad"
  input_arg {
    name: "grad"
    type_attr: "T"
  }
  input_arg {
    name: "indices"
    type_attr: "Tidx"
  }
  input_arg {
    name: "segment_ids"
    type: DT_INT32
  }
  input_arg {
    name: "output_dim0"
    type: DT_INT32
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
    name: "Tidx"
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
  name: "SparseSegmentSqrtN"
  input_arg {
    name: "data"
    type_attr: "T"
  }
  input_arg {
    name: "indices"
    type_attr: "Tidx"
  }
  input_arg {
    name: "segment_ids"
    type: DT_INT32
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
    name: "Tidx"
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
  name: "SparseSegmentSqrtNGrad"
  input_arg {
    name: "grad"
    type_attr: "T"
  }
  input_arg {
    name: "indices"
    type_attr: "Tidx"
  }
  input_arg {
    name: "segment_ids"
    type: DT_INT32
  }
  input_arg {
    name: "output_dim0"
    type: DT_INT32
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
    name: "Tidx"
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
  name: "SparseSegmentSum"
  input_arg {
    name: "data"
    type_attr: "T"
  }
  input_arg {
    name: "indices"
    type_attr: "Tidx"
  }
  input_arg {
    name: "segment_ids"
    type: DT_INT32
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
    name: "Tidx"
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
  name: "Sqrt"
  input_arg {
    name: "x"
    type_attr: "T"
  }
  output_arg {
    name: "y"
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
        type: DT_COMPLEX64
        type: DT_COMPLEX128
      }
    }
  }
}
op {
  name: "SqrtGrad"
  input_arg {
    name: "x"
    type_attr: "T"
  }
  input_arg {
    name: "y"
    type_attr: "T"
  }
  output_arg {
    name: "z"
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
        type: DT_COMPLEX64
        type: DT_COMPLEX128
      }
    }
  }
}
op {
  name: "Square"
  input_arg {
    name: "x"
    type_attr: "T"
  }
  output_arg {
    name: "y"
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
        type: DT_INT32
        type: DT_INT64
        type: DT_COMPLEX64
        type: DT_COMPLEX128
      }
    }
  }
}
op {
  name: "SquaredDifference"
  input_arg {
    name: "x"
    type_attr: "T"
  }
  input_arg {
    name: "y"
    type_attr: "T"
  }
  output_arg {
    name: "z"
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
        type: DT_INT32
        type: DT_INT64
        type: DT_COMPLEX64
        type: DT_COMPLEX128
      }
    }
  }
  is_commutative: true
}
op {
  name: "Sub"
  input_arg {
    name: "x"
    type_attr: "T"
  }
  input_arg {
    name: "y"
    type_attr: "T"
  }
  output_arg {
    name: "z"
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
        type: DT_INT32
        type: DT_INT64
        type: DT_COMPLEX64
        type: DT_COMPLEX128
      }
    }
  }
}
op {
  name: "Sum"
  input_arg {
    name: "input"
    type_attr: "T"
  }
  input_arg {
    name: "reduction_indices"
    type_attr: "Tidx"
  }
  output_arg {
    name: "output"
    type_attr: "T"
  }
  attr {
    name: "keep_dims"
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
    name: "Tidx"
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
  name: "Tan"
  input_arg {
    name: "x"
    type_attr: "T"
  }
  output_arg {
    name: "y"
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
        type: DT_INT32
        type: DT_INT64
        type: DT_COMPLEX64
        type: DT_COMPLEX128
      }
    }
  }
}
op {
  name: "Tanh"
  input_arg {
    name: "x"
    type_attr: "T"
  }
  output_arg {
    name: "y"
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
        type: DT_COMPLEX64
        type: DT_COMPLEX128
      }
    }
  }
}
op {
  name: "TanhGrad"
  input_arg {
    name: "x"
    type_attr: "T"
  }
  input_arg {
    name: "y"
    type_attr: "T"
  }
  output_arg {
    name: "z"
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
        type: DT_COMPLEX64
        type: DT_COMPLEX128
      }
    }
  }
}
op {
  name: "TruncateDiv"
  input_arg {
    name: "x"
    type_attr: "T"
  }
  input_arg {
    name: "y"
    type_attr: "T"
  }
  output_arg {
    name: "z"
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
        type: DT_UINT8
        type: DT_INT8
        type: DT_UINT16
        type: DT_INT16
        type: DT_INT32
        type: DT_INT64
        type: DT_COMPLEX64
        type: DT_COMPLEX128
      }
    }
  }
}
op {
  name: "TruncateMod"
  input_arg {
    name: "x"
    type_attr: "T"
  }
  input_arg {
    name: "y"
    type_attr: "T"
  }
  output_arg {
    name: "z"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_INT32
        type: DT_INT64
        type: DT_FLOAT
        type: DT_DOUBLE
      }
    }
  }
}
op {
  name: "UnsortedSegmentSum"
  input_arg {
    name: "data"
    type_attr: "T"
  }
  input_arg {
    name: "segment_ids"
    type_attr: "Tindices"
  }
  input_arg {
    name: "num_segments"
    type: DT_INT32
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
    name: "Tindices"
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
  name: "Zeta"
  input_arg {
    name: "x"
    type_attr: "T"
  }
  input_arg {
    name: "q"
    type_attr: "T"
  }
  output_arg {
    name: "z"
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
"""


_op_def_lib = _InitOpDefLibrary()
