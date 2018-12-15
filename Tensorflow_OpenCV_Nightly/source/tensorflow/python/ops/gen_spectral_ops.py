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



def fft(input, name=None):
  r"""Fast Fourier transform.

  Computes the 1-dimensional discrete Fourier transform over the inner-most
  dimension of `input`.

  Args:
    input: A `Tensor` of type `complex64`. A complex64 tensor.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `complex64`.
    A complex64 tensor of the same shape as `input`. The inner-most
      dimension of `input` is replaced with its 1D Fourier transform.

    @compatibility(numpy)
    Equivalent to np.fft.fft
    @end_compatibility
  """
  result = _op_def_lib.apply_op("FFT", input=input, name=name)
  return result



def fft2d(input, name=None):
  r"""2D fast Fourier transform.

  Computes the 2-dimensional discrete Fourier transform over the inner-most
  2 dimensions of `input`.

  Args:
    input: A `Tensor` of type `complex64`. A complex64 tensor.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `complex64`.
    A complex64 tensor of the same shape as `input`. The inner-most 2
      dimensions of `input` are replaced with their 2D Fourier transform.

    @compatibility(numpy)
    Equivalent to np.fft.fft2
    @end_compatibility
  """
  result = _op_def_lib.apply_op("FFT2D", input=input, name=name)
  return result



def fft3d(input, name=None):
  r"""3D fast Fourier transform.

  Computes the 3-dimensional discrete Fourier transform over the inner-most 3
  dimensions of `input`.

  Args:
    input: A `Tensor` of type `complex64`. A complex64 tensor.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `complex64`.
    A complex64 tensor of the same shape as `input`. The inner-most 3
      dimensions of `input` are replaced with their 3D Fourier transform.

    @compatibility(numpy)
    Equivalent to np.fft.fftn with 3 dimensions.
    @end_compatibility
  """
  result = _op_def_lib.apply_op("FFT3D", input=input, name=name)
  return result



def ifft(input, name=None):
  r"""Inverse fast Fourier transform.

  Computes the inverse 1-dimensional discrete Fourier transform over the
  inner-most dimension of `input`.

  Args:
    input: A `Tensor` of type `complex64`. A complex64 tensor.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `complex64`.
    A complex64 tensor of the same shape as `input`. The inner-most
      dimension of `input` is replaced with its inverse 1D Fourier transform.

    @compatibility(numpy)
    Equivalent to np.fft.ifft
    @end_compatibility
  """
  result = _op_def_lib.apply_op("IFFT", input=input, name=name)
  return result



def ifft2d(input, name=None):
  r"""Inverse 2D fast Fourier transform.

  Computes the inverse 2-dimensional discrete Fourier transform over the
  inner-most 2 dimensions of `input`.

  Args:
    input: A `Tensor` of type `complex64`. A complex64 tensor.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `complex64`.
    A complex64 tensor of the same shape as `input`. The inner-most 2
      dimensions of `input` are replaced with their inverse 2D Fourier transform.

    @compatibility(numpy)
    Equivalent to np.fft.ifft2
    @end_compatibility
  """
  result = _op_def_lib.apply_op("IFFT2D", input=input, name=name)
  return result



def ifft3d(input, name=None):
  r"""Inverse 3D fast Fourier transform.

  Computes the inverse 3-dimensional discrete Fourier transform over the
  inner-most 3 dimensions of `input`.

  Args:
    input: A `Tensor` of type `complex64`. A complex64 tensor.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `complex64`.
    A complex64 tensor of the same shape as `input`. The inner-most 3
      dimensions of `input` are replaced with their inverse 3D Fourier transform.

    @compatibility(numpy)
    Equivalent to np.fft.ifftn with 3 dimensions.
    @end_compatibility
  """
  result = _op_def_lib.apply_op("IFFT3D", input=input, name=name)
  return result



def irfft(input, fft_length, name=None):
  r"""Inverse real-valued fast Fourier transform.

  Computes the inverse 1-dimensional discrete Fourier transform of a real-valued
  signal over the inner-most dimension of `input`.

  The inner-most dimension of `input` is assumed to be the result of `RFFT`: the
  `fft_length / 2 + 1` unique components of the DFT of a real-valued signal. If
  `fft_length` is not provided, it is computed from the size of the inner-most
  dimension of `input` (`fft_length = 2 * (inner - 1)`). If the FFT length used to
  compute `input` is odd, it should be provided since it cannot be inferred
  properly.

  Args:
    input: A `Tensor` of type `complex64`. A complex64 tensor.
    fft_length: A `Tensor` of type `int32`.
      An int32 tensor of shape [1]. The FFT length.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32`.
    A float32 tensor of the same rank as `input`. The inner-most
      dimension of `input` is replaced with the `fft_length` samples of its inverse
      1D Fourier transform.

    @compatibility(numpy)
    Equivalent to np.fft.irfft
    @end_compatibility
  """
  result = _op_def_lib.apply_op("IRFFT", input=input, fft_length=fft_length,
                                name=name)
  return result



def irfft2d(input, fft_length, name=None):
  r"""Inverse 2D real-valued fast Fourier transform.

  Computes the inverse 2-dimensional discrete Fourier transform of a real-valued
  signal over the inner-most 2 dimensions of `input`.

  The inner-most 2 dimensions of `input` are assumed to be the result of `RFFT2D`:
  The inner-most dimension contains the `fft_length / 2 + 1` unique components of
  the DFT of a real-valued signal. If `fft_length` is not provided, it is computed
  from the size of the inner-most 2 dimensions of `input`. If the FFT length used
  to compute `input` is odd, it should be provided since it cannot be inferred
  properly.

  Args:
    input: A `Tensor` of type `complex64`. A complex64 tensor.
    fft_length: A `Tensor` of type `int32`.
      An int32 tensor of shape [2]. The FFT length for each dimension.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32`.
    A float32 tensor of the same rank as `input`. The inner-most 2
      dimensions of `input` are replaced with the `fft_length` samples of their
      inverse 2D Fourier transform.

    @compatibility(numpy)
    Equivalent to np.fft.irfft2
    @end_compatibility
  """
  result = _op_def_lib.apply_op("IRFFT2D", input=input, fft_length=fft_length,
                                name=name)
  return result



def irfft3d(input, fft_length, name=None):
  r"""Inverse 3D real-valued fast Fourier transform.

  Computes the inverse 3-dimensional discrete Fourier transform of a real-valued
  signal over the inner-most 3 dimensions of `input`.

  The inner-most 3 dimensions of `input` are assumed to be the result of `RFFT3D`:
  The inner-most dimension contains the `fft_length / 2 + 1` unique components of
  the DFT of a real-valued signal. If `fft_length` is not provided, it is computed
  from the size of the inner-most 3 dimensions of `input`. If the FFT length used
  to compute `input` is odd, it should be provided since it cannot be inferred
  properly.

  Args:
    input: A `Tensor` of type `complex64`. A complex64 tensor.
    fft_length: A `Tensor` of type `int32`.
      An int32 tensor of shape [3]. The FFT length for each dimension.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32`.
    A float32 tensor of the same rank as `input`. The inner-most 3
      dimensions of `input` are replaced with the `fft_length` samples of their
      inverse 3D real Fourier transform.

    @compatibility(numpy)
    Equivalent to np.irfftn with 3 dimensions.
    @end_compatibility
  """
  result = _op_def_lib.apply_op("IRFFT3D", input=input, fft_length=fft_length,
                                name=name)
  return result



def rfft(input, fft_length, name=None):
  r"""Real-valued fast Fourier transform.

  Computes the 1-dimensional discrete Fourier transform of a real-valued signal
  over the inner-most dimension of `input`.

  Since the DFT of a real signal is Hermitian-symmetric, `RFFT` only returns the
  `fft_length / 2 + 1` unique components of the FFT: the zero-frequency term,
  followed by the `fft_length / 2` positive-frequency terms.

  Args:
    input: A `Tensor` of type `float32`. A float32 tensor.
    fft_length: A `Tensor` of type `int32`.
      An int32 tensor of shape [1]. The FFT length.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `complex64`.
    A complex64 tensor of the same rank as `input`. The inner-most
      dimension of `input` is replaced with the `fft_length / 2 + 1` unique
      frequency components of its 1D Fourier transform.

    @compatibility(numpy)
    Equivalent to np.fft.rfft
    @end_compatibility
  """
  result = _op_def_lib.apply_op("RFFT", input=input, fft_length=fft_length,
                                name=name)
  return result



def rfft2d(input, fft_length, name=None):
  r"""2D real-valued fast Fourier transform.

  Computes the 2-dimensional discrete Fourier transform of a real-valued signal
  over the inner-most 2 dimensions of `input`.

  Since the DFT of a real signal is Hermitian-symmetric, `RFFT2D` only returns the
  `fft_length / 2 + 1` unique components of the FFT for the inner-most dimension
  of `output`: the zero-frequency term, followed by the `fft_length / 2`
  positive-frequency terms.

  Args:
    input: A `Tensor` of type `float32`. A float32 tensor.
    fft_length: A `Tensor` of type `int32`.
      An int32 tensor of shape [2]. The FFT length for each dimension.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `complex64`.
    A complex64 tensor of the same rank as `input`. The inner-most 2
      dimensions of `input` are replaced with their 2D Fourier transform. The
      inner-most dimension contains `fft_length / 2 + 1` unique frequency
      components.

    @compatibility(numpy)
    Equivalent to np.fft.rfft2
    @end_compatibility
  """
  result = _op_def_lib.apply_op("RFFT2D", input=input, fft_length=fft_length,
                                name=name)
  return result



def rfft3d(input, fft_length, name=None):
  r"""3D real-valued fast Fourier transform.

  Computes the 3-dimensional discrete Fourier transform of a real-valued signal
  over the inner-most 3 dimensions of `input`.

  Since the DFT of a real signal is Hermitian-symmetric, `RFFT3D` only returns the
  `fft_length / 2 + 1` unique components of the FFT for the inner-most dimension
  of `output`: the zero-frequency term, followed by the `fft_length / 2`
  positive-frequency terms.

  Args:
    input: A `Tensor` of type `float32`. A float32 tensor.
    fft_length: A `Tensor` of type `int32`.
      An int32 tensor of shape [3]. The FFT length for each dimension.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `complex64`.
    A complex64 tensor of the same rank as `input`. The inner-most 3
      dimensions of `input` are replaced with the their 3D Fourier transform. The
      inner-most dimension contains `fft_length / 2 + 1` unique frequency
      components.

    @compatibility(numpy)
    Equivalent to np.fft.rfftn with 3 dimensions.
    @end_compatibility
  """
  result = _op_def_lib.apply_op("RFFT3D", input=input, fft_length=fft_length,
                                name=name)
  return result


def _InitOpDefLibrary():
  op_list = _op_def_pb2.OpList()
  _text_format.Merge(_InitOpDefLibrary.op_list_ascii, op_list)
  _op_def_registry.register_op_list(op_list)
  op_def_lib = _op_def_library.OpDefLibrary()
  op_def_lib.add_op_list(op_list)
  return op_def_lib


_InitOpDefLibrary.op_list_ascii = """op {
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
  name: "IRFFT"
  input_arg {
    name: "input"
    type: DT_COMPLEX64
  }
  input_arg {
    name: "fft_length"
    type: DT_INT32
  }
  output_arg {
    name: "output"
    type: DT_FLOAT
  }
}
op {
  name: "IRFFT2D"
  input_arg {
    name: "input"
    type: DT_COMPLEX64
  }
  input_arg {
    name: "fft_length"
    type: DT_INT32
  }
  output_arg {
    name: "output"
    type: DT_FLOAT
  }
}
op {
  name: "IRFFT3D"
  input_arg {
    name: "input"
    type: DT_COMPLEX64
  }
  input_arg {
    name: "fft_length"
    type: DT_INT32
  }
  output_arg {
    name: "output"
    type: DT_FLOAT
  }
}
op {
  name: "RFFT"
  input_arg {
    name: "input"
    type: DT_FLOAT
  }
  input_arg {
    name: "fft_length"
    type: DT_INT32
  }
  output_arg {
    name: "output"
    type: DT_COMPLEX64
  }
}
op {
  name: "RFFT2D"
  input_arg {
    name: "input"
    type: DT_FLOAT
  }
  input_arg {
    name: "fft_length"
    type: DT_INT32
  }
  output_arg {
    name: "output"
    type: DT_COMPLEX64
  }
}
op {
  name: "RFFT3D"
  input_arg {
    name: "input"
    type: DT_FLOAT
  }
  input_arg {
    name: "fft_length"
    type: DT_INT32
  }
  output_arg {
    name: "output"
    type: DT_COMPLEX64
  }
}
"""


_op_def_lib = _InitOpDefLibrary()
