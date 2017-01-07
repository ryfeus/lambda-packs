"""Python wrappers around Brain.

This file is MACHINE GENERATED! Do not edit.
"""

import collections

from google.protobuf import text_format

from tensorflow.core.framework import op_def_pb2
from tensorflow.python.framework import op_def_registry
from tensorflow.python.framework import ops
from tensorflow.python.ops import op_def_library


_batch_matrix_band_part_outputs = ["band"]


def batch_matrix_band_part(input, num_lower, num_upper, name=None):
  r"""Copy a tensor setting everything outside a central band in each innermost matrix

  to zero.

  The `band` part is computed as follows:
  Assume `input` has `k` dimensions `[I, J, K, ..., M, N]`, then the output is a
  tensor with the same shape where

  `band[i, j, k, ..., m, n] = in_band(m, n) * input[i, j, k, ..., m, n]`.

  The indicator function 'in_band(m, n)` is one if
  `(num_lower < 0 || (m-n) <= num_lower)) &&
  (num_upper < 0 || (n-m) <= num_upper)`, and zero otherwise.

  For example:

  ```prettyprint
  # if 'input' is [[ 0,  1,  2, 3]
                   [-1,  0,  1, 2]
                   [-2, -1,  0, 1]
                   [-3, -2, -1, 0]],

  tf.batch_matrix_band_part(input, 1, -1) ==> [[ 0,  1,  2, 3]
                                               [-1,  0,  1, 2]
                                               [ 0, -1,  0, 1]
                                               [ 0,  0, -1, 0]],

  tf.batch_matrix_band_part(input, 2, 1) ==> [[ 0,  1,  0, 0]
                                              [-1,  0,  1, 0]
                                              [-2, -1,  0, 1]
                                              [ 0, -2, -1, 0]]
  ```

  Useful special cases:

  ```prettyprint
   tf.batch_matrix_band_part(input, 0, -1) ==> Upper triangular part.
   tf.batch_matrix_band_part(input, -1, 0) ==> Lower triangular part.
   tf.batch_matrix_band_part(input, 0, 0) ==> Diagonal.
  ```

  Args:
    input: A `Tensor`. Rank `k` tensor.
    num_lower: A `Tensor` of type `int64`.
      0-D tensor. Number of subdiagonals to keep. If negative, keep entire
      lower triangle.
    num_upper: A `Tensor` of type `int64`.
      0-D tensor. Number of superdiagonals to keep. If negative, keep
      entire upper triangle.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
    Rank `k` tensor of the same shape as input. The extracted banded tensor.
  """
  result = _op_def_lib.apply_op("BatchMatrixBandPart", input=input,
                                num_lower=num_lower, num_upper=num_upper,
                                name=name)
  return result


_batch_matrix_diag_outputs = ["output"]


def batch_matrix_diag(diagonal, name=None):
  r"""Returns a batched diagonal tensor with a given batched diagonal values.

  Given a `diagonal`, this operation returns a tensor with the `diagonal` and
  everything else padded with zeros. The diagonal is computed as follows:

  Assume `diagonal` has `k` dimensions `[I, J, K, ..., N]`, then the output is a
  tensor of rank `k+1` with dimensions [I, J, K, ..., N, N]` where:

  `output[i, j, k, ..., m, n] = 1{m=n} * diagonal[i, j, k, ..., n]`.

  For example:

  ```prettyprint
  # 'diagonal' is [[1, 2, 3, 4], [5, 6, 7, 8]]

  and diagonal.shape = (2, 4)

  tf.batch_matrix_diag(diagonal) ==> [[[1, 0, 0, 0]
                                       [0, 2, 0, 0]
                                       [0, 0, 3, 0]
                                       [0, 0, 0, 4]],
                                      [[5, 0, 0, 0]
                                       [0, 6, 0, 0]
                                       [0, 0, 7, 0]
                                       [0, 0, 0, 8]]]

  which has shape (2, 4, 4)
  ```

  Args:
    diagonal: A `Tensor`. Rank `k`, where `k >= 1`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `diagonal`.
    Rank `k+1`, with `output.shape = diagonal.shape + [diagonal.shape[-1]]`.
  """
  result = _op_def_lib.apply_op("BatchMatrixDiag", diagonal=diagonal,
                                name=name)
  return result


_batch_matrix_diag_part_outputs = ["diagonal"]


def batch_matrix_diag_part(input, name=None):
  r"""Returns the batched diagonal part of a batched tensor.

  This operation returns a tensor with the `diagonal` part
  of the batched `input`. The `diagonal` part is computed as follows:

  Assume `input` has `k` dimensions `[I, J, K, ..., N, N]`, then the output is a
  tensor of rank `k - 1` with dimensions `[I, J, K, ..., N]` where:

  `diagonal[i, j, k, ..., n] = input[i, j, k, ..., n, n]`.

  The input must be at least a matrix.

  For example:

  ```prettyprint
  # 'input' is [[[1, 0, 0, 0]
                 [0, 2, 0, 0]
                 [0, 0, 3, 0]
                 [0, 0, 0, 4]],
                [[5, 0, 0, 0]
                 [0, 6, 0, 0]
                 [0, 0, 7, 0]
                 [0, 0, 0, 8]]]

  and input.shape = (2, 4, 4)

  tf.batch_matrix_diag_part(input) ==> [[1, 2, 3, 4], [5, 6, 7, 8]]

  which has shape (2, 4)
  ```

  Args:
    input: A `Tensor`.
      Rank `k` tensor where `k >= 2` and the last two dimensions are equal.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
    The extracted diagonal(s) having shape
    `diagonal.shape = input.shape[:-1]`.
  """
  result = _op_def_lib.apply_op("BatchMatrixDiagPart", input=input, name=name)
  return result


_batch_to_space_outputs = ["output"]


def batch_to_space(input, crops, block_size, name=None):
  r"""BatchToSpace for 4-D tensors of type T.

  Rearranges (permutes) data from batch into blocks of spatial data, followed by
  cropping. This is the reverse transformation of SpaceToBatch. More specifically,
  this op outputs a copy of the input tensor where values from the `batch`
  dimension are moved in spatial blocks to the `height` and `width` dimensions,
  followed by cropping along the `height` and `width` dimensions.

  Args:
    input: A `Tensor`. 4-D tensor with shape
      `[batch*block_size*block_size, height_pad/block_size, width_pad/block_size,
        depth]`. Note that the batch size of the input tensor must be divisible by
      `block_size * block_size`.
    crops: A `Tensor` of type `int32`.
      2-D tensor of non-negative integers with shape `[2, 2]`. It specifies
      how many elements to crop from the intermediate result across the spatial
      dimensions as follows:

          crops = [[crop_top, crop_bottom], [crop_left, crop_right]]
    block_size: An `int`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
    4-D with shape `[batch, height, width, depth]`, where:

          height = height_pad - crop_top - crop_bottom
          width = width_pad - crop_left - crop_right

    The attr `block_size` must be greater than one. It indicates the block size.
  """
  result = _op_def_lib.apply_op("BatchToSpace", input=input, crops=crops,
                                block_size=block_size, name=name)
  return result


_bitcast_outputs = ["output"]


def bitcast(input, type, name=None):
  r"""Bitcasts a tensor from one type to another without copying data.

  Given a tensor `input`, this operation returns a tensor that has the same buffer
  data as `input` with datatype `type`.

  If the input datatype `T` is larger than the output datatype `type` then the
  shape changes from [...] to [..., sizeof(`T`)/sizeof(`type`)].

  If `T` is smaller than `type`, the operator requires that the rightmost
  dimension be equal to sizeof(`type`)/sizeof(`T`). The shape then goes from
  [..., sizeof(`type`)/sizeof(`T`)] to [...].

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`, `complex128`, `qint8`, `quint8`, `qint32`, `half`.
    type: A `tf.DType` from: `tf.float32, tf.float64, tf.int64, tf.int32, tf.uint8, tf.uint16, tf.int16, tf.int8, tf.complex64, tf.complex128, tf.qint8, tf.quint8, tf.qint32, tf.half`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `type`.
  """
  result = _op_def_lib.apply_op("Bitcast", input=input, type=type, name=name)
  return result


__broadcast_gradient_args_outputs = ["r0", "r1"]


_BroadcastGradientArgsOutput = collections.namedtuple("BroadcastGradientArgs",
                                                      __broadcast_gradient_args_outputs)


def _broadcast_gradient_args(s0, s1, name=None):
  r"""Return the reduction indices for computing gradients of s0 op s1 with broadcast.

  This is typically used by gradient computations for a broadcasting operation.

  Args:
    s0: A `Tensor` of type `int32`.
    s1: A `Tensor` of type `int32`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (r0, r1).
    r0: A `Tensor` of type `int32`.
    r1: A `Tensor` of type `int32`.
  """
  result = _op_def_lib.apply_op("BroadcastGradientArgs", s0=s0, s1=s1,
                                name=name)
  return _BroadcastGradientArgsOutput._make(result)


_check_numerics_outputs = ["output"]


def check_numerics(tensor, message, name=None):
  r"""Checks a tensor for NaN and Inf values.

  When run, reports an `InvalidArgument` error if `tensor` has any values
  that are not a number (NaN) or infinity (Inf). Otherwise, passes `tensor` as-is.

  Args:
    tensor: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`.
    message: A `string`. Prefix of the error message.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `tensor`.
  """
  result = _op_def_lib.apply_op("CheckNumerics", tensor=tensor,
                                message=message, name=name)
  return result


__concat_outputs = ["output"]


def _concat(concat_dim, values, name=None):
  r"""Concatenates tensors along one dimension.

  Args:
    concat_dim: A `Tensor` of type `int32`.
      0-D.  The dimension along which to concatenate.  Must be in the
      range [0, rank(values)).
    values: A list of at least 2 `Tensor` objects of the same type.
      The `N` Tensors to concatenate. Their ranks and types must match,
      and their sizes must match in all dimensions except `concat_dim`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `values`.
    A `Tensor` with the concatenation of values stacked along the
    `concat_dim` dimension.  This tensor's shape matches that of `values` except
    in `concat_dim` where it has the sum of the sizes.
  """
  result = _op_def_lib.apply_op("Concat", concat_dim=concat_dim,
                                values=values, name=name)
  return result


__concat_offset_outputs = ["offset"]


def _concat_offset(concat_dim, shape, name=None):
  r"""Computes offsets of concat inputs within its output.

  For example:

  ```prettyprint
  # 'x' is [2, 2, 7]
  # 'y' is [2, 3, 7]
  # 'z' is [2, 5, 7]
  concat_offset(2, [x, y, z]) => [0, 0, 0], [0, 2, 0], [0, 5, 0]
  ```

  Args:
    concat_dim: A `Tensor` of type `int32`.
      The dimension along which to concatenate.
    shape: A list of at least 2 `Tensor` objects of type `int32`.
      The `N` int32 vectors representing shape of tensors being concatenated.
    name: A name for the operation (optional).

  Returns:
    A list with the same number of `Tensor` objects as `shape` of `Tensor` objects of type `int32`.
  """
  result = _op_def_lib.apply_op("ConcatOffset", concat_dim=concat_dim,
                                shape=shape, name=name)
  return result


__const_outputs = ["output"]


def _const(value, dtype, name=None):
  r"""Returns a constant tensor.

  Args:
    value: . Attr `value` is the tensor to return.
    dtype: A `tf.DType`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `dtype`.
  """
  result = _op_def_lib.apply_op("Const", value=value, dtype=dtype, name=name)
  return result


_depth_to_space_outputs = ["output"]


def depth_to_space(input, block_size, name=None):
  r"""DepthToSpace for tensors of type T.

  Rearranges data from depth into blocks of spatial data.
  This is the reverse transformation of SpaceToDepth. More specifically,
  this op outputs a copy of the input tensor where values from the `depth`
  dimension are moved in spatial blocks to the `height` and `width` dimensions.
  The attr `block_size` indicates the input block size and how the data is moved.

    * Chunks of data of size `block_size * block_size` from depth are rearranged
      into non-overlapping blocks of size `block_size x block_size`
    * The width the output tensor is `input_depth * block_size`, whereas the
      height is `input_height * block_size`.
    * The depth of the input tensor must be divisible by
      `block_size * block_size`.

  That is, assuming the input is in the shape:
  `[batch, height, width, depth]`,
  the shape of the output will be:
  `[batch, height*block_size, width*block_size, depth/(block_size*block_size)]`

  This operation requires that the input tensor be of rank 4, and that
  `block_size` be >=1 and that `block_size * block_size` be a divisor of the
  input depth.

  This operation is useful for resizing the activations between convolutions
  (but keeping all data), e.g. instead of pooling. It is also useful for training
  purely convolutional models.

  For example, given this input of shape `[1, 1, 1, 4]`, and a block size of 2:

  ```prettyprint
  x = [[[[1, 2, 3, 4]]]]

  ```

  This operation will output a tensor of shape `[1, 2, 2, 1]`:

  ```prettyprint
     [[[[1], [2]],
       [[3], [4]]]]
  ```

  Here, the input has a batch of 1 and each batch element has shape `[1, 1, 4]`,
  the corresponding output will have 2x2 elements and will have a depth of
  1 channel (1 = `4 / (block_size * block_size)`).
  The output element shape is `[2, 2, 1]`.

  For an input tensor with larger depth, here of shape `[1, 1, 1, 12]`, e.g.

  ```prettyprint
  x = [[[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]]]
  ```

  This operation, for block size of 2, will return the following tensor of shape
  `[1, 2, 2, 3]`

  ```prettyprint
     [[[[1, 2, 3], [4, 5, 6]],
       [[7, 8, 9], [10, 11, 12]]]]

  ```

  Similarly, for the following input of shape `[1 2 2 4]`, and a block size of 2:

  ```prettyprint
  x =  [[[[1, 2, 3, 4],
         [5, 6, 7, 8]],
        [[9, 10, 11, 12],
         [13, 14, 15, 16]]]]
  ```

  the operator will return the following tensor of shape `[1 4 4 1]`:

  ```prettyprint
  x = [[ [1],   [2],  [5],  [6]],
       [ [3],   [4],  [7],  [8]],
       [ [9],  [10], [13],  [14]],
       [ [11], [12], [15],  [16]]]

  ```

  Args:
    input: A `Tensor`.
    block_size: An `int`.
      The size of the spatial block, same as in Space2Depth.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  result = _op_def_lib.apply_op("DepthToSpace", input=input,
                                block_size=block_size, name=name)
  return result


_diag_outputs = ["output"]


def diag(diagonal, name=None):
  r"""Returns a diagonal tensor with a given diagonal values.

  Given a `diagonal`, this operation returns a tensor with the `diagonal` and
  everything else padded with zeros. The diagonal is computed as follows:

  Assume `diagonal` has dimensions [D1,..., Dk], then the output is a tensor of
  rank 2k with dimensions [D1,..., Dk, D1,..., Dk] where:

  `output[i1,..., ik, i1,..., ik] = diagonal[i1, ..., ik]` and 0 everywhere else.

  For example:

  ```prettyprint
  # 'diagonal' is [1, 2, 3, 4]
  tf.diag(diagonal) ==> [[1, 0, 0, 0]
                         [0, 2, 0, 0]
                         [0, 0, 3, 0]
                         [0, 0, 0, 4]]
  ```

  Args:
    diagonal: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`, `complex64`.
      Rank k tensor where k is at most 3.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `diagonal`.
  """
  result = _op_def_lib.apply_op("Diag", diagonal=diagonal, name=name)
  return result


_diag_part_outputs = ["diagonal"]


def diag_part(input, name=None):
  r"""Returns the diagonal part of the tensor.

  This operation returns a tensor with the `diagonal` part
  of the `input`. The `diagonal` part is computed as follows:

  Assume `input` has dimensions `[D1,..., Dk, D1,..., Dk]`, then the output is a
  tensor of rank `k` with dimensions `[D1,..., Dk]` where:

  `diagonal[i1,..., ik] = input[i1, ..., ik, i1,..., ik]`.

  For example:

  ```prettyprint
  # 'input' is [[1, 0, 0, 0]
                [0, 2, 0, 0]
                [0, 0, 3, 0]
                [0, 0, 0, 4]]

  tf.diag_part(input) ==> [1, 2, 3, 4]
  ```

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`, `complex64`.
      Rank k tensor where k is 2, 4, or 6.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`. The extracted diagonal.
  """
  result = _op_def_lib.apply_op("DiagPart", input=input, name=name)
  return result


__edit_distance_outputs = ["output"]


def _edit_distance(hypothesis_indices, hypothesis_values, hypothesis_shape,
                   truth_indices, truth_values, truth_shape, normalize=None,
                   name=None):
  r"""Computes the (possibly normalized) Levenshtein Edit Distance.

  The inputs are variable-length sequences provided by SparseTensors
    (hypothesis_indices, hypothesis_values, hypothesis_shape)
  and
    (truth_indices, truth_values, truth_shape).

  The inputs are:

  Args:
    hypothesis_indices: A `Tensor` of type `int64`.
      The indices of the hypothesis list SparseTensor.
      This is an N x R int64 matrix.
    hypothesis_values: A `Tensor`.
      The values of the hypothesis list SparseTensor.
      This is an N-length vector.
    hypothesis_shape: A `Tensor` of type `int64`.
      The shape of the hypothesis list SparseTensor.
      This is an R-length vector.
    truth_indices: A `Tensor` of type `int64`.
      The indices of the truth list SparseTensor.
      This is an M x R int64 matrix.
    truth_values: A `Tensor`. Must have the same type as `hypothesis_values`.
      The values of the truth list SparseTensor.
      This is an M-length vector.
    truth_shape: A `Tensor` of type `int64`. truth indices, vector.
    normalize: An optional `bool`. Defaults to `True`.
      boolean (if true, edit distances are normalized by length of truth).

      The output is:
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32`. A dense float tensor with rank R - 1.

    For the example input:

        // hypothesis represents a 2x1 matrix with variable-length values:
        //   (0,0) = ["a"]
        //   (1,0) = ["b"]
        hypothesis_indices = [[0, 0, 0],
                              [1, 0, 0]]
        hypothesis_values = ["a", "b"]
        hypothesis_shape = [2, 1, 1]

        // truth represents a 2x2 matrix with variable-length values:
        //   (0,0) = []
        //   (0,1) = ["a"]
        //   (1,0) = ["b", "c"]
        //   (1,1) = ["a"]
        truth_indices = [[0, 1, 0],
                         [1, 0, 0],
                         [1, 0, 1],
                         [1, 1, 0]]
        truth_values = ["a", "b", "c", "a"]
        truth_shape = [2, 2, 2]
        normalize = true

    The output will be:

        // output is a 2x2 matrix with edit distances normalized by truth lengths.
        output = [[inf, 1.0],  // (0,0): no truth, (0,1): no hypothesis
                  [0.5, 1.0]]  // (1,0): addition, (1,1): no hypothesis
  """
  result = _op_def_lib.apply_op("EditDistance",
                                hypothesis_indices=hypothesis_indices,
                                hypothesis_values=hypothesis_values,
                                hypothesis_shape=hypothesis_shape,
                                truth_indices=truth_indices,
                                truth_values=truth_values,
                                truth_shape=truth_shape, normalize=normalize,
                                name=name)
  return result


_expand_dims_outputs = ["output"]


def expand_dims(input, dim, name=None):
  r"""Inserts a dimension of 1 into a tensor's shape.

  Given a tensor `input`, this operation inserts a dimension of 1 at the
  dimension index `dim` of `input`'s shape. The dimension index `dim` starts at
  zero; if you specify a negative number for `dim` it is counted backward from
  the end.

  This operation is useful if you want to add a batch dimension to a single
  element. For example, if you have a single image of shape `[height, width,
  channels]`, you can make it a batch of 1 image with `expand_dims(image, 0)`,
  which will make the shape `[1, height, width, channels]`.

  Other examples:

  ```prettyprint
  # 't' is a tensor of shape [2]
  shape(expand_dims(t, 0)) ==> [1, 2]
  shape(expand_dims(t, 1)) ==> [2, 1]
  shape(expand_dims(t, -1)) ==> [2, 1]

  # 't2' is a tensor of shape [2, 3, 5]
  shape(expand_dims(t2, 0)) ==> [1, 2, 3, 5]
  shape(expand_dims(t2, 2)) ==> [2, 3, 1, 5]
  shape(expand_dims(t2, 3)) ==> [2, 3, 5, 1]
  ```

  This operation requires that:

  `-1-input.dims() <= dim <= input.dims()`

  This operation is related to `squeeze()`, which removes dimensions of
  size 1.

  Args:
    input: A `Tensor`.
    dim: A `Tensor` of type `int32`.
      0-D (scalar). Specifies the dimension index at which to
      expand the shape of `input`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
    Contains the same data as `input`, but its shape has an additional
    dimension of size 1 added.
  """
  result = _op_def_lib.apply_op("ExpandDims", input=input, dim=dim, name=name)
  return result


_extract_image_patches_outputs = ["patches"]


def extract_image_patches(images, padding, ksizes=None, strides=None,
                          rates=None, name=None):
  r"""Extract `patches` from `images` and puth them in the "depth" output dimension.

  Args:
    images: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`, `uint8`, `int16`, `int8`, `uint16`, `half`.
      4-D Tensor with shape `[batch, in_rows, in_cols, depth]`.
    padding: A `string` from: `"SAME", "VALID"`.
      The type of padding algorithm to use.

      We specify the size-related attributes as:

            ksizes = [1, ksize_rows, ksize_cols, 1]
            strides = [1, strides_rows, strides_cols, 1]
            rates = [1, rates_rows, rates_cols, 1]
    ksizes: An optional list of `ints`. Defaults to `[]`.
      The size of the sliding window for each dimension of `images`.
    strides: An optional list of `ints`. Defaults to `[]`.
      1-D of length 4. How far the centers of two consecutive patches are in
      the images. Must be: `[1, stride_rows, stride_cols, 1]`.
    rates: An optional list of `ints`. Defaults to `[]`.
      1-D of length 4. Must be: `[1, rate_rows, rate_cols, 1]`. This is the
      input stride, specifying how far two consecutive patch samples are in the
      input. Equivalent to extracting patches with
      `patch_sizes_eff = patch_sizes + (patch_sizes - 1) * (rates - 1), followed by
      subsampling them spatially by a factor of `rates`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `images`.
    4-D Tensor with shape `[batch, out_rows, out_cols, ksize_rows *
    ksize_cols * depth]` containing image patches with size
    `ksize_rows x ksize_cols x depth` vectorized in the "depth" dimension.
  """
  result = _op_def_lib.apply_op("ExtractImagePatches", images=images,
                                padding=padding, ksizes=ksizes,
                                strides=strides, rates=rates, name=name)
  return result


_fill_outputs = ["output"]


def fill(dims, value, name=None):
  r"""Creates a tensor filled with a scalar value.

  This operation creates a tensor of shape `dims` and fills it with `value`.

  For example:

  ```prettyprint
  # Output tensor has shape [2, 3].
  fill([2, 3], 9) ==> [[9, 9, 9]
                       [9, 9, 9]]
  ```

  Args:
    dims: A `Tensor` of type `int32`.
      1-D. Represents the shape of the output tensor.
    value: A `Tensor`. 0-D (scalar). Value to fill the returned tensor.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `value`.
  """
  result = _op_def_lib.apply_op("Fill", dims=dims, value=value, name=name)
  return result


_gather_outputs = ["output"]


def gather(params, indices, validate_indices=None, name=None):
  r"""Gather slices from `params` according to `indices`.

  `indices` must be an integer tensor of any dimension (usually 0-D or 1-D).
  Produces an output tensor with shape `indices.shape + params.shape[1:]` where:

      # Scalar indices
      output[:, ..., :] = params[indices, :, ... :]

      # Vector indices
      output[i, :, ..., :] = params[indices[i], :, ... :]

      # Higher rank indices
      output[i, ..., j, :, ... :] = params[indices[i, ..., j], :, ..., :]

  If `indices` is a permutation and `len(indices) == params.shape[0]` then
  this operation will permute `params` accordingly.

  <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
  <img style="width:100%" src="../../images/Gather.png" alt>
  </div>

  Args:
    params: A `Tensor`.
    indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.
    validate_indices: An optional `bool`. Defaults to `True`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `params`.
  """
  result = _op_def_lib.apply_op("Gather", params=params, indices=indices,
                                validate_indices=validate_indices, name=name)
  return result


_gather_nd_outputs = ["output"]


def gather_nd(params, indices, name=None):
  r"""Gather values from `params` according to `indices`.

  `indices` must be integer tensor, containing indices into `params`.
  It must be shape `[d_0, ..., d_N, R]` where `R` is the rank of `params`.
  The innermost dimension of `indices` (with length `R`) corresponds to the
  indices of `params`.

  Produces an output tensor with shape `[d_0, ..., d_{n-1}]` where:

      output[i, j, k, ...] = params[indices[i, j, k, ..., :]]

  e.g. for `indices` a matrix:

      output[i] = params[indices[i, :]]

  Args:
    params: A `Tensor`. R-D.  The tensor from which to gather values.
    indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      (N+1)-D.  Index tensor having shape `[d_0, ..., d_N, R]`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `params`.
    N-D.  Values from `params` gathered from indices given by `indices`.
  """
  result = _op_def_lib.apply_op("GatherNd", params=params, indices=indices,
                                name=name)
  return result


_identity_outputs = ["output"]


def identity(input, name=None):
  r"""Return a tensor with the same shape and contents as the input tensor or value.

  Args:
    input: A `Tensor`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  result = _op_def_lib.apply_op("Identity", input=input, name=name)
  return result


_immutable_const_outputs = ["tensor"]


def immutable_const(dtype, shape, memory_region_name, name=None):
  r"""Returns immutable tensor from memory region.

  The current implementation memmaps the tensor from a file.

  Args:
    dtype: A `tf.DType`. Type of the returned tensor.
    shape: A `tf.TensorShape` or list of `ints`. Shape of the returned tensor.
    memory_region_name: A `string`.
      Name of readonly memory region used by the tensor, see
      NewReadOnlyMemoryRegionFromFile in tensorflow::Env.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `dtype`.
  """
  result = _op_def_lib.apply_op("ImmutableConst", dtype=dtype, shape=shape,
                                memory_region_name=memory_region_name,
                                name=name)
  return result


_invert_permutation_outputs = ["y"]


def invert_permutation(x, name=None):
  r"""Computes the inverse permutation of a tensor.

  This operation computes the inverse of an index permutation. It takes a 1-D
  integer tensor `x`, which represents the indices of a zero-based array, and
  swaps each value with its index position. In other words, for an output tensor
  `y` and an input tensor `x`, this operation computes the following:

  `y[x[i]] = i for i in [0, 1, ..., len(x) - 1]`

  The values must include 0. There can be no duplicate values or negative values.

  For example:

  ```prettyprint
  # tensor `x` is [3, 4, 0, 2, 1]
  invert_permutation(x) ==> [2, 4, 3, 0, 1]
  ```

  Args:
    x: A `Tensor` of type `int32`. 1-D.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int32`. 1-D.
  """
  result = _op_def_lib.apply_op("InvertPermutation", x=x, name=name)
  return result


_list_diff_outputs = ["out", "idx"]


_ListDiffOutput = collections.namedtuple("ListDiff", _list_diff_outputs)


def list_diff(x, y, name=None):
  r"""Computes the difference between two lists of numbers or strings.

  Given a list `x` and a list `y`, this operation returns a list `out` that
  represents all values that are in `x` but not in `y`. The returned list `out`
  is sorted in the same order that the numbers appear in `x` (duplicates are
  preserved). This operation also returns a list `idx` that represents the
  position of each `out` element in `x`. In other words:

  `out[i] = x[idx[i]] for i in [0, 1, ..., len(out) - 1]`

  For example, given this input:

  ```prettyprint
  x = [1, 2, 3, 4, 5, 6]
  y = [1, 3, 5]
  ```

  This operation would return:

  ```prettyprint
  out ==> [2, 4, 6]
  idx ==> [1, 3, 5]
  ```

  Args:
    x: A `Tensor`. 1-D. Values to keep.
    y: A `Tensor`. Must have the same type as `x`. 1-D. Values to remove.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (out, idx).
    out: A `Tensor`. Has the same type as `x`. 1-D. Values present in `x` but not in `y`.
    idx: A `Tensor` of type `int32`. 1-D. Positions of `x` values preserved in `out`.
  """
  result = _op_def_lib.apply_op("ListDiff", x=x, y=y, name=name)
  return _ListDiffOutput._make(result)


__mirror_pad_outputs = ["output"]


def _mirror_pad(input, paddings, mode, name=None):
  r"""Pads a tensor with mirrored values.

  This operation pads a `input` with mirrored values according to the `paddings`
  you specify. `paddings` is an integer tensor with shape `[n, 2]`, where n is
  the rank of `input`. For each dimension D of `input`, `paddings[D, 0]` indicates
  how many values to add before the contents of `input` in that dimension, and
  `paddings[D, 1]` indicates how many values to add after the contents of `input`
  in that dimension. Both `paddings[D, 0]` and `paddings[D, 1]` must be no greater
  than `input.dim_size(D)` (or `input.dim_size(D) - 1`) if `copy_border` is true
  (if false, respectively).

  The padded size of each dimension D of the output is:

  `paddings(D, 0) + input.dim_size(D) + paddings(D, 1)`

  For example:

  ```prettyprint
  # 't' is [[1, 2, 3], [4, 5, 6]].
  # 'paddings' is [[1, 1]], [2, 2]].
  # 'mode' is SYMMETRIC.
  # rank of 't' is 2.
  pad(t, paddings) ==> [[2, 1, 1, 2, 3, 3, 2]
                        [2, 1, 1, 2, 3, 3, 2]
                        [5, 4, 4, 5, 6, 6, 5]
                        [5, 4, 4, 5, 6, 6, 5]]
  ```

  Args:
    input: A `Tensor`. The input tensor to be padded.
    paddings: A `Tensor` of type `int32`.
      A two-column matrix specifying the padding sizes. The number of
      rows must be the same as the rank of `input`.
    mode: A `string` from: `"REFLECT", "SYMMETRIC"`.
      Either `REFLECT` or `SYMMETRIC`. In reflect mode the padded regions
      do not include the borders, while in symmetric mode the padded regions
      do include the borders. For example, if `input` is `[1, 2, 3]` and `paddings`
      is `[0, 2]`, then the output is `[1, 2, 3, 2, 1]` in reflect mode, and
      it is `[1, 2, 3, 3, 2]` in symmetric mode.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`. The padded tensor.
  """
  result = _op_def_lib.apply_op("MirrorPad", input=input, paddings=paddings,
                                mode=mode, name=name)
  return result


__mirror_pad_grad_outputs = ["output"]


def _mirror_pad_grad(input, paddings, mode, name=None):
  r"""Gradient op for `MirrorPad` op. This op folds a mirror-padded tensor.

  This operation folds the padded areas of `input` by `MirrorPad` according to the
  `paddings` you specify. `paddings` must be the same as `paddings` argument
  given to the corresponding `MirrorPad` op.

  The folded size of each dimension D of the output is:

  `input.dim_size(D) - paddings(D, 0) - paddings(D, 1)`

  For example:

  ```prettyprint
  # 't' is [[1, 2, 3], [4, 5, 6], [7, 8, 9]].
  # 'paddings' is [[0, 1]], [0, 1]].
  # 'mode' is SYMMETRIC.
  # rank of 't' is 2.
  pad(t, paddings) ==> [[ 1,  5]
                        [11, 28]]
  ```

  Args:
    input: A `Tensor`. The input tensor to be folded.
    paddings: A `Tensor` of type `int32`.
      A two-column matrix specifying the padding sizes. The number of
      rows must be the same as the rank of `input`.
    mode: A `string` from: `"REFLECT", "SYMMETRIC"`.
      The mode used in the `MirrorPad` op.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`. The folded tensor.
  """
  result = _op_def_lib.apply_op("MirrorPadGrad", input=input,
                                paddings=paddings, mode=mode, name=name)
  return result


__one_hot_outputs = ["output"]


def _one_hot(indices, depth, on_value, off_value, axis=None, name=None):
  r"""Returns a one-hot tensor.

  The locations represented by indices in `indices` take value `on_value`,
  while all other locations take value `off_value`.

  If the input `indices` is rank `N`, the output will have rank `N+1`,
  The new axis is created at dimension `axis` (default: the new axis is
  appended at the end).

  If `indices` is a scalar the output shape will be a vector of length `depth`.

  If `indices` is a vector of length `features`, the output shape will be:
  ```
    features x depth if axis == -1
    depth x features if axis == 0
  ```

  If `indices` is a matrix (batch) with shape `[batch, features]`,
  the output shape will be:
  ```
    batch x features x depth if axis == -1
    batch x depth x features if axis == 1
    depth x batch x features if axis == 0
  ```


  Examples
  =========

  Suppose that

  ```
    indices = [0, 2, -1, 1]
    depth = 3
    on_value = 5.0
    off_value = 0.0
    axis = -1
  ```

  Then output is `[4 x 3]`:

      ```output =
        [5.0 0.0 0.0]  // one_hot(0)
        [0.0 0.0 5.0]  // one_hot(2)
        [0.0 0.0 0.0]  // one_hot(-1)
        [0.0 5.0 0.0]  // one_hot(1)
      ```

  Suppose that

  ```
    indices = [0, 2, -1, 1]
    depth = 3
    on_value = 0.0
    off_value = 3.0
    axis = 0
  ```

  Then output is `[3 x 4]`:

      ```output =
        [0.0 3.0 3.0 3.0]
        [3.0 3.0 3.0 0.0]
        [3.0 3.0 3.0 3.0]
        [3.0 0.0 3.0 3.0]
      //  ^                one_hot(0)
      //      ^            one_hot(2)
      //          ^        one_hot(-1)
      //              ^    one_hot(1)
      ```
  Suppose that

  ```
    indices = [[0, 2], [1, -1]]
    depth = 3
    on_value = 1.0
    off_value = 0.0
    axis = -1
  ```

  Then output is `[2 x 2 x 3]`:

      ```output =
        [
          [1.0, 0.0, 0.0]  // one_hot(0)
          [0.0, 0.0, 1.0]  // one_hot(2)
        ][
          [0.0, 1.0, 0.0]  // one_hot(1)
          [0.0, 0.0, 0.0]  // one_hot(-1)
        ]```

  Args:
    indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A tensor of indices.
    depth: A `Tensor` of type `int32`.
      A scalar defining the depth of the one hot dimension.
    on_value: A `Tensor`.
      A scalar defining the value to fill in output when `indices[j] = i`.
    off_value: A `Tensor`. Must have the same type as `on_value`.
      A scalar defining the value to fill in output when `indices[j] != i`.
    axis: An optional `int`. Defaults to `-1`.
      The axis to fill (default: -1, a new inner-most axis).
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `on_value`. The one-hot tensor.
  """
  result = _op_def_lib.apply_op("OneHot", indices=indices, depth=depth,
                                on_value=on_value, off_value=off_value,
                                axis=axis, name=name)
  return result


__pack_outputs = ["output"]


def _pack(values, name=None):
  r"""Packs a list of `N` rank-`R` tensors into one rank-`(R+1)` tensor.

  Packs the `N` tensors in `values` into a tensor with rank one higher than each
  tensor in `values` and shape `[N] + values[0].shape`. The output satisfies
  `output[i, ...] = values[i][...]`.

  This is the opposite of `unpack`.

  Args:
    values: A list of at least 1 `Tensor` objects of the same type.
      Must be of same shape and type.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `values`. The packed tensor.
  """
  result = _op_def_lib.apply_op("Pack", values=values, name=name)
  return result


__pad_outputs = ["output"]


def _pad(input, paddings, name=None):
  r"""Pads a tensor with zeros.

  This operation pads a `input` with zeros according to the `paddings` you
  specify. `paddings` is an integer tensor with shape `[Dn, 2]`, where n is the
  rank of `input`. For each dimension D of `input`, `paddings[D, 0]` indicates
  how many zeros to add before the contents of `input` in that dimension, and
  `paddings[D, 1]` indicates how many zeros to add after the contents of `input`
  in that dimension.

  The padded size of each dimension D of the output is:

  `paddings(D, 0) + input.dim_size(D) + paddings(D, 1)`

  For example:

  ```prettyprint
  # 't' is [[1, 1], [2, 2]]
  # 'paddings' is [[1, 1], [2, 2]]
  # rank of 't' is 2
  pad(t, paddings) ==> [[0, 0, 0, 0, 0, 0]
                        [0, 0, 1, 1, 0, 0]
                        [0, 0, 2, 2, 0, 0]
                        [0, 0, 0, 0, 0, 0]]
  ```

  Args:
    input: A `Tensor`.
    paddings: A `Tensor` of type `int32`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  result = _op_def_lib.apply_op("Pad", input=input, paddings=paddings,
                                name=name)
  return result


__placeholder_outputs = ["output"]


def _placeholder(dtype, shape=None, name=None):
  r"""A placeholder op for a value that will be fed into the computation.

  N.B. This operation will fail with an error if it is executed. It is
  intended as a way to represent a value that will always be fed, and to
  provide attrs that enable the fed value to be checked at runtime.

  Args:
    dtype: A `tf.DType`. The type of elements in the tensor.
    shape: An optional `tf.TensorShape` or list of `ints`. Defaults to `[]`.
      (Optional) The shape of the tensor. If the shape has 0 dimensions, the
      shape is unconstrained.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `dtype`.
    A placeholder tensor that must be replaced using the feed mechanism.
  """
  result = _op_def_lib.apply_op("Placeholder", dtype=dtype, shape=shape,
                                name=name)
  return result


_placeholder_with_default_outputs = ["output"]


def placeholder_with_default(input, shape, name=None):
  r"""A placeholder op that passes though `input` when its output is not fed.

  Args:
    input: A `Tensor`. The default value to produce when `output` is not fed.
    shape: A `tf.TensorShape` or list of `ints`.
      The (possibly partial) shape of the tensor.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
    A placeholder tensor that defaults to `input` if it is not fed.
  """
  result = _op_def_lib.apply_op("PlaceholderWithDefault", input=input,
                                shape=shape, name=name)
  return result


_rank_outputs = ["output"]


def rank(input, name=None):
  r"""Returns the rank of a tensor.

  This operation returns an integer representing the rank of `input`.

  For example:

  ```prettyprint
  # 't' is [[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]]
  # shape of tensor 't' is [2, 2, 3]
  rank(t) ==> 3
  ```

  **Note**: The rank of a tensor is not the same as the rank of a matrix. The rank
  of a tensor is the number of indices required to uniquely select each element
  of the tensor. Rank is also known as "order", "degree", or "ndims."

  Args:
    input: A `Tensor`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int32`.
  """
  result = _op_def_lib.apply_op("Rank", input=input, name=name)
  return result


__ref_identity_outputs = ["output"]


def _ref_identity(input, name=None):
  r"""Return the same ref tensor as the input ref tensor.

  Args:
    input: A mutable `Tensor`.
    name: A name for the operation (optional).

  Returns:
    A mutable `Tensor`. Has the same type as `input`.
  """
  result = _op_def_lib.apply_op("RefIdentity", input=input, name=name)
  return result


_reshape_outputs = ["output"]


def reshape(tensor, shape, name=None):
  r"""Reshapes a tensor.

  Given `tensor`, this operation returns a tensor that has the same values
  as `tensor` with shape `shape`.

  If one component of `shape` is the special value -1, the size of that dimension
  is computed so that the total size remains constant.  In particular, a `shape`
  of `[-1]` flattens into 1-D.  At most one component of `shape` can be -1.

  If `shape` is 1-D or higher, then the operation returns a tensor with shape
  `shape` filled with the values of `tensor`. In this case, the number of elements
  implied by `shape` must be the same as the number of elements in `tensor`.

  For example:

  ```prettyprint
  # tensor 't' is [1, 2, 3, 4, 5, 6, 7, 8, 9]
  # tensor 't' has shape [9]
  reshape(t, [3, 3]) ==> [[1, 2, 3],
                          [4, 5, 6],
                          [7, 8, 9]]

  # tensor 't' is [[[1, 1], [2, 2]],
  #                [[3, 3], [4, 4]]]
  # tensor 't' has shape [2, 2, 2]
  reshape(t, [2, 4]) ==> [[1, 1, 2, 2],
                          [3, 3, 4, 4]]

  # tensor 't' is [[[1, 1, 1],
  #                 [2, 2, 2]],
  #                [[3, 3, 3],
  #                 [4, 4, 4]],
  #                [[5, 5, 5],
  #                 [6, 6, 6]]]
  # tensor 't' has shape [3, 2, 3]
  # pass '[-1]' to flatten 't'
  reshape(t, [-1]) ==> [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6]

  # -1 can also be used to infer the shape

  # -1 is inferred to be 9:
  reshape(t, [2, -1]) ==> [[1, 1, 1, 2, 2, 2, 3, 3, 3],
                           [4, 4, 4, 5, 5, 5, 6, 6, 6]]
  # -1 is inferred to be 2:
  reshape(t, [-1, 9]) ==> [[1, 1, 1, 2, 2, 2, 3, 3, 3],
                           [4, 4, 4, 5, 5, 5, 6, 6, 6]]
  # -1 is inferred to be 3:
  reshape(t, [ 2, -1, 3]) ==> [[[1, 1, 1],
                                [2, 2, 2],
                                [3, 3, 3]],
                               [[4, 4, 4],
                                [5, 5, 5],
                                [6, 6, 6]]]

  # tensor 't' is [7]
  # shape `[]` reshapes to a scalar
  reshape(t, []) ==> 7
  ```

  Args:
    tensor: A `Tensor`.
    shape: A `Tensor` of type `int32`. Defines the shape of the output tensor.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `tensor`.
  """
  result = _op_def_lib.apply_op("Reshape", tensor=tensor, shape=shape,
                                name=name)
  return result


_reverse_outputs = ["output"]


def reverse(tensor, dims, name=None):
  r"""Reverses specific dimensions of a tensor.

  Given a `tensor`, and a `bool` tensor `dims` representing the dimensions
  of `tensor`, this operation reverses each dimension i of `tensor` where
  `dims[i]` is `True`.

  `tensor` can have up to 8 dimensions. The number of dimensions
  of `tensor` must equal the number of elements in `dims`. In other words:

  `rank(tensor) = size(dims)`

  For example:

  ```prettyprint
  # tensor 't' is [[[[ 0,  1,  2,  3],
  #                  [ 4,  5,  6,  7],
  #                  [ 8,  9, 10, 11]],
  #                 [[12, 13, 14, 15],
  #                  [16, 17, 18, 19],
  #                  [20, 21, 22, 23]]]]
  # tensor 't' shape is [1, 2, 3, 4]

  # 'dims' is [False, False, False, True]
  reverse(t, dims) ==> [[[[ 3,  2,  1,  0],
                          [ 7,  6,  5,  4],
                          [ 11, 10, 9, 8]],
                         [[15, 14, 13, 12],
                          [19, 18, 17, 16],
                          [23, 22, 21, 20]]]]

  # 'dims' is [False, True, False, False]
  reverse(t, dims) ==> [[[[12, 13, 14, 15],
                          [16, 17, 18, 19],
                          [20, 21, 22, 23]
                         [[ 0,  1,  2,  3],
                          [ 4,  5,  6,  7],
                          [ 8,  9, 10, 11]]]]

  # 'dims' is [False, False, True, False]
  reverse(t, dims) ==> [[[[8, 9, 10, 11],
                          [4, 5, 6, 7],
                          [0, 1, 2, 3]]
                         [[20, 21, 22, 23],
                          [16, 17, 18, 19],
                          [12, 13, 14, 15]]]]
  ```

  Args:
    tensor: A `Tensor`. Must be one of the following types: `uint8`, `int8`, `int32`, `bool`, `half`, `float32`, `float64`.
      Up to 8-D.
    dims: A `Tensor` of type `bool`. 1-D. The dimensions to reverse.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `tensor`. The same shape as `tensor`.
  """
  result = _op_def_lib.apply_op("Reverse", tensor=tensor, dims=dims,
                                name=name)
  return result


_reverse_sequence_outputs = ["output"]


def reverse_sequence(input, seq_lengths, seq_dim, batch_dim=None, name=None):
  r"""Reverses variable length slices.

  This op first slices `input` along the dimension `batch_dim`, and for each
  slice `i`, reverses the first `seq_lengths[i]` elements along
  the dimension `seq_dim`.

  The elements of `seq_lengths` must obey `seq_lengths[i] < input.dims[seq_dim]`,
  and `seq_lengths` must be a vector of length `input.dims[batch_dim]`.

  The output slice `i` along dimension `batch_dim` is then given by input
  slice `i`, with the first `seq_lengths[i]` slices along dimension
  `seq_dim` reversed.

  For example:

  ```prettyprint
  # Given this:
  batch_dim = 0
  seq_dim = 1
  input.dims = (4, 8, ...)
  seq_lengths = [7, 2, 3, 5]

  # then slices of input are reversed on seq_dim, but only up to seq_lengths:
  output[0, 0:7, :, ...] = input[0, 7:0:-1, :, ...]
  output[1, 0:2, :, ...] = input[1, 2:0:-1, :, ...]
  output[2, 0:3, :, ...] = input[2, 3:0:-1, :, ...]
  output[3, 0:5, :, ...] = input[3, 5:0:-1, :, ...]

  # while entries past seq_lens are copied through:
  output[0, 7:, :, ...] = input[0, 7:, :, ...]
  output[1, 2:, :, ...] = input[1, 2:, :, ...]
  output[2, 3:, :, ...] = input[2, 3:, :, ...]
  output[3, 2:, :, ...] = input[3, 2:, :, ...]
  ```

  In contrast, if:

  ```prettyprint
  # Given this:
  batch_dim = 2
  seq_dim = 0
  input.dims = (8, ?, 4, ...)
  seq_lengths = [7, 2, 3, 5]

  # then slices of input are reversed on seq_dim, but only up to seq_lengths:
  output[0:7, :, 0, :, ...] = input[7:0:-1, :, 0, :, ...]
  output[0:2, :, 1, :, ...] = input[2:0:-1, :, 1, :, ...]
  output[0:3, :, 2, :, ...] = input[3:0:-1, :, 2, :, ...]
  output[0:5, :, 3, :, ...] = input[5:0:-1, :, 3, :, ...]

  # while entries past seq_lens are copied through:
  output[7:, :, 0, :, ...] = input[7:, :, 0, :, ...]
  output[2:, :, 1, :, ...] = input[2:, :, 1, :, ...]
  output[3:, :, 2, :, ...] = input[3:, :, 2, :, ...]
  output[2:, :, 3, :, ...] = input[2:, :, 3, :, ...]
  ```

  Args:
    input: A `Tensor`. The input to reverse.
    seq_lengths: A `Tensor` of type `int64`.
      1-D with length `input.dims(batch_dim)` and
      `max(seq_lengths) < input.dims(seq_dim)`
    seq_dim: An `int`. The dimension which is partially reversed.
    batch_dim: An optional `int`. Defaults to `0`.
      The dimension along which reversal is performed.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
    The partially reversed input. It has the same shape as `input`.
  """
  result = _op_def_lib.apply_op("ReverseSequence", input=input,
                                seq_lengths=seq_lengths, seq_dim=seq_dim,
                                batch_dim=batch_dim, name=name)
  return result


_shape_outputs = ["output"]


def shape(input, name=None):
  r"""Returns the shape of a tensor.

  This operation returns a 1-D integer tensor representing the shape of `input`.

  For example:

  ```prettyprint
  # 't' is [[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]]
  shape(t) ==> [2, 2, 3]
  ```

  Args:
    input: A `Tensor`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int32`.
  """
  result = _op_def_lib.apply_op("Shape", input=input, name=name)
  return result


_shape_n_outputs = ["output"]


def shape_n(input, name=None):
  r"""Returns shape of tensors.

  This operation returns N 1-D integer tensors representing shape of `input[i]s`.

  Args:
    input: A list of at least 1 `Tensor` objects of the same type.
    name: A name for the operation (optional).

  Returns:
    A list with the same number of `Tensor` objects as `input` of `Tensor` objects of type `int32`.
  """
  result = _op_def_lib.apply_op("ShapeN", input=input, name=name)
  return result


_size_outputs = ["output"]


def size(input, name=None):
  r"""Returns the size of a tensor.

  This operation returns an integer representing the number of elements in
  `input`.

  For example:

  ```prettyprint
  # 't' is [[[1, 1,, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]]]
  size(t) ==> 12
  ```

  Args:
    input: A `Tensor`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int32`.
  """
  result = _op_def_lib.apply_op("Size", input=input, name=name)
  return result


__slice_outputs = ["output"]


def _slice(input, begin, size, name=None):
  r"""Return a slice from 'input'.

  The output tensor is a tensor with dimensions described by 'size'
  whose values are extracted from 'input' starting at the offsets in
  'begin'.

  *Requirements*:
    0 <= begin[i] <= begin[i] + size[i] <= Di  for i in [0, n)

  Args:
    input: A `Tensor`.
    begin: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      begin[i] specifies the offset into the 'i'th dimension of
      'input' to slice from.
    size: A `Tensor`. Must have the same type as `begin`.
      size[i] specifies the number of elements of the 'i'th dimension
      of 'input' to slice. If size[i] is -1, all remaining elements in dimension
      i are included in the slice (i.e. this is equivalent to setting
      size[i] = input.dim_size(i) - begin[i]).
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  result = _op_def_lib.apply_op("Slice", input=input, begin=begin, size=size,
                                name=name)
  return result


_space_to_batch_outputs = ["output"]


def space_to_batch(input, paddings, block_size, name=None):
  r"""SpaceToBatch for 4-D tensors of type T.

  Zero-pads and then rearranges (permutes) blocks of spatial data into batch.
  More specifically, this op outputs a copy of the input tensor where values from
  the `height` and `width` dimensions are moved to the `batch` dimension. After
  the zero-padding, both `height` and `width` of the input must be divisible by the
  block size.

  Args:
    input: A `Tensor`. 4-D with shape `[batch, height, width, depth]`.
    paddings: A `Tensor` of type `int32`.
      2-D tensor of non-negative integers with shape `[2, 2]`. It specifies
        the padding of the input with zeros across the spatial dimensions as follows:

            paddings = [[pad_top, pad_bottom], [pad_left, pad_right]]

        The effective spatial dimensions of the zero-padded input tensor will be:

            height_pad = pad_top + height + pad_bottom
            width_pad = pad_left + width + pad_right

      The attr `block_size` must be greater than one. It indicates the block size.

        * Non-overlapping blocks of size `block_size x block size` in the height and
          width dimensions are rearranged into the batch dimension at each location.
        * The batch of the output tensor is `batch * block_size * block_size`.
        * Both height_pad and width_pad must be divisible by block_size.

      The shape of the output will be:

          [batch*block_size*block_size, height_pad/block_size, width_pad/block_size,
           depth]
    block_size: An `int`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  result = _op_def_lib.apply_op("SpaceToBatch", input=input,
                                paddings=paddings, block_size=block_size,
                                name=name)
  return result


_space_to_depth_outputs = ["output"]


def space_to_depth(input, block_size, name=None):
  r"""SpaceToDepth for tensors of type T.

  Rearranges blocks of spatial data, into depth. More specifically,
  this op outputs a copy of the input tensor where values from the `height`
  and `width` dimensions are moved to the `depth` dimension.
  The attr `block_size` indicates the input block size and how the data is moved.

    * Non-overlapping blocks of size `block_size x block size` are rearranged
      into depth at each location.
    * The depth of the output tensor is `input_depth * block_size * block_size`.
    * The input tensor's height and width must be divisible by block_size.

  That is, assuming the input is in the shape:
  `[batch, height, width, depth]`,
  the shape of the output will be:
  `[batch, height/block_size, width/block_size, depth*block_size*block_size]`

  This operation requires that the input tensor be of rank 4, and that
  `block_size` be >=1 and a divisor of both the input `height` and `width`.

  This operation is useful for resizing the activations between convolutions
  (but keeping all data), e.g. instead of pooling. It is also useful for training
  purely convolutional models.

  For example, given this input of shape `[1, 2, 2, 1]`, and block_size of 2:

  ```prettyprint
  x = [[[[1], [2]],
        [[3], [4]]]]
  ```

  This operation will output a tensor of shape `[1, 1, 1, 4]`:

  ```prettyprint
  [[[[1, 2, 3, 4]]]]
  ```

  Here, the input has a batch of 1 and each batch element has shape `[2, 2, 1]`,
  the corresponding output will have a single element (i.e. width and height are
  both 1) and will have a depth of 4 channels (1 * block_size * block_size).
  The output element shape is `[1, 1, 4]`.

  For an input tensor with larger depth, here of shape `[1, 2, 2, 3]`, e.g.

  ```prettyprint
  x = [[[[1, 2, 3], [4, 5, 6]],
        [[7, 8, 9], [10, 11, 12]]]]
  ```

  This operation, for block_size of 2, will return the following tensor of shape
  `[1, 1, 1, 12]`

  ```prettyprint
  [[[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]]]
  ```

  Similarly, for the following input of shape `[1 4 4 1]`, and a block size of 2:

  ```prettyprint
  x = [[[[1],   [2],  [5],  [6]],
        [[3],   [4],  [7],  [8]],
        [[9],  [10], [13],  [14]],
        [[11], [12], [15],  [16]]]]
  ```

  the operator will return the following tensor of shape `[1 2 2 4]`:

  ```prettyprint
  x = [[[[1, 2, 3, 4],
         [5, 6, 7, 8]],
        [[9, 10, 11, 12],
         [13, 14, 15, 16]]]]
  ```

  Args:
    input: A `Tensor`.
    block_size: An `int`. The size of the spatial block.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  result = _op_def_lib.apply_op("SpaceToDepth", input=input,
                                block_size=block_size, name=name)
  return result


__split_outputs = ["output"]


def _split(split_dim, value, num_split, name=None):
  r"""Splits a tensor into `num_split` tensors along one dimension.

  Args:
    split_dim: A `Tensor` of type `int32`.
      0-D.  The dimension along which to split.  Must be in the range
      `[0, rank(value))`.
    value: A `Tensor`. The tensor to split.
    num_split: An `int` that is `>= 1`.
      The number of ways to split.  Must evenly divide
      `value.shape[split_dim]`.
    name: A name for the operation (optional).

  Returns:
    A list of `num_split` `Tensor` objects of the same type as value.
    They are identically shaped tensors, whose shape matches that of `value`
    except along `split_dim`, where their sizes are
    `values.shape[split_dim] / num_split`.
  """
  result = _op_def_lib.apply_op("Split", split_dim=split_dim, value=value,
                                num_split=num_split, name=name)
  return result


_squeeze_outputs = ["output"]


def squeeze(input, squeeze_dims=None, name=None):
  r"""Removes dimensions of size 1 from the shape of a tensor.

  Given a tensor `input`, this operation returns a tensor of the same type with
  all dimensions of size 1 removed. If you don't want to remove all size 1
  dimensions, you can remove specific size 1 dimensions by specifying
  `squeeze_dims`.

  For example:

  ```prettyprint
  # 't' is a tensor of shape [1, 2, 1, 3, 1, 1]
  shape(squeeze(t)) ==> [2, 3]
  ```

  Or, to remove specific size 1 dimensions:

  ```prettyprint
  # 't' is a tensor of shape [1, 2, 1, 3, 1, 1]
  shape(squeeze(t, [2, 4])) ==> [1, 2, 3, 1]
  ```

  Args:
    input: A `Tensor`. The `input` to squeeze.
    squeeze_dims: An optional list of `ints`. Defaults to `[]`.
      If specified, only squeezes the dimensions listed. The dimension
      index starts at 0. It is an error to squeeze a dimension that is not 1.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
    Contains the same data as `input`, but has one or more dimensions of
    size 1 removed.
  """
  result = _op_def_lib.apply_op("Squeeze", input=input,
                                squeeze_dims=squeeze_dims, name=name)
  return result


_stop_gradient_outputs = ["output"]


def stop_gradient(input, name=None):
  r"""Stops gradient computation.

  When executed in a graph, this op outputs its input tensor as-is.

  When building ops to compute gradients, this op prevents the contribution of
  its inputs to be taken into account.  Normally, the gradient generator adds ops
  to a graph to compute the derivatives of a specified 'loss' by recursively
  finding out inputs that contributed to its computation.  If you insert this op
  in the graph it inputs are masked from the gradient generator.  They are not
  taken into account for computing gradients.

  This is useful any time you want to compute a value with TensorFlow but need
  to pretend that the value was a constant. Some examples include:

  *  The *EM* algorithm where the *M-step* should not involve backpropagation
     through the output of the *E-step*.
  *  Contrastive divergence training of Boltzmann machines where, when
     differentiating the energy function, the training must not backpropagate
     through the graph that generated the samples from the model.
  *  Adversarial training, where no backprop should happen through the adversarial
     example generation process.

  Args:
    input: A `Tensor`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  result = _op_def_lib.apply_op("StopGradient", input=input, name=name)
  return result


_tile_outputs = ["output"]


def tile(input, multiples, name=None):
  r"""Constructs a tensor by tiling a given tensor.

  This operation creates a new tensor by replicating `input` `multiples` times.
  The output tensor's i'th dimension has `input.dims(i) * multiples[i]` elements,
  and the values of `input` are replicated `multiples[i]` times along the 'i'th
  dimension. For example, tiling `[a b c d]` by `[2]` produces
  `[a b c d a b c d]`.

  Args:
    input: A `Tensor`. 1-D or higher.
    multiples: A `Tensor` of type `int32`.
      1-D. Length must be the same as the number of dimensions in `input`
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  result = _op_def_lib.apply_op("Tile", input=input, multiples=multiples,
                                name=name)
  return result


__tile_grad_outputs = ["output"]


def _tile_grad(input, multiples, name=None):
  r"""Returns the gradient of `Tile`.

  Since `Tile` takes an input and repeats the input `multiples` times
  along each dimension, `TileGrad` takes in `multiples` and aggregates
  each repeated tile of `input` into `output`.

  Args:
    input: A `Tensor`.
    multiples: A `Tensor` of type `int32`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  result = _op_def_lib.apply_op("TileGrad", input=input, multiples=multiples,
                                name=name)
  return result


_transpose_outputs = ["y"]


def transpose(x, perm, name=None):
  r"""Shuffle dimensions of x according to a permutation.

  The output `y` has the same rank as `x`. The shapes of `x` and `y` satisfy:
    `y.shape[i] == x.shape[perm[i]] for i in [0, 1, ..., rank(x) - 1]`

  Args:
    x: A `Tensor`.
    perm: A `Tensor` of type `int32`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
  result = _op_def_lib.apply_op("Transpose", x=x, perm=perm, name=name)
  return result


_unique_outputs = ["y", "idx"]


_UniqueOutput = collections.namedtuple("Unique", _unique_outputs)


def unique(x, name=None):
  r"""Finds unique elements in a 1-D tensor.

  This operation returns a tensor `y` containing all of the unique elements of `x`
  sorted in the same order that they occur in `x`. This operation also returns a
  tensor `idx` the same size as `x` that contains the index of each value of `x`
  in the unique output `y`. In other words:

  `y[idx[i]] = x[i] for i in [0, 1,...,rank(x) - 1]`

  For example:

  ```prettyprint
  # tensor 'x' is [1, 1, 2, 4, 4, 4, 7, 8, 8]
  y, idx = unique(x)
  y ==> [1, 2, 4, 7, 8]
  idx ==> [0, 0, 1, 2, 2, 2, 3, 4, 4]
  ```

  Args:
    x: A `Tensor`. 1-D.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (y, idx).
    y: A `Tensor`. Has the same type as `x`. 1-D.
    idx: A `Tensor` of type `int32`. 1-D.
  """
  result = _op_def_lib.apply_op("Unique", x=x, name=name)
  return _UniqueOutput._make(result)


_unique_with_counts_outputs = ["y", "idx", "count"]


_UniqueWithCountsOutput = collections.namedtuple("UniqueWithCounts",
                                                 _unique_with_counts_outputs)


def unique_with_counts(x, name=None):
  r"""Finds unique elements in a 1-D tensor.

  This operation returns a tensor `y` containing all of the unique elements of `x`
  sorted in the same order that they occur in `x`. This operation also returns a
  tensor `idx` the same size as `x` that contains the index of each value of `x`
  in the unique output `y`. Finally, it returns a third tensor `count` that
  contains the count of each element of `y` in `x`. In other words:

  `y[idx[i]] = x[i] for i in [0, 1,...,rank(x) - 1]`

  For example:

  ```prettyprint
  # tensor 'x' is [1, 1, 2, 4, 4, 4, 7, 8, 8]
  y, idx, count = unique_with_counts(x)
  y ==> [1, 2, 4, 7, 8]
  idx ==> [0, 0, 1, 2, 2, 2, 3, 4, 4]
  count ==> [2, 1, 3, 1, 2]
  ```

  Args:
    x: A `Tensor`. 1-D.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (y, idx, count).
    y: A `Tensor`. Has the same type as `x`. 1-D.
    idx: A `Tensor` of type `int32`. 1-D.
    count: A `Tensor` of type `int32`. 1-D.
  """
  result = _op_def_lib.apply_op("UniqueWithCounts", x=x, name=name)
  return _UniqueWithCountsOutput._make(result)


__unpack_outputs = ["output"]


def _unpack(value, num, name=None):
  r"""Unpacks the outer dimension of a rank-`R` tensor into `num` rank-`(R-1)` tensors.

  Unpacks `num` tensors from `value` by chipping it along the first dimension.
  The i'th tensor in `output` is the slice `value[i, ...]`. Each tensor in
  `output` has shape `value.shape[1:]`.

  This is the opposite of `pack`.

  Args:
    value: A `Tensor`. 1-D or higher, with first dimension `num`.
    num: An `int` that is `>= 0`.
    name: A name for the operation (optional).

  Returns:
    A list of `num` `Tensor` objects of the same type as value.
    The list of tensors unpacked from `value`.
  """
  result = _op_def_lib.apply_op("Unpack", value=value, num=num, name=name)
  return result


_where_outputs = ["index"]


def where(input, name=None):
  r"""Returns locations of true values in a boolean tensor.

  This operation returns the coordinates of true elements in `input`. The
  coordinates are returned in a 2-D tensor where the first dimension (rows)
  represents the number of true elements, and the second dimension (columns)
  represents the coordinates of the true elements. Keep in mind, the shape of
  the output tensor can vary depending on how many true values there are in
  `input`. Indices are output in row-major order.

  For example:

  ```prettyprint
  # 'input' tensor is [[True, False]
  #                    [True, False]]
  # 'input' has two true values, so output has two coordinates.
  # 'input' has rank of 2, so coordinates have two indices.
  where(input) ==> [[0, 0],
                    [1, 0]]

  # `input` tensor is [[[True, False]
  #                     [True, False]]
  #                    [[False, True]
  #                     [False, True]]
  #                    [[False, False]
  #                     [False, True]]]
  # 'input' has 5 true values, so output has 5 coordinates.
  # 'input' has rank of 3, so coordinates have three indices.
  where(input) ==> [[0, 0, 0],
                    [0, 1, 0],
                    [1, 0, 1],
                    [1, 1, 1],
                    [2, 1, 1]]
  ```

  Args:
    input: A `Tensor` of type `bool`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int64`.
  """
  result = _op_def_lib.apply_op("Where", input=input, name=name)
  return result


__zeros_like_outputs = ["y"]


def _zeros_like(x, name=None):
  r"""Returns a tensor of zeros with the same shape and type as x.

  Args:
    x: A `Tensor`. a tensor of type T.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
    a tensor of the same shape and type as x but filled with zeros.
  """
  result = _op_def_lib.apply_op("ZerosLike", x=x, name=name)
  return result


def _InitOpDefLibrary():
  op_list = op_def_pb2.OpList()
  text_format.Merge(_InitOpDefLibrary.op_list_ascii, op_list)
  op_def_registry.register_op_list(op_list)
  op_def_lib = op_def_library.OpDefLibrary()
  op_def_lib.add_op_list(op_list)
  return op_def_lib


_InitOpDefLibrary.op_list_ascii = """op {
  name: "BatchMatrixBandPart"
  input_arg {
    name: "input"
    type_attr: "T"
  }
  input_arg {
    name: "num_lower"
    type: DT_INT64
  }
  input_arg {
    name: "num_upper"
    type: DT_INT64
  }
  output_arg {
    name: "band"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
  }
}
op {
  name: "BatchMatrixDiag"
  input_arg {
    name: "diagonal"
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
  name: "BatchMatrixDiagPart"
  input_arg {
    name: "input"
    type_attr: "T"
  }
  output_arg {
    name: "diagonal"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
  }
}
op {
  name: "BatchToSpace"
  input_arg {
    name: "input"
    type_attr: "T"
  }
  input_arg {
    name: "crops"
    type: DT_INT32
  }
  output_arg {
    name: "output"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
  }
  attr {
    name: "block_size"
    type: "int"
  }
}
op {
  name: "Bitcast"
  input_arg {
    name: "input"
    type_attr: "T"
  }
  output_arg {
    name: "output"
    type_attr: "type"
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
    name: "type"
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
  name: "BroadcastGradientArgs"
  input_arg {
    name: "s0"
    type: DT_INT32
  }
  input_arg {
    name: "s1"
    type: DT_INT32
  }
  output_arg {
    name: "r0"
    type: DT_INT32
  }
  output_arg {
    name: "r1"
    type: DT_INT32
  }
}
op {
  name: "CheckNumerics"
  input_arg {
    name: "tensor"
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
    name: "message"
    type: "string"
  }
}
op {
  name: "Concat"
  input_arg {
    name: "concat_dim"
    type: DT_INT32
  }
  input_arg {
    name: "values"
    type_attr: "T"
    number_attr: "N"
  }
  output_arg {
    name: "output"
    type_attr: "T"
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
op {
  name: "ConcatOffset"
  input_arg {
    name: "concat_dim"
    type: DT_INT32
  }
  input_arg {
    name: "shape"
    type: DT_INT32
    number_attr: "N"
  }
  output_arg {
    name: "offset"
    type: DT_INT32
    number_attr: "N"
  }
  attr {
    name: "N"
    type: "int"
    has_minimum: true
    minimum: 2
  }
}
op {
  name: "Const"
  output_arg {
    name: "output"
    type_attr: "dtype"
  }
  attr {
    name: "value"
    type: "tensor"
  }
  attr {
    name: "dtype"
    type: "type"
  }
}
op {
  name: "DepthToSpace"
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
  }
  attr {
    name: "block_size"
    type: "int"
  }
}
op {
  name: "Diag"
  input_arg {
    name: "diagonal"
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
        type: DT_COMPLEX64
      }
    }
  }
}
op {
  name: "DiagPart"
  input_arg {
    name: "input"
    type_attr: "T"
  }
  output_arg {
    name: "diagonal"
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
        type: DT_COMPLEX64
      }
    }
  }
}
op {
  name: "EditDistance"
  input_arg {
    name: "hypothesis_indices"
    type: DT_INT64
  }
  input_arg {
    name: "hypothesis_values"
    type_attr: "T"
  }
  input_arg {
    name: "hypothesis_shape"
    type: DT_INT64
  }
  input_arg {
    name: "truth_indices"
    type: DT_INT64
  }
  input_arg {
    name: "truth_values"
    type_attr: "T"
  }
  input_arg {
    name: "truth_shape"
    type: DT_INT64
  }
  output_arg {
    name: "output"
    type: DT_FLOAT
  }
  attr {
    name: "normalize"
    type: "bool"
    default_value {
      b: true
    }
  }
  attr {
    name: "T"
    type: "type"
  }
}
op {
  name: "ExpandDims"
  input_arg {
    name: "input"
    type_attr: "T"
  }
  input_arg {
    name: "dim"
    type: DT_INT32
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
  name: "ExtractImagePatches"
  input_arg {
    name: "images"
    type_attr: "T"
  }
  output_arg {
    name: "patches"
    type_attr: "T"
  }
  attr {
    name: "ksizes"
    type: "list(int)"
    default_value {
    }
  }
  attr {
    name: "strides"
    type: "list(int)"
    default_value {
    }
  }
  attr {
    name: "rates"
    type: "list(int)"
    default_value {
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
  name: "Fill"
  input_arg {
    name: "dims"
    type: DT_INT32
  }
  input_arg {
    name: "value"
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
  name: "Gather"
  input_arg {
    name: "params"
    type_attr: "Tparams"
  }
  input_arg {
    name: "indices"
    type_attr: "Tindices"
  }
  output_arg {
    name: "output"
    type_attr: "Tparams"
  }
  attr {
    name: "validate_indices"
    type: "bool"
    default_value {
      b: true
    }
  }
  attr {
    name: "Tparams"
    type: "type"
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
  name: "GatherNd"
  input_arg {
    name: "params"
    type_attr: "Tparams"
  }
  input_arg {
    name: "indices"
    type_attr: "Tindices"
  }
  output_arg {
    name: "output"
    type_attr: "Tparams"
  }
  attr {
    name: "Tparams"
    type: "type"
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
  name: "Identity"
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
  }
}
op {
  name: "ImmutableConst"
  output_arg {
    name: "tensor"
    type_attr: "dtype"
  }
  attr {
    name: "dtype"
    type: "type"
  }
  attr {
    name: "shape"
    type: "shape"
  }
  attr {
    name: "memory_region_name"
    type: "string"
  }
}
op {
  name: "InvertPermutation"
  input_arg {
    name: "x"
    type: DT_INT32
  }
  output_arg {
    name: "y"
    type: DT_INT32
  }
}
op {
  name: "ListDiff"
  input_arg {
    name: "x"
    type_attr: "T"
  }
  input_arg {
    name: "y"
    type_attr: "T"
  }
  output_arg {
    name: "out"
    type_attr: "T"
  }
  output_arg {
    name: "idx"
    type: DT_INT32
  }
  attr {
    name: "T"
    type: "type"
  }
}
op {
  name: "MirrorPad"
  input_arg {
    name: "input"
    type_attr: "T"
  }
  input_arg {
    name: "paddings"
    type: DT_INT32
  }
  output_arg {
    name: "output"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
  }
  attr {
    name: "mode"
    type: "string"
    allowed_values {
      list {
        s: "REFLECT"
        s: "SYMMETRIC"
      }
    }
  }
}
op {
  name: "MirrorPadGrad"
  input_arg {
    name: "input"
    type_attr: "T"
  }
  input_arg {
    name: "paddings"
    type: DT_INT32
  }
  output_arg {
    name: "output"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
  }
  attr {
    name: "mode"
    type: "string"
    allowed_values {
      list {
        s: "REFLECT"
        s: "SYMMETRIC"
      }
    }
  }
}
op {
  name: "OneHot"
  input_arg {
    name: "indices"
    type_attr: "TI"
  }
  input_arg {
    name: "depth"
    type: DT_INT32
  }
  input_arg {
    name: "on_value"
    type_attr: "T"
  }
  input_arg {
    name: "off_value"
    type_attr: "T"
  }
  output_arg {
    name: "output"
    type_attr: "T"
  }
  attr {
    name: "axis"
    type: "int"
    default_value {
      i: -1
    }
  }
  attr {
    name: "T"
    type: "type"
  }
  attr {
    name: "TI"
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
  name: "Pack"
  input_arg {
    name: "values"
    type_attr: "T"
    number_attr: "N"
  }
  output_arg {
    name: "output"
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
  }
}
op {
  name: "Pad"
  input_arg {
    name: "input"
    type_attr: "T"
  }
  input_arg {
    name: "paddings"
    type: DT_INT32
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
  name: "Placeholder"
  output_arg {
    name: "output"
    type_attr: "dtype"
  }
  attr {
    name: "dtype"
    type: "type"
  }
  attr {
    name: "shape"
    type: "shape"
    default_value {
      shape {
      }
    }
  }
}
op {
  name: "PlaceholderWithDefault"
  input_arg {
    name: "input"
    type_attr: "dtype"
  }
  output_arg {
    name: "output"
    type_attr: "dtype"
  }
  attr {
    name: "dtype"
    type: "type"
  }
  attr {
    name: "shape"
    type: "shape"
  }
}
op {
  name: "Rank"
  input_arg {
    name: "input"
    type_attr: "T"
  }
  output_arg {
    name: "output"
    type: DT_INT32
  }
  attr {
    name: "T"
    type: "type"
  }
}
op {
  name: "RefIdentity"
  input_arg {
    name: "input"
    type_attr: "T"
    is_ref: true
  }
  output_arg {
    name: "output"
    type_attr: "T"
    is_ref: true
  }
  attr {
    name: "T"
    type: "type"
  }
}
op {
  name: "Reshape"
  input_arg {
    name: "tensor"
    type_attr: "T"
  }
  input_arg {
    name: "shape"
    type: DT_INT32
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
  name: "Reverse"
  input_arg {
    name: "tensor"
    type_attr: "T"
  }
  input_arg {
    name: "dims"
    type: DT_BOOL
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
        type: DT_UINT8
        type: DT_INT8
        type: DT_INT32
        type: DT_BOOL
        type: DT_HALF
        type: DT_FLOAT
        type: DT_DOUBLE
      }
    }
  }
}
op {
  name: "ReverseSequence"
  input_arg {
    name: "input"
    type_attr: "T"
  }
  input_arg {
    name: "seq_lengths"
    type: DT_INT64
  }
  output_arg {
    name: "output"
    type_attr: "T"
  }
  attr {
    name: "seq_dim"
    type: "int"
  }
  attr {
    name: "batch_dim"
    type: "int"
    default_value {
      i: 0
    }
  }
  attr {
    name: "T"
    type: "type"
  }
}
op {
  name: "Shape"
  input_arg {
    name: "input"
    type_attr: "T"
  }
  output_arg {
    name: "output"
    type: DT_INT32
  }
  attr {
    name: "T"
    type: "type"
  }
}
op {
  name: "ShapeN"
  input_arg {
    name: "input"
    type_attr: "T"
    number_attr: "N"
  }
  output_arg {
    name: "output"
    type: DT_INT32
    number_attr: "N"
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
  }
}
op {
  name: "Size"
  input_arg {
    name: "input"
    type_attr: "T"
  }
  output_arg {
    name: "output"
    type: DT_INT32
  }
  attr {
    name: "T"
    type: "type"
  }
}
op {
  name: "Slice"
  input_arg {
    name: "input"
    type_attr: "T"
  }
  input_arg {
    name: "begin"
    type_attr: "Index"
  }
  input_arg {
    name: "size"
    type_attr: "Index"
  }
  output_arg {
    name: "output"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
  }
  attr {
    name: "Index"
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
  name: "SpaceToBatch"
  input_arg {
    name: "input"
    type_attr: "T"
  }
  input_arg {
    name: "paddings"
    type: DT_INT32
  }
  output_arg {
    name: "output"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
  }
  attr {
    name: "block_size"
    type: "int"
  }
}
op {
  name: "SpaceToDepth"
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
  }
  attr {
    name: "block_size"
    type: "int"
  }
}
op {
  name: "Split"
  input_arg {
    name: "split_dim"
    type: DT_INT32
  }
  input_arg {
    name: "value"
    type_attr: "T"
  }
  output_arg {
    name: "output"
    type_attr: "T"
    number_attr: "num_split"
  }
  attr {
    name: "num_split"
    type: "int"
    has_minimum: true
    minimum: 1
  }
  attr {
    name: "T"
    type: "type"
  }
}
op {
  name: "Squeeze"
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
  }
  attr {
    name: "squeeze_dims"
    type: "list(int)"
    default_value {
      list {
      }
    }
    has_minimum: true
  }
}
op {
  name: "StopGradient"
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
  }
}
op {
  name: "Tile"
  input_arg {
    name: "input"
    type_attr: "T"
  }
  input_arg {
    name: "multiples"
    type: DT_INT32
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
  name: "TileGrad"
  input_arg {
    name: "input"
    type_attr: "T"
  }
  input_arg {
    name: "multiples"
    type: DT_INT32
  }
  output_arg {
    name: "output"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
  }
  deprecation {
    version: 3
    explanation: "TileGrad has been replaced with reduce_sum"
  }
}
op {
  name: "Transpose"
  input_arg {
    name: "x"
    type_attr: "T"
  }
  input_arg {
    name: "perm"
    type: DT_INT32
  }
  output_arg {
    name: "y"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
  }
}
op {
  name: "Unique"
  input_arg {
    name: "x"
    type_attr: "T"
  }
  output_arg {
    name: "y"
    type_attr: "T"
  }
  output_arg {
    name: "idx"
    type: DT_INT32
  }
  attr {
    name: "T"
    type: "type"
  }
}
op {
  name: "UniqueWithCounts"
  input_arg {
    name: "x"
    type_attr: "T"
  }
  output_arg {
    name: "y"
    type_attr: "T"
  }
  output_arg {
    name: "idx"
    type: DT_INT32
  }
  output_arg {
    name: "count"
    type: DT_INT32
  }
  attr {
    name: "T"
    type: "type"
  }
}
op {
  name: "Unpack"
  input_arg {
    name: "value"
    type_attr: "T"
  }
  output_arg {
    name: "output"
    type_attr: "T"
    number_attr: "num"
  }
  attr {
    name: "num"
    type: "int"
    has_minimum: true
  }
  attr {
    name: "T"
    type: "type"
  }
}
op {
  name: "Where"
  input_arg {
    name: "input"
    type: DT_BOOL
  }
  output_arg {
    name: "index"
    type: DT_INT64
  }
}
op {
  name: "ZerosLike"
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
  }
}
"""


_op_def_lib = _InitOpDefLibrary()
