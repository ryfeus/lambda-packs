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
__add_many_sparse_to_tensors_map_outputs = ["sparse_handles"]


def _add_many_sparse_to_tensors_map(sparse_indices, sparse_values,
                                    sparse_shape, container=None,
                                    shared_name=None, name=None):
  r"""Add an `N`-minibatch `SparseTensor` to a `SparseTensorsMap`, return `N` handles.

  A `SparseTensor` of rank `R` is represented by three tensors: `sparse_indices`,
  `sparse_values`, and `sparse_shape`, where

  ```sparse_indices.shape[1] == sparse_shape.shape[0] == R```

  An `N`-minibatch of `SparseTensor` objects is represented as a `SparseTensor`
  having a first `sparse_indices` column taking values between `[0, N)`, where
  the minibatch size `N == sparse_shape[0]`.

  The input `SparseTensor` must have rank `R` greater than 1, and the first
  dimension is treated as the minibatch dimension.  Elements of the `SparseTensor`
  must be sorted in increasing order of this first dimension.  The stored
  `SparseTensor` objects pointed to by each row of the output `sparse_handles`
  will have rank `R-1`.

  The `SparseTensor` values can then be read out as part of a minibatch by passing
  the given keys as vector elements to `TakeManySparseFromTensorsMap`.  To ensure
  the correct `SparseTensorsMap` is accessed, ensure that the same
  `container` and `shared_name` are passed to that Op.  If no `shared_name`
  is provided here, instead use the *name* of the Operation created by calling
  `AddManySparseToTensorsMap` as the `shared_name` passed to
  `TakeManySparseFromTensorsMap`.  Ensure the Operations are colocated.

  Args:
    sparse_indices: A `Tensor` of type `int64`.
      2-D.  The `indices` of the minibatch `SparseTensor`.
      `sparse_indices[:, 0]` must be ordered values in `[0, N)`.
    sparse_values: A `Tensor`.
      1-D.  The `values` of the minibatch `SparseTensor`.
    sparse_shape: A `Tensor` of type `int64`.
      1-D.  The `shape` of the minibatch `SparseTensor`.
      The minibatch size `N == sparse_shape[0]`.
    container: An optional `string`. Defaults to `""`.
      The container name for the `SparseTensorsMap` created by this op.
    shared_name: An optional `string`. Defaults to `""`.
      The shared name for the `SparseTensorsMap` created by this op.
      If blank, the new Operation's unique name is used.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int64`.
    1-D.  The handles of the `SparseTensor` now stored in the
    `SparseTensorsMap`.  Shape: `[N]`.
  """
  result = _op_def_lib.apply_op("AddManySparseToTensorsMap",
                                sparse_indices=sparse_indices,
                                sparse_values=sparse_values,
                                sparse_shape=sparse_shape,
                                container=container, shared_name=shared_name,
                                name=name)
  return result


__add_sparse_to_tensors_map_outputs = ["sparse_handle"]


def _add_sparse_to_tensors_map(sparse_indices, sparse_values, sparse_shape,
                               container=None, shared_name=None, name=None):
  r"""Add a `SparseTensor` to a `SparseTensorsMap` return its handle.

  A `SparseTensor` is represented by three tensors: `sparse_indices`,
  `sparse_values`, and `sparse_shape`.

  This operator takes the given `SparseTensor` and adds it to a container
  object (a `SparseTensorsMap`).  A unique key within this container is generated
  in the form of an `int64`, and this is the value that is returned.

  The `SparseTensor` can then be read out as part of a minibatch by passing
  the key as a vector element to `TakeManySparseFromTensorsMap`.  To ensure
  the correct `SparseTensorsMap` is accessed, ensure that the same
  `container` and `shared_name` are passed to that Op.  If no `shared_name`
  is provided here, instead use the *name* of the Operation created by calling
  `AddSparseToTensorsMap` as the `shared_name` passed to
  `TakeManySparseFromTensorsMap`.  Ensure the Operations are colocated.

  Args:
    sparse_indices: A `Tensor` of type `int64`.
      2-D.  The `indices` of the `SparseTensor`.
    sparse_values: A `Tensor`. 1-D.  The `values` of the `SparseTensor`.
    sparse_shape: A `Tensor` of type `int64`.
      1-D.  The `shape` of the `SparseTensor`.
    container: An optional `string`. Defaults to `""`.
      The container name for the `SparseTensorsMap` created by this op.
    shared_name: An optional `string`. Defaults to `""`.
      The shared name for the `SparseTensorsMap` created by this op.
      If blank, the new Operation's unique name is used.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int64`.
    0-D.  The handle of the `SparseTensor` now stored in the
    `SparseTensorsMap`.
  """
  result = _op_def_lib.apply_op("AddSparseToTensorsMap",
                                sparse_indices=sparse_indices,
                                sparse_values=sparse_values,
                                sparse_shape=sparse_shape,
                                container=container, shared_name=shared_name,
                                name=name)
  return result


__deserialize_many_sparse_outputs = ["sparse_indices", "sparse_values",
                                    "sparse_shape"]


_DeserializeManySparseOutput = _collections.namedtuple("DeserializeManySparse",
                                                       __deserialize_many_sparse_outputs)


def _deserialize_many_sparse(serialized_sparse, dtype, name=None):
  r"""Deserialize and concatenate `SparseTensors` from a serialized minibatch.

  The input `serialized_sparse` must be a string matrix of shape `[N x 3]` where
  `N` is the minibatch size and the rows correspond to packed outputs of
  `SerializeSparse`.  The ranks of the original `SparseTensor` objects
  must all match.  When the final `SparseTensor` is created, it has rank one
  higher than the ranks of the incoming `SparseTensor` objects
  (they have been concatenated along a new row dimension).

  The output `SparseTensor` object's shape values for all dimensions but the
  first are the max across the input `SparseTensor` objects' shape values
  for the corresponding dimensions.  Its first shape value is `N`, the minibatch
  size.

  The input `SparseTensor` objects' indices are assumed ordered in
  standard lexicographic order.  If this is not the case, after this
  step run `SparseReorder` to restore index ordering.

  For example, if the serialized input is a `[2 x 3]` matrix representing two
  original `SparseTensor` objects:

      index = [ 0]
              [10]
              [20]
      values = [1, 2, 3]
      shape = [50]

  and

      index = [ 2]
              [10]
      values = [4, 5]
      shape = [30]

  then the final deserialized `SparseTensor` will be:

      index = [0  0]
              [0 10]
              [0 20]
              [1  2]
              [1 10]
      values = [1, 2, 3, 4, 5]
      shape = [2 50]

  Args:
    serialized_sparse: A `Tensor` of type `string`.
      2-D, The `N` serialized `SparseTensor` objects.
      Must have 3 columns.
    dtype: A `tf.DType`. The `dtype` of the serialized `SparseTensor` objects.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (sparse_indices, sparse_values, sparse_shape).
    sparse_indices: A `Tensor` of type `int64`.
    sparse_values: A `Tensor` of type `dtype`.
    sparse_shape: A `Tensor` of type `int64`.
  """
  result = _op_def_lib.apply_op("DeserializeManySparse",
                                serialized_sparse=serialized_sparse,
                                dtype=dtype, name=name)
  return _DeserializeManySparseOutput._make(result)


__serialize_many_sparse_outputs = ["serialized_sparse"]


def _serialize_many_sparse(sparse_indices, sparse_values, sparse_shape,
                           name=None):
  r"""Serialize an `N`-minibatch `SparseTensor` into an `[N, 3]` string `Tensor`.

  The `SparseTensor` must have rank `R` greater than 1, and the first dimension
  is treated as the minibatch dimension.  Elements of the `SparseTensor`
  must be sorted in increasing order of this first dimension.  The serialized
  `SparseTensor` objects going into each row of `serialized_sparse` will have
  rank `R-1`.

  The minibatch size `N` is extracted from `sparse_shape[0]`.

  Args:
    sparse_indices: A `Tensor` of type `int64`.
      2-D.  The `indices` of the minibatch `SparseTensor`.
    sparse_values: A `Tensor`.
      1-D.  The `values` of the minibatch `SparseTensor`.
    sparse_shape: A `Tensor` of type `int64`.
      1-D.  The `shape` of the minibatch `SparseTensor`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`.
  """
  result = _op_def_lib.apply_op("SerializeManySparse",
                                sparse_indices=sparse_indices,
                                sparse_values=sparse_values,
                                sparse_shape=sparse_shape, name=name)
  return result


__serialize_sparse_outputs = ["serialized_sparse"]


def _serialize_sparse(sparse_indices, sparse_values, sparse_shape, name=None):
  r"""Serialize a `SparseTensor` into a string 3-vector (1-D `Tensor`) object.

  Args:
    sparse_indices: A `Tensor` of type `int64`.
      2-D.  The `indices` of the `SparseTensor`.
    sparse_values: A `Tensor`. 1-D.  The `values` of the `SparseTensor`.
    sparse_shape: A `Tensor` of type `int64`.
      1-D.  The `shape` of the `SparseTensor`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`.
  """
  result = _op_def_lib.apply_op("SerializeSparse",
                                sparse_indices=sparse_indices,
                                sparse_values=sparse_values,
                                sparse_shape=sparse_shape, name=name)
  return result


__sparse_add_outputs = ["sum_indices", "sum_values", "sum_shape"]


_SparseAddOutput = _collections.namedtuple("SparseAdd", __sparse_add_outputs)


def _sparse_add(a_indices, a_values, a_shape, b_indices, b_values, b_shape,
                thresh, name=None):
  r"""Adds two `SparseTensor` objects to produce another `SparseTensor`.

  The input `SparseTensor` objects' indices are assumed ordered in standard
  lexicographic order.  If this is not the case, before this step run
  `SparseReorder` to restore index ordering.

  By default, if two values sum to zero at some index, the output `SparseTensor`
  would still include that particular location in its index, storing a zero in the
  corresponding value slot.  To override this, callers can specify `thresh`,
  indicating that if the sum has a magnitude strictly smaller than `thresh`, its
  corresponding value and index would then not be included.  In particular,
  `thresh == 0` (default) means everything is kept and actual thresholding happens
  only for a positive value.

  In the following shapes, `nnz` is the count after taking `thresh` into account.

  Args:
    a_indices: A `Tensor` of type `int64`.
      2-D.  The `indices` of the first `SparseTensor`, size `[nnz, ndims]` Matrix.
    a_values: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`, `complex128`, `qint8`, `quint8`, `qint32`, `half`.
      1-D.  The `values` of the first `SparseTensor`, size `[nnz]` Vector.
    a_shape: A `Tensor` of type `int64`.
      1-D.  The `shape` of the first `SparseTensor`, size `[ndims]` Vector.
    b_indices: A `Tensor` of type `int64`.
      2-D.  The `indices` of the second `SparseTensor`, size `[nnz, ndims]` Matrix.
    b_values: A `Tensor`. Must have the same type as `a_values`.
      1-D.  The `values` of the second `SparseTensor`, size `[nnz]` Vector.
    b_shape: A `Tensor` of type `int64`.
      1-D.  The `shape` of the second `SparseTensor`, size `[ndims]` Vector.
    thresh: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`, `uint8`, `int16`, `int8`, `uint16`, `half`.
      0-D.  The magnitude threshold that determines if an output value/index
      pair takes space.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (sum_indices, sum_values, sum_shape).
    sum_indices: A `Tensor` of type `int64`.
    sum_values: A `Tensor`. Has the same type as `a_values`.
    sum_shape: A `Tensor` of type `int64`.
  """
  result = _op_def_lib.apply_op("SparseAdd", a_indices=a_indices,
                                a_values=a_values, a_shape=a_shape,
                                b_indices=b_indices, b_values=b_values,
                                b_shape=b_shape, thresh=thresh, name=name)
  return _SparseAddOutput._make(result)


__sparse_add_grad_outputs = ["a_val_grad", "b_val_grad"]


_SparseAddGradOutput = _collections.namedtuple("SparseAddGrad",
                                               __sparse_add_grad_outputs)


def _sparse_add_grad(backprop_val_grad, a_indices, b_indices, sum_indices,
                     name=None):
  r"""The gradient operator for the SparseAdd op.

  The SparseAdd op calculates A + B, where A, B, and the sum are all represented
  as `SparseTensor` objects.  This op takes in the upstream gradient w.r.t.
  non-empty values of the sum, and outputs the gradients w.r.t. the non-empty
  values of A and B.

  Args:
    backprop_val_grad: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`, `complex128`, `qint8`, `quint8`, `qint32`, `half`.
      1-D with shape `[nnz(sum)]`.  The gradient with respect to
      the non-empty values of the sum.
    a_indices: A `Tensor` of type `int64`.
      2-D.  The `indices` of the `SparseTensor` A, size `[nnz(A), ndims]`.
    b_indices: A `Tensor` of type `int64`.
      2-D.  The `indices` of the `SparseTensor` B, size `[nnz(B), ndims]`.
    sum_indices: A `Tensor` of type `int64`.
      2-D.  The `indices` of the sum `SparseTensor`, size
      `[nnz(sum), ndims]`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (a_val_grad, b_val_grad).
    a_val_grad: A `Tensor`. Has the same type as `backprop_val_grad`. 1-D with shape `[nnz(A)]`. The gradient with respect to the
      non-empty values of A.
    b_val_grad: A `Tensor`. Has the same type as `backprop_val_grad`. 1-D with shape `[nnz(B)]`. The gradient with respect to the
      non-empty values of B.
  """
  result = _op_def_lib.apply_op("SparseAddGrad",
                                backprop_val_grad=backprop_val_grad,
                                a_indices=a_indices, b_indices=b_indices,
                                sum_indices=sum_indices, name=name)
  return _SparseAddGradOutput._make(result)


__sparse_concat_outputs = ["output_indices", "output_values", "output_shape"]


_SparseConcatOutput = _collections.namedtuple("SparseConcat",
                                              __sparse_concat_outputs)


def _sparse_concat(indices, values, shapes, concat_dim, name=None):
  r"""Concatenates a list of `SparseTensor` along the specified dimension.

  Concatenation is with respect to the dense versions of these sparse tensors.
  It is assumed that each input is a `SparseTensor` whose elements are ordered
  along increasing dimension number.

  All inputs' shapes must match, except for the concat dimension.  The
  `indices`, `values`, and `shapes` lists must have the same length.

  The output shape is identical to the inputs', except along the concat
  dimension, where it is the sum of the inputs' sizes along that dimension.

  The output elements will be resorted to preserve the sort order along
  increasing dimension number.

  This op runs in `O(M log M)` time, where `M` is the total number of non-empty
  values across all inputs. This is due to the need for an internal sort in
  order to concatenate efficiently across an arbitrary dimension.

  For example, if `concat_dim = 1` and the inputs are

      sp_inputs[0]: shape = [2, 3]
      [0, 2]: "a"
      [1, 0]: "b"
      [1, 1]: "c"

      sp_inputs[1]: shape = [2, 4]
      [0, 1]: "d"
      [0, 2]: "e"

  then the output will be

      shape = [2, 7]
      [0, 2]: "a"
      [0, 4]: "d"
      [0, 5]: "e"
      [1, 0]: "b"
      [1, 1]: "c"

  Graphically this is equivalent to doing

      [    a] concat [  d e  ] = [    a   d e  ]
      [b c  ]        [       ]   [b c          ]

  Args:
    indices: A list of at least 2 `Tensor` objects of type `int64`.
      2-D.  Indices of each input `SparseTensor`.
    values: A list with the same number of `Tensor` objects as `indices` of `Tensor` objects of the same type.
      1-D.  Non-empty values of each `SparseTensor`.
    shapes: A list with the same number of `Tensor` objects as `indices` of `Tensor` objects of type `int64`.
      1-D.  Shapes of each `SparseTensor`.
    concat_dim: An `int`.
      Dimension to concatenate along. Must be in range [-rank, rank),
      where rank is the number of dimensions in each input `SparseTensor`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output_indices, output_values, output_shape).
    output_indices: A `Tensor` of type `int64`. 2-D.  Indices of the concatenated `SparseTensor`.
    output_values: A `Tensor`. Has the same type as `values`. 1-D.  Non-empty values of the concatenated `SparseTensor`.
    output_shape: A `Tensor` of type `int64`. 1-D.  Shape of the concatenated `SparseTensor`.
  """
  result = _op_def_lib.apply_op("SparseConcat", indices=indices,
                                values=values, shapes=shapes,
                                concat_dim=concat_dim, name=name)
  return _SparseConcatOutput._make(result)


_sparse_dense_cwise_add_outputs = ["output"]


def sparse_dense_cwise_add(sp_indices, sp_values, sp_shape, dense, name=None):
  r"""Adds up a SparseTensor and a dense Tensor, using these special rules:

  (1) Broadcasts the dense side to have the same shape as the sparse side, if
      eligible;
  (2) Then, only the dense values pointed to by the indices of the SparseTensor
      participate in the cwise addition.

  By these rules, the result is a logical SparseTensor with exactly the same
  indices and shape, but possibly with different non-zero values.  The output of
  this Op is the resultant non-zero values.

  Args:
    sp_indices: A `Tensor` of type `int64`.
      2-D.  `N x R` matrix with the indices of non-empty values in a
      SparseTensor, possibly not in canonical ordering.
    sp_values: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`, `complex128`, `qint8`, `quint8`, `qint32`, `half`.
      1-D.  `N` non-empty values corresponding to `sp_indices`.
    sp_shape: A `Tensor` of type `int64`.
      1-D.  Shape of the input SparseTensor.
    dense: A `Tensor`. Must have the same type as `sp_values`.
      `R`-D.  The dense Tensor operand.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `sp_values`.
    1-D.  The `N` values that are operated on.
  """
  result = _op_def_lib.apply_op("SparseDenseCwiseAdd", sp_indices=sp_indices,
                                sp_values=sp_values, sp_shape=sp_shape,
                                dense=dense, name=name)
  return result


_sparse_dense_cwise_div_outputs = ["output"]


def sparse_dense_cwise_div(sp_indices, sp_values, sp_shape, dense, name=None):
  r"""Component-wise divides a SparseTensor by a dense Tensor.

  *Limitation*: this Op only broadcasts the dense side to the sparse side, but not
  the other direction.

  Args:
    sp_indices: A `Tensor` of type `int64`.
      2-D.  `N x R` matrix with the indices of non-empty values in a
      SparseTensor, possibly not in canonical ordering.
    sp_values: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`, `complex128`, `qint8`, `quint8`, `qint32`, `half`.
      1-D.  `N` non-empty values corresponding to `sp_indices`.
    sp_shape: A `Tensor` of type `int64`.
      1-D.  Shape of the input SparseTensor.
    dense: A `Tensor`. Must have the same type as `sp_values`.
      `R`-D.  The dense Tensor operand.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `sp_values`.
    1-D.  The `N` values that are operated on.
  """
  result = _op_def_lib.apply_op("SparseDenseCwiseDiv", sp_indices=sp_indices,
                                sp_values=sp_values, sp_shape=sp_shape,
                                dense=dense, name=name)
  return result


_sparse_dense_cwise_mul_outputs = ["output"]


def sparse_dense_cwise_mul(sp_indices, sp_values, sp_shape, dense, name=None):
  r"""Component-wise multiplies a SparseTensor by a dense Tensor.

  The output locations corresponding to the implicitly zero elements in the sparse
  tensor will be zero (i.e., will not take up storage space), regardless of the
  contents of the dense tensor (even if it's +/-INF and that INF*0 == NaN).

  *Limitation*: this Op only broadcasts the dense side to the sparse side, but not
  the other direction.

  Args:
    sp_indices: A `Tensor` of type `int64`.
      2-D.  `N x R` matrix with the indices of non-empty values in a
      SparseTensor, possibly not in canonical ordering.
    sp_values: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`, `complex128`, `qint8`, `quint8`, `qint32`, `half`.
      1-D.  `N` non-empty values corresponding to `sp_indices`.
    sp_shape: A `Tensor` of type `int64`.
      1-D.  Shape of the input SparseTensor.
    dense: A `Tensor`. Must have the same type as `sp_values`.
      `R`-D.  The dense Tensor operand.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `sp_values`.
    1-D.  The `N` values that are operated on.
  """
  result = _op_def_lib.apply_op("SparseDenseCwiseMul", sp_indices=sp_indices,
                                sp_values=sp_values, sp_shape=sp_shape,
                                dense=dense, name=name)
  return result


_sparse_reduce_sum_outputs = ["output"]


def sparse_reduce_sum(input_indices, input_values, input_shape,
                      reduction_axes, keep_dims=None, name=None):
  r"""Computes the sum of elements across dimensions of a SparseTensor.

  This Op takes a SparseTensor and is the sparse counterpart to
  `tf.reduce_sum()`.  In particular, this Op also returns a dense `Tensor`
  instead of a sparse one.

  Reduces `sp_input` along the dimensions given in `reduction_axes`.  Unless
  `keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
  `reduction_axes`. If `keep_dims` is true, the reduced dimensions are retained
  with length 1.

  If `reduction_axes` has no entries, all dimensions are reduced, and a tensor
  with a single element is returned.  Additionally, the axes can be negative,
  which are interpreted according to the indexing rules in Python.

  Args:
    input_indices: A `Tensor` of type `int64`.
      2-D.  `N x R` matrix with the indices of non-empty values in a
      SparseTensor, possibly not in canonical ordering.
    input_values: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`, `complex128`, `qint8`, `quint8`, `qint32`, `half`.
      1-D.  `N` non-empty values corresponding to `input_indices`.
    input_shape: A `Tensor` of type `int64`.
      1-D.  Shape of the input SparseTensor.
    reduction_axes: A `Tensor` of type `int32`.
      1-D.  Length-`K` vector containing the reduction axes.
    keep_dims: An optional `bool`. Defaults to `False`.
      If true, retain reduced dimensions with length 1.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input_values`.
    `R-K`-D.  The reduced Tensor.
  """
  result = _op_def_lib.apply_op("SparseReduceSum",
                                input_indices=input_indices,
                                input_values=input_values,
                                input_shape=input_shape,
                                reduction_axes=reduction_axes,
                                keep_dims=keep_dims, name=name)
  return result


_sparse_reduce_sum_sparse_outputs = ["output_indices", "output_values",
                                    "output_shape"]


_SparseReduceSumSparseOutput = _collections.namedtuple("SparseReduceSumSparse",
                                                       _sparse_reduce_sum_sparse_outputs)


def sparse_reduce_sum_sparse(input_indices, input_values, input_shape,
                             reduction_axes, keep_dims=None, name=None):
  r"""Computes the sum of elements across dimensions of a SparseTensor.

  This Op takes a SparseTensor and is the sparse counterpart to
  `tf.reduce_sum()`.  In contrast to SparseReduceSum, this Op returns a
  SparseTensor.

  Reduces `sp_input` along the dimensions given in `reduction_axes`.  Unless
  `keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
  `reduction_axes`. If `keep_dims` is true, the reduced dimensions are retained
  with length 1.

  If `reduction_axes` has no entries, all dimensions are reduced, and a tensor
  with a single element is returned.  Additionally, the axes can be negative,
  which are interpreted according to the indexing rules in Python.

  Args:
    input_indices: A `Tensor` of type `int64`.
      2-D.  `N x R` matrix with the indices of non-empty values in a
      SparseTensor, possibly not in canonical ordering.
    input_values: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`, `complex128`, `qint8`, `quint8`, `qint32`, `half`.
      1-D.  `N` non-empty values corresponding to `input_indices`.
    input_shape: A `Tensor` of type `int64`.
      1-D.  Shape of the input SparseTensor.
    reduction_axes: A `Tensor` of type `int32`.
      1-D.  Length-`K` vector containing the reduction axes.
    keep_dims: An optional `bool`. Defaults to `False`.
      If true, retain reduced dimensions with length 1.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output_indices, output_values, output_shape).
    output_indices: A `Tensor` of type `int64`.
    output_values: A `Tensor`. Has the same type as `input_values`.
    output_shape: A `Tensor` of type `int64`.
  """
  result = _op_def_lib.apply_op("SparseReduceSumSparse",
                                input_indices=input_indices,
                                input_values=input_values,
                                input_shape=input_shape,
                                reduction_axes=reduction_axes,
                                keep_dims=keep_dims, name=name)
  return _SparseReduceSumSparseOutput._make(result)


__sparse_reorder_outputs = ["output_indices", "output_values"]


_SparseReorderOutput = _collections.namedtuple("SparseReorder",
                                               __sparse_reorder_outputs)


def _sparse_reorder(input_indices, input_values, input_shape, name=None):
  r"""Reorders a SparseTensor into the canonical, row-major ordering.

  Note that by convention, all sparse ops preserve the canonical ordering along
  increasing dimension number. The only time ordering can be violated is during
  manual manipulation of the indices and values vectors to add entries.

  Reordering does not affect the shape of the SparseTensor.

  If the tensor has rank `R` and `N` non-empty values, `input_indices` has
  shape `[N, R]`, input_values has length `N`, and input_shape has length `R`.

  Args:
    input_indices: A `Tensor` of type `int64`.
      2-D.  `N x R` matrix with the indices of non-empty values in a
      SparseTensor, possibly not in canonical ordering.
    input_values: A `Tensor`.
      1-D.  `N` non-empty values corresponding to `input_indices`.
    input_shape: A `Tensor` of type `int64`.
      1-D.  Shape of the input SparseTensor.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output_indices, output_values).
    output_indices: A `Tensor` of type `int64`. 2-D.  `N x R` matrix with the same indices as input_indices, but
      in canonical row-major ordering.
    output_values: A `Tensor`. Has the same type as `input_values`. 1-D.  `N` non-empty values corresponding to `output_indices`.
  """
  result = _op_def_lib.apply_op("SparseReorder", input_indices=input_indices,
                                input_values=input_values,
                                input_shape=input_shape, name=name)
  return _SparseReorderOutput._make(result)


__sparse_reshape_outputs = ["output_indices", "output_shape"]


_SparseReshapeOutput = _collections.namedtuple("SparseReshape",
                                               __sparse_reshape_outputs)


def _sparse_reshape(input_indices, input_shape, new_shape, name=None):
  r"""Reshapes a SparseTensor to represent values in a new dense shape.

  This operation has the same semantics as reshape on the represented dense
  tensor.  The `input_indices` are recomputed based on the requested `new_shape`.

  If one component of `new_shape` is the special value -1, the size of that
  dimension is computed so that the total dense size remains constant.  At
  most one component of `new_shape` can be -1.  The number of dense elements
  implied by `new_shape` must be the same as the number of dense elements
  originally implied by `input_shape`.

  Reshaping does not affect the order of values in the SparseTensor.

  If the input tensor has rank `R_in` and `N` non-empty values, and `new_shape`
  has length `R_out`, then `input_indices` has shape `[N, R_in]`,
  `input_shape` has length `R_in`, `output_indices` has shape `[N, R_out]`, and
  `output_shape` has length `R_out`.

  Args:
    input_indices: A `Tensor` of type `int64`.
      2-D.  `N x R_in` matrix with the indices of non-empty values in a
      SparseTensor.
    input_shape: A `Tensor` of type `int64`.
      1-D.  `R_in` vector with the input SparseTensor's dense shape.
    new_shape: A `Tensor` of type `int64`.
      1-D.  `R_out` vector with the requested new dense shape.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output_indices, output_shape).
    output_indices: A `Tensor` of type `int64`. 2-D.  `N x R_out` matrix with the updated indices of non-empty
      values in the output SparseTensor.
    output_shape: A `Tensor` of type `int64`. 1-D.  `R_out` vector with the full dense shape of the output
      SparseTensor.  This is the same as `new_shape` but with any -1 dimensions
      filled in.
  """
  result = _op_def_lib.apply_op("SparseReshape", input_indices=input_indices,
                                input_shape=input_shape, new_shape=new_shape,
                                name=name)
  return _SparseReshapeOutput._make(result)


_sparse_softmax_outputs = ["output"]


def sparse_softmax(sp_indices, sp_values, sp_shape, name=None):
  r"""Applies softmax to a batched N-D `SparseTensor`.

  The inputs represent an N-D SparseTensor  with logical shape `[..., B, C]`
  (where `N >= 2`), and with indices sorted in the canonical lexicographic order.

  This op is equivalent to applying the normal `tf.nn.softmax()` to each innermost
  logical submatrix with shape `[B, C]`, but with the catch that *the implicitly
  zero elements do not participate*.  Specifically, the algorithm is equivalent
  to the following:

    (1) Applies `tf.nn.softmax()` to a densified view of each innermost submatrix
        with shape `[B, C]`, along the size-C dimension;
    (2) Masks out the original implicitly-zero locations;
    (3) Renormalizes the remaining elements.

  Hence, the `SparseTensor` result has exactly the same non-zero indices and
  shape.

  Args:
    sp_indices: A `Tensor` of type `int64`.
      2-D.  `NNZ x R` matrix with the indices of non-empty values in a
      SparseTensor, in canonical ordering.
    sp_values: A `Tensor`. Must be one of the following types: `float32`, `float64`.
      1-D.  `NNZ` non-empty values corresponding to `sp_indices`.
    sp_shape: A `Tensor` of type `int64`.
      1-D.  Shape of the input SparseTensor.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `sp_values`.
    1-D.  The `NNZ` values for the result `SparseTensor`.
  """
  result = _op_def_lib.apply_op("SparseSoftmax", sp_indices=sp_indices,
                                sp_values=sp_values, sp_shape=sp_shape,
                                name=name)
  return result


_sparse_sparse_maximum_outputs = ["output_indices", "output_values"]


_SparseSparseMaximumOutput = _collections.namedtuple("SparseSparseMaximum",
                                                     _sparse_sparse_maximum_outputs)


def sparse_sparse_maximum(a_indices, a_values, a_shape, b_indices, b_values,
                          b_shape, name=None):
  r"""Returns the element-wise max of two SparseTensors.

  Assumes the two SparseTensors have the same shape, i.e., no broadcasting.

  Args:
    a_indices: A `Tensor` of type `int64`.
      2-D.  `N x R` matrix with the indices of non-empty values in a
      SparseTensor, in the canonical lexicographic ordering.
    a_values: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`, `uint8`, `int16`, `int8`, `uint16`, `half`.
      1-D.  `N` non-empty values corresponding to `a_indices`.
    a_shape: A `Tensor` of type `int64`.
      1-D.  Shape of the input SparseTensor.
    b_indices: A `Tensor` of type `int64`.
      counterpart to `a_indices` for the other operand.
    b_values: A `Tensor`. Must have the same type as `a_values`.
      counterpart to `a_values` for the other operand; must be of the same dtype.
    b_shape: A `Tensor` of type `int64`.
      counterpart to `a_shape` for the other operand; the two shapes must be equal.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output_indices, output_values).
    output_indices: A `Tensor` of type `int64`. 2-D.  The indices of the output SparseTensor.
    output_values: A `Tensor`. Has the same type as `a_values`. 1-D.  The values of the output SparseTensor.
  """
  result = _op_def_lib.apply_op("SparseSparseMaximum", a_indices=a_indices,
                                a_values=a_values, a_shape=a_shape,
                                b_indices=b_indices, b_values=b_values,
                                b_shape=b_shape, name=name)
  return _SparseSparseMaximumOutput._make(result)


_sparse_sparse_minimum_outputs = ["output_indices", "output_values"]


_SparseSparseMinimumOutput = _collections.namedtuple("SparseSparseMinimum",
                                                     _sparse_sparse_minimum_outputs)


def sparse_sparse_minimum(a_indices, a_values, a_shape, b_indices, b_values,
                          b_shape, name=None):
  r"""Returns the element-wise min of two SparseTensors.

  Assumes the two SparseTensors have the same shape, i.e., no broadcasting.

  Args:
    a_indices: A `Tensor` of type `int64`.
      2-D.  `N x R` matrix with the indices of non-empty values in a
      SparseTensor, in the canonical lexicographic ordering.
    a_values: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`, `complex128`, `qint8`, `quint8`, `qint32`, `half`.
      1-D.  `N` non-empty values corresponding to `a_indices`.
    a_shape: A `Tensor` of type `int64`.
      1-D.  Shape of the input SparseTensor.
    b_indices: A `Tensor` of type `int64`.
      counterpart to `a_indices` for the other operand.
    b_values: A `Tensor`. Must have the same type as `a_values`.
      counterpart to `a_values` for the other operand; must be of the same dtype.
    b_shape: A `Tensor` of type `int64`.
      counterpart to `a_shape` for the other operand; the two shapes must be equal.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output_indices, output_values).
    output_indices: A `Tensor` of type `int64`. 2-D.  The indices of the output SparseTensor.
    output_values: A `Tensor`. Has the same type as `a_values`. 1-D.  The values of the output SparseTensor.
  """
  result = _op_def_lib.apply_op("SparseSparseMinimum", a_indices=a_indices,
                                a_values=a_values, a_shape=a_shape,
                                b_indices=b_indices, b_values=b_values,
                                b_shape=b_shape, name=name)
  return _SparseSparseMinimumOutput._make(result)


__sparse_split_outputs = ["output_indices", "output_values", "output_shape"]


_SparseSplitOutput = _collections.namedtuple("SparseSplit",
                                             __sparse_split_outputs)


def _sparse_split(split_dim, indices, values, shape, num_split, name=None):
  r"""Split a `SparseTensor` into `num_split` tensors along one dimension.

  If the `shape[split_dim]` is not an integer multiple of `num_split`. Slices
  `[0 : shape[split_dim] % num_split]` gets one extra dimension.
  For example, if `split_dim = 1` and `num_split = 2` and the input is

      input_tensor = shape = [2, 7]
      [    a   d e  ]
      [b c          ]

  Graphically the output tensors are:

      output_tensor[0] = shape = [2, 4]
      [    a  ]
      [b c    ]

      output_tensor[1] = shape = [2, 3]
      [ d e  ]
      [      ]

  Args:
    split_dim: A `Tensor` of type `int64`.
      0-D.  The dimension along which to split.  Must be in the range
      `[0, rank(shape))`.
    indices: A `Tensor` of type `int64`.
      2-D tensor represents the indices of the sparse tensor.
    values: A `Tensor`. 1-D tensor represents the values of the sparse tensor.
    shape: A `Tensor` of type `int64`.
      1-D. tensor represents the shape of the sparse tensor.
      output indices: A list of 1-D tensors represents the indices of the output
      sparse tensors.
    num_split: An `int` that is `>= 1`. The number of ways to split.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output_indices, output_values, output_shape).
    output_indices: A list of `num_split` `Tensor` objects of type `int64`.
    output_values: A list of `num_split` `Tensor` objects of the same type as values. A list of 1-D tensors represents the values of the output sparse
      tensors.
    output_shape: A list of `num_split` `Tensor` objects of type `int64`. A list of 1-D tensors represents the shape of the output sparse
      tensors.
  """
  result = _op_def_lib.apply_op("SparseSplit", split_dim=split_dim,
                                indices=indices, values=values, shape=shape,
                                num_split=num_split, name=name)
  return _SparseSplitOutput._make(result)


__sparse_tensor_dense_add_outputs = ["output"]


def _sparse_tensor_dense_add(a_indices, a_values, a_shape, b, name=None):
  r"""Adds up a `SparseTensor` and a dense `Tensor`, producing a dense `Tensor`.

  This Op does not require `a_indices` be sorted in standard lexicographic order.

  Args:
    a_indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      2-D.  The `indices` of the `SparseTensor`, with shape `[nnz, ndims]`.
    a_values: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`, `complex128`, `qint8`, `quint8`, `qint32`, `half`.
      1-D.  The `values` of the `SparseTensor`, with shape `[nnz]`.
    a_shape: A `Tensor`. Must have the same type as `a_indices`.
      1-D.  The `shape` of the `SparseTensor`, with shape `[ndims]`.
    b: A `Tensor`. Must have the same type as `a_values`.
      `ndims`-D Tensor.  With shape `a_shape`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `a_values`.
  """
  result = _op_def_lib.apply_op("SparseTensorDenseAdd", a_indices=a_indices,
                                a_values=a_values, a_shape=a_shape, b=b,
                                name=name)
  return result


__sparse_tensor_dense_mat_mul_outputs = ["product"]


def _sparse_tensor_dense_mat_mul(a_indices, a_values, a_shape, b,
                                 adjoint_a=None, adjoint_b=None, name=None):
  r"""Multiply SparseTensor (of rank 2) "A" by dense matrix "B".

  No validity checking is performed on the indices of A.  However, the following
  input format is recommended for optimal behavior:

  if adjoint_a == false:
    A should be sorted in lexicographically increasing order.  Use SparseReorder
    if you're not sure.
  if adjoint_a == true:
    A should be sorted in order of increasing dimension 1 (i.e., "column major"
    order instead of "row major" order).

  Args:
    a_indices: A `Tensor` of type `int64`.
      2-D.  The `indices` of the `SparseTensor`, size `[nnz, 2]` Matrix.
    a_values: A `Tensor`.
      1-D.  The `values` of the `SparseTensor`, size `[nnz]` Vector.
    a_shape: A `Tensor` of type `int64`.
      1-D.  The `shape` of the `SparseTensor`, size `[2]` Vector.
    b: A `Tensor`. Must have the same type as `a_values`.
      2-D.  A dense Matrix.
    adjoint_a: An optional `bool`. Defaults to `False`.
      Use the adjoint of A in the matrix multiply.  If A is complex, this
      is transpose(conj(A)).  Otherwise it's transpose(A).
    adjoint_b: An optional `bool`. Defaults to `False`.
      Use the adjoint of B in the matrix multiply.  If B is complex, this
      is transpose(conj(B)).  Otherwise it's transpose(B).
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `a_values`.
  """
  result = _op_def_lib.apply_op("SparseTensorDenseMatMul",
                                a_indices=a_indices, a_values=a_values,
                                a_shape=a_shape, b=b, adjoint_a=adjoint_a,
                                adjoint_b=adjoint_b, name=name)
  return result


__sparse_to_dense_outputs = ["dense"]


def _sparse_to_dense(sparse_indices, output_shape, sparse_values,
                     default_value, validate_indices=None, name=None):
  r"""Converts a sparse representation into a dense tensor.

  Builds an array `dense` with shape `output_shape` such that

  ```prettyprint
  # If sparse_indices is scalar
  dense[i] = (i == sparse_indices ? sparse_values : default_value)

  # If sparse_indices is a vector, then for each i
  dense[sparse_indices[i]] = sparse_values[i]

  # If sparse_indices is an n by d matrix, then for each i in [0, n)
  dense[sparse_indices[i][0], ..., sparse_indices[i][d-1]] = sparse_values[i]
  ```

  All other values in `dense` are set to `default_value`.  If `sparse_values` is a
  scalar, all sparse indices are set to this single value.

  Indices should be sorted in lexicographic order, and indices must not
  contain any repeats. If `validate_indices` is true, these properties
  are checked during execution.

  Args:
    sparse_indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      0-D, 1-D, or 2-D.  `sparse_indices[i]` contains the complete
      index where `sparse_values[i]` will be placed.
    output_shape: A `Tensor`. Must have the same type as `sparse_indices`.
      1-D.  Shape of the dense output tensor.
    sparse_values: A `Tensor`.
      1-D.  Values corresponding to each row of `sparse_indices`,
      or a scalar value to be used for all sparse indices.
    default_value: A `Tensor`. Must have the same type as `sparse_values`.
      Scalar value to set for indices not specified in
      `sparse_indices`.
    validate_indices: An optional `bool`. Defaults to `True`.
      If true, indices are checked to make sure they are sorted in
      lexicographic order and that there are no repeats.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `sparse_values`.
    Dense output tensor of shape `output_shape`.
  """
  result = _op_def_lib.apply_op("SparseToDense",
                                sparse_indices=sparse_indices,
                                output_shape=output_shape,
                                sparse_values=sparse_values,
                                default_value=default_value,
                                validate_indices=validate_indices, name=name)
  return result


__take_many_sparse_from_tensors_map_outputs = ["sparse_indices",
                                              "sparse_values", "sparse_shape"]


_TakeManySparseFromTensorsMapOutput = _collections.namedtuple("TakeManySparseFromTensorsMap",
                                                              __take_many_sparse_from_tensors_map_outputs)


def _take_many_sparse_from_tensors_map(sparse_handles, dtype, container=None,
                                       shared_name=None, name=None):
  r"""Read `SparseTensors` from a `SparseTensorsMap` and concatenate them.

  The input `sparse_handles` must be an `int64` matrix of shape `[N, 1]` where
  `N` is the minibatch size and the rows correspond to the output handles of
  `AddSparseToTensorsMap` or `AddManySparseToTensorsMap`.  The ranks of the
  original `SparseTensor` objects that went into the given input ops must all
  match.  When the final `SparseTensor` is created, it has rank one
  higher than the ranks of the incoming `SparseTensor` objects
  (they have been concatenated along a new row dimension on the left).

  The output `SparseTensor` object's shape values for all dimensions but the
  first are the max across the input `SparseTensor` objects' shape values
  for the corresponding dimensions.  Its first shape value is `N`, the minibatch
  size.

  The input `SparseTensor` objects' indices are assumed ordered in
  standard lexicographic order.  If this is not the case, after this
  step run `SparseReorder` to restore index ordering.

  For example, if the handles represent an input, which is a `[2, 3]` matrix
  representing two original `SparseTensor` objects:

  ```
      index = [ 0]
              [10]
              [20]
      values = [1, 2, 3]
      shape = [50]
  ```

  and

  ```
      index = [ 2]
              [10]
      values = [4, 5]
      shape = [30]
  ```

  then the final `SparseTensor` will be:

  ```
      index = [0  0]
              [0 10]
              [0 20]
              [1  2]
              [1 10]
      values = [1, 2, 3, 4, 5]
      shape = [2 50]
  ```

  Args:
    sparse_handles: A `Tensor` of type `int64`.
      1-D, The `N` serialized `SparseTensor` objects.
      Shape: `[N]`.
    dtype: A `tf.DType`.
      The `dtype` of the `SparseTensor` objects stored in the
      `SparseTensorsMap`.
    container: An optional `string`. Defaults to `""`.
      The container name for the `SparseTensorsMap` read by this op.
    shared_name: An optional `string`. Defaults to `""`.
      The shared name for the `SparseTensorsMap` read by this op.
      It should not be blank; rather the `shared_name` or unique Operation name
      of the Op that created the original `SparseTensorsMap` should be used.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (sparse_indices, sparse_values, sparse_shape).
    sparse_indices: A `Tensor` of type `int64`. 2-D.  The `indices` of the minibatch `SparseTensor`.
    sparse_values: A `Tensor` of type `dtype`. 1-D.  The `values` of the minibatch `SparseTensor`.
    sparse_shape: A `Tensor` of type `int64`. 1-D.  The `shape` of the minibatch `SparseTensor`.
  """
  result = _op_def_lib.apply_op("TakeManySparseFromTensorsMap",
                                sparse_handles=sparse_handles, dtype=dtype,
                                container=container, shared_name=shared_name,
                                name=name)
  return _TakeManySparseFromTensorsMapOutput._make(result)


def _InitOpDefLibrary():
  op_list = _op_def_pb2.OpList()
  _text_format.Merge(_InitOpDefLibrary.op_list_ascii, op_list)
  _op_def_registry.register_op_list(op_list)
  op_def_lib = _op_def_library.OpDefLibrary()
  op_def_lib.add_op_list(op_list)
  return op_def_lib


_InitOpDefLibrary.op_list_ascii = """op {
  name: "AddManySparseToTensorsMap"
  input_arg {
    name: "sparse_indices"
    type: DT_INT64
  }
  input_arg {
    name: "sparse_values"
    type_attr: "T"
  }
  input_arg {
    name: "sparse_shape"
    type: DT_INT64
  }
  output_arg {
    name: "sparse_handles"
    type: DT_INT64
  }
  attr {
    name: "T"
    type: "type"
  }
  attr {
    name: "container"
    type: "string"
    default_value {
      s: ""
    }
  }
  attr {
    name: "shared_name"
    type: "string"
    default_value {
      s: ""
    }
  }
  is_stateful: true
}
op {
  name: "AddSparseToTensorsMap"
  input_arg {
    name: "sparse_indices"
    type: DT_INT64
  }
  input_arg {
    name: "sparse_values"
    type_attr: "T"
  }
  input_arg {
    name: "sparse_shape"
    type: DT_INT64
  }
  output_arg {
    name: "sparse_handle"
    type: DT_INT64
  }
  attr {
    name: "T"
    type: "type"
  }
  attr {
    name: "container"
    type: "string"
    default_value {
      s: ""
    }
  }
  attr {
    name: "shared_name"
    type: "string"
    default_value {
      s: ""
    }
  }
  is_stateful: true
}
op {
  name: "DeserializeManySparse"
  input_arg {
    name: "serialized_sparse"
    type: DT_STRING
  }
  output_arg {
    name: "sparse_indices"
    type: DT_INT64
  }
  output_arg {
    name: "sparse_values"
    type_attr: "dtype"
  }
  output_arg {
    name: "sparse_shape"
    type: DT_INT64
  }
  attr {
    name: "dtype"
    type: "type"
  }
}
op {
  name: "SerializeManySparse"
  input_arg {
    name: "sparse_indices"
    type: DT_INT64
  }
  input_arg {
    name: "sparse_values"
    type_attr: "T"
  }
  input_arg {
    name: "sparse_shape"
    type: DT_INT64
  }
  output_arg {
    name: "serialized_sparse"
    type: DT_STRING
  }
  attr {
    name: "T"
    type: "type"
  }
}
op {
  name: "SerializeSparse"
  input_arg {
    name: "sparse_indices"
    type: DT_INT64
  }
  input_arg {
    name: "sparse_values"
    type_attr: "T"
  }
  input_arg {
    name: "sparse_shape"
    type: DT_INT64
  }
  output_arg {
    name: "serialized_sparse"
    type: DT_STRING
  }
  attr {
    name: "T"
    type: "type"
  }
}
op {
  name: "SparseAdd"
  input_arg {
    name: "a_indices"
    type: DT_INT64
  }
  input_arg {
    name: "a_values"
    type_attr: "T"
  }
  input_arg {
    name: "a_shape"
    type: DT_INT64
  }
  input_arg {
    name: "b_indices"
    type: DT_INT64
  }
  input_arg {
    name: "b_values"
    type_attr: "T"
  }
  input_arg {
    name: "b_shape"
    type: DT_INT64
  }
  input_arg {
    name: "thresh"
    type_attr: "Treal"
  }
  output_arg {
    name: "sum_indices"
    type: DT_INT64
  }
  output_arg {
    name: "sum_values"
    type_attr: "T"
  }
  output_arg {
    name: "sum_shape"
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
    name: "Treal"
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
  name: "SparseAddGrad"
  input_arg {
    name: "backprop_val_grad"
    type_attr: "T"
  }
  input_arg {
    name: "a_indices"
    type: DT_INT64
  }
  input_arg {
    name: "b_indices"
    type: DT_INT64
  }
  input_arg {
    name: "sum_indices"
    type: DT_INT64
  }
  output_arg {
    name: "a_val_grad"
    type_attr: "T"
  }
  output_arg {
    name: "b_val_grad"
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
  name: "SparseConcat"
  input_arg {
    name: "indices"
    type: DT_INT64
    number_attr: "N"
  }
  input_arg {
    name: "values"
    type_attr: "T"
    number_attr: "N"
  }
  input_arg {
    name: "shapes"
    type: DT_INT64
    number_attr: "N"
  }
  output_arg {
    name: "output_indices"
    type: DT_INT64
  }
  output_arg {
    name: "output_values"
    type_attr: "T"
  }
  output_arg {
    name: "output_shape"
    type: DT_INT64
  }
  attr {
    name: "concat_dim"
    type: "int"
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
  name: "SparseDenseCwiseAdd"
  input_arg {
    name: "sp_indices"
    type: DT_INT64
  }
  input_arg {
    name: "sp_values"
    type_attr: "T"
  }
  input_arg {
    name: "sp_shape"
    type: DT_INT64
  }
  input_arg {
    name: "dense"
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
  name: "SparseDenseCwiseDiv"
  input_arg {
    name: "sp_indices"
    type: DT_INT64
  }
  input_arg {
    name: "sp_values"
    type_attr: "T"
  }
  input_arg {
    name: "sp_shape"
    type: DT_INT64
  }
  input_arg {
    name: "dense"
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
  name: "SparseDenseCwiseMul"
  input_arg {
    name: "sp_indices"
    type: DT_INT64
  }
  input_arg {
    name: "sp_values"
    type_attr: "T"
  }
  input_arg {
    name: "sp_shape"
    type: DT_INT64
  }
  input_arg {
    name: "dense"
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
  name: "SparseReduceSum"
  input_arg {
    name: "input_indices"
    type: DT_INT64
  }
  input_arg {
    name: "input_values"
    type_attr: "T"
  }
  input_arg {
    name: "input_shape"
    type: DT_INT64
  }
  input_arg {
    name: "reduction_axes"
    type: DT_INT32
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
}
op {
  name: "SparseReduceSumSparse"
  input_arg {
    name: "input_indices"
    type: DT_INT64
  }
  input_arg {
    name: "input_values"
    type_attr: "T"
  }
  input_arg {
    name: "input_shape"
    type: DT_INT64
  }
  input_arg {
    name: "reduction_axes"
    type: DT_INT32
  }
  output_arg {
    name: "output_indices"
    type: DT_INT64
  }
  output_arg {
    name: "output_values"
    type_attr: "T"
  }
  output_arg {
    name: "output_shape"
    type: DT_INT64
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
}
op {
  name: "SparseReorder"
  input_arg {
    name: "input_indices"
    type: DT_INT64
  }
  input_arg {
    name: "input_values"
    type_attr: "T"
  }
  input_arg {
    name: "input_shape"
    type: DT_INT64
  }
  output_arg {
    name: "output_indices"
    type: DT_INT64
  }
  output_arg {
    name: "output_values"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
  }
}
op {
  name: "SparseReshape"
  input_arg {
    name: "input_indices"
    type: DT_INT64
  }
  input_arg {
    name: "input_shape"
    type: DT_INT64
  }
  input_arg {
    name: "new_shape"
    type: DT_INT64
  }
  output_arg {
    name: "output_indices"
    type: DT_INT64
  }
  output_arg {
    name: "output_shape"
    type: DT_INT64
  }
}
op {
  name: "SparseSoftmax"
  input_arg {
    name: "sp_indices"
    type: DT_INT64
  }
  input_arg {
    name: "sp_values"
    type_attr: "T"
  }
  input_arg {
    name: "sp_shape"
    type: DT_INT64
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
}
op {
  name: "SparseSparseMaximum"
  input_arg {
    name: "a_indices"
    type: DT_INT64
  }
  input_arg {
    name: "a_values"
    type_attr: "T"
  }
  input_arg {
    name: "a_shape"
    type: DT_INT64
  }
  input_arg {
    name: "b_indices"
    type: DT_INT64
  }
  input_arg {
    name: "b_values"
    type_attr: "T"
  }
  input_arg {
    name: "b_shape"
    type: DT_INT64
  }
  output_arg {
    name: "output_indices"
    type: DT_INT64
  }
  output_arg {
    name: "output_values"
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
  name: "SparseSparseMinimum"
  input_arg {
    name: "a_indices"
    type: DT_INT64
  }
  input_arg {
    name: "a_values"
    type_attr: "T"
  }
  input_arg {
    name: "a_shape"
    type: DT_INT64
  }
  input_arg {
    name: "b_indices"
    type: DT_INT64
  }
  input_arg {
    name: "b_values"
    type_attr: "T"
  }
  input_arg {
    name: "b_shape"
    type: DT_INT64
  }
  output_arg {
    name: "output_indices"
    type: DT_INT64
  }
  output_arg {
    name: "output_values"
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
  name: "SparseSplit"
  input_arg {
    name: "split_dim"
    type: DT_INT64
  }
  input_arg {
    name: "indices"
    type: DT_INT64
  }
  input_arg {
    name: "values"
    type_attr: "T"
  }
  input_arg {
    name: "shape"
    type: DT_INT64
  }
  output_arg {
    name: "output_indices"
    type: DT_INT64
    number_attr: "num_split"
  }
  output_arg {
    name: "output_values"
    type_attr: "T"
    number_attr: "num_split"
  }
  output_arg {
    name: "output_shape"
    type: DT_INT64
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
  name: "SparseTensorDenseAdd"
  input_arg {
    name: "a_indices"
    type_attr: "Tindices"
  }
  input_arg {
    name: "a_values"
    type_attr: "T"
  }
  input_arg {
    name: "a_shape"
    type_attr: "Tindices"
  }
  input_arg {
    name: "b"
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
  name: "SparseTensorDenseMatMul"
  input_arg {
    name: "a_indices"
    type: DT_INT64
  }
  input_arg {
    name: "a_values"
    type_attr: "T"
  }
  input_arg {
    name: "a_shape"
    type: DT_INT64
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
  }
  attr {
    name: "adjoint_a"
    type: "bool"
    default_value {
      b: false
    }
  }
  attr {
    name: "adjoint_b"
    type: "bool"
    default_value {
      b: false
    }
  }
}
op {
  name: "SparseToDense"
  input_arg {
    name: "sparse_indices"
    type_attr: "Tindices"
  }
  input_arg {
    name: "output_shape"
    type_attr: "Tindices"
  }
  input_arg {
    name: "sparse_values"
    type_attr: "T"
  }
  input_arg {
    name: "default_value"
    type_attr: "T"
  }
  output_arg {
    name: "dense"
    type_attr: "T"
  }
  attr {
    name: "validate_indices"
    type: "bool"
    default_value {
      b: true
    }
  }
  attr {
    name: "T"
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
  name: "TakeManySparseFromTensorsMap"
  input_arg {
    name: "sparse_handles"
    type: DT_INT64
  }
  output_arg {
    name: "sparse_indices"
    type: DT_INT64
  }
  output_arg {
    name: "sparse_values"
    type_attr: "dtype"
  }
  output_arg {
    name: "sparse_shape"
    type: DT_INT64
  }
  attr {
    name: "dtype"
    type: "type"
  }
  attr {
    name: "container"
    type: "string"
    default_value {
      s: ""
    }
  }
  attr {
    name: "shared_name"
    type: "string"
    default_value {
      s: ""
    }
  }
  is_stateful: true
}
"""


_op_def_lib = _InitOpDefLibrary()
