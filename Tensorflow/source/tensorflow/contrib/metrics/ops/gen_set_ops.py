"""Python wrappers around Brain.

This file is MACHINE GENERATED! Do not edit.
"""

import collections

from google.protobuf import text_format

from tensorflow.core.framework import op_def_pb2
from tensorflow.python.framework import op_def_registry
from tensorflow.python.framework import ops
from tensorflow.python.ops import op_def_library


__dense_to_dense_set_operation_outputs = ["result_indices", "result_values",
                                         "result_shape"]


_DenseToDenseSetOperationOutput = collections.namedtuple("DenseToDenseSetOperation",
                                                         __dense_to_dense_set_operation_outputs)


def _dense_to_dense_set_operation(set1, set2, set_operation,
                                  validate_indices=None, name=None):
  r"""Applies set operation along last dimension of 2 `Tensor` inputs.

  See SetOperationOp::SetOperationFromContext for values of `set_operation`.

  Output `result` is a `SparseTensor` represented by `result_indices`,
  `result_values`, and `result_shape`. For `set1` and `set2` ranked `n`, this
  has rank `n` and the same 1st `n-1` dimensions as `set1` and `set2`. The `nth`
  dimension contains the result of `set_operation` applied to the corresponding
  `[0...n-1]` dimension of `set`.

  Args:
    set1: A `Tensor`. Must be one of the following types: `int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`, `string`.
      `Tensor` with rank `n`. 1st `n-1` dimensions must be the same as `set2`.
      Dimension `n` contains values in a set, duplicates are allowed but ignored.
    set2: A `Tensor`. Must have the same type as `set1`.
      `Tensor` with rank `n`. 1st `n-1` dimensions must be the same as `set1`.
      Dimension `n` contains values in a set, duplicates are allowed but ignored.
    set_operation: A `string`.
    validate_indices: An optional `bool`. Defaults to `True`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (result_indices, result_values, result_shape).
    result_indices: A `Tensor` of type `int64`. 2D indices of a `SparseTensor`.
    result_values: A `Tensor`. Has the same type as `set1`. 1D values of a `SparseTensor`.
    result_shape: A `Tensor` of type `int64`. 1D `Tensor` shape of a `SparseTensor`. `result_shape[0...n-1]` is
      the same as the 1st `n-1` dimensions of `set1` and `set2`, `result_shape[n]`
      is the max result set size across all `0...n-1` dimensions.
  """
  result = _op_def_lib.apply_op("DenseToDenseSetOperation", set1=set1,
                                set2=set2, set_operation=set_operation,
                                validate_indices=validate_indices, name=name)
  return _DenseToDenseSetOperationOutput._make(result)


ops.RegisterShape("DenseToDenseSetOperation")(None)
__dense_to_sparse_set_operation_outputs = ["result_indices", "result_values",
                                          "result_shape"]


_DenseToSparseSetOperationOutput = collections.namedtuple("DenseToSparseSetOperation",
                                                          __dense_to_sparse_set_operation_outputs)


def _dense_to_sparse_set_operation(set1, set2_indices, set2_values,
                                   set2_shape, set_operation,
                                   validate_indices=None, name=None):
  r"""Applies set operation along last dimension of `Tensor` and `SparseTensor`.

  See SetOperationOp::SetOperationFromContext for values of `set_operation`.

  Input `set2` is a `SparseTensor` represented by `set2_indices`, `set2_values`,
  and `set2_shape`. For `set2` ranked `n`, 1st `n-1` dimensions must be the same
  as `set1`. Dimension `n` contains values in a set, duplicates are allowed but
  ignored.

  If `validate_indices` is `True`, this op validates the order and range of `set2`
  indices.

  Output `result` is a `SparseTensor` represented by `result_indices`,
  `result_values`, and `result_shape`. For `set1` and `set2` ranked `n`, this
  has rank `n` and the same 1st `n-1` dimensions as `set1` and `set2`. The `nth`
  dimension contains the result of `set_operation` applied to the corresponding
  `[0...n-1]` dimension of `set`.

  Args:
    set1: A `Tensor`. Must be one of the following types: `int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`, `string`.
      `Tensor` with rank `n`. 1st `n-1` dimensions must be the same as `set2`.
      Dimension `n` contains values in a set, duplicates are allowed but ignored.
    set2_indices: A `Tensor` of type `int64`.
      2D `Tensor`, indices of a `SparseTensor`. Must be in row-major
      order.
    set2_values: A `Tensor`. Must have the same type as `set1`.
      1D `Tensor`, values of a `SparseTensor`. Must be in row-major
      order.
    set2_shape: A `Tensor` of type `int64`.
      1D `Tensor`, shape of a `SparseTensor`. `set2_shape[0...n-1]` must
      be the same as the 1st `n-1` dimensions of `set1`, `result_shape[n]` is the
      max set size across `n-1` dimensions.
    set_operation: A `string`.
    validate_indices: An optional `bool`. Defaults to `True`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (result_indices, result_values, result_shape).
    result_indices: A `Tensor` of type `int64`. 2D indices of a `SparseTensor`.
    result_values: A `Tensor`. Has the same type as `set1`. 1D values of a `SparseTensor`.
    result_shape: A `Tensor` of type `int64`. 1D `Tensor` shape of a `SparseTensor`. `result_shape[0...n-1]` is
      the same as the 1st `n-1` dimensions of `set1` and `set2`, `result_shape[n]`
      is the max result set size across all `0...n-1` dimensions.
  """
  result = _op_def_lib.apply_op("DenseToSparseSetOperation", set1=set1,
                                set2_indices=set2_indices,
                                set2_values=set2_values,
                                set2_shape=set2_shape,
                                set_operation=set_operation,
                                validate_indices=validate_indices, name=name)
  return _DenseToSparseSetOperationOutput._make(result)


ops.RegisterShape("DenseToSparseSetOperation")(None)
__set_size_outputs = ["size"]


def _set_size(set_indices, set_values, set_shape, validate_indices=None,
              name=None):
  r"""Number of unique elements along last dimension of input `set`.

  Input `set` is a `SparseTensor` represented by `set_indices`, `set_values`,
  and `set_shape`. The last dimension contains values in a set, duplicates are
  allowed but ignored.

  If `validate_indices` is `True`, this op validates the order and range of `set`
  indices.

  Args:
    set_indices: A `Tensor` of type `int64`.
      2D `Tensor`, indices of a `SparseTensor`.
    set_values: A `Tensor`. Must be one of the following types: `int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`, `string`.
      1D `Tensor`, values of a `SparseTensor`.
    set_shape: A `Tensor` of type `int64`.
      1D `Tensor`, shape of a `SparseTensor`.
    validate_indices: An optional `bool`. Defaults to `True`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int32`.
    For `set` ranked `n`, this is a `Tensor` with rank `n-1`, and the same 1st
    `n-1` dimensions as `set`. Each value is the number of unique elements in
    the corresponding `[0...n-1]` dimension of `set`.
  """
  result = _op_def_lib.apply_op("SetSize", set_indices=set_indices,
                                set_values=set_values, set_shape=set_shape,
                                validate_indices=validate_indices, name=name)
  return result


ops.RegisterShape("SetSize")(None)
__sparse_to_sparse_set_operation_outputs = ["result_indices", "result_values",
                                           "result_shape"]


_SparseToSparseSetOperationOutput = collections.namedtuple("SparseToSparseSetOperation",
                                                           __sparse_to_sparse_set_operation_outputs)


def _sparse_to_sparse_set_operation(set1_indices, set1_values, set1_shape,
                                    set2_indices, set2_values, set2_shape,
                                    set_operation, validate_indices=None,
                                    name=None):
  r"""Applies set operation along last dimension of 2 `SparseTensor` inputs.

  See SetOperationOp::SetOperationFromContext for values of `set_operation`.

  If `validate_indices` is `True`, `SparseToSparseSetOperation` validates the
  order and range of `set1` and `set2` indices.

  Input `set1` is a `SparseTensor` represented by `set1_indices`, `set1_values`,
  and `set1_shape`. For `set1` ranked `n`, 1st `n-1` dimensions must be the same
  as `set2`. Dimension `n` contains values in a set, duplicates are allowed but
  ignored.

  Input `set2` is a `SparseTensor` represented by `set2_indices`, `set2_values`,
  and `set2_shape`. For `set2` ranked `n`, 1st `n-1` dimensions must be the same
  as `set1`. Dimension `n` contains values in a set, duplicates are allowed but
  ignored.

  If `validate_indices` is `True`, this op validates the order and range of `set1`
  and `set2` indices.

  Output `result` is a `SparseTensor` represented by `result_indices`,
  `result_values`, and `result_shape`. For `set1` and `set2` ranked `n`, this
  has rank `n` and the same 1st `n-1` dimensions as `set1` and `set2`. The `nth`
  dimension contains the result of `set_operation` applied to the corresponding
  `[0...n-1]` dimension of `set`.

  Args:
    set1_indices: A `Tensor` of type `int64`.
      2D `Tensor`, indices of a `SparseTensor`. Must be in row-major
      order.
    set1_values: A `Tensor`. Must be one of the following types: `int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`, `string`.
      1D `Tensor`, values of a `SparseTensor`. Must be in row-major
      order.
    set1_shape: A `Tensor` of type `int64`.
      1D `Tensor`, shape of a `SparseTensor`. `set1_shape[0...n-1]` must
      be the same as `set2_shape[0...n-1]`, `set1_shape[n]` is the
      max set size across `0...n-1` dimensions.
    set2_indices: A `Tensor` of type `int64`.
      2D `Tensor`, indices of a `SparseTensor`. Must be in row-major
      order.
    set2_values: A `Tensor`. Must have the same type as `set1_values`.
      1D `Tensor`, values of a `SparseTensor`. Must be in row-major
      order.
    set2_shape: A `Tensor` of type `int64`.
      1D `Tensor`, shape of a `SparseTensor`. `set2_shape[0...n-1]` must
      be the same as `set1_shape[0...n-1]`, `set2_shape[n]` is the
      max set size across `0...n-1` dimensions.
    set_operation: A `string`.
    validate_indices: An optional `bool`. Defaults to `True`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (result_indices, result_values, result_shape).
    result_indices: A `Tensor` of type `int64`. 2D indices of a `SparseTensor`.
    result_values: A `Tensor`. Has the same type as `set1_values`. 1D values of a `SparseTensor`.
    result_shape: A `Tensor` of type `int64`. 1D `Tensor` shape of a `SparseTensor`. `result_shape[0...n-1]` is
      the same as the 1st `n-1` dimensions of `set1` and `set2`, `result_shape[n]`
      is the max result set size across all `0...n-1` dimensions.
  """
  result = _op_def_lib.apply_op("SparseToSparseSetOperation",
                                set1_indices=set1_indices,
                                set1_values=set1_values,
                                set1_shape=set1_shape,
                                set2_indices=set2_indices,
                                set2_values=set2_values,
                                set2_shape=set2_shape,
                                set_operation=set_operation,
                                validate_indices=validate_indices, name=name)
  return _SparseToSparseSetOperationOutput._make(result)


ops.RegisterShape("SparseToSparseSetOperation")(None)
def _InitOpDefLibrary():
  op_list = op_def_pb2.OpList()
  text_format.Merge(_InitOpDefLibrary.op_list_ascii, op_list)
  op_def_registry.register_op_list(op_list)
  op_def_lib = op_def_library.OpDefLibrary()
  op_def_lib.add_op_list(op_list)
  return op_def_lib


_InitOpDefLibrary.op_list_ascii = """op {
  name: "DenseToDenseSetOperation"
  input_arg {
    name: "set1"
    type_attr: "T"
  }
  input_arg {
    name: "set2"
    type_attr: "T"
  }
  output_arg {
    name: "result_indices"
    type: DT_INT64
  }
  output_arg {
    name: "result_values"
    type_attr: "T"
  }
  output_arg {
    name: "result_shape"
    type: DT_INT64
  }
  attr {
    name: "set_operation"
    type: "string"
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
    allowed_values {
      list {
        type: DT_INT8
        type: DT_INT16
        type: DT_INT32
        type: DT_INT64
        type: DT_UINT8
        type: DT_UINT16
        type: DT_STRING
      }
    }
  }
}
op {
  name: "DenseToSparseSetOperation"
  input_arg {
    name: "set1"
    type_attr: "T"
  }
  input_arg {
    name: "set2_indices"
    type: DT_INT64
  }
  input_arg {
    name: "set2_values"
    type_attr: "T"
  }
  input_arg {
    name: "set2_shape"
    type: DT_INT64
  }
  output_arg {
    name: "result_indices"
    type: DT_INT64
  }
  output_arg {
    name: "result_values"
    type_attr: "T"
  }
  output_arg {
    name: "result_shape"
    type: DT_INT64
  }
  attr {
    name: "set_operation"
    type: "string"
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
    allowed_values {
      list {
        type: DT_INT8
        type: DT_INT16
        type: DT_INT32
        type: DT_INT64
        type: DT_UINT8
        type: DT_UINT16
        type: DT_STRING
      }
    }
  }
}
op {
  name: "SetSize"
  input_arg {
    name: "set_indices"
    type: DT_INT64
  }
  input_arg {
    name: "set_values"
    type_attr: "T"
  }
  input_arg {
    name: "set_shape"
    type: DT_INT64
  }
  output_arg {
    name: "size"
    type: DT_INT32
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
    allowed_values {
      list {
        type: DT_INT8
        type: DT_INT16
        type: DT_INT32
        type: DT_INT64
        type: DT_UINT8
        type: DT_UINT16
        type: DT_STRING
      }
    }
  }
}
op {
  name: "SparseToSparseSetOperation"
  input_arg {
    name: "set1_indices"
    type: DT_INT64
  }
  input_arg {
    name: "set1_values"
    type_attr: "T"
  }
  input_arg {
    name: "set1_shape"
    type: DT_INT64
  }
  input_arg {
    name: "set2_indices"
    type: DT_INT64
  }
  input_arg {
    name: "set2_values"
    type_attr: "T"
  }
  input_arg {
    name: "set2_shape"
    type: DT_INT64
  }
  output_arg {
    name: "result_indices"
    type: DT_INT64
  }
  output_arg {
    name: "result_values"
    type_attr: "T"
  }
  output_arg {
    name: "result_shape"
    type: DT_INT64
  }
  attr {
    name: "set_operation"
    type: "string"
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
    allowed_values {
      list {
        type: DT_INT8
        type: DT_INT16
        type: DT_INT32
        type: DT_INT64
        type: DT_UINT8
        type: DT_UINT16
        type: DT_STRING
      }
    }
  }
}
"""


_op_def_lib = _InitOpDefLibrary()
