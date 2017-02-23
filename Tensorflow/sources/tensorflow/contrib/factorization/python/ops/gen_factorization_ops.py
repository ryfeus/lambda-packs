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
_wals_compute_partial_lhs_and_rhs_outputs = ["partial_lhs", "partial_rhs"]


_WALSComputePartialLhsAndRhsOutput = _collections.namedtuple("WALSComputePartialLhsAndRhs",
                                                             _wals_compute_partial_lhs_and_rhs_outputs)


def wals_compute_partial_lhs_and_rhs(factors, factor_weights,
                                     unobserved_weights, input_weights,
                                     input_indices, input_values,
                                     input_block_size, input_is_transpose,
                                     name=None):
  r"""Computes the partial left-hand side and right-hand side of WALS update.

  Args:
    factors: A `Tensor` of type `float32`. Matrix of size m * k.
    factor_weights: A `Tensor` of type `float32`.
      Vector of size m. Corresponds to column weights
    unobserved_weights: A `Tensor` of type `float32`.
      Scalar. Weight for unobserved input entries.
    input_weights: A `Tensor` of type `float32`.
      Vector of size n. Corresponds to row weights.
    input_indices: A `Tensor` of type `int64`.
      Indices for the input SparseTensor.
    input_values: A `Tensor` of type `float32`.
      Values for the input SparseTensor.
    input_block_size: A `Tensor` of type `int64`.
      Scalar. Number of rows spanned by input.
    input_is_transpose: A `Tensor` of type `bool`.
      If true, logically transposes the input for processing.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (partial_lhs, partial_rhs).
    partial_lhs: A `Tensor` of type `float32`. 3-D tensor with size input_block_size x k x k.
    partial_rhs: A `Tensor` of type `float32`. Matrix with size input_block_size x k.
  """
  result = _op_def_lib.apply_op("WALSComputePartialLhsAndRhs",
                                factors=factors,
                                factor_weights=factor_weights,
                                unobserved_weights=unobserved_weights,
                                input_weights=input_weights,
                                input_indices=input_indices,
                                input_values=input_values,
                                input_block_size=input_block_size,
                                input_is_transpose=input_is_transpose,
                                name=name)
  return _WALSComputePartialLhsAndRhsOutput._make(result)


_ops.RegisterShape("WALSComputePartialLhsAndRhs")(None)
def _InitOpDefLibrary():
  op_list = _op_def_pb2.OpList()
  _text_format.Merge(_InitOpDefLibrary.op_list_ascii, op_list)
  _op_def_registry.register_op_list(op_list)
  op_def_lib = _op_def_library.OpDefLibrary()
  op_def_lib.add_op_list(op_list)
  return op_def_lib


_InitOpDefLibrary.op_list_ascii = """op {
  name: "WALSComputePartialLhsAndRhs"
  input_arg {
    name: "factors"
    type: DT_FLOAT
  }
  input_arg {
    name: "factor_weights"
    type: DT_FLOAT
  }
  input_arg {
    name: "unobserved_weights"
    type: DT_FLOAT
  }
  input_arg {
    name: "input_weights"
    type: DT_FLOAT
  }
  input_arg {
    name: "input_indices"
    type: DT_INT64
  }
  input_arg {
    name: "input_values"
    type: DT_FLOAT
  }
  input_arg {
    name: "input_block_size"
    type: DT_INT64
  }
  input_arg {
    name: "input_is_transpose"
    type: DT_BOOL
  }
  output_arg {
    name: "partial_lhs"
    type: DT_FLOAT
  }
  output_arg {
    name: "partial_rhs"
    type: DT_FLOAT
  }
}
"""


_op_def_lib = _InitOpDefLibrary()
