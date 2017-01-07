"""Python wrappers around Brain.

This file is MACHINE GENERATED! Do not edit.
"""

import collections

from google.protobuf import text_format

from tensorflow.core.framework import op_def_pb2
from tensorflow.python.framework import op_def_registry
from tensorflow.python.framework import ops
from tensorflow.python.ops import op_def_library


_sparse_feature_cross_outputs = ["output_indices", "output_values",
                                "output_shape"]


_SparseFeatureCrossOutput = collections.namedtuple("SparseFeatureCross",
                                                   _sparse_feature_cross_outputs)


def sparse_feature_cross(indices, values, shapes, dense, hashed_output,
                         num_buckets, out_type, internal_type, name=None):
  r"""Generates sparse cross form a list of sparse tensors.

  The op takes two lists, one of 2D `SparseTensor` and one of 2D `Tensor`, each
  representing features of one feature column. It outputs a 2D `SparseTensor` with
  the batchwise crosses of these features.

  For example, if the inputs are

      inputs[0]: SparseTensor with shape = [2, 2]
      [0, 0]: "a"
      [1, 0]: "b"
      [1, 1]: "c"

      inputs[1]: SparseTensor with shape = [2, 1]
      [0, 0]: "d"
      [1, 0]: "e"

      inputs[2]: Tensor [["f"], ["g"]]

  then the output will be

      shape = [2, 2]
      [0, 0]: "a_X_d_X_f"
      [1, 0]: "b_X_e_X_g"
      [1, 1]: "c_X_e_X_g"

  if hashed_output=true then the output will be

      shape = [2, 2]
      [0, 0]: Hash64("f", Hash64("d", Hash64("a")))
      [1, 0]: Hash64("g", Hash64("e", Hash64("b")))
      [1, 1]: Hash64("g", Hash64("e", Hash64("c")))

  Args:
    indices: A list of `Tensor` objects of type `int64`.
      2-D.  Indices of each input `SparseTensor`.
    values: A list of `Tensor` objects with types from: `int64`, `string`.
      1-D.   values of each `SparseTensor`.
    shapes: A list with the same number of `Tensor` objects as `indices` of `Tensor` objects of type `int64`.
      1-D.   Shapes of each `SparseTensor`.
    dense: A list of `Tensor` objects with types from: `int64`, `string`.
      2-D.    Columns represented by dense `Tensor`.
    hashed_output: A `bool`.
    num_buckets: An `int` that is `>= 0`.
    out_type: A `tf.DType` from: `tf.int64, tf.string`.
    internal_type: A `tf.DType` from: `tf.int64, tf.string`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output_indices, output_values, output_shape).
    output_indices: A `Tensor` of type `int64`. 2-D.  Indices of the concatenated `SparseTensor`.
    output_values: A `Tensor` of type `out_type`. 1-D.  Non-empty values of the concatenated or hashed
      `SparseTensor`.
    output_shape: A `Tensor` of type `int64`. 1-D.  Shape of the concatenated `SparseTensor`.
  """
  result = _op_def_lib.apply_op("SparseFeatureCross", indices=indices,
                                values=values, shapes=shapes, dense=dense,
                                hashed_output=hashed_output,
                                num_buckets=num_buckets, out_type=out_type,
                                internal_type=internal_type, name=name)
  return _SparseFeatureCrossOutput._make(result)


ops.RegisterShape("SparseFeatureCross")(None)
def _InitOpDefLibrary():
  op_list = op_def_pb2.OpList()
  text_format.Merge(_InitOpDefLibrary.op_list_ascii, op_list)
  op_def_registry.register_op_list(op_list)
  op_def_lib = op_def_library.OpDefLibrary()
  op_def_lib.add_op_list(op_list)
  return op_def_lib


_InitOpDefLibrary.op_list_ascii = """op {
  name: "SparseFeatureCross"
  input_arg {
    name: "indices"
    type: DT_INT64
    number_attr: "N"
  }
  input_arg {
    name: "values"
    type_list_attr: "sparse_types"
  }
  input_arg {
    name: "shapes"
    type: DT_INT64
    number_attr: "N"
  }
  input_arg {
    name: "dense"
    type_list_attr: "dense_types"
  }
  output_arg {
    name: "output_indices"
    type: DT_INT64
  }
  output_arg {
    name: "output_values"
    type_attr: "out_type"
  }
  output_arg {
    name: "output_shape"
    type: DT_INT64
  }
  attr {
    name: "N"
    type: "int"
    has_minimum: true
  }
  attr {
    name: "hashed_output"
    type: "bool"
  }
  attr {
    name: "num_buckets"
    type: "int"
    has_minimum: true
  }
  attr {
    name: "sparse_types"
    type: "list(type)"
    has_minimum: true
    allowed_values {
      list {
        type: DT_INT64
        type: DT_STRING
      }
    }
  }
  attr {
    name: "dense_types"
    type: "list(type)"
    has_minimum: true
    allowed_values {
      list {
        type: DT_INT64
        type: DT_STRING
      }
    }
  }
  attr {
    name: "out_type"
    type: "type"
    allowed_values {
      list {
        type: DT_INT64
        type: DT_STRING
      }
    }
  }
  attr {
    name: "internal_type"
    type: "type"
    allowed_values {
      list {
        type: DT_INT64
        type: DT_STRING
      }
    }
  }
}
"""


_op_def_lib = _InitOpDefLibrary()
