"""Python wrappers around TensorFlow ops.

This file is MACHINE GENERATED! Do not edit.
Original C++ source file: image_ops.cc
"""

import collections as _collections

from tensorflow.python.eager import execute as _execute
from tensorflow.python.eager import context as _context
from tensorflow.python.eager import core as _core
from tensorflow.python.framework import dtypes as _dtypes
from tensorflow.python.framework import tensor_shape as _tensor_shape

from tensorflow.core.framework import op_def_pb2 as _op_def_pb2
# Needed to trigger the call to _set_call_cpp_shape_fn.
from tensorflow.python.framework import common_shapes as _common_shapes
from tensorflow.python.framework import op_def_registry as _op_def_registry
from tensorflow.python.framework import ops as _ops
from tensorflow.python.framework import op_def_library as _op_def_library


_bipartite_match_outputs = ["row_to_col_match_indices",
                           "col_to_row_match_indices"]
_BipartiteMatchOutput = _collections.namedtuple(
    "BipartiteMatch", _bipartite_match_outputs)


def bipartite_match(distance_mat, num_valid_rows, top_k=-1, name=None):
  r"""Find bipartite matching based on a given distance matrix.

  A greedy bi-partite matching algorithm is used to obtain the matching with the
  (greedy) minimum distance.

  Args:
    distance_mat: A `Tensor` of type `float32`.
      A 2-D float tensor of shape `[num_rows, num_columns]`. It is a
      pair-wise distance matrix between the entities represented by each row and
      each column. It is an asymmetric matrix. The smaller the distance is, the more
      similar the pairs are. The bipartite matching is to minimize the distances.
    num_valid_rows: A `Tensor` of type `float32`.
      A scalar or a 1-D tensor with one element describing the
      number of valid rows of distance_mat to consider for the bipartite matching.
      If set to be negative, then all rows from `distance_mat` are used.
    top_k: An optional `int`. Defaults to `-1`.
      A scalar that specifies the number of top-k matches to retrieve.
      If set to be negative, then is set according to the maximum number of
      matches from `distance_mat`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (row_to_col_match_indices, col_to_row_match_indices).

    row_to_col_match_indices: A `Tensor` of type `int32`. A vector of length num_rows, which is the number of
      rows of the input `distance_matrix`.
      If `row_to_col_match_indices[i]` is not -1, row i is matched to column
      `row_to_col_match_indices[i]`.
    col_to_row_match_indices: A `Tensor` of type `int32`. A vector of length num_columns, which is the number
      of columns of the input ditance matrix.
      If `col_to_row_match_indices[j]` is not -1, column j is matched to row
      `col_to_row_match_indices[j]`.
  """
  if top_k is None:
    top_k = -1
  top_k = _execute.make_int(top_k, "top_k")
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "BipartiteMatch", distance_mat=distance_mat,
        num_valid_rows=num_valid_rows, top_k=top_k, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("top_k", _op.get_attr("top_k"))
  else:
    distance_mat = _ops.convert_to_tensor(distance_mat, _dtypes.float32)
    num_valid_rows = _ops.convert_to_tensor(num_valid_rows, _dtypes.float32)
    _inputs_flat = [distance_mat, num_valid_rows]
    _attrs = ("top_k", top_k)
    _result = _execute.execute(b"BipartiteMatch", 2, inputs=_inputs_flat,
                               attrs=_attrs, ctx=_ctx, name=name)
  _execute.record_gradient(
      "BipartiteMatch", _inputs_flat, _attrs, _result, name)
  _result = _BipartiteMatchOutput._make(_result)
  return _result

_ops.RegisterShape("BipartiteMatch")(None)


def image_projective_transform(images, transforms, interpolation, name=None):
  r"""Applies the given transform to each of the images.

  Input `image` is a `Tensor` in NHWC format (where the axes are image in batch,
  rows, columns, and channels. Input `transforms` is a num_images x 8 or 1 x 8
  matrix, where each row corresponds to a 3 x 3 projective transformation matrix,
  with the last entry assumed to be 1. If there is one row, the same
  transformation will be applied to all images.

  If one row of `transforms` is `[a0, a1, a2, b0, b1, b2, c0, c1]`, then it maps
  the *output* point `(x, y)` to a transformed *input* point
  `(x', y') = ((a0 x + a1 y + a2) / k, (b0 x + b1 y + b2) / k)`, where
  `k = c0 x + c1 y + 1`. If the transformed point lays outside of the input
  image, the output pixel is set to 0. The output is the same size as the input,

  Args:
    images: A `Tensor`. Must be one of the following types: `uint8`, `int32`, `int64`, `float32`, `float64`.
      4D `Tensor`, input image(s) in NHWC format.
    transforms: A `Tensor` of type `float32`.
      2D `Tensor`, projective transform(s) to apply to the image(s).
    interpolation: A `string`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `images`.
    4D `Tensor`, image(s) in NHWC format, generated by applying
    the `transforms` to the `images`. Satisfies the description above.
  """
  interpolation = _execute.make_str(interpolation, "interpolation")
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "ImageProjectiveTransform", images=images, transforms=transforms,
        interpolation=interpolation, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("dtype", _op.get_attr("dtype"), "interpolation",
              _op.get_attr("interpolation"))
  else:
    _attr_dtype, (images,) = _execute.args_to_matching_eager([images], _ctx)
    _attr_dtype = _attr_dtype.as_datatype_enum
    transforms = _ops.convert_to_tensor(transforms, _dtypes.float32)
    _inputs_flat = [images, transforms]
    _attrs = ("dtype", _attr_dtype, "interpolation", interpolation)
    _result = _execute.execute(b"ImageProjectiveTransform", 1,
                               inputs=_inputs_flat, attrs=_attrs, ctx=_ctx,
                               name=name)
  _execute.record_gradient(
      "ImageProjectiveTransform", _inputs_flat, _attrs, _result, name)
  _result, = _result
  return _result

_ops.RegisterShape("ImageProjectiveTransform")(None)

def _InitOpDefLibrary(op_list_proto_bytes):
  op_list = _op_def_pb2.OpList()
  op_list.ParseFromString(op_list_proto_bytes)
  _op_def_registry.register_op_list(op_list)
  op_def_lib = _op_def_library.OpDefLibrary()
  op_def_lib.add_op_list(op_list)
  return op_def_lib
# op {
#   name: "BipartiteMatch"
#   input_arg {
#     name: "distance_mat"
#     type: DT_FLOAT
#   }
#   input_arg {
#     name: "num_valid_rows"
#     type: DT_FLOAT
#   }
#   output_arg {
#     name: "row_to_col_match_indices"
#     type: DT_INT32
#   }
#   output_arg {
#     name: "col_to_row_match_indices"
#     type: DT_INT32
#   }
#   attr {
#     name: "top_k"
#     type: "int"
#     default_value {
#       i: -1
#     }
#   }
#   is_stateful: true
# }
# op {
#   name: "ImageProjectiveTransform"
#   input_arg {
#     name: "images"
#     type_attr: "dtype"
#   }
#   input_arg {
#     name: "transforms"
#     type: DT_FLOAT
#   }
#   output_arg {
#     name: "transformed_images"
#     type_attr: "dtype"
#   }
#   attr {
#     name: "dtype"
#     type: "type"
#     allowed_values {
#       list {
#         type: DT_UINT8
#         type: DT_INT32
#         type: DT_INT64
#         type: DT_FLOAT
#         type: DT_DOUBLE
#       }
#     }
#   }
#   attr {
#     name: "interpolation"
#     type: "string"
#   }
# }
_op_def_lib = _InitOpDefLibrary(b"\n\220\001\n\016BipartiteMatch\022\020\n\014distance_mat\030\001\022\022\n\016num_valid_rows\030\001\032\034\n\030row_to_col_match_indices\030\003\032\034\n\030col_to_row_match_indices\030\003\"\031\n\005top_k\022\003int\032\013\030\377\377\377\377\377\377\377\377\377\001\210\001\001\n\213\001\n\030ImageProjectiveTransform\022\017\n\006images\"\005dtype\022\016\n\ntransforms\030\001\032\033\n\022transformed_images\"\005dtype\"\030\n\005dtype\022\004type:\t\n\0072\005\004\003\t\001\002\"\027\n\rinterpolation\022\006string")
