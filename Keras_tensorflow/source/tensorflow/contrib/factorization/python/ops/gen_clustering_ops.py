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
_kmeans_plus_plus_initialization_outputs = ["samples"]


def kmeans_plus_plus_initialization(points, num_to_sample, seed,
                                    num_retries_per_sample, name=None):
  r"""Selects num_to_sample rows of input using the KMeans++ criterion.

  Rows of points are assumed to be input points. One row is selected at random.
  Subsequent rows are sampled with probability proportional to the squared L2
  distance from the nearest row selected thus far till num_to_sample rows have
  been sampled.

  Args:
    points: A `Tensor` of type `float32`.
      Matrix of shape (n, d). Rows are assumed to be input points.
    num_to_sample: A `Tensor` of type `int64`.
      Scalar. The number of rows to sample. This value must not be
      larger than n.
    seed: A `Tensor` of type `int64`.
      Scalar. Seed for initializing the random number generator.
    num_retries_per_sample: A `Tensor` of type `int64`.
      Scalar. For each row that is sampled, this parameter
      specifies the number of additional points to draw from the current
      distribution before selecting the best. If a negative value is specified, a
      heuristic is used to sample O(log(num_to_sample)) additional points.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32`.
    Matrix of shape (num_to_sample, d). The sampled rows.
  """
  result = _op_def_lib.apply_op("KmeansPlusPlusInitialization", points=points,
                                num_to_sample=num_to_sample, seed=seed,
                                num_retries_per_sample=num_retries_per_sample,
                                name=name)
  return result


_ops.RegisterShape("KmeansPlusPlusInitialization")(None)
_nearest_neighbors_outputs = ["nearest_center_indices",
                             "nearest_center_distances"]


_NearestNeighborsOutput = _collections.namedtuple("NearestNeighbors",
                                                  _nearest_neighbors_outputs)


def nearest_neighbors(points, centers, k, name=None):
  r"""Selects the k nearest centers for each point.

  Rows of points are assumed to be input points. Rows of centers are assumed to be
  the list of candidate centers. For each point, the k centers that have least L2
  distance to it are computed.

  Args:
    points: A `Tensor` of type `float32`.
      Matrix of shape (n, d). Rows are assumed to be input points.
    centers: A `Tensor` of type `float32`.
      Matrix of shape (m, d). Rows are assumed to be centers.
    k: A `Tensor` of type `int64`.
      Scalar. Number of nearest centers to return for each point. If k is larger
      than m, then only m centers are returned.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (nearest_center_indices, nearest_center_distances).
    nearest_center_indices: A `Tensor` of type `int64`. Matrix of shape (n, min(m, k)). Each row contains the
      indices of the centers closest to the corresponding point, ordered by
      increasing distance.
    nearest_center_distances: A `Tensor` of type `float32`. Matrix of shape (n, min(m, k)). Each row contains the
      squared L2 distance to the corresponding center in nearest_center_indices.
  """
  result = _op_def_lib.apply_op("NearestNeighbors", points=points,
                                centers=centers, k=k, name=name)
  return _NearestNeighborsOutput._make(result)


_ops.RegisterShape("NearestNeighbors")(None)
def _InitOpDefLibrary():
  op_list = _op_def_pb2.OpList()
  _text_format.Merge(_InitOpDefLibrary.op_list_ascii, op_list)
  _op_def_registry.register_op_list(op_list)
  op_def_lib = _op_def_library.OpDefLibrary()
  op_def_lib.add_op_list(op_list)
  return op_def_lib


_InitOpDefLibrary.op_list_ascii = """op {
  name: "KmeansPlusPlusInitialization"
  input_arg {
    name: "points"
    type: DT_FLOAT
  }
  input_arg {
    name: "num_to_sample"
    type: DT_INT64
  }
  input_arg {
    name: "seed"
    type: DT_INT64
  }
  input_arg {
    name: "num_retries_per_sample"
    type: DT_INT64
  }
  output_arg {
    name: "samples"
    type: DT_FLOAT
  }
}
op {
  name: "NearestNeighbors"
  input_arg {
    name: "points"
    type: DT_FLOAT
  }
  input_arg {
    name: "centers"
    type: DT_FLOAT
  }
  input_arg {
    name: "k"
    type: DT_INT64
  }
  output_arg {
    name: "nearest_center_indices"
    type: DT_INT64
  }
  output_arg {
    name: "nearest_center_distances"
    type: DT_FLOAT
  }
}
"""


_op_def_lib = _InitOpDefLibrary()
