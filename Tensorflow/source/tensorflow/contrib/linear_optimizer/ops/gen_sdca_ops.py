"""Python wrappers around Brain.

This file is MACHINE GENERATED! Do not edit.
"""

import collections

from google.protobuf import text_format

from tensorflow.core.framework import op_def_pb2
from tensorflow.python.framework import op_def_registry
from tensorflow.python.framework import ops
from tensorflow.python.ops import op_def_library


_sdca_shrink_l1_outputs = [""]


def sdca_shrink_l1(sparse_weights, dense_weights, l1, l2, name=None):
  r"""Applies L1 regularization shrink step on the parameters.

  Args:
    sparse_weights: A list of `Tensor` objects of type mutable `float32`.
      a list of vectors where each value is the weight associated with
      a feature group.
    dense_weights: A list of `Tensor` objects of type mutable `float32`.
      a list of vectors where the value is the weight associated with
      a dense feature group.
    l1: A `float`. Symmetric l1 regularization strength.
    l2: A `float`. Symmetric l2 regularization strength.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  result = _op_def_lib.apply_op("SdcaShrinkL1", sparse_weights=sparse_weights,
                                dense_weights=dense_weights, l1=l1, l2=l2,
                                name=name)
  return result


ops.RegisterShape("SdcaShrinkL1")(None)
_sdca_solver_outputs = [""]


def sdca_solver(sparse_features_indices, sparse_features_values,
                dense_features, example_weights, example_labels, example_ids,
                sparse_weights, dense_weights, loss_type, l1, l2,
                num_inner_iterations, container, solver_uuid, name=None):
  r"""Stochastic Dual Coordinate Ascent (SDCA) optimizer for linear models with

  L1 + L2 regularization. As global optimization objective is strongly-convex, the
  optimizer optimizes the dual objective at each step. The optimizer applies each
  update one example at a time. Examples are sampled uniformly, and the optimizer
  is learning rate free and enjoys linear convergence rate.

  Proximal Stochastic Dual Coordinate Ascent, Shalev-Shwartz, Shai; Zhang, Tong.
  2012arXiv1211.2717S: http://arxiv.org/pdf/1211.2717v1.pdf

    Loss objective = \sum f_{i}(wx_{i}) + l2 * |w|^2 + l1 * |w|

  Args:
    sparse_features_indices: A list of `Tensor` objects of type `int64`.
      a list of matrices with two columns that contain
      example_indices, and feature_indices.
    sparse_features_values: A list with the same number of `Tensor` objects as `sparse_features_indices` of `Tensor` objects of type `float32`.
      a list of vectors which contains feature value
      associated with each feature group.
    dense_features: A list of `Tensor` objects of type `float32`.
      a list of vectors which contains the dense feature values.
    example_weights: A `Tensor` of type `float32`.
      a vector which contains the weight associated with each
      example.
    example_labels: A `Tensor` of type `float32`.
      a vector which contains the label/target associated with each
      example.
    example_ids: A `Tensor` of type `string`.
      a vector which contains the unique identifier associated with each
      example.
    sparse_weights: A list with the same number of `Tensor` objects as `sparse_features_indices` of `Tensor` objects of type mutable `float32`.
      a list of vectors where each value is the weight associated with
      a feature group.
    dense_weights: A list with the same number of `Tensor` objects as `dense_features` of `Tensor` objects of type mutable `float32`.
      a list of vectors where the value is the weight associated with
      a dense feature group.
    loss_type: A `string` from: `"logistic_loss", "squared_loss", "hinge_loss"`.
      Type of the primal loss. Currently SdcaSolver supports logistic,
      squared and hinge losses.
    l1: A `float`. Symmetric l1 regularization strength.
    l2: A `float`. Symmetric l2 regularization strength.
    num_inner_iterations: An `int` that is `>= 1`.
      Number of iterations per mini-batch.
    container: A `string`.
      Name of the Container that stores data across invocations of this
      Kernel. Together with SolverUUID form an isolation unit for this solver.
    solver_uuid: A `string`. Universally Unique Identifier for this solver.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  result = _op_def_lib.apply_op("SdcaSolver",
                                sparse_features_indices=sparse_features_indices,
                                sparse_features_values=sparse_features_values,
                                dense_features=dense_features,
                                example_weights=example_weights,
                                example_labels=example_labels,
                                example_ids=example_ids,
                                sparse_weights=sparse_weights,
                                dense_weights=dense_weights,
                                loss_type=loss_type, l1=l1, l2=l2,
                                num_inner_iterations=num_inner_iterations,
                                container=container, solver_uuid=solver_uuid,
                                name=name)
  return result


ops.RegisterShape("SdcaSolver")(None)
_sdca_training_stats_outputs = ["primal_loss", "dual_loss", "example_weights"]


_SdcaTrainingStatsOutput = collections.namedtuple("SdcaTrainingStats",
                                                  _sdca_training_stats_outputs)


def sdca_training_stats(container, solver_uuid, name=None):
  r"""Computes statistics over all examples seen by the optimizer.

  Args:
    container: A `string`.
      Name of the Container that stores data across invocations of this
      Kernel. Together with SolverUUID form an isolation unit for this solver.
    solver_uuid: A `string`. Universally Unique Identifier for this solver.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (primal_loss, dual_loss, example_weights).
    primal_loss: A `Tensor` of type `float64`. total primal loss of all examples seen by the optimizer.
    dual_loss: A `Tensor` of type `float64`. total dual loss of all examples seen by the optimizer.
    example_weights: A `Tensor` of type `float64`. total example weights of all examples seen by the optimizer
      (guaranteed to be positive; otherwise returns FAILED_PRECONDITION as it
       probably indicates a bug in the training data).
  """
  result = _op_def_lib.apply_op("SdcaTrainingStats", container=container,
                                solver_uuid=solver_uuid, name=name)
  return _SdcaTrainingStatsOutput._make(result)


ops.RegisterShape("SdcaTrainingStats")(None)
def _InitOpDefLibrary():
  op_list = op_def_pb2.OpList()
  text_format.Merge(_InitOpDefLibrary.op_list_ascii, op_list)
  op_def_registry.register_op_list(op_list)
  op_def_lib = op_def_library.OpDefLibrary()
  op_def_lib.add_op_list(op_list)
  return op_def_lib


_InitOpDefLibrary.op_list_ascii = """op {
  name: "SdcaShrinkL1"
  input_arg {
    name: "sparse_weights"
    type: DT_FLOAT
    number_attr: "num_sparse_features"
    is_ref: true
  }
  input_arg {
    name: "dense_weights"
    type: DT_FLOAT
    number_attr: "num_dense_features"
    is_ref: true
  }
  attr {
    name: "num_sparse_features"
    type: "int"
    has_minimum: true
  }
  attr {
    name: "num_dense_features"
    type: "int"
    has_minimum: true
  }
  attr {
    name: "l1"
    type: "float"
  }
  attr {
    name: "l2"
    type: "float"
  }
}
op {
  name: "SdcaSolver"
  input_arg {
    name: "sparse_features_indices"
    type: DT_INT64
    number_attr: "num_sparse_features"
  }
  input_arg {
    name: "sparse_features_values"
    type: DT_FLOAT
    number_attr: "num_sparse_features"
  }
  input_arg {
    name: "dense_features"
    type: DT_FLOAT
    number_attr: "num_dense_features"
  }
  input_arg {
    name: "example_weights"
    type: DT_FLOAT
  }
  input_arg {
    name: "example_labels"
    type: DT_FLOAT
  }
  input_arg {
    name: "example_ids"
    type: DT_STRING
  }
  input_arg {
    name: "sparse_weights"
    type: DT_FLOAT
    number_attr: "num_sparse_features"
    is_ref: true
  }
  input_arg {
    name: "dense_weights"
    type: DT_FLOAT
    number_attr: "num_dense_features"
    is_ref: true
  }
  attr {
    name: "loss_type"
    type: "string"
    allowed_values {
      list {
        s: "logistic_loss"
        s: "squared_loss"
        s: "hinge_loss"
      }
    }
  }
  attr {
    name: "num_sparse_features"
    type: "int"
    has_minimum: true
  }
  attr {
    name: "num_dense_features"
    type: "int"
    has_minimum: true
  }
  attr {
    name: "l1"
    type: "float"
  }
  attr {
    name: "l2"
    type: "float"
  }
  attr {
    name: "num_inner_iterations"
    type: "int"
    has_minimum: true
    minimum: 1
  }
  attr {
    name: "container"
    type: "string"
  }
  attr {
    name: "solver_uuid"
    type: "string"
  }
}
op {
  name: "SdcaTrainingStats"
  output_arg {
    name: "primal_loss"
    type: DT_DOUBLE
  }
  output_arg {
    name: "dual_loss"
    type: DT_DOUBLE
  }
  output_arg {
    name: "example_weights"
    type: DT_DOUBLE
  }
  attr {
    name: "container"
    type: "string"
  }
  attr {
    name: "solver_uuid"
    type: "string"
  }
}
"""


_op_def_lib = _InitOpDefLibrary()
