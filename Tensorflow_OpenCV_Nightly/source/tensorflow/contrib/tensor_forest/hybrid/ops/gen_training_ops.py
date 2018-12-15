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

_hard_routing_function_outputs = ["path_probability", "path"]
_HardRoutingFunctionOutput = _collections.namedtuple(
    "HardRoutingFunction", _hard_routing_function_outputs)


def hard_routing_function(input_data, tree_parameters, tree_biases, max_nodes,
                          tree_depth, name=None):
  r"""  Chooses a single path for each instance in `input_data` and returns the leaf

    the probability of the path and the path taken.

    tree_depth: The depth of the decision tree.

    input_data: The training batch's features as a 2-d tensor; `input_data[i][j]`
     gives the j-th feature of the i-th input.
    tree_parameters: `tree_parameters[i]` gives the weight of
     the logistic regression model that translates from node features to
     probabilities.
    tree_biases: `tree_biases[i]` gives the bias of the logistic
     regression model that translates from node features to
     probabilities.

    path_probility: `path_probability[i]` gives the probability of reaching each
     node in `path[i]`.
    path: `path[i][j]` gives the jth node in the path taken by the ith data
     instance.

  Args:
    input_data: A `Tensor` of type `float32`.
    tree_parameters: A `Tensor` of type `float32`.
    tree_biases: A `Tensor` of type `float32`.
    max_nodes: An `int`.
    tree_depth: An `int`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (path_probability, path).

    path_probability: A `Tensor` of type `float32`.
    path: A `Tensor` of type `int32`.
  """
  result = _op_def_lib.apply_op("HardRoutingFunction", input_data=input_data,
                                tree_parameters=tree_parameters,
                                tree_biases=tree_biases, max_nodes=max_nodes,
                                tree_depth=tree_depth, name=name)
  return _HardRoutingFunctionOutput._make(result)


_ops.RegisterShape("HardRoutingFunction")(None)

_k_feature_gradient_outputs = ["routing_gradient", "data_gradient",
                              "weight_gradient"]
_KFeatureGradientOutput = _collections.namedtuple(
    "KFeatureGradient", _k_feature_gradient_outputs)


def k_feature_gradient(input_data, tree_parameters, tree_biases, routes,
                       layer_num, random_seed, name=None):
  r"""    Computes the derivative of the routing loss with respect to each decision

      node.  Each decision node is constrained to make a decision based on only
      k features.

      layer_num: The layer number of this tree.
      random_seed: The base random seed.

      input_data: The training batch's features as a 2-d tensor;
       `input_data[i][j]` gives the j-th feature of the i-th input.
      tree_parameters: `tree_parameters[i]` gives the weight of
       the logistic regression model that translates from node features to
       probabilities.
      tree_biases: `tree_biases[i]` gives the bias of the logistic
       regression model that translates from node features to
       probabilities.
      routes: The routes computed by routing_function_op.

      routing_gradient: `routing_gradient` provides du / df, where u is the
       routing function and f is the (vector of) decision functions.  A decision
       function f_i computes the routing decision at node i.

      data_gradient: `data_gradient` provides df / dx, where f is the (vector
       of) decision functions and x is a batch of data.

      weights_gradient: `weights_gradient` provides df / dw, where f is the
       (vector of) decision functions and w is the matrix of parameters that
       determine how instances are routed through a tree.

      f_i, the decision function at node i, is parameterized by t_i (parameters)
      and b_i (bias) and takes data x as input.  This op is called in
      training_ops.py to compute du / df, and we use that to compute

      du / dx = du / df * df / dx,
      du / dt = du / df * df / dt, and
      du / db = du / df * df / db.

  Args:
    input_data: A `Tensor` of type `float32`.
    tree_parameters: A `Tensor` of type `float32`.
    tree_biases: A `Tensor` of type `float32`.
    routes: A `Tensor` of type `float32`.
    layer_num: An `int`.
    random_seed: An `int`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (routing_gradient, data_gradient, weight_gradient).

    routing_gradient: A `Tensor` of type `float32`.
    data_gradient: A `Tensor` of type `float32`.
    weight_gradient: A `Tensor` of type `float32`.
  """
  result = _op_def_lib.apply_op("KFeatureGradient", input_data=input_data,
                                tree_parameters=tree_parameters,
                                tree_biases=tree_biases, routes=routes,
                                layer_num=layer_num, random_seed=random_seed,
                                name=name)
  return _KFeatureGradientOutput._make(result)


_ops.RegisterShape("KFeatureGradient")(None)

def k_feature_routing_function(input_data, tree_parameters, tree_biases,
                               layer_num, max_nodes, num_features_per_node,
                               random_seed, name=None):
  r"""  Returns the probability that each input will reach each leaf node.  Each

    decision is made based on k features.

    layer_num: The layer number of this tree.
    max_nodes: The number of nodes in the tree.
    num_features_per_node: The number of features each node can use to make a
     decision.
    random_seed: The base random seed.

    input_data: The training batch's features as a 2-d tensor; `input_data[i][j]`
     gives the j-th feature of the i-th input.
    tree_parameters: `tree_parameters[i]` gives the weight of
     the logistic regression model that translates from node features to
     probabilities.
    tree_biases: `tree_biases[i]` gives the bias of the logistic
     regression model that translates from node features to
     probabilities.
    tree_features: `tree_features[i]` gives the decision feature for node i.

    probabilities: `probabilities[i][j]` is the probability that input i
     will reach node j.

  Args:
    input_data: A `Tensor` of type `float32`.
    tree_parameters: A `Tensor` of type `float32`.
    tree_biases: A `Tensor` of type `float32`.
    layer_num: An `int`.
    max_nodes: An `int`.
    num_features_per_node: An `int`.
    random_seed: An `int`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32`.
  """
  result = _op_def_lib.apply_op("KFeatureRoutingFunction",
                                input_data=input_data,
                                tree_parameters=tree_parameters,
                                tree_biases=tree_biases, layer_num=layer_num,
                                max_nodes=max_nodes,
                                num_features_per_node=num_features_per_node,
                                random_seed=random_seed, name=name)
  return result


_ops.RegisterShape("KFeatureRoutingFunction")(None)

def routing_function(input_data, tree_parameters, tree_biases, max_nodes,
                     name=None):
  r"""  Returns the probability that each input will reach each leaf node.

    max_nodes: The number of nodes in the tree.

    input_data: The training batch's features as a 2-d tensor; `input_data[i][j]`
     gives the j-th feature of the i-th input.
    tree_parameters: `tree_parameters[i]` gives the weight of
     the logistic regression model that translates from node features to
     probabilities.
    tree_biases: `tree_biases[i]` gives the bias of the logistic
     regression model that translates from node features to
     probabilities.

    probabilities: `probabilities[i][j]` is the probability that input i
     will reach node j.

  Args:
    input_data: A `Tensor` of type `float32`.
    tree_parameters: A `Tensor` of type `float32`.
    tree_biases: A `Tensor` of type `float32`.
    max_nodes: An `int`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32`.
  """
  result = _op_def_lib.apply_op("RoutingFunction", input_data=input_data,
                                tree_parameters=tree_parameters,
                                tree_biases=tree_biases, max_nodes=max_nodes,
                                name=name)
  return result


_ops.RegisterShape("RoutingFunction")(None)

def routing_gradient(input_data, tree_parameters, tree_biases, routes,
                     max_nodes, name=None):
  r"""  Computes the derivative of the routing loss with respect to each decision

    node.

    max_nodes: The number of nodes in the tree.

    tree_parameters: `tree_parameters[i]` gives the weight of
     the logistic regression model that translates from node features to
     probabilities.
    tree_biases: `tree_biases[i]` gives the bias of the logistic
     regression model that translates from node features to
     probabilities.
    routes: The routes computed by routing_function_op.

    routing_gradient: `routing_gradient` provides du / df, where u is the routing
     function and f is the (vector of) decision functions.  A decision function
     f_i computes the routing decision at node i.

     f_i is parameterized by t_i (parameters) and b_i (bias) and takes data x as
     input.  This op is called in training_ops.py to compute du / df, and we use
     that to compute

       du / dx = du / df * df / dx,
       du / dt = du / df * df / dt, and
       du / db = du / df * df / db.

  Args:
    input_data: A `Tensor` of type `float32`.
    tree_parameters: A `Tensor` of type `float32`.
    tree_biases: A `Tensor` of type `float32`.
    routes: A `Tensor` of type `float32`.
    max_nodes: An `int`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32`.
  """
  result = _op_def_lib.apply_op("RoutingGradient", input_data=input_data,
                                tree_parameters=tree_parameters,
                                tree_biases=tree_biases, routes=routes,
                                max_nodes=max_nodes, name=name)
  return result


_ops.RegisterShape("RoutingGradient")(None)

_stochastic_hard_routing_function_outputs = ["path_probability", "path"]
_StochasticHardRoutingFunctionOutput = _collections.namedtuple(
    "StochasticHardRoutingFunction",
    _stochastic_hard_routing_function_outputs)


def stochastic_hard_routing_function(input_data, tree_parameters, tree_biases,
                                     tree_depth, random_seed, name=None):
  r"""  Samples a path for each instance in `input_data` and returns the

    probability of the path and the path taken.

    tree_depth: The depth of the decision tree.
    random_seed: The base random seed.

    input_data: The training batch's features as a 2-d tensor; `input_data[i][j]`
     gives the j-th feature of the i-th input.
    tree_parameters: `tree_parameters[i]` gives the weight of
     the logistic regression model that translates from node features to
     probabilities.
    tree_biases: `tree_biases[i]` gives the bias of the logistic
     regression model that translates from node features to
     probabilities.

    path_probility: `path_probability[i]` gives the probability of reaching each
     node in `path[i]`.
    path: `path[i][j]` gives the jth node in the path taken by the ith data
     instance.

  Args:
    input_data: A `Tensor` of type `float32`.
    tree_parameters: A `Tensor` of type `float32`.
    tree_biases: A `Tensor` of type `float32`.
    tree_depth: An `int`.
    random_seed: An `int`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (path_probability, path).

    path_probability: A `Tensor` of type `float32`.
    path: A `Tensor` of type `int32`.
  """
  result = _op_def_lib.apply_op("StochasticHardRoutingFunction",
                                input_data=input_data,
                                tree_parameters=tree_parameters,
                                tree_biases=tree_biases,
                                tree_depth=tree_depth,
                                random_seed=random_seed, name=name)
  return _StochasticHardRoutingFunctionOutput._make(result)


_ops.RegisterShape("StochasticHardRoutingFunction")(None)

_stochastic_hard_routing_gradient_outputs = ["routing_gradient",
                                            "data_gradient",
                                            "parameter_gradient",
                                            "bias_gradient"]
_StochasticHardRoutingGradientOutput = _collections.namedtuple(
    "StochasticHardRoutingGradient",
    _stochastic_hard_routing_gradient_outputs)


def stochastic_hard_routing_gradient(input_data, tree_parameters, tree_biases,
                                     path_probability, path, tree_depth,
                                     name=None):
  r"""  Computes the derivative of the routing loss with respect to each decision

    node.

    tree_depth: The depth of the decision tree.

    input_data: The training batch's features as a 2-d tensor; `input_data[i][j]`
     gives the j-th feature of the i-th input
    tree_parameters: `tree_parameters[i]` gives the weight of
     the logistic regression model that translates from node features to
     probabilities.
    tree_biases: `tree_biases[i]` gives the bias of the logistic
     regression model that translates from node features to
     probabilities.
    path_probility: `path_probability[i]` gives the probability of reaching each
     node in `path[i]`.
    path: `path[i][j]` gives the jth node in the path taken by the ith data
     instance.

    routing_gradient: `routing_gradient` provides du / df, where u is the routing
     function and f is the (vector of) decision functions.  A decision function
     f_i computes the routing decision at node i.
    data_gradient: `data_gradient` provides df / dx, where f is the (vector
     of) decision functions and x is a batch of data.
    parameter_gradient: `parameter_gradient` provides df / dw, where f is the
     (vector of) decision functions and w is the matrix of parameters that
     determine how instances are routed through a tree.
    bias_gradient: `bias_gradient` provides df / db, where f is the
     (vector of) decision functions and b is the vector of bias parameters that
     determine how instances are routed through a tree.

    f_i is parameterized by t_i (parameters) and b_i (bias) and takes data x as
    input.  This op is called in training_ops.py to compute du / df, and we use
    that to compute

       du / dx = du / df * df / dx,
       du / dt = du / df * df / dt, and
       du / db = du / df * df / db.

  Args:
    input_data: A `Tensor` of type `float32`.
    tree_parameters: A `Tensor` of type `float32`.
    tree_biases: A `Tensor` of type `float32`.
    path_probability: A `Tensor` of type `float32`.
    path: A `Tensor` of type `int32`.
    tree_depth: An `int`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (routing_gradient, data_gradient, parameter_gradient, bias_gradient).

    routing_gradient: A `Tensor` of type `float32`.
    data_gradient: A `Tensor` of type `float32`.
    parameter_gradient: A `Tensor` of type `float32`.
    bias_gradient: A `Tensor` of type `float32`.
  """
  result = _op_def_lib.apply_op("StochasticHardRoutingGradient",
                                input_data=input_data,
                                tree_parameters=tree_parameters,
                                tree_biases=tree_biases,
                                path_probability=path_probability, path=path,
                                tree_depth=tree_depth, name=name)
  return _StochasticHardRoutingGradientOutput._make(result)


_ops.RegisterShape("StochasticHardRoutingGradient")(None)

def unpack_path(path, path_values, name=None):
  r"""  Takes a batch of paths through a tree and a batch of values along those paths

    and returns a batch_size by num_nodes encoding of the path values.

    path: `path[i][j]` gives the jth node in the path taken by the ith data
     instance.
    path_values: `path_values[i][j]` gives the value associated with node j in the
     path defined by the ith instance

    unpacked_paths: `unpacked_paths[i][path[i][k]]` is path_values[i][k] for k in
     [0, tree_depth).  All other elements of unpacked_paths are zero.

  Args:
    path: A `Tensor` of type `int32`.
    path_values: A `Tensor` of type `float32`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32`.
  """
  result = _op_def_lib.apply_op("UnpackPath", path=path,
                                path_values=path_values, name=name)
  return result


_ops.RegisterShape("UnpackPath")(None)
def _InitOpDefLibrary():
  op_list = _op_def_pb2.OpList()
  _text_format.Merge(_InitOpDefLibrary.op_list_ascii, op_list)
  _op_def_registry.register_op_list(op_list)
  op_def_lib = _op_def_library.OpDefLibrary()
  op_def_lib.add_op_list(op_list)
  return op_def_lib


_InitOpDefLibrary.op_list_ascii = """op {
  name: "HardRoutingFunction"
  input_arg {
    name: "input_data"
    type: DT_FLOAT
  }
  input_arg {
    name: "tree_parameters"
    type: DT_FLOAT
  }
  input_arg {
    name: "tree_biases"
    type: DT_FLOAT
  }
  output_arg {
    name: "path_probability"
    type: DT_FLOAT
  }
  output_arg {
    name: "path"
    type: DT_INT32
  }
  attr {
    name: "max_nodes"
    type: "int"
  }
  attr {
    name: "tree_depth"
    type: "int"
  }
}
op {
  name: "KFeatureGradient"
  input_arg {
    name: "input_data"
    type: DT_FLOAT
  }
  input_arg {
    name: "tree_parameters"
    type: DT_FLOAT
  }
  input_arg {
    name: "tree_biases"
    type: DT_FLOAT
  }
  input_arg {
    name: "routes"
    type: DT_FLOAT
  }
  output_arg {
    name: "routing_gradient"
    type: DT_FLOAT
  }
  output_arg {
    name: "data_gradient"
    type: DT_FLOAT
  }
  output_arg {
    name: "weight_gradient"
    type: DT_FLOAT
  }
  attr {
    name: "layer_num"
    type: "int"
  }
  attr {
    name: "random_seed"
    type: "int"
  }
}
op {
  name: "KFeatureRoutingFunction"
  input_arg {
    name: "input_data"
    type: DT_FLOAT
  }
  input_arg {
    name: "tree_parameters"
    type: DT_FLOAT
  }
  input_arg {
    name: "tree_biases"
    type: DT_FLOAT
  }
  output_arg {
    name: "probabilities"
    type: DT_FLOAT
  }
  attr {
    name: "layer_num"
    type: "int"
  }
  attr {
    name: "max_nodes"
    type: "int"
  }
  attr {
    name: "num_features_per_node"
    type: "int"
  }
  attr {
    name: "random_seed"
    type: "int"
  }
}
op {
  name: "RoutingFunction"
  input_arg {
    name: "input_data"
    type: DT_FLOAT
  }
  input_arg {
    name: "tree_parameters"
    type: DT_FLOAT
  }
  input_arg {
    name: "tree_biases"
    type: DT_FLOAT
  }
  output_arg {
    name: "probabilities"
    type: DT_FLOAT
  }
  attr {
    name: "max_nodes"
    type: "int"
  }
}
op {
  name: "RoutingGradient"
  input_arg {
    name: "input_data"
    type: DT_FLOAT
  }
  input_arg {
    name: "tree_parameters"
    type: DT_FLOAT
  }
  input_arg {
    name: "tree_biases"
    type: DT_FLOAT
  }
  input_arg {
    name: "routes"
    type: DT_FLOAT
  }
  output_arg {
    name: "routing_gradient"
    type: DT_FLOAT
  }
  attr {
    name: "max_nodes"
    type: "int"
  }
}
op {
  name: "StochasticHardRoutingFunction"
  input_arg {
    name: "input_data"
    type: DT_FLOAT
  }
  input_arg {
    name: "tree_parameters"
    type: DT_FLOAT
  }
  input_arg {
    name: "tree_biases"
    type: DT_FLOAT
  }
  output_arg {
    name: "path_probability"
    type: DT_FLOAT
  }
  output_arg {
    name: "path"
    type: DT_INT32
  }
  attr {
    name: "tree_depth"
    type: "int"
  }
  attr {
    name: "random_seed"
    type: "int"
  }
}
op {
  name: "StochasticHardRoutingGradient"
  input_arg {
    name: "input_data"
    type: DT_FLOAT
  }
  input_arg {
    name: "tree_parameters"
    type: DT_FLOAT
  }
  input_arg {
    name: "tree_biases"
    type: DT_FLOAT
  }
  input_arg {
    name: "path_probability"
    type: DT_FLOAT
  }
  input_arg {
    name: "path"
    type: DT_INT32
  }
  output_arg {
    name: "routing_gradient"
    type: DT_FLOAT
  }
  output_arg {
    name: "data_gradient"
    type: DT_FLOAT
  }
  output_arg {
    name: "parameter_gradient"
    type: DT_FLOAT
  }
  output_arg {
    name: "bias_gradient"
    type: DT_FLOAT
  }
  attr {
    name: "tree_depth"
    type: "int"
  }
}
op {
  name: "UnpackPath"
  input_arg {
    name: "path"
    type: DT_INT32
  }
  input_arg {
    name: "path_values"
    type: DT_FLOAT
  }
  output_arg {
    name: "unpacked_path"
    type: DT_FLOAT
  }
}
"""


_op_def_lib = _InitOpDefLibrary()
