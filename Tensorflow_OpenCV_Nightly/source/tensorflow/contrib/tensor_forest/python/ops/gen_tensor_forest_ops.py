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

def best_splits(finished_nodes, node_to_accumulator, split_sums,
                split_squares, accumulator_sums, accumulator_sqaures,
                regression=None, name=None):
  r"""  Returns the index of the best split for each finished node.

    For classification, the best split is the split with the lowest weighted
    Gini impurity, as calculated from the statistics in `split_sums` and
    `accumulator_sums`. For regression we use the lowest variance, incoporating
    the *_squares as well.

    finished_nodes:= A 1-d int32 tensor containing the indices of finished nodes.
    node_to_accumulator: `node_to_accumulator[i]` is the accumulator slot used by
      fertile node i, or -1 if node i isn't fertile.
    split_sums:= a 3-d tensor where `split_sums[a][s]` summarizes the
      training labels for examples that fall into the fertile node associated with
      accumulator slot s and have then taken the *left* branch of candidate split
      s.  For a classification problem, `split_sums[a][s][c]` is the count of such
      examples with class c and for regression problems, `split_sums[a][s]` is the
      sum of the regression labels for such examples.
    split_squares: Same as split_sums, but it contains the sum of the
      squares of the regression labels.  Only used for regression.  For
      classification problems, pass a dummy tensor into this.
    accumulator_sums:= a 2-d tensor where `accumulator_sums[a]` summarizes the
      training labels for examples that fall into the fertile node associated with
      accumulator slot s.  For a classification problem, `accumulator_sums[a][c]`
      is the count of such examples with class c and for regression problems,
      `accumulator_sums[a]` is the sum of the regression labels for such examples.
    accumulator_squares: Same as accumulator_sums, but it contains the sum of the
      squares of the regression labels.  Only used for regression.  For
      classification problems, pass a dummy tensor into this.
    split_indices: `split_indices[i]` contains the index of the split to use for
      `finished_nodes[i]`.

  Args:
    finished_nodes: A `Tensor` of type `int32`.
    node_to_accumulator: A `Tensor` of type `int32`.
    split_sums: A `Tensor` of type `float32`.
    split_squares: A `Tensor` of type `float32`.
    accumulator_sums: A `Tensor` of type `float32`.
    accumulator_sqaures: A `Tensor` of type `float32`.
    regression: An optional `bool`. Defaults to `False`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int32`.
  """
  result = _op_def_lib.apply_op("BestSplits", finished_nodes=finished_nodes,
                                node_to_accumulator=node_to_accumulator,
                                split_sums=split_sums,
                                split_squares=split_squares,
                                accumulator_sums=accumulator_sums,
                                accumulator_sqaures=accumulator_sqaures,
                                regression=regression, name=name)
  return result


_ops.RegisterShape("BestSplits")(None)

_count_extremely_random_stats_outputs = ["pcw_node_sums_delta",
                                        "pcw_node_squares_delta",
                                        "pcw_splits_indices",
                                        "pcw_candidate_splits_sums_delta",
                                        "pcw_candidate_splits_squares_delta",
                                        "pcw_totals_indices",
                                        "pcw_totals_sums_delta",
                                        "pcw_totals_squares_delta", "leaves"]
_CountExtremelyRandomStatsOutput = _collections.namedtuple(
    "CountExtremelyRandomStats", _count_extremely_random_stats_outputs)


def count_extremely_random_stats(input_data, sparse_input_indices,
                                 sparse_input_values, sparse_input_shape,
                                 input_labels, input_weights, tree,
                                 tree_thresholds, node_to_accumulator,
                                 candidate_split_features,
                                 candidate_split_thresholds, birth_epochs,
                                 current_epoch, input_spec, num_classes,
                                 regression=None, name=None):
  r"""Calculates incremental statistics for a batch of training data.

  Each training example in `input_data` is sent through the decision tree
  represented by `tree` and `tree_thresholds`.
  The shape and contents of the outputs differ depending on whether
  `regression` is true or not.

  For `regression` = false (classification), `pcw_node_sums_delta[i]` is
  incremented for every node i that it passes through, and the leaf it ends up
  in is recorded in `leaves[i]`.  Then, if the leaf is fertile and
  initialized, the statistics for its corresponding accumulator slot
  are updated in `pcw_candidate_sums_delta` and `pcw_totals_sums_delta`.

  For `regression` = true, outputs contain the sum of the input_labels
  for the appropriate nodes.  In adddition, the *_squares outputs are filled
  in with the sums of the squares of the input_labels. Since outputs are
  all updated at once, the *_indices outputs don't specify the output
  dimension to update, rather the *_delta output contains updates for all the
  outputs.  For example, `pcw_totals_indices` specifies the accumulators to
  update, and `pcw_total_splits_sums_delta` contains the complete output
  updates for each of those accumulators.

  The attr `num_classes` is needed to appropriately size the outputs.

  Args:
    input_data: A `Tensor` of type `float32`.
      The training batch's features as a 2-d tensor; `input_data[i][j]`
      gives the j-th feature of the i-th input.
    sparse_input_indices: A `Tensor` of type `int64`.
      The indices tensor from the SparseTensor input.
    sparse_input_values: A `Tensor` of type `float32`.
      The values tensor from the SparseTensor input.
    sparse_input_shape: A `Tensor` of type `int64`.
      The shape tensor from the SparseTensor input.
    input_labels: A `Tensor` of type `float32`.
      The training batch's labels; `input_labels[i]` is the class
      of the i-th input.
    input_weights: 
      A 1-D float tensor.  If non-empty, `input_weights[i]` gives
      the weight of the i-th input.
    tree:  A 2-d int32 tensor.  `tree[i][0]` gives the index of the left child
      of the i-th node, `tree[i][0] + 1` gives the index of the right child of
      the i-th node, and `tree[i][1]` gives the index of the feature used to
      split the i-th node.
    tree_thresholds: A `Tensor` of type `float32`.
      `tree_thresholds[i]` is the value used to split the i-th
      node.
    node_to_accumulator: A `Tensor` of type `int32`.
      If the i-th node is fertile, `node_to_accumulator[i]`
      is it's accumulator slot.  Otherwise, `node_to_accumulator[i]` is -1.
    candidate_split_features: A `Tensor` of type `int32`.
      `candidate_split_features[a][s]` is the
      index of the feature being considered by split s of accumulator slot a.
    candidate_split_thresholds: A `Tensor` of type `float32`.
      `candidate_split_thresholds[a][s]` is the
      threshold value being considered by split s of accumulator slot a.
    birth_epochs: A `Tensor` of type `int32`.
      `birth_epoch[i]` is the epoch node i was born in.  Only
      nodes satisfying `current_epoch - birth_epoch <= 1` accumulate statistics.
    current_epoch: 
      A 1-d int32 tensor with shape (1).  current_epoch[0] contains
      the current epoch.
    input_spec: A `string`.
      A 1-D tensor containing the type of each column in input_data,
      (e.g. continuous float, categorical).  Index 0 should contain the default
      type, individual feature types start at index 1.
    num_classes: An `int`.
    regression: An optional `bool`. Defaults to `False`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (pcw_node_sums_delta, pcw_node_squares_delta, pcw_splits_indices, pcw_candidate_splits_sums_delta, pcw_candidate_splits_squares_delta, pcw_totals_indices, pcw_totals_sums_delta, pcw_totals_squares_delta, leaves).

    pcw_node_sums_delta: A `Tensor` of type `float32`. `pcw_node_sums_delta[i][c]` is the number of training
      examples in this training batch with class c that passed through node i for
      classification.  For regression, it is the sum of the input_labels that
      have passed through node i.
    pcw_node_squares_delta: A `Tensor` of type `float32`. `pcw_node_squares_delta[i][c]` is the sum of the
      squares of the input labels that have passed through node i for
      regression.  Not set for classification.
    pcw_splits_indices: A 2-d tensor of shape (?, 3) for classification and
      (?, 2) for regression.
      `pcw_splits_indices[i]` gives the coordinates of an entry in
      candidate_split_pcw_sums and candidate_split_pcw_squares that need to be
      updated.  This is meant to be passed with `pcw_candidate_splits_*_delta` to
      a scatter_add for candidate_split_pcw_*:
        training_ops.scatter_add_ndim(candidate_split_pcw_sums
            pcw_splits_indices, pcw_candidate_splits_sums_delta)
    pcw_candidate_splits_sums_delta: A `Tensor` of type `float32`. For classification,
      `pcw_candidate_splits_sums_delta[i]` is the
      number of training examples in this training batch that correspond to
      the i-th entry in `pcw_splits_indices` which took the *left* branch of
      candidate split. For regression, it is the same but a 2-D tensor that has
      the sum of the input_labels for each i-th entry in the indices.
    pcw_candidate_splits_squares_delta: A `Tensor` of type `float32`. For regression, same as
      `pcw_candidate_splits_sums_delta` but the sum of the squares. Not set
      for classification.
    pcw_totals_indices: A `Tensor` of type `int32`. For classification, 'pcw_totals_indices` contains the
      indices (accumulator, class) into total_pcw_sums to update with
      pcw_totals_sums_delta.  For regression, it only contains the accumulator
      (not the class), because pcw_totals_*_delta will contain all the outputs.
    pcw_totals_sums_delta: A `Tensor` of type `float32`. For classification, `pcw_totals_sums_delta[i]` is the
      number of training examples in this batch that ended up in the fertile
      node with accumulator and class indicated by `pcw_totals_indices[i]`.
      For regression, it is the sum of the input_labels corresponding to the
      entries in `pcw_totals_indices[i]`.
    pcw_totals_squares_delta: A `Tensor` of type `float32`. For regression, same as
      `pcw_totals_sums_delta` but the sum of the squares. Not set
      for classification.
    leaves: A `Tensor` of type `int32`. `leaves[i]` is the leaf that input i ended up in.
  """
  result = _op_def_lib.apply_op("CountExtremelyRandomStats",
                                input_data=input_data,
                                sparse_input_indices=sparse_input_indices,
                                sparse_input_values=sparse_input_values,
                                sparse_input_shape=sparse_input_shape,
                                input_labels=input_labels,
                                input_weights=input_weights, tree=tree,
                                tree_thresholds=tree_thresholds,
                                node_to_accumulator=node_to_accumulator,
                                candidate_split_features=candidate_split_features,
                                candidate_split_thresholds=candidate_split_thresholds,
                                birth_epochs=birth_epochs,
                                current_epoch=current_epoch,
                                input_spec=input_spec,
                                num_classes=num_classes,
                                regression=regression, name=name)
  return _CountExtremelyRandomStatsOutput._make(result)


_ops.RegisterShape("CountExtremelyRandomStats")(None)

_finished_nodes_outputs = ["finished", "stale"]
_FinishedNodesOutput = _collections.namedtuple(
    "FinishedNodes", _finished_nodes_outputs)


def finished_nodes(leaves, node_to_accumulator, split_sums, split_squares,
                   accumulator_sums, accumulator_squares, birth_epochs,
                   current_epoch, num_split_after_samples, min_split_samples,
                   regression=None, dominate_fraction=None,
                   dominate_method=None, random_seed=None,
                   check_dominates_every_samples=None, name=None):
  r"""Determines which of the given leaf nodes are done accumulating.

  The `regression` attribute should be set to true for regression problems, and
  false for classification problems.

  If dominate_method is not set to none, then every
  `check_dominates_every_samples` steps the specified method will be used to
  see if the current best split has probability `dominate_fraction` of being
  asymptotically better than the second best split.  If so, the best split
  is picked now, rather than waiting until `num_split_after_samples` samples
  have been seen.  WARNING:  for weighted input data, only `dominate_method` =
  none is safe.

  Args:
    leaves:  A 1-d int32 tensor.  Lists the nodes that are currently leaves.
    node_to_accumulator: A `Tensor` of type `int32`.
      If the i-th node is fertile, `node_to_accumulator[i]`
      is it's accumulator slot.  Otherwise, `node_to_accumulator[i]` is -1.
    split_sums:  a 3-d tensor where `split_sums[a][s]` summarizes the
      training labels for examples that fall into the fertile node associated with
      accumulator slot s and have then taken the *left* branch of candidate split
      s.  For a classification problem, `split_sums[a][s][c]` is the count of such
      examples with class c and for regression problems, `split_sums[a][s]` is the
      sum of the regression labels for such examples.
    split_squares: A `Tensor` of type `float32`.
      Same as split_sums, but it contains the sum of the
      squares of the regression labels.  Only used for regression.  For
      classification problems, pass a dummy tensor into this.
    accumulator_sums: A `Tensor` of type `float32`.
      For classification, `accumulator_sums[a][c]` records how
      many training examples have class c and have ended up in the fertile node
      associated with accumulator slot a.  It has the total sum in entry 0 for
      convenience. For regression, it is the same except it contains the sum
      of the input labels that have been seen, and entry 0 contains the number
      of training examples that have been seen.
    accumulator_squares: A `Tensor` of type `float32`.
      Same as accumulator_sums, but it contains the sum of the
      squares of the regression labels.  Only used for regression.  For
      classification problems, pass a dummy tensor into this.
    birth_epochs:  A 1-d int32 tensor.  `birth_epochs[i]` contains the epoch
      the i-th node was created in.
    current_epoch:  A 1-d int32 tensor with shape (1).  `current_epoch[0]`
      stores the current epoch number.
    num_split_after_samples: An `int`.
    min_split_samples: An `int`.
    regression: An optional `bool`. Defaults to `False`.
    dominate_fraction: An optional `float`. Defaults to `0.99`.
    dominate_method: An optional `string` from: `"none", "hoeffding", "bootstrap", "chebyshev"`. Defaults to `"bootstrap"`.
    random_seed: An optional `int`. Defaults to `0`.
    check_dominates_every_samples: An optional `int`. Defaults to `75`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (finished, stale).

    finished: A 1-d int32 tensor containing the indices of the finished nodes.
      Nodes are finished if they have received at least num_split_after_samples
      samples, or if they have received min_split_samples and the best scoring
      split is sufficiently greater than the next best split.
    stale: A 1-d int32 tensor containing the fertile nodes that were created two
      or more epochs ago.
  """
  result = _op_def_lib.apply_op("FinishedNodes", leaves=leaves,
                                node_to_accumulator=node_to_accumulator,
                                split_sums=split_sums,
                                split_squares=split_squares,
                                accumulator_sums=accumulator_sums,
                                accumulator_squares=accumulator_squares,
                                birth_epochs=birth_epochs,
                                current_epoch=current_epoch,
                                num_split_after_samples=num_split_after_samples,
                                min_split_samples=min_split_samples,
                                regression=regression,
                                dominate_fraction=dominate_fraction,
                                dominate_method=dominate_method,
                                random_seed=random_seed,
                                check_dominates_every_samples=check_dominates_every_samples,
                                name=name)
  return _FinishedNodesOutput._make(result)


_ops.RegisterShape("FinishedNodes")(None)

_grow_tree_outputs = ["nodes_to_update", "tree_updates", "threshold_updates",
                     "new_end_of_tree"]
_GrowTreeOutput = _collections.namedtuple(
    "GrowTree", _grow_tree_outputs)


def grow_tree(end_of_tree, node_to_accumulator, finished_nodes, best_splits,
              candidate_split_features, candidate_split_thresholds,
              name=None):
  r"""  Output the tree changes needed to resolve fertile nodes.

    Previous Ops have already decided which fertile nodes want to stop being
    fertile and what their best candidate split should be and have passed that
    information to this Op in `finished_nodes` and `best_splits`.  This Op
    merely checks that there is still space in tree to add new nodes, and if
    so, writes out the sparse updates needed for the fertile nodes to be
    resolved to the tree and threshold tensors.

    end_of_tree: `end_of_tree[0]` is the number of allocated nodes, or
      equivalently the index of the first free node in the tree tensor.
    node_to_accumulator: `node_to_accumulator[i]` is the accumulator slot used by
      fertile node i, or -1 if node i isn't fertile.
    finished_nodes:= A 1-d int32 tensor containing the indices of finished nodes.
    best_splits: `best_splits[i]` is the index of the best split for
      `finished_nodes[i]`.
    candidate_split_features: `candidate_split_features[a][s]` is the feature
      being considered for split s of the fertile node associated with
      accumulator slot a.
    candidate_split_thresholds: `candidate_split_thresholds[a][s]` is the
      threshold value being considered for split s of the fertile node associated
      with accumulator slot a.
    nodes_to_update:= A 1-d int32 tensor containing the node indices that need
      updating.
    tree_updates: The updates to apply to the 2-d tree tensor.  Intended to be
      used with `tf.scatter_update(tree, nodes_to_update, tree_updates)`.
    threshold_updates: The updates to apply to the 1-d thresholds tensor.
      Intended to be used with
      `tf.scatter_update(thresholds, nodes_to_update, threshold_updates)`.
    new_end_of_tree: `new_end_of_tree[0]` is the new size of the tree.

  Args:
    end_of_tree: A `Tensor` of type `int32`.
    node_to_accumulator: A `Tensor` of type `int32`.
    finished_nodes: A `Tensor` of type `int32`.
    best_splits: A `Tensor` of type `int32`.
    candidate_split_features: A `Tensor` of type `int32`.
    candidate_split_thresholds: A `Tensor` of type `float32`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (nodes_to_update, tree_updates, threshold_updates, new_end_of_tree).

    nodes_to_update: A `Tensor` of type `int32`.
    tree_updates: A `Tensor` of type `int32`.
    threshold_updates: A `Tensor` of type `float32`.
    new_end_of_tree: A `Tensor` of type `int32`.
  """
  result = _op_def_lib.apply_op("GrowTree", end_of_tree=end_of_tree,
                                node_to_accumulator=node_to_accumulator,
                                finished_nodes=finished_nodes,
                                best_splits=best_splits,
                                candidate_split_features=candidate_split_features,
                                candidate_split_thresholds=candidate_split_thresholds,
                                name=name)
  return _GrowTreeOutput._make(result)


_ops.RegisterShape("GrowTree")(None)

def reinterpret_string_to_float(input_data, name=None):
  r"""   Converts byte arrays represented by strings to 32-bit

     floating point numbers. The output numbers themselves are meaningless, and
     should only be used in == comparisons.

     input_data: A batch of string features as a 2-d tensor; `input_data[i][j]`
       gives the j-th feature of the i-th input.
     output_data: A tensor of the same shape as input_data but the values are
       float32.

  Args:
    input_data: A `Tensor` of type `string`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32`.
  """
  result = _op_def_lib.apply_op("ReinterpretStringToFloat",
                                input_data=input_data, name=name)
  return result


_ops.RegisterShape("ReinterpretStringToFloat")(None)

_sample_inputs_outputs = ["accumulators_to_update", "new_split_feature_rows",
                         "new_split_threshold_rows"]
_SampleInputsOutput = _collections.namedtuple(
    "SampleInputs", _sample_inputs_outputs)


def sample_inputs(input_data, sparse_input_indices, sparse_input_values,
                  sparse_input_shape, input_weights, node_to_accumulator,
                  leaves, candidate_split_features,
                  candidate_split_thresholds, input_spec,
                  split_initializations_per_input, split_sampling_random_seed,
                  name=None):
  r"""Initializes candidate splits for newly fertile nodes.

  In an extremely random forest, we don't consider all possible threshold
  values for a candidate split feature, but rather only a sampling of them.
  This Op takes those samples from the training data in `input_data`.  The
  feature and threshold samples are stored in tensors that are indexed by
  accumulator slot, so for each input, we must first look up which leaf
  it ended up in (using `leaves`) and then which accumulator slot if any
  that leaf maps to (using `node_to_accumulator`).

  The attribute `split_initializations_per_input` controls how many splits
  a single training example can initialize, and the attribute
  `split_sampling_random_seed` sets the random number generator's seed
  (a value of 0 means use the current time as the seed).

  Args:
    input_data: A `Tensor` of type `float32`.
      The features for the current batch of training data.
      `input_data[i][j]` is the j-th feature of the i-th input.
    sparse_input_indices: A `Tensor` of type `int64`.
      The indices tensor from the SparseTensor input.
    sparse_input_values: A `Tensor` of type `float32`.
      The values tensor from the SparseTensor input.
    sparse_input_shape: A `Tensor` of type `int64`.
      The shape tensor from the SparseTensor input.
    input_weights: A `Tensor` of type `float32`.
      For a dense input, input_weights[i] is the weight associated
      with input_data[i].  For sparse input, input_weights[i] is the weight
      associated with sparse_input_values[i].  Or in either case, if all the
      weights are 1, input_weights can be empty.  SampleInputs will reject inputs
      with weight less than Uniform([0,1)), so weights outside of that range may
      not be what you want.
    node_to_accumulator: A `Tensor` of type `int32`.
      For a fertile node i, node_to_accumulator[i] is the
      associated accumulator slot.  For non-fertile nodes, it is -1.
    leaves: A `Tensor` of type `int32`.
      `leaves[i]` is the leaf that the i-th input landed in, as
      calculated by CountExtremelyRandomStats.
    candidate_split_features: A `Tensor` of type `int32`.
      The current features for the candidate splits;
      `candidate_split_features[a][s]` is the index of the feature being
      considered by split s in accumulator slot a.
    candidate_split_thresholds: A `Tensor` of type `float32`.
      The current thresholds for the candidate splits;
      `candidate_split_thresholds[a][s]` is the threshold value being
      considered by split s in accumulator slot a.
    input_spec: A `string`.
    split_initializations_per_input: An `int`.
    split_sampling_random_seed: An `int`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (accumulators_to_update, new_split_feature_rows, new_split_threshold_rows).

    accumulators_to_update: A `Tensor` of type `int32`. A list of the accumulators to change in the
      candidate_split_features and candidate_split_thresholds tensors.
    new_split_feature_rows: A `Tensor` of type `int32`. The new values for the candidate_split_features
      tensor.  Intended to be used with
      `tf.scatter_update(candidate_split_features,
                         accumulators_to_update,
                         new_split_feature_rows)`
    new_split_threshold_rows: A `Tensor` of type `float32`. The new values for the candidate_split_thresholds
      tensor.  Intended to be used with
      `tf.scatter_update(candidate_split_thresholds,
                         accumulators_to_update,
                         new_split_feature_thresholds)`
  """
  result = _op_def_lib.apply_op("SampleInputs", input_data=input_data,
                                sparse_input_indices=sparse_input_indices,
                                sparse_input_values=sparse_input_values,
                                sparse_input_shape=sparse_input_shape,
                                input_weights=input_weights,
                                node_to_accumulator=node_to_accumulator,
                                leaves=leaves,
                                candidate_split_features=candidate_split_features,
                                candidate_split_thresholds=candidate_split_thresholds,
                                input_spec=input_spec,
                                split_initializations_per_input=split_initializations_per_input,
                                split_sampling_random_seed=split_sampling_random_seed,
                                name=name)
  return _SampleInputsOutput._make(result)


_ops.RegisterShape("SampleInputs")(None)

def scatter_add_ndim(input, indices, deltas, name=None):
  r"""  Add elements in deltas to mutable input according to indices.

    input: A N-dimensional float tensor to mutate.
    indices:= A 2-D int32 tensor. The size of dimension 0 is the number of
      deltas, the size of dimension 1 is the rank of the input.  `indices[i]`
      gives the coordinates of input that `deltas[i]` should add to.  If
      `indices[i]` does not fully specify a location (it has less indices than
      there are dimensions in `input`), it is assumed that they are start
      indices and that deltas contains enough values to fill in the remaining
      input dimensions.
    deltas: `deltas[i]` is the value to add to input at index indices[i][:]

  Args:
    input: A `Tensor` of type mutable `float32`.
    indices: A `Tensor` of type `int32`.
    deltas: A `Tensor` of type `float32`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  result = _op_def_lib.apply_op("ScatterAddNdim", input=input,
                                indices=indices, deltas=deltas, name=name)
  return result


_ops.RegisterShape("ScatterAddNdim")(None)

_top_n_insert_outputs = ["shortlist_ids", "update_ids", "update_scores"]
_TopNInsertOutput = _collections.namedtuple(
    "TopNInsert", _top_n_insert_outputs)


def top_n_insert(ids, scores, new_ids, new_scores, name=None):
  r"""  Outputs update Tensors for adding new_ids and new_scores to the shortlist.

    ids:= A 1-D int64 tensor containing the ids on the shortlist (except for
      ids[0], which is the current size of the shortlist.
    scores:= A 1-D float32 tensor containing the scores on the shortlist.
    new_ids:= A 1-D int64 tensor containing the new ids to add to the shortlist.
    shortlist_ids:= A 1-D int64 tensor containing the ids of the shortlist entries
      to update.  Intended to be used with
      tf.scatter_update(shortlist_scores, shortlist_ids, new_scores).
    update_ids:= A 1-D int64 tensor containing ...
    update_scores:= A 1-D float32 tensor containing ...

  Args:
    ids: A `Tensor` of type `int64`.
    scores: A `Tensor` of type `float32`.
    new_ids: A `Tensor` of type `int64`.
    new_scores: A `Tensor` of type `float32`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (shortlist_ids, update_ids, update_scores).

    shortlist_ids: A `Tensor` of type `int64`.
    update_ids: A `Tensor` of type `int64`.
    update_scores: A `Tensor` of type `float32`.
  """
  result = _op_def_lib.apply_op("TopNInsert", ids=ids, scores=scores,
                                new_ids=new_ids, new_scores=new_scores,
                                name=name)
  return _TopNInsertOutput._make(result)


_ops.RegisterShape("TopNInsert")(None)

_top_n_remove_outputs = ["shortlist_ids", "new_length"]
_TopNRemoveOutput = _collections.namedtuple(
    "TopNRemove", _top_n_remove_outputs)


def top_n_remove(ids, remove_ids, name=None):
  r"""  Remove ids from a shortlist.

    ids:= A 1-D int64 tensor containing the ids on the shortlist (except for
      ids[0], which is the current size of the shortlist.
    remove_ids:= A 1-D int64 tensor containing the ids to remove.
    shortlist_ids:= A 1-D int64 tensor containing the shortlist entries that
      need to be removed.
    new_length:= A length 1 1-D int64 tensor containing the new length of the
      shortlist.

  Args:
    ids: A `Tensor` of type `int64`.
    remove_ids: A `Tensor` of type `int64`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (shortlist_ids, new_length).

    shortlist_ids: A `Tensor` of type `int64`.
    new_length: A `Tensor` of type `int64`.
  """
  result = _op_def_lib.apply_op("TopNRemove", ids=ids, remove_ids=remove_ids,
                                name=name)
  return _TopNRemoveOutput._make(result)


_ops.RegisterShape("TopNRemove")(None)

def tree_predictions(input_data, sparse_input_indices, sparse_input_values,
                     sparse_input_shape, tree, tree_thresholds,
                     node_per_class_weights, input_spec, valid_leaf_threshold,
                     name=None):
  r"""  Returns the per-class probabilities for each input.

    input_spec: A serialized TensorForestDataSpec proto.
    input_data: The training batch's features as a 2-d tensor; `input_data[i][j]`
     gives the j-th feature of the i-th input.
    sparse_input_indices: The indices tensor from the SparseTensor input.
    sparse_input_values: The values tensor from the SparseTensor input.
    sparse_input_shape: The shape tensor from the SparseTensor input.
    tree:= A 2-d int32 tensor.  `tree[i][0]` gives the index of the left child
     of the i-th node, `tree[i][0] + 1` gives the index of the right child of
     the i-th node, and `tree[i][1]` gives the index of the feature used to
     split the i-th node.
    tree_thresholds: `tree_thresholds[i]` is the value used to split the i-th
     node.
    node_per_class_weights: `node_per_class_weights[n][c]` records how many
     training examples have class c and have ended up in node n.
    predictions: `predictions[i][j]` is the probability that input i is class j.
    valid_leaf_threshold: Minimum number of samples that have arrived to a leaf
      to be considered a valid leaf, otherwise use the parent.

  Args:
    input_data: A `Tensor` of type `float32`.
    sparse_input_indices: A `Tensor` of type `int64`.
    sparse_input_values: A `Tensor` of type `float32`.
    sparse_input_shape: A `Tensor` of type `int64`.
    tree: A `Tensor` of type `int32`.
    tree_thresholds: A `Tensor` of type `float32`.
    node_per_class_weights: A `Tensor` of type `float32`.
    input_spec: A `string`.
    valid_leaf_threshold: A `float`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32`.
  """
  result = _op_def_lib.apply_op("TreePredictions", input_data=input_data,
                                sparse_input_indices=sparse_input_indices,
                                sparse_input_values=sparse_input_values,
                                sparse_input_shape=sparse_input_shape,
                                tree=tree, tree_thresholds=tree_thresholds,
                                node_per_class_weights=node_per_class_weights,
                                input_spec=input_spec,
                                valid_leaf_threshold=valid_leaf_threshold,
                                name=name)
  return result


_ops.RegisterShape("TreePredictions")(None)

_update_fertile_slots_outputs = ["node_to_accumulator_map_updates",
                                "accumulator_to_node_map_updates",
                                "accumulators_cleared",
                                "accumulators_allocated"]
_UpdateFertileSlotsOutput = _collections.namedtuple(
    "UpdateFertileSlots", _update_fertile_slots_outputs)


def update_fertile_slots(finished, non_fertile_leaves,
                         non_fertile_leaf_scores, end_of_tree,
                         accumulator_sums, node_to_accumulator, stale_leaves,
                         node_sums, regression=None, name=None):
  r"""Updates accumulator slots to reflect finished or newly fertile nodes.

  Args:
    finished:  A 1-d int32 tensor containing the indices of fertile nodes that
      are ready to decide on a split.
    non_fertile_leaves:  A 1-d int32 tensor containing the indices of all the
      currently non-fertile leaves.  If there are free accumulator slots after
      deallocation, UpdateFertileSlots will consider these nodes (plus the ones
      in new_leaves) and potentially turn some of them fertile.
    non_fertile_leaf_scores: A `Tensor` of type `float32`.
      `non_fertile_leaf_scores[i]` is the splitting score
      of the non-fertile leaf `non_fertile_leaves[i]`.
    end_of_tree: A `Tensor` of type `int32`.
      The end of tree tensor from the previous training iteration, used
      with the finished input to calculate a list of new leaf indices created by
      GrowTree, which will be considered to become fertile if there are free
      slots.
    accumulator_sums: A `Tensor` of type `float32`.
      For classification, `accumulator_sums[a][c]` records how
      many training examples have class c and have ended up in the fertile node
      associated with accumulator slot a.  It has the total sum in entry 0 for
      convenience. For regression, it is the same except it contains the sum
      of the input labels that have been seen, and entry 0 contains the number
      of training examples that have been seen.
    node_to_accumulator: A `Tensor` of type `int32`.
      `node_to_accumulator[i]` is the accumulator slot used by
      fertile node i, or -1 if node i isn't fertile.
    stale_leaves: 
      A 1-d int32 tensor containing the indices of all leaves that
      have stopped accumulating statistics because they are too old.
    node_sums: A `Tensor` of type `float32`.
      `node_sums[n][c]` records how many
      training examples have class c and have ended up in node n.
    regression: An optional `bool`. Defaults to `False`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (node_to_accumulator_map_updates, accumulator_to_node_map_updates, accumulators_cleared, accumulators_allocated).

    node_to_accumulator_map_updates: A 2-d int32 tensor describing the changes
      that need to be applied to the node_to_accumulator map.  Intended to be used
      with
      `tf.scatter_update(node_to_accumulator,
                         node_to_accumulator_map_updates[0],
                         node_to_accumulator_map_updates[1])`.
    accumulator_to_node_map_updates: A 2-d int32 tensor describing the changes
      that need to be applied to the node_to_accumulator map.  Intended to be used
      with
      `tf.scatter_update(accumulator_to_node_map,
                         accumulator_to_node_map_updates[0],
                         accumulator_to_node_map_updates[1])`.
    accumulators_cleared: A 1-d int32 tensor containing the indices of all
      the accumulator slots that need to be cleared.
    accumulators_allocated: A 1-d int32 tensor containing the indices of all
      the accumulator slots that need to be allocated.
  """
  result = _op_def_lib.apply_op("UpdateFertileSlots", finished=finished,
                                non_fertile_leaves=non_fertile_leaves,
                                non_fertile_leaf_scores=non_fertile_leaf_scores,
                                end_of_tree=end_of_tree,
                                accumulator_sums=accumulator_sums,
                                node_to_accumulator=node_to_accumulator,
                                stale_leaves=stale_leaves,
                                node_sums=node_sums, regression=regression,
                                name=name)
  return _UpdateFertileSlotsOutput._make(result)


_ops.RegisterShape("UpdateFertileSlots")(None)
def _InitOpDefLibrary():
  op_list = _op_def_pb2.OpList()
  _text_format.Merge(_InitOpDefLibrary.op_list_ascii, op_list)
  _op_def_registry.register_op_list(op_list)
  op_def_lib = _op_def_library.OpDefLibrary()
  op_def_lib.add_op_list(op_list)
  return op_def_lib


_InitOpDefLibrary.op_list_ascii = """op {
  name: "BestSplits"
  input_arg {
    name: "finished_nodes"
    type: DT_INT32
  }
  input_arg {
    name: "node_to_accumulator"
    type: DT_INT32
  }
  input_arg {
    name: "split_sums"
    type: DT_FLOAT
  }
  input_arg {
    name: "split_squares"
    type: DT_FLOAT
  }
  input_arg {
    name: "accumulator_sums"
    type: DT_FLOAT
  }
  input_arg {
    name: "accumulator_sqaures"
    type: DT_FLOAT
  }
  output_arg {
    name: "split_indices"
    type: DT_INT32
  }
  attr {
    name: "regression"
    type: "bool"
    default_value {
      b: false
    }
  }
}
op {
  name: "CountExtremelyRandomStats"
  input_arg {
    name: "input_data"
    type: DT_FLOAT
  }
  input_arg {
    name: "sparse_input_indices"
    type: DT_INT64
  }
  input_arg {
    name: "sparse_input_values"
    type: DT_FLOAT
  }
  input_arg {
    name: "sparse_input_shape"
    type: DT_INT64
  }
  input_arg {
    name: "input_labels"
    type: DT_FLOAT
  }
  input_arg {
    name: "input_weights"
    type: DT_FLOAT
  }
  input_arg {
    name: "tree"
    type: DT_INT32
  }
  input_arg {
    name: "tree_thresholds"
    type: DT_FLOAT
  }
  input_arg {
    name: "node_to_accumulator"
    type: DT_INT32
  }
  input_arg {
    name: "candidate_split_features"
    type: DT_INT32
  }
  input_arg {
    name: "candidate_split_thresholds"
    type: DT_FLOAT
  }
  input_arg {
    name: "birth_epochs"
    type: DT_INT32
  }
  input_arg {
    name: "current_epoch"
    type: DT_INT32
  }
  output_arg {
    name: "pcw_node_sums_delta"
    type: DT_FLOAT
  }
  output_arg {
    name: "pcw_node_squares_delta"
    type: DT_FLOAT
  }
  output_arg {
    name: "pcw_splits_indices"
    type: DT_INT32
  }
  output_arg {
    name: "pcw_candidate_splits_sums_delta"
    type: DT_FLOAT
  }
  output_arg {
    name: "pcw_candidate_splits_squares_delta"
    type: DT_FLOAT
  }
  output_arg {
    name: "pcw_totals_indices"
    type: DT_INT32
  }
  output_arg {
    name: "pcw_totals_sums_delta"
    type: DT_FLOAT
  }
  output_arg {
    name: "pcw_totals_squares_delta"
    type: DT_FLOAT
  }
  output_arg {
    name: "leaves"
    type: DT_INT32
  }
  attr {
    name: "input_spec"
    type: "string"
  }
  attr {
    name: "num_classes"
    type: "int"
  }
  attr {
    name: "regression"
    type: "bool"
    default_value {
      b: false
    }
  }
}
op {
  name: "FinishedNodes"
  input_arg {
    name: "leaves"
    type: DT_INT32
  }
  input_arg {
    name: "node_to_accumulator"
    type: DT_INT32
  }
  input_arg {
    name: "split_sums"
    type: DT_FLOAT
  }
  input_arg {
    name: "split_squares"
    type: DT_FLOAT
  }
  input_arg {
    name: "accumulator_sums"
    type: DT_FLOAT
  }
  input_arg {
    name: "accumulator_squares"
    type: DT_FLOAT
  }
  input_arg {
    name: "birth_epochs"
    type: DT_INT32
  }
  input_arg {
    name: "current_epoch"
    type: DT_INT32
  }
  output_arg {
    name: "finished"
    type: DT_INT32
  }
  output_arg {
    name: "stale"
    type: DT_INT32
  }
  attr {
    name: "regression"
    type: "bool"
    default_value {
      b: false
    }
  }
  attr {
    name: "num_split_after_samples"
    type: "int"
  }
  attr {
    name: "min_split_samples"
    type: "int"
  }
  attr {
    name: "dominate_fraction"
    type: "float"
    default_value {
      f: 0.99
    }
  }
  attr {
    name: "dominate_method"
    type: "string"
    default_value {
      s: "bootstrap"
    }
    allowed_values {
      list {
        s: "none"
        s: "hoeffding"
        s: "bootstrap"
        s: "chebyshev"
      }
    }
  }
  attr {
    name: "random_seed"
    type: "int"
    default_value {
      i: 0
    }
  }
  attr {
    name: "check_dominates_every_samples"
    type: "int"
    default_value {
      i: 75
    }
  }
}
op {
  name: "GrowTree"
  input_arg {
    name: "end_of_tree"
    type: DT_INT32
  }
  input_arg {
    name: "node_to_accumulator"
    type: DT_INT32
  }
  input_arg {
    name: "finished_nodes"
    type: DT_INT32
  }
  input_arg {
    name: "best_splits"
    type: DT_INT32
  }
  input_arg {
    name: "candidate_split_features"
    type: DT_INT32
  }
  input_arg {
    name: "candidate_split_thresholds"
    type: DT_FLOAT
  }
  output_arg {
    name: "nodes_to_update"
    type: DT_INT32
  }
  output_arg {
    name: "tree_updates"
    type: DT_INT32
  }
  output_arg {
    name: "threshold_updates"
    type: DT_FLOAT
  }
  output_arg {
    name: "new_end_of_tree"
    type: DT_INT32
  }
}
op {
  name: "ReinterpretStringToFloat"
  input_arg {
    name: "input_data"
    type: DT_STRING
  }
  output_arg {
    name: "output_data"
    type: DT_FLOAT
  }
}
op {
  name: "SampleInputs"
  input_arg {
    name: "input_data"
    type: DT_FLOAT
  }
  input_arg {
    name: "sparse_input_indices"
    type: DT_INT64
  }
  input_arg {
    name: "sparse_input_values"
    type: DT_FLOAT
  }
  input_arg {
    name: "sparse_input_shape"
    type: DT_INT64
  }
  input_arg {
    name: "input_weights"
    type: DT_FLOAT
  }
  input_arg {
    name: "node_to_accumulator"
    type: DT_INT32
  }
  input_arg {
    name: "leaves"
    type: DT_INT32
  }
  input_arg {
    name: "candidate_split_features"
    type: DT_INT32
  }
  input_arg {
    name: "candidate_split_thresholds"
    type: DT_FLOAT
  }
  output_arg {
    name: "accumulators_to_update"
    type: DT_INT32
  }
  output_arg {
    name: "new_split_feature_rows"
    type: DT_INT32
  }
  output_arg {
    name: "new_split_threshold_rows"
    type: DT_FLOAT
  }
  attr {
    name: "input_spec"
    type: "string"
  }
  attr {
    name: "split_initializations_per_input"
    type: "int"
  }
  attr {
    name: "split_sampling_random_seed"
    type: "int"
  }
}
op {
  name: "ScatterAddNdim"
  input_arg {
    name: "input"
    type: DT_FLOAT
    is_ref: true
  }
  input_arg {
    name: "indices"
    type: DT_INT32
  }
  input_arg {
    name: "deltas"
    type: DT_FLOAT
  }
}
op {
  name: "TopNInsert"
  input_arg {
    name: "ids"
    type: DT_INT64
  }
  input_arg {
    name: "scores"
    type: DT_FLOAT
  }
  input_arg {
    name: "new_ids"
    type: DT_INT64
  }
  input_arg {
    name: "new_scores"
    type: DT_FLOAT
  }
  output_arg {
    name: "shortlist_ids"
    type: DT_INT64
  }
  output_arg {
    name: "update_ids"
    type: DT_INT64
  }
  output_arg {
    name: "update_scores"
    type: DT_FLOAT
  }
}
op {
  name: "TopNRemove"
  input_arg {
    name: "ids"
    type: DT_INT64
  }
  input_arg {
    name: "remove_ids"
    type: DT_INT64
  }
  output_arg {
    name: "shortlist_ids"
    type: DT_INT64
  }
  output_arg {
    name: "new_length"
    type: DT_INT64
  }
}
op {
  name: "TreePredictions"
  input_arg {
    name: "input_data"
    type: DT_FLOAT
  }
  input_arg {
    name: "sparse_input_indices"
    type: DT_INT64
  }
  input_arg {
    name: "sparse_input_values"
    type: DT_FLOAT
  }
  input_arg {
    name: "sparse_input_shape"
    type: DT_INT64
  }
  input_arg {
    name: "tree"
    type: DT_INT32
  }
  input_arg {
    name: "tree_thresholds"
    type: DT_FLOAT
  }
  input_arg {
    name: "node_per_class_weights"
    type: DT_FLOAT
  }
  output_arg {
    name: "predictions"
    type: DT_FLOAT
  }
  attr {
    name: "input_spec"
    type: "string"
  }
  attr {
    name: "valid_leaf_threshold"
    type: "float"
  }
}
op {
  name: "UpdateFertileSlots"
  input_arg {
    name: "finished"
    type: DT_INT32
  }
  input_arg {
    name: "non_fertile_leaves"
    type: DT_INT32
  }
  input_arg {
    name: "non_fertile_leaf_scores"
    type: DT_FLOAT
  }
  input_arg {
    name: "end_of_tree"
    type: DT_INT32
  }
  input_arg {
    name: "accumulator_sums"
    type: DT_FLOAT
  }
  input_arg {
    name: "node_to_accumulator"
    type: DT_INT32
  }
  input_arg {
    name: "stale_leaves"
    type: DT_INT32
  }
  input_arg {
    name: "node_sums"
    type: DT_FLOAT
  }
  output_arg {
    name: "node_to_accumulator_map_updates"
    type: DT_INT32
  }
  output_arg {
    name: "accumulator_to_node_map_updates"
    type: DT_INT32
  }
  output_arg {
    name: "accumulators_cleared"
    type: DT_INT32
  }
  output_arg {
    name: "accumulators_allocated"
    type: DT_INT32
  }
  attr {
    name: "regression"
    type: "bool"
    default_value {
      b: false
    }
  }
}
"""


_op_def_lib = _InitOpDefLibrary()
