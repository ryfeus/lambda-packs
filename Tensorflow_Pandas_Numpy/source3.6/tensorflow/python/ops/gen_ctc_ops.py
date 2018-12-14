"""Python wrappers around TensorFlow ops.

This file is MACHINE GENERATED! Do not edit.
Original C++ source file: ctc_ops.cc
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


__ctc_beam_search_decoder_outputs = ["decoded_indices", "decoded_values",
                                    "decoded_shape", "log_probability"]
_CTCBeamSearchDecoderOutput = _collections.namedtuple(
    "CTCBeamSearchDecoder", __ctc_beam_search_decoder_outputs)


def _ctc_beam_search_decoder(inputs, sequence_length, beam_width, top_paths, merge_repeated=True, name=None):
  r"""Performs beam search decoding on the logits given in input.

  A note about the attribute merge_repeated: For the beam search decoder,
  this means that if consecutive entries in a beam are the same, only
  the first of these is emitted.  That is, when the top path is "A B B B B",
  "A B" is returned if merge_repeated = True but "A B B B B" is
  returned if merge_repeated = False.

  Args:
    inputs: A `Tensor` of type `float32`.
      3-D, shape: `(max_time x batch_size x num_classes)`, the logits.
    sequence_length: A `Tensor` of type `int32`.
      A vector containing sequence lengths, size `(batch)`.
    beam_width: An `int` that is `>= 1`.
      A scalar >= 0 (beam search beam width).
    top_paths: An `int` that is `>= 1`.
      A scalar >= 0, <= beam_width (controls output size).
    merge_repeated: An optional `bool`. Defaults to `True`.
      If true, merge repeated classes in output.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (decoded_indices, decoded_values, decoded_shape, log_probability).

    decoded_indices: A list of `top_paths` `Tensor` objects with type `int64`. A list (length: top_paths) of indices matrices.  Matrix j,
      size `(total_decoded_outputs[j] x 2)`, has indices of a
      `SparseTensor<int64, 2>`.  The rows store: [batch, time].
    decoded_values: A list of `top_paths` `Tensor` objects with type `int64`. A list (length: top_paths) of values vectors.  Vector j,
      size `(length total_decoded_outputs[j])`, has the values of a
      `SparseTensor<int64, 2>`.  The vector stores the decoded classes for beam j.
    decoded_shape: A list of `top_paths` `Tensor` objects with type `int64`. A list (length: top_paths) of shape vector.  Vector j,
      size `(2)`, stores the shape of the decoded `SparseTensor[j]`.
      Its values are: `[batch_size, max_decoded_length[j]]`.
    log_probability: A `Tensor` of type `float32`. A matrix, shaped: `(batch_size x top_paths)`.  The
      sequence log-probabilities.
  """
  beam_width = _execute.make_int(beam_width, "beam_width")
  top_paths = _execute.make_int(top_paths, "top_paths")
  if merge_repeated is None:
    merge_repeated = True
  merge_repeated = _execute.make_bool(merge_repeated, "merge_repeated")
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "CTCBeamSearchDecoder", inputs=inputs,
        sequence_length=sequence_length, beam_width=beam_width,
        top_paths=top_paths, merge_repeated=merge_repeated, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("beam_width", _op.get_attr("beam_width"), "top_paths",
              _op.get_attr("top_paths"), "merge_repeated",
              _op.get_attr("merge_repeated"))
  else:
    inputs = _ops.convert_to_tensor(inputs, _dtypes.float32)
    sequence_length = _ops.convert_to_tensor(sequence_length, _dtypes.int32)
    _inputs_flat = [inputs, sequence_length]
    _attrs = ("beam_width", beam_width, "top_paths", top_paths,
              "merge_repeated", merge_repeated)
    _result = _execute.execute(b"CTCBeamSearchDecoder", top_paths + top_paths
                               + top_paths + 1, inputs=_inputs_flat,
                               attrs=_attrs, ctx=_ctx, name=name)
  _execute.record_gradient(
      "CTCBeamSearchDecoder", _inputs_flat, _attrs, _result, name)
  _result = [_result[:top_paths]] + _result[top_paths:]
  _result = _result[:1] + [_result[1:1 + top_paths]] + _result[1 + top_paths:]
  _result = _result[:2] + [_result[2:2 + top_paths]] + _result[2 + top_paths:]
  _result = _CTCBeamSearchDecoderOutput._make(_result)
  return _result


__ctc_greedy_decoder_outputs = ["decoded_indices", "decoded_values",
                               "decoded_shape", "log_probability"]
_CTCGreedyDecoderOutput = _collections.namedtuple(
    "CTCGreedyDecoder", __ctc_greedy_decoder_outputs)


def _ctc_greedy_decoder(inputs, sequence_length, merge_repeated=False, name=None):
  r"""Performs greedy decoding on the logits given in inputs.

  A note about the attribute merge_repeated: if enabled, when
  consecutive logits' maximum indices are the same, only the first of
  these is emitted.  Labeling the blank '*', the sequence "A B B * B B"
  becomes "A B B" if merge_repeated = True and "A B B B B" if
  merge_repeated = False.

  Regardless of the value of merge_repeated, if the maximum index of a given
  time and batch corresponds to the blank, index `(num_classes - 1)`, no new
  element is emitted.

  Args:
    inputs: A `Tensor` of type `float32`.
      3-D, shape: `(max_time x batch_size x num_classes)`, the logits.
    sequence_length: A `Tensor` of type `int32`.
      A vector containing sequence lengths, size `(batch_size)`.
    merge_repeated: An optional `bool`. Defaults to `False`.
      If True, merge repeated classes in output.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (decoded_indices, decoded_values, decoded_shape, log_probability).

    decoded_indices: A `Tensor` of type `int64`. Indices matrix, size `(total_decoded_outputs x 2)`,
      of a `SparseTensor<int64, 2>`.  The rows store: [batch, time].
    decoded_values: A `Tensor` of type `int64`. Values vector, size: `(total_decoded_outputs)`,
      of a `SparseTensor<int64, 2>`.  The vector stores the decoded classes.
    decoded_shape: A `Tensor` of type `int64`. Shape vector, size `(2)`, of the decoded SparseTensor.
      Values are: `[batch_size, max_decoded_length]`.
    log_probability: A `Tensor` of type `float32`. Matrix, size `(batch_size x 1)`, containing sequence
      log-probabilities.
  """
  if merge_repeated is None:
    merge_repeated = False
  merge_repeated = _execute.make_bool(merge_repeated, "merge_repeated")
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "CTCGreedyDecoder", inputs=inputs, sequence_length=sequence_length,
        merge_repeated=merge_repeated, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("merge_repeated", _op.get_attr("merge_repeated"))
  else:
    inputs = _ops.convert_to_tensor(inputs, _dtypes.float32)
    sequence_length = _ops.convert_to_tensor(sequence_length, _dtypes.int32)
    _inputs_flat = [inputs, sequence_length]
    _attrs = ("merge_repeated", merge_repeated)
    _result = _execute.execute(b"CTCGreedyDecoder", 4, inputs=_inputs_flat,
                               attrs=_attrs, ctx=_ctx, name=name)
  _execute.record_gradient(
      "CTCGreedyDecoder", _inputs_flat, _attrs, _result, name)
  _result = _CTCGreedyDecoderOutput._make(_result)
  return _result


__ctc_loss_outputs = ["loss", "gradient"]
_CTCLossOutput = _collections.namedtuple(
    "CTCLoss", __ctc_loss_outputs)


def _ctc_loss(inputs, labels_indices, labels_values, sequence_length, preprocess_collapse_repeated=False, ctc_merge_repeated=True, ignore_longer_outputs_than_inputs=False, name=None):
  r"""Calculates the CTC Loss (log probability) for each batch entry.  Also calculates

  the gradient.  This class performs the softmax operation for you, so inputs
  should be e.g. linear projections of outputs by an LSTM.

  Args:
    inputs: A `Tensor` of type `float32`.
      3-D, shape: `(max_time x batch_size x num_classes)`, the logits.
    labels_indices: A `Tensor` of type `int64`.
      The indices of a `SparseTensor<int32, 2>`.
      `labels_indices(i, :) == [b, t]` means `labels_values(i)` stores the id for
      `(batch b, time t)`.
    labels_values: A `Tensor` of type `int32`.
      The values (labels) associated with the given batch and time.
    sequence_length: A `Tensor` of type `int32`.
      A vector containing sequence lengths (batch).
    preprocess_collapse_repeated: An optional `bool`. Defaults to `False`.
      Scalar, if true then repeated labels are
      collapsed prior to the CTC calculation.
    ctc_merge_repeated: An optional `bool`. Defaults to `True`.
      Scalar.  If set to false, *during* CTC calculation
      repeated non-blank labels will not be merged and are interpreted as
      individual labels.  This is a simplified version of CTC.
    ignore_longer_outputs_than_inputs: An optional `bool`. Defaults to `False`.
      Scalar. If set to true, during CTC
      calculation, items that have longer output sequences than input sequences
      are skipped: they don't contribute to the loss term and have zero-gradient.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (loss, gradient).

    loss: A `Tensor` of type `float32`. A vector (batch) containing log-probabilities.
    gradient: A `Tensor` of type `float32`. The gradient of `loss`.  3-D, shape:
      `(max_time x batch_size x num_classes)`.
  """
  if preprocess_collapse_repeated is None:
    preprocess_collapse_repeated = False
  preprocess_collapse_repeated = _execute.make_bool(preprocess_collapse_repeated, "preprocess_collapse_repeated")
  if ctc_merge_repeated is None:
    ctc_merge_repeated = True
  ctc_merge_repeated = _execute.make_bool(ctc_merge_repeated, "ctc_merge_repeated")
  if ignore_longer_outputs_than_inputs is None:
    ignore_longer_outputs_than_inputs = False
  ignore_longer_outputs_than_inputs = _execute.make_bool(ignore_longer_outputs_than_inputs, "ignore_longer_outputs_than_inputs")
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "CTCLoss", inputs=inputs, labels_indices=labels_indices,
        labels_values=labels_values, sequence_length=sequence_length,
        preprocess_collapse_repeated=preprocess_collapse_repeated,
        ctc_merge_repeated=ctc_merge_repeated,
        ignore_longer_outputs_than_inputs=ignore_longer_outputs_than_inputs,
        name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("preprocess_collapse_repeated",
              _op.get_attr("preprocess_collapse_repeated"),
              "ctc_merge_repeated", _op.get_attr("ctc_merge_repeated"),
              "ignore_longer_outputs_than_inputs",
              _op.get_attr("ignore_longer_outputs_than_inputs"))
  else:
    inputs = _ops.convert_to_tensor(inputs, _dtypes.float32)
    labels_indices = _ops.convert_to_tensor(labels_indices, _dtypes.int64)
    labels_values = _ops.convert_to_tensor(labels_values, _dtypes.int32)
    sequence_length = _ops.convert_to_tensor(sequence_length, _dtypes.int32)
    _inputs_flat = [inputs, labels_indices, labels_values, sequence_length]
    _attrs = ("preprocess_collapse_repeated", preprocess_collapse_repeated,
              "ctc_merge_repeated", ctc_merge_repeated,
              "ignore_longer_outputs_than_inputs",
              ignore_longer_outputs_than_inputs)
    _result = _execute.execute(b"CTCLoss", 2, inputs=_inputs_flat,
                               attrs=_attrs, ctx=_ctx, name=name)
  _execute.record_gradient(
      "CTCLoss", _inputs_flat, _attrs, _result, name)
  _result = _CTCLossOutput._make(_result)
  return _result

def _InitOpDefLibrary(op_list_proto_bytes):
  op_list = _op_def_pb2.OpList()
  op_list.ParseFromString(op_list_proto_bytes)
  _op_def_registry.register_op_list(op_list)
  op_def_lib = _op_def_library.OpDefLibrary()
  op_def_lib.add_op_list(op_list)
  return op_def_lib
# op {
#   name: "CTCBeamSearchDecoder"
#   input_arg {
#     name: "inputs"
#     type: DT_FLOAT
#   }
#   input_arg {
#     name: "sequence_length"
#     type: DT_INT32
#   }
#   output_arg {
#     name: "decoded_indices"
#     type: DT_INT64
#     number_attr: "top_paths"
#   }
#   output_arg {
#     name: "decoded_values"
#     type: DT_INT64
#     number_attr: "top_paths"
#   }
#   output_arg {
#     name: "decoded_shape"
#     type: DT_INT64
#     number_attr: "top_paths"
#   }
#   output_arg {
#     name: "log_probability"
#     type: DT_FLOAT
#   }
#   attr {
#     name: "beam_width"
#     type: "int"
#     has_minimum: true
#     minimum: 1
#   }
#   attr {
#     name: "top_paths"
#     type: "int"
#     has_minimum: true
#     minimum: 1
#   }
#   attr {
#     name: "merge_repeated"
#     type: "bool"
#     default_value {
#       b: true
#     }
#   }
# }
# op {
#   name: "CTCGreedyDecoder"
#   input_arg {
#     name: "inputs"
#     type: DT_FLOAT
#   }
#   input_arg {
#     name: "sequence_length"
#     type: DT_INT32
#   }
#   output_arg {
#     name: "decoded_indices"
#     type: DT_INT64
#   }
#   output_arg {
#     name: "decoded_values"
#     type: DT_INT64
#   }
#   output_arg {
#     name: "decoded_shape"
#     type: DT_INT64
#   }
#   output_arg {
#     name: "log_probability"
#     type: DT_FLOAT
#   }
#   attr {
#     name: "merge_repeated"
#     type: "bool"
#     default_value {
#       b: false
#     }
#   }
# }
# op {
#   name: "CTCLoss"
#   input_arg {
#     name: "inputs"
#     type: DT_FLOAT
#   }
#   input_arg {
#     name: "labels_indices"
#     type: DT_INT64
#   }
#   input_arg {
#     name: "labels_values"
#     type: DT_INT32
#   }
#   input_arg {
#     name: "sequence_length"
#     type: DT_INT32
#   }
#   output_arg {
#     name: "loss"
#     type: DT_FLOAT
#   }
#   output_arg {
#     name: "gradient"
#     type: DT_FLOAT
#   }
#   attr {
#     name: "preprocess_collapse_repeated"
#     type: "bool"
#     default_value {
#       b: false
#     }
#   }
#   attr {
#     name: "ctc_merge_repeated"
#     type: "bool"
#     default_value {
#       b: true
#     }
#   }
#   attr {
#     name: "ignore_longer_outputs_than_inputs"
#     type: "bool"
#     default_value {
#       b: false
#     }
#   }
# }
_op_def_lib = _InitOpDefLibrary(b"\n\362\001\n\024CTCBeamSearchDecoder\022\n\n\006inputs\030\001\022\023\n\017sequence_length\030\003\032\036\n\017decoded_indices\030\t*\ttop_paths\032\035\n\016decoded_values\030\t*\ttop_paths\032\034\n\rdecoded_shape\030\t*\ttop_paths\032\023\n\017log_probability\030\001\"\025\n\nbeam_width\022\003int(\0010\001\"\024\n\ttop_paths\022\003int(\0010\001\"\032\n\016merge_repeated\022\004bool\032\002(\001\n\240\001\n\020CTCGreedyDecoder\022\n\n\006inputs\030\001\022\023\n\017sequence_length\030\003\032\023\n\017decoded_indices\030\t\032\022\n\016decoded_values\030\t\032\021\n\rdecoded_shape\030\t\032\023\n\017log_probability\030\001\"\032\n\016merge_repeated\022\004bool\032\002(\000\n\342\001\n\007CTCLoss\022\n\n\006inputs\030\001\022\022\n\016labels_indices\030\t\022\021\n\rlabels_values\030\003\022\023\n\017sequence_length\030\003\032\010\n\004loss\030\001\032\014\n\010gradient\030\001\"(\n\034preprocess_collapse_repeated\022\004bool\032\002(\000\"\036\n\022ctc_merge_repeated\022\004bool\032\002(\001\"-\n!ignore_longer_outputs_than_inputs\022\004bool\032\002(\000")
