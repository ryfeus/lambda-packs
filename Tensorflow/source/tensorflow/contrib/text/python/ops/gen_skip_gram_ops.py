"""Python wrappers around TensorFlow ops.

This file is MACHINE GENERATED! Do not edit.
Original C++ source file: gen_skip_gram_ops.cc
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


_skip_gram_generate_candidates_outputs = ["tokens", "labels"]
_SkipGramGenerateCandidatesOutput = _collections.namedtuple(
    "SkipGramGenerateCandidates", _skip_gram_generate_candidates_outputs)


def skip_gram_generate_candidates(input_tensor, min_skips, max_skips, start, limit, emit_self_as_target, seed=0, seed2=0, name=None):
  r"""Generates skip-gram token and label paired Tensors from the input tensor.

  See docs for the public-facing skip_gram_sample() Python op for more details.

  Args:
    input_tensor: A `Tensor`.
    min_skips: A `Tensor` of type `int32`.
    max_skips: A `Tensor` of type `int32`.
    start: A `Tensor` of type `int32`.
    limit: A `Tensor` of type `int32`.
    emit_self_as_target: A `Tensor` of type `bool`.
    seed: An optional `int`. Defaults to `0`.
    seed2: An optional `int`. Defaults to `0`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (tokens, labels).

    tokens: A `Tensor`. Has the same type as `input_tensor`.
    labels: A `Tensor`. Has the same type as `input_tensor`.
  """
  if seed is None:
    seed = 0
  seed = _execute.make_int(seed, "seed")
  if seed2 is None:
    seed2 = 0
  seed2 = _execute.make_int(seed2, "seed2")
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "SkipGramGenerateCandidates", input_tensor=input_tensor,
        min_skips=min_skips, max_skips=max_skips, start=start, limit=limit,
        emit_self_as_target=emit_self_as_target, seed=seed, seed2=seed2,
        name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("T", _op.get_attr("T"), "seed", _op.get_attr("seed"), "seed2",
              _op.get_attr("seed2"))
  else:
    _attr_T, (input_tensor,) = _execute.args_to_matching_eager([input_tensor], _ctx)
    _attr_T = _attr_T.as_datatype_enum
    min_skips = _ops.convert_to_tensor(min_skips, _dtypes.int32)
    max_skips = _ops.convert_to_tensor(max_skips, _dtypes.int32)
    start = _ops.convert_to_tensor(start, _dtypes.int32)
    limit = _ops.convert_to_tensor(limit, _dtypes.int32)
    emit_self_as_target = _ops.convert_to_tensor(emit_self_as_target, _dtypes.bool)
    _inputs_flat = [input_tensor, min_skips, max_skips, start, limit, emit_self_as_target]
    _attrs = ("T", _attr_T, "seed", seed, "seed2", seed2)
    _result = _execute.execute(b"SkipGramGenerateCandidates", 2,
                               inputs=_inputs_flat, attrs=_attrs, ctx=_ctx,
                               name=name)
  _execute.record_gradient(
      "SkipGramGenerateCandidates", _inputs_flat, _attrs, _result, name)
  _result = _SkipGramGenerateCandidatesOutput._make(_result)
  return _result

_ops.RegisterShape("SkipGramGenerateCandidates")(None)

def _InitOpDefLibrary(op_list_proto_bytes):
  op_list = _op_def_pb2.OpList()
  op_list.ParseFromString(op_list_proto_bytes)
  _op_def_registry.register_op_list(op_list)
  op_def_lib = _op_def_library.OpDefLibrary()
  op_def_lib.add_op_list(op_list)
  return op_def_lib
# op {
#   name: "SkipGramGenerateCandidates"
#   input_arg {
#     name: "input_tensor"
#     type_attr: "T"
#   }
#   input_arg {
#     name: "min_skips"
#     type: DT_INT32
#   }
#   input_arg {
#     name: "max_skips"
#     type: DT_INT32
#   }
#   input_arg {
#     name: "start"
#     type: DT_INT32
#   }
#   input_arg {
#     name: "limit"
#     type: DT_INT32
#   }
#   input_arg {
#     name: "emit_self_as_target"
#     type: DT_BOOL
#   }
#   output_arg {
#     name: "tokens"
#     type_attr: "T"
#   }
#   output_arg {
#     name: "labels"
#     type_attr: "T"
#   }
#   attr {
#     name: "T"
#     type: "type"
#   }
#   attr {
#     name: "seed"
#     type: "int"
#     default_value {
#       i: 0
#     }
#   }
#   attr {
#     name: "seed2"
#     type: "int"
#     default_value {
#       i: 0
#     }
#   }
#   is_stateful: true
# }
_op_def_lib = _InitOpDefLibrary(b"\n\307\001\n\032SkipGramGenerateCandidates\022\021\n\014input_tensor\"\001T\022\r\n\tmin_skips\030\003\022\r\n\tmax_skips\030\003\022\t\n\005start\030\003\022\t\n\005limit\030\003\022\027\n\023emit_self_as_target\030\n\032\013\n\006tokens\"\001T\032\013\n\006labels\"\001T\"\t\n\001T\022\004type\"\017\n\004seed\022\003int\032\002\030\000\"\020\n\005seed2\022\003int\032\002\030\000\210\001\001")
