"""Python wrappers around TensorFlow ops.

This file is MACHINE GENERATED! Do not edit.
Original C++ source file: beam_search_ops.cc
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


def gather_tree(step_ids, parent_ids, sequence_length, name=None):
  r"""Calculates the full beams from the per-step ids and parent beam ids.

  This op implements the following mathematical equations:

  ```python
  TODO(ebrevdo): fill in
  ```

  Args:
    step_ids: A `Tensor`. Must be one of the following types: `int32`.
      `[max_time, batch_size, beam_width]`.
    parent_ids: A `Tensor`. Must have the same type as `step_ids`.
      `[max_time, batch_size, beam_width]`.
    sequence_length: A `Tensor`. Must have the same type as `step_ids`.
      `[batch_size, beam_width]`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `step_ids`.
    `[max_time, batch_size, beam_width]`.
  """
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "GatherTree", step_ids=step_ids, parent_ids=parent_ids,
        sequence_length=sequence_length, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("T", _op.get_attr("T"))
  else:
    _attr_T, _inputs_T = _execute.args_to_matching_eager([step_ids, parent_ids, sequence_length], _ctx)
    (step_ids, parent_ids, sequence_length) = _inputs_T
    _attr_T = _attr_T.as_datatype_enum
    _inputs_flat = [step_ids, parent_ids, sequence_length]
    _attrs = ("T", _attr_T)
    _result = _execute.execute(b"GatherTree", 1, inputs=_inputs_flat,
                               attrs=_attrs, ctx=_ctx, name=name)
  _execute.record_gradient(
      "GatherTree", _inputs_flat, _attrs, _result, name)
  _result, = _result
  return _result

_ops.RegisterShape("GatherTree")(None)

def _InitOpDefLibrary(op_list_proto_bytes):
  op_list = _op_def_pb2.OpList()
  op_list.ParseFromString(op_list_proto_bytes)
  _op_def_registry.register_op_list(op_list)
  op_def_lib = _op_def_library.OpDefLibrary()
  op_def_lib.add_op_list(op_list)
  return op_def_lib
# op {
#   name: "GatherTree"
#   input_arg {
#     name: "step_ids"
#     type_attr: "T"
#   }
#   input_arg {
#     name: "parent_ids"
#     type_attr: "T"
#   }
#   input_arg {
#     name: "sequence_length"
#     type_attr: "T"
#   }
#   output_arg {
#     name: "beams"
#     type_attr: "T"
#   }
#   attr {
#     name: "T"
#     type: "type"
#     allowed_values {
#       list {
#         type: DT_INT32
#       }
#     }
#   }
# }
_op_def_lib = _InitOpDefLibrary(b"\n`\n\nGatherTree\022\r\n\010step_ids\"\001T\022\017\n\nparent_ids\"\001T\022\024\n\017sequence_length\"\001T\032\n\n\005beams\"\001T\"\020\n\001T\022\004type:\005\n\0032\001\003")
