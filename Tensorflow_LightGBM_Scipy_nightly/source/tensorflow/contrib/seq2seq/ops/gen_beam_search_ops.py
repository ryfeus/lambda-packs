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
  result = _op_def_lib.apply_op("GatherTree", step_ids=step_ids,
                                parent_ids=parent_ids,
                                sequence_length=sequence_length, name=name)
  return result


_ops.RegisterShape("GatherTree")(None)
def _InitOpDefLibrary():
  op_list = _op_def_pb2.OpList()
  _text_format.Merge(_InitOpDefLibrary.op_list_ascii, op_list)
  _op_def_registry.register_op_list(op_list)
  op_def_lib = _op_def_library.OpDefLibrary()
  op_def_lib.add_op_list(op_list)
  return op_def_lib


_InitOpDefLibrary.op_list_ascii = """op {
  name: "GatherTree"
  input_arg {
    name: "step_ids"
    type_attr: "T"
  }
  input_arg {
    name: "parent_ids"
    type_attr: "T"
  }
  input_arg {
    name: "sequence_length"
    type_attr: "T"
  }
  output_arg {
    name: "beams"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_INT32
      }
    }
  }
}
"""


_op_def_lib = _InitOpDefLibrary()
