"""Python wrappers around TensorFlow ops.

This file is MACHINE GENERATED! Do not edit.
Original C++ source file: control_flow_ops.cc
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


def abort(error_msg="", exit_without_error=False, name=None):
  r"""Raise a exception to abort the process when called.

  If exit_without_error is true, the process will exit normally,
  otherwise it will exit with a SIGABORT signal.

  Returns nothing but an exception.

  Args:
    error_msg: An optional `string`. Defaults to `""`.
      A string which is the message associated with the exception.
    exit_without_error: An optional `bool`. Defaults to `False`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  if error_msg is None:
    error_msg = ""
  error_msg = _execute.make_str(error_msg, "error_msg")
  if exit_without_error is None:
    exit_without_error = False
  exit_without_error = _execute.make_bool(exit_without_error, "exit_without_error")
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "Abort", error_msg=error_msg, exit_without_error=exit_without_error,
        name=name)
    return _op
  else:
    _inputs_flat = []
    _attrs = ("error_msg", error_msg, "exit_without_error",
              exit_without_error)
    _result = _execute.execute(b"Abort", 0, inputs=_inputs_flat, attrs=_attrs,
                               ctx=_ctx, name=name)
  return _result


def control_trigger(name=None):
  r"""Does nothing. Serves as a control trigger for scheduling.

  Only useful as a placeholder for control edges.

  Args:
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "ControlTrigger", name=name)
    return _op
  else:
    _inputs_flat = []
    _attrs = None
    _result = _execute.execute(b"ControlTrigger", 0, inputs=_inputs_flat,
                               attrs=_attrs, ctx=_ctx, name=name)
  return _result


def enter(data, frame_name, is_constant=False, parallel_iterations=10, name=None):
  r"""Creates or finds a child frame, and makes `data` available to the child frame.

  This op is used together with `Exit` to create loops in the graph.
  The unique `frame_name` is used by the `Executor` to identify frames. If
  `is_constant` is true, `output` is a constant in the child frame; otherwise
  it may be changed in the child frame. At most `parallel_iterations` iterations
  are run in parallel in the child frame.

  Args:
    data: A `Tensor`. The tensor to be made available to the child frame.
    frame_name: A `string`. The name of the child frame.
    is_constant: An optional `bool`. Defaults to `False`.
      If true, the output is constant within the child frame.
    parallel_iterations: An optional `int`. Defaults to `10`.
      The number of iterations allowed to run in parallel.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `data`. The same tensor as `data`.
  """
  frame_name = _execute.make_str(frame_name, "frame_name")
  if is_constant is None:
    is_constant = False
  is_constant = _execute.make_bool(is_constant, "is_constant")
  if parallel_iterations is None:
    parallel_iterations = 10
  parallel_iterations = _execute.make_int(parallel_iterations, "parallel_iterations")
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "Enter", data=data, frame_name=frame_name, is_constant=is_constant,
        parallel_iterations=parallel_iterations, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("T", _op.get_attr("T"), "frame_name",
              _op.get_attr("frame_name"), "is_constant",
              _op.get_attr("is_constant"), "parallel_iterations",
              _op.get_attr("parallel_iterations"))
  else:
    _attr_T, (data,) = _execute.args_to_matching_eager([data], _ctx)
    _attr_T = _attr_T.as_datatype_enum
    _inputs_flat = [data]
    _attrs = ("T", _attr_T, "frame_name", frame_name, "is_constant",
              is_constant, "parallel_iterations", parallel_iterations)
    _result = _execute.execute(b"Enter", 1, inputs=_inputs_flat, attrs=_attrs,
                               ctx=_ctx, name=name)
  _execute.record_gradient(
      "Enter", _inputs_flat, _attrs, _result, name)
  _result, = _result
  return _result


def _exit(data, name=None):
  r"""Exits the current frame to its parent frame.

  Exit makes its input `data` available to the parent frame.

  Args:
    data: A `Tensor`. The tensor to be made available to the parent frame.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `data`. The same tensor as `data`.
  """
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "Exit", data=data, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("T", _op.get_attr("T"))
  else:
    _attr_T, (data,) = _execute.args_to_matching_eager([data], _ctx)
    _attr_T = _attr_T.as_datatype_enum
    _inputs_flat = [data]
    _attrs = ("T", _attr_T)
    _result = _execute.execute(b"Exit", 1, inputs=_inputs_flat, attrs=_attrs,
                               ctx=_ctx, name=name)
  _execute.record_gradient(
      "Exit", _inputs_flat, _attrs, _result, name)
  _result, = _result
  return _result


def loop_cond(input, name=None):
  r"""Forwards the input to the output.

  This operator represents the loop termination condition used by the
  "pivot" switches of a loop.

  Args:
    input: A `Tensor` of type `bool`.
      A boolean scalar, representing the branch predicate of the Switch op.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `bool`. The same tensor as `input`.
  """
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "LoopCond", input=input, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = None
  else:
    input = _ops.convert_to_tensor(input, _dtypes.bool)
    _inputs_flat = [input]
    _attrs = None
    _result = _execute.execute(b"LoopCond", 1, inputs=_inputs_flat,
                               attrs=_attrs, ctx=_ctx, name=name)
  _execute.record_gradient(
      "LoopCond", _inputs_flat, _attrs, _result, name)
  _result, = _result
  return _result


__merge_outputs = ["output", "value_index"]
_MergeOutput = _collections.namedtuple(
    "Merge", __merge_outputs)


def _merge(inputs, name=None):
  r"""Forwards the value of an available tensor from `inputs` to `output`.

  `Merge` waits for at least one of the tensors in `inputs` to become available.
  It is usually combined with `Switch` to implement branching.

  `Merge` forwards the first tensor to become available to `output`, and sets
  `value_index` to its index in `inputs`.

  Args:
    inputs: A list of at least 1 `Tensor` objects with the same type.
      The input tensors, exactly one of which will become available.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output, value_index).

    output: A `Tensor`. Has the same type as `inputs`. Will be set to the available input tensor.
    value_index: A `Tensor` of type `int32`. The index of the chosen input tensor in `inputs`.
  """
  if not isinstance(inputs, (list, tuple)):
    raise TypeError(
        "Expected list for 'inputs' argument to "
        "'merge' Op, not %r." % inputs)
  _attr_N = len(inputs)
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "Merge", inputs=inputs, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("T", _op.get_attr("T"), "N", _op.get_attr("N"))
  else:
    _attr_T, inputs = _execute.args_to_matching_eager(list(inputs), _ctx)
    _attr_T = _attr_T.as_datatype_enum
    _inputs_flat = list(inputs)
    _attrs = ("T", _attr_T, "N", _attr_N)
    _result = _execute.execute(b"Merge", 2, inputs=_inputs_flat, attrs=_attrs,
                               ctx=_ctx, name=name)
  _execute.record_gradient(
      "Merge", _inputs_flat, _attrs, _result, name)
  _result = _MergeOutput._make(_result)
  return _result


def next_iteration(data, name=None):
  r"""Makes its input available to the next iteration.

  Args:
    data: A `Tensor`. The tensor to be made available to the next iteration.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `data`. The same tensor as `data`.
  """
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "NextIteration", data=data, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("T", _op.get_attr("T"))
  else:
    _attr_T, (data,) = _execute.args_to_matching_eager([data], _ctx)
    _attr_T = _attr_T.as_datatype_enum
    _inputs_flat = [data]
    _attrs = ("T", _attr_T)
    _result = _execute.execute(b"NextIteration", 1, inputs=_inputs_flat,
                               attrs=_attrs, ctx=_ctx, name=name)
  _execute.record_gradient(
      "NextIteration", _inputs_flat, _attrs, _result, name)
  _result, = _result
  return _result


def no_op(name=None):
  r"""Does nothing. Only useful as a placeholder for control edges.

  Args:
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "NoOp", name=name)
    return _op
  else:
    _inputs_flat = []
    _attrs = None
    _result = _execute.execute(b"NoOp", 0, inputs=_inputs_flat, attrs=_attrs,
                               ctx=_ctx, name=name)
  return _result


def ref_enter(data, frame_name, is_constant=False, parallel_iterations=10, name=None):
  r"""Creates or finds a child frame, and makes `data` available to the child frame.

  The unique `frame_name` is used by the `Executor` to identify frames. If
  `is_constant` is true, `output` is a constant in the child frame; otherwise
  it may be changed in the child frame. At most `parallel_iterations` iterations
  are run in parallel in the child frame.

  Args:
    data: A mutable `Tensor`.
      The tensor to be made available to the child frame.
    frame_name: A `string`. The name of the child frame.
    is_constant: An optional `bool`. Defaults to `False`.
      If true, the output is constant within the child frame.
    parallel_iterations: An optional `int`. Defaults to `10`.
      The number of iterations allowed to run in parallel.
    name: A name for the operation (optional).

  Returns:
    A mutable `Tensor`. Has the same type as `data`.
    The same tensor as `data`.
  """
  frame_name = _execute.make_str(frame_name, "frame_name")
  if is_constant is None:
    is_constant = False
  is_constant = _execute.make_bool(is_constant, "is_constant")
  if parallel_iterations is None:
    parallel_iterations = 10
  parallel_iterations = _execute.make_int(parallel_iterations, "parallel_iterations")
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "RefEnter", data=data, frame_name=frame_name, is_constant=is_constant,
        parallel_iterations=parallel_iterations, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("T", _op.get_attr("T"), "frame_name",
              _op.get_attr("frame_name"), "is_constant",
              _op.get_attr("is_constant"), "parallel_iterations",
              _op.get_attr("parallel_iterations"))
  else:
    raise RuntimeError(
        "ref_enter op does not support eager execution. Arg 'output'' is a ref.")
  _execute.record_gradient(
      "RefEnter", _inputs_flat, _attrs, _result, name)
  _result, = _result
  return _result


def _ref_exit(data, name=None):
  r"""Exits the current frame to its parent frame.

  Exit makes its input `data` available to the parent frame.

  Args:
    data: A mutable `Tensor`.
      The tensor to be made available to the parent frame.
    name: A name for the operation (optional).

  Returns:
    A mutable `Tensor`. Has the same type as `data`.
    The same tensor as `data`.
  """
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "RefExit", data=data, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("T", _op.get_attr("T"))
  else:
    raise RuntimeError(
        "ref_exit op does not support eager execution. Arg 'output'' is a ref.")
  _execute.record_gradient(
      "RefExit", _inputs_flat, _attrs, _result, name)
  _result, = _result
  return _result


__ref_merge_outputs = ["output", "value_index"]
_RefMergeOutput = _collections.namedtuple(
    "RefMerge", __ref_merge_outputs)


def _ref_merge(inputs, name=None):
  r"""Forwards the value of an available tensor from `inputs` to `output`.

  `Merge` waits for at least one of the tensors in `inputs` to become available.
  It is usually combined with `Switch` to implement branching.

  `Merge` forwards the first tensor for become available to `output`, and sets
  `value_index` to its index in `inputs`.

  Args:
    inputs: A list of at least 1 mutable `Tensor` objects with the same type.
      The input tensors, exactly one of which will become available.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output, value_index).

    output: A mutable `Tensor`. Has the same type as `inputs`. Will be set to the available input tensor.
    value_index: A `Tensor` of type `int32`. The index of the chosen input tensor in `inputs`.
  """
  if not isinstance(inputs, (list, tuple)):
    raise TypeError(
        "Expected list for 'inputs' argument to "
        "'ref_merge' Op, not %r." % inputs)
  _attr_N = len(inputs)
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "RefMerge", inputs=inputs, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("T", _op.get_attr("T"), "N", _op.get_attr("N"))
  else:
    raise RuntimeError(
        "ref_merge op does not support eager execution. Arg 'output'' is a ref.")
  _execute.record_gradient(
      "RefMerge", _inputs_flat, _attrs, _result, name)
  _result = _RefMergeOutput._make(_result)
  return _result


def ref_next_iteration(data, name=None):
  r"""Makes its input available to the next iteration.

  Args:
    data: A mutable `Tensor`.
      The tensor to be made available to the next iteration.
    name: A name for the operation (optional).

  Returns:
    A mutable `Tensor`. Has the same type as `data`.
    The same tensor as `data`.
  """
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "RefNextIteration", data=data, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("T", _op.get_attr("T"))
  else:
    raise RuntimeError(
        "ref_next_iteration op does not support eager execution. Arg 'output'' is a ref.")
  _execute.record_gradient(
      "RefNextIteration", _inputs_flat, _attrs, _result, name)
  _result, = _result
  return _result


def ref_select(index, inputs, name=None):
  r"""Forwards the `index`th element of `inputs` to `output`.

  Args:
    index: A `Tensor` of type `int32`.
      A scalar that determines the input that gets selected.
    inputs: A list of at least 1 mutable `Tensor` objects with the same type.
      A list of ref tensors, one of which will be forwarded to `output`.
    name: A name for the operation (optional).

  Returns:
    A mutable `Tensor`. Has the same type as `inputs`. The forwarded tensor.
  """
  if not isinstance(inputs, (list, tuple)):
    raise TypeError(
        "Expected list for 'inputs' argument to "
        "'ref_select' Op, not %r." % inputs)
  _attr_N = len(inputs)
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "RefSelect", index=index, inputs=inputs, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("T", _op.get_attr("T"), "N", _op.get_attr("N"))
  else:
    raise RuntimeError(
        "ref_select op does not support eager execution. Arg 'output'' is a ref.")
  _execute.record_gradient(
      "RefSelect", _inputs_flat, _attrs, _result, name)
  _result, = _result
  return _result


_ref_switch_outputs = ["output_false", "output_true"]
_RefSwitchOutput = _collections.namedtuple(
    "RefSwitch", _ref_switch_outputs)


def ref_switch(data, pred, name=None):
  r"""Forwards the ref tensor `data` to the output port determined by `pred`.

  If `pred` is true, the `data` input is forwarded to `output_true`. Otherwise,
  the data goes to `output_false`.

  See also `Switch` and `Merge`.

  Args:
    data: A mutable `Tensor`.
      The ref tensor to be forwarded to the appropriate output.
    pred: A `Tensor` of type `bool`.
      A scalar that specifies which output port will receive data.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output_false, output_true).

    output_false: A mutable `Tensor`. Has the same type as `data`. If `pred` is false, data will be forwarded to this output.
    output_true: A mutable `Tensor`. Has the same type as `data`. If `pred` is true, data will be forwarded to this output.
  """
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "RefSwitch", data=data, pred=pred, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("T", _op.get_attr("T"))
  else:
    raise RuntimeError(
        "ref_switch op does not support eager execution. Arg 'output_true'' is a ref.")
  _execute.record_gradient(
      "RefSwitch", _inputs_flat, _attrs, _result, name)
  _result = _RefSwitchOutput._make(_result)
  return _result


__switch_outputs = ["output_false", "output_true"]
_SwitchOutput = _collections.namedtuple(
    "Switch", __switch_outputs)


def _switch(data, pred, name=None):
  r"""Forwards `data` to the output port determined by `pred`.

  If `pred` is true, the `data` input is forwarded to `output_true`. Otherwise,
  the data goes to `output_false`.

  See also `RefSwitch` and `Merge`.

  Args:
    data: A `Tensor`. The tensor to be forwarded to the appropriate output.
    pred: A `Tensor` of type `bool`.
      A scalar that specifies which output port will receive data.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output_false, output_true).

    output_false: A `Tensor`. Has the same type as `data`. If `pred` is false, data will be forwarded to this output.
    output_true: A `Tensor`. Has the same type as `data`. If `pred` is true, data will be forwarded to this output.
  """
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "Switch", data=data, pred=pred, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("T", _op.get_attr("T"))
  else:
    _attr_T, (data,) = _execute.args_to_matching_eager([data], _ctx)
    _attr_T = _attr_T.as_datatype_enum
    pred = _ops.convert_to_tensor(pred, _dtypes.bool)
    _inputs_flat = [data, pred]
    _attrs = ("T", _attr_T)
    _result = _execute.execute(b"Switch", 2, inputs=_inputs_flat,
                               attrs=_attrs, ctx=_ctx, name=name)
  _execute.record_gradient(
      "Switch", _inputs_flat, _attrs, _result, name)
  _result = _SwitchOutput._make(_result)
  return _result

def _InitOpDefLibrary(op_list_proto_bytes):
  op_list = _op_def_pb2.OpList()
  op_list.ParseFromString(op_list_proto_bytes)
  _op_def_registry.register_op_list(op_list)
  op_def_lib = _op_def_library.OpDefLibrary()
  op_def_lib.add_op_list(op_list)
  return op_def_lib
# op {
#   name: "Abort"
#   attr {
#     name: "error_msg"
#     type: "string"
#     default_value {
#       s: ""
#     }
#   }
#   attr {
#     name: "exit_without_error"
#     type: "bool"
#     default_value {
#       b: false
#     }
#   }
# }
# op {
#   name: "ControlTrigger"
# }
# op {
#   name: "Enter"
#   input_arg {
#     name: "data"
#     type_attr: "T"
#   }
#   output_arg {
#     name: "output"
#     type_attr: "T"
#   }
#   attr {
#     name: "T"
#     type: "type"
#   }
#   attr {
#     name: "frame_name"
#     type: "string"
#   }
#   attr {
#     name: "is_constant"
#     type: "bool"
#     default_value {
#       b: false
#     }
#   }
#   attr {
#     name: "parallel_iterations"
#     type: "int"
#     default_value {
#       i: 10
#     }
#   }
# }
# op {
#   name: "Exit"
#   input_arg {
#     name: "data"
#     type_attr: "T"
#   }
#   output_arg {
#     name: "output"
#     type_attr: "T"
#   }
#   attr {
#     name: "T"
#     type: "type"
#   }
# }
# op {
#   name: "LoopCond"
#   input_arg {
#     name: "input"
#     type: DT_BOOL
#   }
#   output_arg {
#     name: "output"
#     type: DT_BOOL
#   }
# }
# op {
#   name: "Merge"
#   input_arg {
#     name: "inputs"
#     type_attr: "T"
#     number_attr: "N"
#   }
#   output_arg {
#     name: "output"
#     type_attr: "T"
#   }
#   output_arg {
#     name: "value_index"
#     type: DT_INT32
#   }
#   attr {
#     name: "T"
#     type: "type"
#   }
#   attr {
#     name: "N"
#     type: "int"
#     has_minimum: true
#     minimum: 1
#   }
# }
# op {
#   name: "NextIteration"
#   input_arg {
#     name: "data"
#     type_attr: "T"
#   }
#   output_arg {
#     name: "output"
#     type_attr: "T"
#   }
#   attr {
#     name: "T"
#     type: "type"
#   }
# }
# op {
#   name: "NoOp"
# }
# op {
#   name: "RefEnter"
#   input_arg {
#     name: "data"
#     type_attr: "T"
#     is_ref: true
#   }
#   output_arg {
#     name: "output"
#     type_attr: "T"
#     is_ref: true
#   }
#   attr {
#     name: "T"
#     type: "type"
#   }
#   attr {
#     name: "frame_name"
#     type: "string"
#   }
#   attr {
#     name: "is_constant"
#     type: "bool"
#     default_value {
#       b: false
#     }
#   }
#   attr {
#     name: "parallel_iterations"
#     type: "int"
#     default_value {
#       i: 10
#     }
#   }
# }
# op {
#   name: "RefExit"
#   input_arg {
#     name: "data"
#     type_attr: "T"
#     is_ref: true
#   }
#   output_arg {
#     name: "output"
#     type_attr: "T"
#     is_ref: true
#   }
#   attr {
#     name: "T"
#     type: "type"
#   }
# }
# op {
#   name: "RefMerge"
#   input_arg {
#     name: "inputs"
#     type_attr: "T"
#     number_attr: "N"
#     is_ref: true
#   }
#   output_arg {
#     name: "output"
#     type_attr: "T"
#     is_ref: true
#   }
#   output_arg {
#     name: "value_index"
#     type: DT_INT32
#   }
#   attr {
#     name: "T"
#     type: "type"
#   }
#   attr {
#     name: "N"
#     type: "int"
#     has_minimum: true
#     minimum: 1
#   }
# }
# op {
#   name: "RefNextIteration"
#   input_arg {
#     name: "data"
#     type_attr: "T"
#     is_ref: true
#   }
#   output_arg {
#     name: "output"
#     type_attr: "T"
#     is_ref: true
#   }
#   attr {
#     name: "T"
#     type: "type"
#   }
# }
# op {
#   name: "RefSelect"
#   input_arg {
#     name: "index"
#     type: DT_INT32
#   }
#   input_arg {
#     name: "inputs"
#     type_attr: "T"
#     number_attr: "N"
#     is_ref: true
#   }
#   output_arg {
#     name: "output"
#     type_attr: "T"
#     is_ref: true
#   }
#   attr {
#     name: "T"
#     type: "type"
#   }
#   attr {
#     name: "N"
#     type: "int"
#     has_minimum: true
#     minimum: 1
#   }
# }
# op {
#   name: "RefSwitch"
#   input_arg {
#     name: "data"
#     type_attr: "T"
#     is_ref: true
#   }
#   input_arg {
#     name: "pred"
#     type: DT_BOOL
#   }
#   output_arg {
#     name: "output_false"
#     type_attr: "T"
#     is_ref: true
#   }
#   output_arg {
#     name: "output_true"
#     type_attr: "T"
#     is_ref: true
#   }
#   attr {
#     name: "T"
#     type: "type"
#   }
#   allows_uninitialized_input: true
# }
# op {
#   name: "Switch"
#   input_arg {
#     name: "data"
#     type_attr: "T"
#   }
#   input_arg {
#     name: "pred"
#     type: DT_BOOL
#   }
#   output_arg {
#     name: "output_false"
#     type_attr: "T"
#   }
#   output_arg {
#     name: "output_true"
#     type_attr: "T"
#   }
#   attr {
#     name: "T"
#     type: "type"
#   }
# }
_op_def_lib = _InitOpDefLibrary(b"\n@\n\005Abort\"\027\n\terror_msg\022\006string\032\002\022\000\"\036\n\022exit_without_error\022\004bool\032\002(\000\n\020\n\016ControlTrigger\ny\n\005Enter\022\t\n\004data\"\001T\032\013\n\006output\"\001T\"\t\n\001T\022\004type\"\024\n\nframe_name\022\006string\"\027\n\013is_constant\022\004bool\032\002(\000\"\036\n\023parallel_iterations\022\003int\032\002\030\n\n)\n\004Exit\022\t\n\004data\"\001T\032\013\n\006output\"\001T\"\t\n\001T\022\004type\n!\n\010LoopCond\022\t\n\005input\030\n\032\n\n\006output\030\n\nN\n\005Merge\022\016\n\006inputs\"\001T*\001N\032\013\n\006output\"\001T\032\017\n\013value_index\030\003\"\t\n\001T\022\004type\"\014\n\001N\022\003int(\0010\001\n2\n\rNextIteration\022\t\n\004data\"\001T\032\013\n\006output\"\001T\"\t\n\001T\022\004type\n\006\n\004NoOp\n\202\001\n\010RefEnter\022\014\n\004data\"\001T\200\001\001\032\016\n\006output\"\001T\200\001\001\"\t\n\001T\022\004type\"\024\n\nframe_name\022\006string\"\027\n\013is_constant\022\004bool\032\002(\000\"\036\n\023parallel_iterations\022\003int\032\002\030\n\n2\n\007RefExit\022\014\n\004data\"\001T\200\001\001\032\016\n\006output\"\001T\200\001\001\"\t\n\001T\022\004type\nW\n\010RefMerge\022\021\n\006inputs\"\001T*\001N\200\001\001\032\016\n\006output\"\001T\200\001\001\032\017\n\013value_index\030\003\"\t\n\001T\022\004type\"\014\n\001N\022\003int(\0010\001\n;\n\020RefNextIteration\022\014\n\004data\"\001T\200\001\001\032\016\n\006output\"\001T\200\001\001\"\t\n\001T\022\004type\nR\n\tRefSelect\022\t\n\005index\030\003\022\021\n\006inputs\"\001T*\001N\200\001\001\032\016\n\006output\"\001T\200\001\001\"\t\n\001T\022\004type\"\014\n\001N\022\003int(\0010\001\n\\\n\tRefSwitch\022\014\n\004data\"\001T\200\001\001\022\010\n\004pred\030\n\032\024\n\014output_false\"\001T\200\001\001\032\023\n\013output_true\"\001T\200\001\001\"\t\n\001T\022\004type\230\001\001\nM\n\006Switch\022\t\n\004data\"\001T\022\010\n\004pred\030\n\032\021\n\014output_false\"\001T\032\020\n\013output_true\"\001T\"\t\n\001T\022\004type")
