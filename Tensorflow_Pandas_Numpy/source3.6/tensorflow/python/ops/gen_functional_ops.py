"""Python wrappers around TensorFlow ops.

This file is MACHINE GENERATED! Do not edit.
Original C++ source file: functional_ops.cc
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


def remote_call(target, args, Tout, f, name=None):
  r"""Runs function `f` on a remote device indicated by `target`.

  Args:
    target: A `Tensor` of type `string`.
      A fully specified device name where we want to run the function.
    args: A list of `Tensor` objects. A list of arguments for the function.
    Tout: A list of `tf.DTypes` that has length `>= 1`.
      The type list for the return values.
    f: A function decorated with @Defun. The function to run remotely.
    name: A name for the operation (optional).

  Returns:
    A list of `Tensor` objects of type `Tout`. A list of return values.
  """
  if not isinstance(Tout, (list, tuple)):
    raise TypeError(
        "Expected list for 'Tout' argument to "
        "'remote_call' Op, not %r." % Tout)
  Tout = [_execute.make_type(_t, "Tout") for _t in Tout]
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "RemoteCall", target=target, args=args, Tout=Tout, f=f, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("Tin", _op.get_attr("Tin"), "Tout", _op.get_attr("Tout"), "f",
              _op.get_attr("f"))
  else:
    _attr_Tin, args = _execute.convert_to_mixed_eager_tensors(args, _ctx)
    _attr_Tin = [_t.as_datatype_enum for _t in _attr_Tin]
    target = _ops.convert_to_tensor(target, _dtypes.string)
    _inputs_flat = [target] + list(args)
    _attrs = ("Tin", _attr_Tin, "Tout", Tout, "f", f)
    _result = _execute.execute(b"RemoteCall", len(Tout), inputs=_inputs_flat,
                               attrs=_attrs, ctx=_ctx, name=name)
  _execute.record_gradient(
      "RemoteCall", _inputs_flat, _attrs, _result, name)
  return _result


def _symbolic_gradient(input, Tout, f, name=None):
  r"""Computes the gradient function for function f via backpropagation.

  Args:
    input: A list of `Tensor` objects. a list of input tensors of size N + M;
    Tout: A list of `tf.DTypes` that has length `>= 1`.
      the type list for the input list.
    f: A function decorated with @Defun.
      The function we want to compute the gradient for.

      The function 'f' must be a numerical function which takes N inputs and
      produces M outputs. Its gradient function 'g', which is computed by
      this SymbolicGradient op is a function taking N + M inputs and
      produces N outputs.

      I.e. if we have
         (y1, y2, ..., y_M) = f(x1, x2, ..., x_N),
      then, g is
         (dL/dx1, dL/dx2, ..., dL/dx_N) = g(x1, x2, ..., x_N,
                                           dL/dy1, dL/dy2, ..., dL/dy_M),

      where L is a scalar-value function of (x1, x2, ..., xN) (e.g., the
      loss function). dL/dx_i is the partial derivative of L with respect
      to x_i.

      (Needs some math expert to say the comment above better.)
    name: A name for the operation (optional).

  Returns:
    A list of `Tensor` objects of type `Tout`.
    a list of output tensors of size N;
  """
  if not isinstance(Tout, (list, tuple)):
    raise TypeError(
        "Expected list for 'Tout' argument to "
        "'symbolic_gradient' Op, not %r." % Tout)
  Tout = [_execute.make_type(_t, "Tout") for _t in Tout]
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "SymbolicGradient", input=input, Tout=Tout, f=f, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("Tin", _op.get_attr("Tin"), "Tout", _op.get_attr("Tout"), "f",
              _op.get_attr("f"))
  else:
    _attr_Tin, input = _execute.convert_to_mixed_eager_tensors(input, _ctx)
    _attr_Tin = [_t.as_datatype_enum for _t in _attr_Tin]
    _inputs_flat = list(input)
    _attrs = ("Tin", _attr_Tin, "Tout", Tout, "f", f)
    _result = _execute.execute(b"SymbolicGradient", len(Tout),
                               inputs=_inputs_flat, attrs=_attrs, ctx=_ctx,
                               name=name)
  _execute.record_gradient(
      "SymbolicGradient", _inputs_flat, _attrs, _result, name)
  return _result

def _InitOpDefLibrary(op_list_proto_bytes):
  op_list = _op_def_pb2.OpList()
  op_list.ParseFromString(op_list_proto_bytes)
  _op_def_registry.register_op_list(op_list)
  op_def_lib = _op_def_library.OpDefLibrary()
  op_def_lib.add_op_list(op_list)
  return op_def_lib
# op {
#   name: "RemoteCall"
#   input_arg {
#     name: "target"
#     type: DT_STRING
#   }
#   input_arg {
#     name: "args"
#     type_list_attr: "Tin"
#   }
#   output_arg {
#     name: "output"
#     type_list_attr: "Tout"
#   }
#   attr {
#     name: "Tin"
#     type: "list(type)"
#     has_minimum: true
#     minimum: 1
#   }
#   attr {
#     name: "Tout"
#     type: "list(type)"
#     has_minimum: true
#     minimum: 1
#   }
#   attr {
#     name: "f"
#     type: "func"
#   }
# }
# op {
#   name: "SymbolicGradient"
#   input_arg {
#     name: "input"
#     type_list_attr: "Tin"
#   }
#   output_arg {
#     name: "output"
#     type_list_attr: "Tout"
#   }
#   attr {
#     name: "Tin"
#     type: "list(type)"
#     has_minimum: true
#     minimum: 1
#   }
#   attr {
#     name: "Tout"
#     type: "list(type)"
#     has_minimum: true
#     minimum: 1
#   }
#   attr {
#     name: "f"
#     type: "func"
#   }
# }
_op_def_lib = _InitOpDefLibrary(b"\no\n\nRemoteCall\022\n\n\006target\030\007\022\013\n\004args2\003Tin\032\016\n\006output2\004Tout\"\025\n\003Tin\022\nlist(type)(\0010\001\"\026\n\004Tout\022\nlist(type)(\0010\001\"\t\n\001f\022\004func\nj\n\020SymbolicGradient\022\014\n\005input2\003Tin\032\016\n\006output2\004Tout\"\025\n\003Tin\022\nlist(type)(\0010\001\"\026\n\004Tout\022\nlist(type)(\0010\001\"\t\n\001f\022\004func")
