"""Python wrappers around TensorFlow ops.

This file is MACHINE GENERATED! Do not edit.
Original C++ source file: test_ops.cc
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


def graph_def_version(name=None):
  r"""TODO: add doc.

  Args:
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int32`.
  """
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "GraphDefVersion", name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = None
  else:
    _inputs_flat = []
    _attrs = None
    _result = _execute.execute(b"GraphDefVersion", 1, inputs=_inputs_flat,
                               attrs=_attrs, ctx=_ctx, name=name)
  _execute.record_gradient(
      "GraphDefVersion", _inputs_flat, _attrs, _result, name)
  _result, = _result
  return _result

_ops.RegisterShape("GraphDefVersion")(None)


def kernel_label(name=None):
  r"""TODO: add doc.

  Args:
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`.
  """
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "KernelLabel", name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = None
  else:
    _inputs_flat = []
    _attrs = None
    _result = _execute.execute(b"KernelLabel", 1, inputs=_inputs_flat,
                               attrs=_attrs, ctx=_ctx, name=name)
  _execute.record_gradient(
      "KernelLabel", _inputs_flat, _attrs, _result, name)
  _result, = _result
  return _result

_ops.RegisterShape("KernelLabel")(None)


def old(name=None):
  r"""TODO: add doc.

  Args:
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "Old", name=name)
    return _op
  else:
    _inputs_flat = []
    _attrs = None
    _result = _execute.execute(b"Old", 0, inputs=_inputs_flat, attrs=_attrs,
                               ctx=_ctx, name=name)
  return _result

_ops.RegisterShape("Old")(None)


def requires_older_graph_version(name=None):
  r"""TODO: add doc.

  Args:
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int32`.
  """
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "RequiresOlderGraphVersion", name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = None
  else:
    _inputs_flat = []
    _attrs = None
    _result = _execute.execute(b"RequiresOlderGraphVersion", 1,
                               inputs=_inputs_flat, attrs=_attrs, ctx=_ctx,
                               name=name)
  _execute.record_gradient(
      "RequiresOlderGraphVersion", _inputs_flat, _attrs, _result, name)
  _result, = _result
  return _result

_ops.RegisterShape("RequiresOlderGraphVersion")(None)


def resource_create_op(resource, name=None):
  r"""TODO: add doc.

  Args:
    resource: A `Tensor` of type `resource`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "ResourceCreateOp", resource=resource, name=name)
    return _op
  else:
    resource = _ops.convert_to_tensor(resource, _dtypes.resource)
    _inputs_flat = [resource]
    _attrs = None
    _result = _execute.execute(b"ResourceCreateOp", 0, inputs=_inputs_flat,
                               attrs=_attrs, ctx=_ctx, name=name)
  return _result

_ops.RegisterShape("ResourceCreateOp")(None)


def resource_initialized_op(resource, name=None):
  r"""TODO: add doc.

  Args:
    resource: A `Tensor` of type `resource`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `bool`.
  """
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "ResourceInitializedOp", resource=resource, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = None
  else:
    resource = _ops.convert_to_tensor(resource, _dtypes.resource)
    _inputs_flat = [resource]
    _attrs = None
    _result = _execute.execute(b"ResourceInitializedOp", 1,
                               inputs=_inputs_flat, attrs=_attrs, ctx=_ctx,
                               name=name)
  _execute.record_gradient(
      "ResourceInitializedOp", _inputs_flat, _attrs, _result, name)
  _result, = _result
  return _result

_ops.RegisterShape("ResourceInitializedOp")(None)


def resource_using_op(resource, name=None):
  r"""TODO: add doc.

  Args:
    resource: A `Tensor` of type `resource`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "ResourceUsingOp", resource=resource, name=name)
    return _op
  else:
    resource = _ops.convert_to_tensor(resource, _dtypes.resource)
    _inputs_flat = [resource]
    _attrs = None
    _result = _execute.execute(b"ResourceUsingOp", 0, inputs=_inputs_flat,
                               attrs=_attrs, ctx=_ctx, name=name)
  return _result

_ops.RegisterShape("ResourceUsingOp")(None)


def stub_resource_handle_op(container="", shared_name="", name=None):
  r"""Creates a handle to a StubResource

  Args:
    container: An optional `string`. Defaults to `""`.
    shared_name: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `resource`.
  """
  if container is None:
    container = ""
  container = _execute.make_str(container, "container")
  if shared_name is None:
    shared_name = ""
  shared_name = _execute.make_str(shared_name, "shared_name")
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "StubResourceHandleOp", container=container, shared_name=shared_name,
        name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("container", _op.get_attr("container"), "shared_name",
              _op.get_attr("shared_name"))
  else:
    _inputs_flat = []
    _attrs = ("container", container, "shared_name", shared_name)
    _result = _execute.execute(b"StubResourceHandleOp", 1,
                               inputs=_inputs_flat, attrs=_attrs, ctx=_ctx,
                               name=name)
  _execute.record_gradient(
      "StubResourceHandleOp", _inputs_flat, _attrs, _result, name)
  _result, = _result
  return _result

_ops.RegisterShape("StubResourceHandleOp")(None)


_test_string_output_outputs = ["output1", "output2"]
_TestStringOutputOutput = _collections.namedtuple(
    "TestStringOutput", _test_string_output_outputs)


def test_string_output(input, name=None):
  r"""TODO: add doc.

  Args:
    input: A `Tensor` of type `float32`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output1, output2).

    output1: A `Tensor` of type `float32`.
    output2: A `Tensor` of type `string`.
  """
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "TestStringOutput", input=input, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = None
  else:
    input = _ops.convert_to_tensor(input, _dtypes.float32)
    _inputs_flat = [input]
    _attrs = None
    _result = _execute.execute(b"TestStringOutput", 2, inputs=_inputs_flat,
                               attrs=_attrs, ctx=_ctx, name=name)
  _execute.record_gradient(
      "TestStringOutput", _inputs_flat, _attrs, _result, name)
  _result = _TestStringOutputOutput._make(_result)
  return _result

_ops.RegisterShape("TestStringOutput")(None)

def _InitOpDefLibrary(op_list_proto_bytes):
  op_list = _op_def_pb2.OpList()
  op_list.ParseFromString(op_list_proto_bytes)
  _op_def_registry.register_op_list(op_list)
  op_def_lib = _op_def_library.OpDefLibrary()
  op_def_lib.add_op_list(op_list)
  return op_def_lib
# op {
#   name: "GraphDefVersion"
#   output_arg {
#     name: "version"
#     type: DT_INT32
#   }
#   is_stateful: true
# }
# op {
#   name: "KernelLabel"
#   output_arg {
#     name: "result"
#     type: DT_STRING
#   }
# }
# op {
#   name: "Old"
#   deprecation {
#     version: 8
#     explanation: "For reasons"
#   }
# }
# op {
#   name: "RequiresOlderGraphVersion"
#   output_arg {
#     name: "version"
#     type: DT_INT32
#   }
#   is_stateful: true
# }
# op {
#   name: "ResourceCreateOp"
#   input_arg {
#     name: "resource"
#     type: DT_RESOURCE
#   }
#   is_stateful: true
# }
# op {
#   name: "ResourceInitializedOp"
#   input_arg {
#     name: "resource"
#     type: DT_RESOURCE
#   }
#   output_arg {
#     name: "initialized"
#     type: DT_BOOL
#   }
#   is_stateful: true
# }
# op {
#   name: "ResourceUsingOp"
#   input_arg {
#     name: "resource"
#     type: DT_RESOURCE
#   }
#   is_stateful: true
# }
# op {
#   name: "StubResourceHandleOp"
#   output_arg {
#     name: "resource"
#     type: DT_RESOURCE
#   }
#   attr {
#     name: "container"
#     type: "string"
#     default_value {
#       s: ""
#     }
#   }
#   attr {
#     name: "shared_name"
#     type: "string"
#     default_value {
#       s: ""
#     }
#   }
#   is_stateful: true
# }
# op {
#   name: "TestStringOutput"
#   input_arg {
#     name: "input"
#     type: DT_FLOAT
#   }
#   output_arg {
#     name: "output1"
#     type: DT_FLOAT
#   }
#   output_arg {
#     name: "output2"
#     type: DT_STRING
#   }
# }
_op_def_lib = _InitOpDefLibrary(b"\n!\n\017GraphDefVersion\032\013\n\007version\030\003\210\001\001\n\031\n\013KernelLabel\032\n\n\006result\030\007\n\026\n\003OldB\017\010\010\022\013For reasons\n+\n\031RequiresOlderGraphVersion\032\013\n\007version\030\003\210\001\001\n#\n\020ResourceCreateOp\022\014\n\010resource\030\024\210\001\001\n9\n\025ResourceInitializedOp\022\014\n\010resource\030\024\032\017\n\013initialized\030\n\210\001\001\n\"\n\017ResourceUsingOp\022\014\n\010resource\030\024\210\001\001\n[\n\024StubResourceHandleOp\032\014\n\010resource\030\024\"\027\n\tcontainer\022\006string\032\002\022\000\"\031\n\013shared_name\022\006string\032\002\022\000\210\001\001\n7\n\020TestStringOutput\022\t\n\005input\030\001\032\013\n\007output1\030\001\032\013\n\007output2\030\007")
