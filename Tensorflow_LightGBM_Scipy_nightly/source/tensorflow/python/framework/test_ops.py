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

def graph_def_version(name=None):
  r"""TODO: add doc.

  Args:
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int32`.
  """
  result = _op_def_lib.apply_op("GraphDefVersion", name=name)
  return result


_ops.RegisterShape("GraphDefVersion")(None)

def kernel_label(name=None):
  r"""TODO: add doc.

  Args:
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`.
  """
  result = _op_def_lib.apply_op("KernelLabel", name=name)
  return result


_ops.RegisterShape("KernelLabel")(None)

def old(name=None):
  r"""TODO: add doc.

  Args:
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  result = _op_def_lib.apply_op("Old", name=name)
  return result


_ops.RegisterShape("Old")(None)

def requires_older_graph_version(name=None):
  r"""TODO: add doc.

  Args:
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int32`.
  """
  result = _op_def_lib.apply_op("RequiresOlderGraphVersion", name=name)
  return result


_ops.RegisterShape("RequiresOlderGraphVersion")(None)

def resource_create_op(resource, name=None):
  r"""TODO: add doc.

  Args:
    resource: A `Tensor` of type `resource`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  result = _op_def_lib.apply_op("ResourceCreateOp", resource=resource,
                                name=name)
  return result


_ops.RegisterShape("ResourceCreateOp")(None)

def resource_initialized_op(resource, name=None):
  r"""TODO: add doc.

  Args:
    resource: A `Tensor` of type `resource`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `bool`.
  """
  result = _op_def_lib.apply_op("ResourceInitializedOp", resource=resource,
                                name=name)
  return result


_ops.RegisterShape("ResourceInitializedOp")(None)

def resource_using_op(resource, name=None):
  r"""TODO: add doc.

  Args:
    resource: A `Tensor` of type `resource`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  result = _op_def_lib.apply_op("ResourceUsingOp", resource=resource,
                                name=name)
  return result


_ops.RegisterShape("ResourceUsingOp")(None)

def stub_resource_handle_op(container=None, shared_name=None, name=None):
  r"""Creates a handle to a StubResource

  Args:
    container: An optional `string`. Defaults to `""`.
    shared_name: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `resource`.
  """
  result = _op_def_lib.apply_op("StubResourceHandleOp", container=container,
                                shared_name=shared_name, name=name)
  return result


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
  result = _op_def_lib.apply_op("TestStringOutput", input=input, name=name)
  return _TestStringOutputOutput._make(result)


_ops.RegisterShape("TestStringOutput")(None)
def _InitOpDefLibrary():
  op_list = _op_def_pb2.OpList()
  _text_format.Merge(_InitOpDefLibrary.op_list_ascii, op_list)
  _op_def_registry.register_op_list(op_list)
  op_def_lib = _op_def_library.OpDefLibrary()
  op_def_lib.add_op_list(op_list)
  return op_def_lib


_InitOpDefLibrary.op_list_ascii = """op {
  name: "GraphDefVersion"
  output_arg {
    name: "version"
    type: DT_INT32
  }
  is_stateful: true
}
op {
  name: "KernelLabel"
  output_arg {
    name: "result"
    type: DT_STRING
  }
}
op {
  name: "Old"
  deprecation {
    version: 8
    explanation: "For reasons"
  }
}
op {
  name: "RequiresOlderGraphVersion"
  output_arg {
    name: "version"
    type: DT_INT32
  }
  is_stateful: true
}
op {
  name: "ResourceCreateOp"
  input_arg {
    name: "resource"
    type: DT_RESOURCE
  }
  is_stateful: true
}
op {
  name: "ResourceInitializedOp"
  input_arg {
    name: "resource"
    type: DT_RESOURCE
  }
  output_arg {
    name: "initialized"
    type: DT_BOOL
  }
  is_stateful: true
}
op {
  name: "ResourceUsingOp"
  input_arg {
    name: "resource"
    type: DT_RESOURCE
  }
  is_stateful: true
}
op {
  name: "StubResourceHandleOp"
  output_arg {
    name: "resource"
    type: DT_RESOURCE
  }
  attr {
    name: "container"
    type: "string"
    default_value {
      s: ""
    }
  }
  attr {
    name: "shared_name"
    type: "string"
    default_value {
      s: ""
    }
  }
  is_stateful: true
}
op {
  name: "TestStringOutput"
  input_arg {
    name: "input"
    type: DT_FLOAT
  }
  output_arg {
    name: "output1"
    type: DT_FLOAT
  }
  output_arg {
    name: "output2"
    type: DT_STRING
  }
}
"""


_op_def_lib = _InitOpDefLibrary()
