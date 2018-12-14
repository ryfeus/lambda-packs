"""Python wrappers around TensorFlow ops.

This file is MACHINE GENERATED! Do not edit.
Original C++ source file: resource_variable_ops.cc
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


def assign_add_variable_op(resource, value, name=None):
  r"""Adds a value to the current value of a variable.

  Any ReadVariableOp which depends directly or indirectly on this assign is
  guaranteed to see the incremented value or a subsequent newer one.

  Outputs the incremented value, which can be used to totally order the
  increments to this variable.

  Args:
    resource: A `Tensor` of type `resource`.
      handle to the resource in which to store the variable.
    value: A `Tensor`. the value by which the variable will be incremented.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "AssignAddVariableOp", resource=resource, value=value, name=name)
    return _op
  else:
    _attr_dtype, (value,) = _execute.args_to_matching_eager([value], _ctx)
    _attr_dtype = _attr_dtype.as_datatype_enum
    resource = _ops.convert_to_tensor(resource, _dtypes.resource)
    _inputs_flat = [resource, value]
    _attrs = ("dtype", _attr_dtype)
    _result = _execute.execute(b"AssignAddVariableOp", 0, inputs=_inputs_flat,
                               attrs=_attrs, ctx=_ctx, name=name)
  return _result


def assign_sub_variable_op(resource, value, name=None):
  r"""Subtracts a value from the current value of a variable.

  Any ReadVariableOp which depends directly or indirectly on this assign is
  guaranteed to see the incremented value or a subsequent newer one.

  Outputs the incremented value, which can be used to totally order the
  increments to this variable.

  Args:
    resource: A `Tensor` of type `resource`.
      handle to the resource in which to store the variable.
    value: A `Tensor`. the value by which the variable will be incremented.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "AssignSubVariableOp", resource=resource, value=value, name=name)
    return _op
  else:
    _attr_dtype, (value,) = _execute.args_to_matching_eager([value], _ctx)
    _attr_dtype = _attr_dtype.as_datatype_enum
    resource = _ops.convert_to_tensor(resource, _dtypes.resource)
    _inputs_flat = [resource, value]
    _attrs = ("dtype", _attr_dtype)
    _result = _execute.execute(b"AssignSubVariableOp", 0, inputs=_inputs_flat,
                               attrs=_attrs, ctx=_ctx, name=name)
  return _result


def assign_variable_op(resource, value, name=None):
  r"""Assigns a new value to a variable.

  Any ReadVariableOp with a control dependency on this op is guaranteed to return
  this value or a subsequent newer value of the variable.

  Args:
    resource: A `Tensor` of type `resource`.
      handle to the resource in which to store the variable.
    value: A `Tensor`. the value to set the new tensor to use.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "AssignVariableOp", resource=resource, value=value, name=name)
    return _op
  else:
    _attr_dtype, (value,) = _execute.args_to_matching_eager([value], _ctx)
    _attr_dtype = _attr_dtype.as_datatype_enum
    resource = _ops.convert_to_tensor(resource, _dtypes.resource)
    _inputs_flat = [resource, value]
    _attrs = ("dtype", _attr_dtype)
    _result = _execute.execute(b"AssignVariableOp", 0, inputs=_inputs_flat,
                               attrs=_attrs, ctx=_ctx, name=name)
  return _result


def destroy_resource_op(resource, ignore_lookup_error=True, name=None):
  r"""Deletes the resource specified by the handle.

  All subsequent operations using the resource will result in a NotFound
  error status.

  Args:
    resource: A `Tensor` of type `resource`. handle to the resource to delete.
    ignore_lookup_error: An optional `bool`. Defaults to `True`.
      whether to ignore the error when the resource
      doesn't exist.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  if ignore_lookup_error is None:
    ignore_lookup_error = True
  ignore_lookup_error = _execute.make_bool(ignore_lookup_error, "ignore_lookup_error")
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "DestroyResourceOp", resource=resource,
        ignore_lookup_error=ignore_lookup_error, name=name)
    return _op
  else:
    resource = _ops.convert_to_tensor(resource, _dtypes.resource)
    _inputs_flat = [resource]
    _attrs = ("ignore_lookup_error", ignore_lookup_error)
    _result = _execute.execute(b"DestroyResourceOp", 0, inputs=_inputs_flat,
                               attrs=_attrs, ctx=_ctx, name=name)
  return _result


def read_variable_op(resource, dtype, name=None):
  r"""Reads the value of a variable.

  The tensor returned by this operation is immutable.

  The value returned by this operation is guaranteed to be influenced by all the
  writes on which this operation depends directly or indirectly, and to not be
  influenced by any of the writes which depend directly or indirectly on this
  operation.

  Args:
    resource: A `Tensor` of type `resource`.
      handle to the resource in which to store the variable.
    dtype: A `tf.DType`. the dtype of the value.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `dtype`.
  """
  dtype = _execute.make_type(dtype, "dtype")
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "ReadVariableOp", resource=resource, dtype=dtype, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("dtype", _op.get_attr("dtype"))
  else:
    resource = _ops.convert_to_tensor(resource, _dtypes.resource)
    _inputs_flat = [resource]
    _attrs = ("dtype", dtype)
    _result = _execute.execute(b"ReadVariableOp", 1, inputs=_inputs_flat,
                               attrs=_attrs, ctx=_ctx, name=name)
  _execute.record_gradient(
      "ReadVariableOp", _inputs_flat, _attrs, _result, name)
  _result, = _result
  return _result


def resource_gather(resource, indices, dtype, validate_indices=True, name=None):
  r"""Gather slices from the variable pointed to by `resource` according to `indices`.

  `indices` must be an integer tensor of any dimension (usually 0-D or 1-D).
  Produces an output tensor with shape `indices.shape + params.shape[1:]` where:

  ```python
      # Scalar indices
      output[:, ..., :] = params[indices, :, ... :]

      # Vector indices
      output[i, :, ..., :] = params[indices[i], :, ... :]

      # Higher rank indices
      output[i, ..., j, :, ... :] = params[indices[i, ..., j], :, ..., :]
  ```

  Args:
    resource: A `Tensor` of type `resource`.
    indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.
    dtype: A `tf.DType`.
    validate_indices: An optional `bool`. Defaults to `True`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `dtype`.
  """
  dtype = _execute.make_type(dtype, "dtype")
  if validate_indices is None:
    validate_indices = True
  validate_indices = _execute.make_bool(validate_indices, "validate_indices")
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "ResourceGather", resource=resource, indices=indices, dtype=dtype,
        validate_indices=validate_indices, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("validate_indices", _op.get_attr("validate_indices"), "dtype",
              _op.get_attr("dtype"), "Tindices", _op.get_attr("Tindices"))
  else:
    _attr_Tindices, (indices,) = _execute.args_to_matching_eager([indices], _ctx)
    _attr_Tindices = _attr_Tindices.as_datatype_enum
    resource = _ops.convert_to_tensor(resource, _dtypes.resource)
    _inputs_flat = [resource, indices]
    _attrs = ("validate_indices", validate_indices, "dtype", dtype,
              "Tindices", _attr_Tindices)
    _result = _execute.execute(b"ResourceGather", 1, inputs=_inputs_flat,
                               attrs=_attrs, ctx=_ctx, name=name)
  _execute.record_gradient(
      "ResourceGather", _inputs_flat, _attrs, _result, name)
  _result, = _result
  return _result


def resource_scatter_add(resource, indices, updates, name=None):
  r"""Adds sparse updates to the variable referenced by `resource`.

  This operation computes

      # Scalar indices
      ref[indices, ...] += updates[...]

      # Vector indices (for each i)
      ref[indices[i], ...] += updates[i, ...]

      # High rank indices (for each i, ..., j)
      ref[indices[i, ..., j], ...] += updates[i, ..., j, ...]

  Duplicate entries are handled correctly: if multiple `indices` reference
  the same location, their contributions add.

  Requires `updates.shape = indices.shape + ref.shape[1:]`.

  <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
  <img style="width:100%" src="https://www.tensorflow.org/images/ScatterAdd.png" alt>
  </div>

  Args:
    resource: A `Tensor` of type `resource`. Should be from a `Variable` node.
    indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A tensor of indices into the first dimension of `ref`.
    updates: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`, `complex128`, `qint8`, `quint8`, `qint32`, `half`.
      A tensor of updated values to add to `ref`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "ResourceScatterAdd", resource=resource, indices=indices,
        updates=updates, name=name)
    return _op
  else:
    _attr_dtype, (updates,) = _execute.args_to_matching_eager([updates], _ctx)
    _attr_dtype = _attr_dtype.as_datatype_enum
    _attr_Tindices, (indices,) = _execute.args_to_matching_eager([indices], _ctx)
    _attr_Tindices = _attr_Tindices.as_datatype_enum
    resource = _ops.convert_to_tensor(resource, _dtypes.resource)
    _inputs_flat = [resource, indices, updates]
    _attrs = ("dtype", _attr_dtype, "Tindices", _attr_Tindices)
    _result = _execute.execute(b"ResourceScatterAdd", 0, inputs=_inputs_flat,
                               attrs=_attrs, ctx=_ctx, name=name)
  return _result


def var_handle_op(dtype, shape, container="", shared_name="", name=None):
  r"""Creates a handle to a Variable resource.

  Args:
    dtype: A `tf.DType`. the type of this variable. Must agree with the dtypes
      of all ops using this variable.
    shape: A `tf.TensorShape` or list of `ints`.
      The (possibly partially specified) shape of this variable.
    container: An optional `string`. Defaults to `""`.
      the container this variable is placed in.
    shared_name: An optional `string`. Defaults to `""`.
      the name by which this variable is referred to.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `resource`.
  """
  dtype = _execute.make_type(dtype, "dtype")
  shape = _execute.make_shape(shape, "shape")
  if container is None:
    container = ""
  container = _execute.make_str(container, "container")
  if shared_name is None:
    shared_name = ""
  shared_name = _execute.make_str(shared_name, "shared_name")
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "VarHandleOp", dtype=dtype, shape=shape, container=container,
        shared_name=shared_name, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("container", _op.get_attr("container"), "shared_name",
              _op.get_attr("shared_name"), "dtype", _op.get_attr("dtype"),
              "shape", _op.get_attr("shape"))
  else:
    _inputs_flat = []
    _attrs = ("container", container, "shared_name", shared_name, "dtype",
              dtype, "shape", shape)
    _result = _execute.execute(b"VarHandleOp", 1, inputs=_inputs_flat,
                               attrs=_attrs, ctx=_ctx, name=name)
  _execute.record_gradient(
      "VarHandleOp", _inputs_flat, _attrs, _result, name)
  _result, = _result
  return _result


def var_is_initialized_op(resource, name=None):
  r"""Checks whether a resource handle-based variable has been initialized.

  Args:
    resource: A `Tensor` of type `resource`. the input resource handle.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `bool`.
    a scalar boolean which is true if the variable has been
    initialized.
  """
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "VarIsInitializedOp", resource=resource, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = None
  else:
    resource = _ops.convert_to_tensor(resource, _dtypes.resource)
    _inputs_flat = [resource]
    _attrs = None
    _result = _execute.execute(b"VarIsInitializedOp", 1, inputs=_inputs_flat,
                               attrs=_attrs, ctx=_ctx, name=name)
  _execute.record_gradient(
      "VarIsInitializedOp", _inputs_flat, _attrs, _result, name)
  _result, = _result
  return _result


def variable_shape(input, out_type=_dtypes.int32, name=None):
  r"""Returns the shape of the variable pointed to by `resource`.

  This operation returns a 1-D integer tensor representing the shape of `input`.

  For example:

  ```
  # 't' is [[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]]
  shape(t) ==> [2, 2, 3]
  ```

  Args:
    input: A `Tensor` of type `resource`.
    out_type: An optional `tf.DType` from: `tf.int32, tf.int64`. Defaults to `tf.int32`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `out_type`.
  """
  if out_type is None:
    out_type = _dtypes.int32
  out_type = _execute.make_type(out_type, "out_type")
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "VariableShape", input=input, out_type=out_type, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("out_type", _op.get_attr("out_type"))
  else:
    input = _ops.convert_to_tensor(input, _dtypes.resource)
    _inputs_flat = [input]
    _attrs = ("out_type", out_type)
    _result = _execute.execute(b"VariableShape", 1, inputs=_inputs_flat,
                               attrs=_attrs, ctx=_ctx, name=name)
  _execute.record_gradient(
      "VariableShape", _inputs_flat, _attrs, _result, name)
  _result, = _result
  return _result

def _InitOpDefLibrary(op_list_proto_bytes):
  op_list = _op_def_pb2.OpList()
  op_list.ParseFromString(op_list_proto_bytes)
  _op_def_registry.register_op_list(op_list)
  op_def_lib = _op_def_library.OpDefLibrary()
  op_def_lib.add_op_list(op_list)
  return op_def_lib
# op {
#   name: "AssignAddVariableOp"
#   input_arg {
#     name: "resource"
#     type: DT_RESOURCE
#   }
#   input_arg {
#     name: "value"
#     type_attr: "dtype"
#   }
#   attr {
#     name: "dtype"
#     type: "type"
#   }
#   is_stateful: true
# }
# op {
#   name: "AssignSubVariableOp"
#   input_arg {
#     name: "resource"
#     type: DT_RESOURCE
#   }
#   input_arg {
#     name: "value"
#     type_attr: "dtype"
#   }
#   attr {
#     name: "dtype"
#     type: "type"
#   }
#   is_stateful: true
# }
# op {
#   name: "AssignVariableOp"
#   input_arg {
#     name: "resource"
#     type: DT_RESOURCE
#   }
#   input_arg {
#     name: "value"
#     type_attr: "dtype"
#   }
#   attr {
#     name: "dtype"
#     type: "type"
#   }
#   is_stateful: true
# }
# op {
#   name: "DestroyResourceOp"
#   input_arg {
#     name: "resource"
#     type: DT_RESOURCE
#   }
#   attr {
#     name: "ignore_lookup_error"
#     type: "bool"
#     default_value {
#       b: true
#     }
#   }
#   is_stateful: true
# }
# op {
#   name: "ReadVariableOp"
#   input_arg {
#     name: "resource"
#     type: DT_RESOURCE
#   }
#   output_arg {
#     name: "value"
#     type_attr: "dtype"
#   }
#   attr {
#     name: "dtype"
#     type: "type"
#   }
#   is_stateful: true
# }
# op {
#   name: "ResourceGather"
#   input_arg {
#     name: "resource"
#     type: DT_RESOURCE
#   }
#   input_arg {
#     name: "indices"
#     type_attr: "Tindices"
#   }
#   output_arg {
#     name: "output"
#     type_attr: "dtype"
#   }
#   attr {
#     name: "validate_indices"
#     type: "bool"
#     default_value {
#       b: true
#     }
#   }
#   attr {
#     name: "dtype"
#     type: "type"
#   }
#   attr {
#     name: "Tindices"
#     type: "type"
#     allowed_values {
#       list {
#         type: DT_INT32
#         type: DT_INT64
#       }
#     }
#   }
#   is_stateful: true
# }
# op {
#   name: "ResourceScatterAdd"
#   input_arg {
#     name: "resource"
#     type: DT_RESOURCE
#   }
#   input_arg {
#     name: "indices"
#     type_attr: "Tindices"
#   }
#   input_arg {
#     name: "updates"
#     type_attr: "dtype"
#   }
#   attr {
#     name: "dtype"
#     type: "type"
#     allowed_values {
#       list {
#         type: DT_FLOAT
#         type: DT_DOUBLE
#         type: DT_INT64
#         type: DT_INT32
#         type: DT_UINT8
#         type: DT_UINT16
#         type: DT_INT16
#         type: DT_INT8
#         type: DT_COMPLEX64
#         type: DT_COMPLEX128
#         type: DT_QINT8
#         type: DT_QUINT8
#         type: DT_QINT32
#         type: DT_HALF
#       }
#     }
#   }
#   attr {
#     name: "Tindices"
#     type: "type"
#     allowed_values {
#       list {
#         type: DT_INT32
#         type: DT_INT64
#       }
#     }
#   }
#   is_stateful: true
# }
# op {
#   name: "VarHandleOp"
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
#   attr {
#     name: "dtype"
#     type: "type"
#   }
#   attr {
#     name: "shape"
#     type: "shape"
#   }
#   is_stateful: true
# }
# op {
#   name: "VarIsInitializedOp"
#   input_arg {
#     name: "resource"
#     type: DT_RESOURCE
#   }
#   output_arg {
#     name: "is_initialized"
#     type: DT_BOOL
#   }
#   is_stateful: true
# }
# op {
#   name: "VariableShape"
#   input_arg {
#     name: "input"
#     type: DT_RESOURCE
#   }
#   output_arg {
#     name: "output"
#     type_attr: "out_type"
#   }
#   attr {
#     name: "out_type"
#     type: "type"
#     default_value {
#       type: DT_INT32
#     }
#     allowed_values {
#       list {
#         type: DT_INT32
#         type: DT_INT64
#       }
#     }
#   }
#   is_stateful: true
# }
_op_def_lib = _InitOpDefLibrary(b"\nE\n\023AssignAddVariableOp\022\014\n\010resource\030\024\022\016\n\005value\"\005dtype\"\r\n\005dtype\022\004type\210\001\001\nE\n\023AssignSubVariableOp\022\014\n\010resource\030\024\022\016\n\005value\"\005dtype\"\r\n\005dtype\022\004type\210\001\001\nB\n\020AssignVariableOp\022\014\n\010resource\030\024\022\016\n\005value\"\005dtype\"\r\n\005dtype\022\004type\210\001\001\nE\n\021DestroyResourceOp\022\014\n\010resource\030\024\"\037\n\023ignore_lookup_error\022\004bool\032\002(\001\210\001\001\n@\n\016ReadVariableOp\022\014\n\010resource\030\024\032\016\n\005value\"\005dtype\"\r\n\005dtype\022\004type\210\001\001\n\216\001\n\016ResourceGather\022\014\n\010resource\030\024\022\023\n\007indices\"\010Tindices\032\017\n\006output\"\005dtype\"\034\n\020validate_indices\022\004bool\032\002(\001\"\r\n\005dtype\022\004type\"\030\n\010Tindices\022\004type:\006\n\0042\002\003\t\210\001\001\n\211\001\n\022ResourceScatterAdd\022\014\n\010resource\030\024\022\023\n\007indices\"\010Tindices\022\020\n\007updates\"\005dtype\"!\n\005dtype\022\004type:\022\n\0202\016\001\002\t\003\004\021\005\006\010\022\013\014\r\023\"\030\n\010Tindices\022\004type:\006\n\0042\002\003\t\210\001\001\nq\n\013VarHandleOp\032\014\n\010resource\030\024\"\027\n\tcontainer\022\006string\032\002\022\000\"\031\n\013shared_name\022\006string\032\002\022\000\"\r\n\005dtype\022\004type\"\016\n\005shape\022\005shape\210\001\001\n9\n\022VarIsInitializedOp\022\014\n\010resource\030\024\032\022\n\016is_initialized\030\n\210\001\001\nO\n\rVariableShape\022\t\n\005input\030\024\032\022\n\006output\"\010out_type\"\034\n\010out_type\022\004type\032\0020\003:\006\n\0042\002\003\t\210\001\001")
