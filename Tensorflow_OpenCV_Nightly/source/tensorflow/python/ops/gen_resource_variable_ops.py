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
  result = _op_def_lib.apply_op("AssignAddVariableOp", resource=resource,
                                value=value, name=name)
  return result



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
  result = _op_def_lib.apply_op("AssignSubVariableOp", resource=resource,
                                value=value, name=name)
  return result



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
  result = _op_def_lib.apply_op("AssignVariableOp", resource=resource,
                                value=value, name=name)
  return result



def destroy_resource_op(resource, ignore_lookup_error=None, name=None):
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
  result = _op_def_lib.apply_op("DestroyResourceOp", resource=resource,
                                ignore_lookup_error=ignore_lookup_error,
                                name=name)
  return result



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
  result = _op_def_lib.apply_op("ReadVariableOp", resource=resource,
                                dtype=dtype, name=name)
  return result



def resource_gather(resource, indices, dtype, validate_indices=None,
                    name=None):
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
  result = _op_def_lib.apply_op("ResourceGather", resource=resource,
                                indices=indices, dtype=dtype,
                                validate_indices=validate_indices, name=name)
  return result



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
  result = _op_def_lib.apply_op("ResourceScatterAdd", resource=resource,
                                indices=indices, updates=updates, name=name)
  return result



def var_handle_op(dtype, shape, container=None, shared_name=None, name=None):
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
  result = _op_def_lib.apply_op("VarHandleOp", dtype=dtype, shape=shape,
                                container=container, shared_name=shared_name,
                                name=name)
  return result



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
  result = _op_def_lib.apply_op("VarIsInitializedOp", resource=resource,
                                name=name)
  return result


def _InitOpDefLibrary():
  op_list = _op_def_pb2.OpList()
  _text_format.Merge(_InitOpDefLibrary.op_list_ascii, op_list)
  _op_def_registry.register_op_list(op_list)
  op_def_lib = _op_def_library.OpDefLibrary()
  op_def_lib.add_op_list(op_list)
  return op_def_lib


_InitOpDefLibrary.op_list_ascii = """op {
  name: "AssignAddVariableOp"
  input_arg {
    name: "resource"
    type: DT_RESOURCE
  }
  input_arg {
    name: "value"
    type_attr: "dtype"
  }
  attr {
    name: "dtype"
    type: "type"
  }
  is_stateful: true
}
op {
  name: "AssignSubVariableOp"
  input_arg {
    name: "resource"
    type: DT_RESOURCE
  }
  input_arg {
    name: "value"
    type_attr: "dtype"
  }
  attr {
    name: "dtype"
    type: "type"
  }
  is_stateful: true
}
op {
  name: "AssignVariableOp"
  input_arg {
    name: "resource"
    type: DT_RESOURCE
  }
  input_arg {
    name: "value"
    type_attr: "dtype"
  }
  attr {
    name: "dtype"
    type: "type"
  }
  is_stateful: true
}
op {
  name: "DestroyResourceOp"
  input_arg {
    name: "resource"
    type: DT_RESOURCE
  }
  attr {
    name: "ignore_lookup_error"
    type: "bool"
    default_value {
      b: true
    }
  }
  is_stateful: true
}
op {
  name: "ReadVariableOp"
  input_arg {
    name: "resource"
    type: DT_RESOURCE
  }
  output_arg {
    name: "value"
    type_attr: "dtype"
  }
  attr {
    name: "dtype"
    type: "type"
  }
  is_stateful: true
}
op {
  name: "ResourceGather"
  input_arg {
    name: "resource"
    type: DT_RESOURCE
  }
  input_arg {
    name: "indices"
    type_attr: "Tindices"
  }
  output_arg {
    name: "output"
    type_attr: "dtype"
  }
  attr {
    name: "validate_indices"
    type: "bool"
    default_value {
      b: true
    }
  }
  attr {
    name: "dtype"
    type: "type"
  }
  attr {
    name: "Tindices"
    type: "type"
    allowed_values {
      list {
        type: DT_INT32
        type: DT_INT64
      }
    }
  }
  is_stateful: true
}
op {
  name: "ResourceScatterAdd"
  input_arg {
    name: "resource"
    type: DT_RESOURCE
  }
  input_arg {
    name: "indices"
    type_attr: "Tindices"
  }
  input_arg {
    name: "updates"
    type_attr: "dtype"
  }
  attr {
    name: "dtype"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT64
        type: DT_INT32
        type: DT_UINT8
        type: DT_UINT16
        type: DT_INT16
        type: DT_INT8
        type: DT_COMPLEX64
        type: DT_COMPLEX128
        type: DT_QINT8
        type: DT_QUINT8
        type: DT_QINT32
        type: DT_HALF
      }
    }
  }
  attr {
    name: "Tindices"
    type: "type"
    allowed_values {
      list {
        type: DT_INT32
        type: DT_INT64
      }
    }
  }
  is_stateful: true
}
op {
  name: "VarHandleOp"
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
  attr {
    name: "dtype"
    type: "type"
  }
  attr {
    name: "shape"
    type: "shape"
  }
  is_stateful: true
}
op {
  name: "VarIsInitializedOp"
  input_arg {
    name: "resource"
    type: DT_RESOURCE
  }
  output_arg {
    name: "is_initialized"
    type: DT_BOOL
  }
  is_stateful: true
}
"""


_op_def_lib = _InitOpDefLibrary()
