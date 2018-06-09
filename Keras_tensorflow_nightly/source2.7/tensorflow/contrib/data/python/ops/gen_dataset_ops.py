"""Python wrappers around TensorFlow ops.

This file is MACHINE GENERATED! Do not edit.
Original C++ source file: gen_dataset_ops.cc
"""

import collections as _collections
import six as _six

from tensorflow.python import pywrap_tensorflow as _pywrap_tensorflow
from tensorflow.python.eager import context as _context
from tensorflow.python.eager import core as _core
from tensorflow.python.eager import execute as _execute
from tensorflow.python.framework import dtypes as _dtypes
from tensorflow.python.framework import errors as _errors
from tensorflow.python.framework import tensor_shape as _tensor_shape

from tensorflow.core.framework import op_def_pb2 as _op_def_pb2
# Needed to trigger the call to _set_call_cpp_shape_fn.
from tensorflow.python.framework import common_shapes as _common_shapes
from tensorflow.python.framework import op_def_registry as _op_def_registry
from tensorflow.python.framework import ops as _ops
from tensorflow.python.framework import op_def_library as _op_def_library
from tensorflow.python.util.tf_export import tf_export


@tf_export('function_buffering_resource')
def function_buffering_resource(string_arg, target_device, shared_name, container, f, buffer_size, name=None):
  r"""Creates a resource that fills up a buffer by making function calls.

  Args:
    string_arg: A `Tensor` of type `string`.
      String argument to the function call.
    target_device: A `Tensor` of type `string`.
      Target device to execute the function on.
    shared_name: A `string`.
      If non-empty, this resource will be shared under the given name
      across multiple sessions.
    container: A `string`.
      If non-empty, this resource is placed in the given container.
      Otherwise, a default container is used.
    f: A function decorated with @Defun. Function to be executed.
    buffer_size: An `int`. Size of the buffer.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `resource`. Handle to the resource created.
  """
  _ctx = _context._context
  if _ctx is None or not _ctx._eager_context.is_eager:
    shared_name = _execute.make_str(shared_name, "shared_name")
    container = _execute.make_str(container, "container")
    buffer_size = _execute.make_int(buffer_size, "buffer_size")
    _, _, _op = _op_def_lib._apply_op_helper(
        "FunctionBufferingResource", string_arg=string_arg,
        target_device=target_device, shared_name=shared_name,
        container=container, f=f, buffer_size=buffer_size, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("shared_name", _op.get_attr("shared_name"), "container",
              _op.get_attr("container"), "f", _op.get_attr("f"),
              "buffer_size", _op.get_attr("buffer_size"))
    _execute.record_gradient(
      "FunctionBufferingResource", _inputs_flat, _attrs, _result, name)
    _result, = _result
    return _result

  else:
    try:
      _result = _pywrap_tensorflow.TFE_Py_FastPathExecute(
        _ctx._context_handle, _ctx._eager_context.device_name,
        "FunctionBufferingResource", name, _ctx._post_execution_callbacks,
        string_arg, target_device, "shared_name", shared_name, "container",
        container, "f", f, "buffer_size", buffer_size)
      return _result
    except _core._FallbackException:
      return function_buffering_resource_eager_fallback(
          string_arg, target_device, shared_name=shared_name,
          container=container, f=f, buffer_size=buffer_size, name=name,
          ctx=_ctx)
    except _core._NotOkStatusException as e:
      if name is not None:
        message = e.message + " name: " + name
      else:
        message = e.message
      _six.raise_from(_core._status_to_exception(e.code, message), None)


def function_buffering_resource_eager_fallback(string_arg, target_device, shared_name, container, f, buffer_size, name=None, ctx=None):
  r"""This is the slowpath function for Eager mode.
  This is for function function_buffering_resource
  """
  _ctx = ctx if ctx else _context.context()
  shared_name = _execute.make_str(shared_name, "shared_name")
  container = _execute.make_str(container, "container")
  buffer_size = _execute.make_int(buffer_size, "buffer_size")
  string_arg = _ops.convert_to_tensor(string_arg, _dtypes.string)
  target_device = _ops.convert_to_tensor(target_device, _dtypes.string)
  _inputs_flat = [string_arg, target_device]
  _attrs = ("shared_name", shared_name, "container", container, "f", f,
  "buffer_size", buffer_size)
  _result = _execute.execute(b"FunctionBufferingResource", 1,
                             inputs=_inputs_flat, attrs=_attrs, ctx=_ctx,
                             name=name)
  _execute.record_gradient(
      "FunctionBufferingResource", _inputs_flat, _attrs, _result, name)
  _result, = _result
  return _result

_ops.RegisterShape("FunctionBufferingResource")(None)


@tf_export('function_buffering_resource_get_next')
def function_buffering_resource_get_next(function_buffer_resource, output_types, name=None):
  r"""Gets the next element from a FunctionBufferingResource.

  Args:
    function_buffer_resource: A `Tensor` of type `resource`.
      The FunctionBufferingResource handle.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
      The type list for the return values.
    name: A name for the operation (optional).

  Returns:
    A list of `Tensor` objects of type `output_types`.
    A list of return values.
  """
  _ctx = _context._context
  if _ctx is None or not _ctx._eager_context.is_eager:
    if not isinstance(output_types, (list, tuple)):
      raise TypeError(
          "Expected list for 'output_types' argument to "
          "'function_buffering_resource_get_next' Op, not %r." % output_types)
    output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
    _, _, _op = _op_def_lib._apply_op_helper(
        "FunctionBufferingResourceGetNext",
        function_buffer_resource=function_buffer_resource,
        output_types=output_types, name=name)
    _result = _op.outputs[:]
    if not _result:
      return _op
    _inputs_flat = _op.inputs
    _attrs = ("output_types", _op.get_attr("output_types"))
    _execute.record_gradient(
      "FunctionBufferingResourceGetNext", _inputs_flat, _attrs, _result, name)
    return _result

  else:
    try:
      _result = _pywrap_tensorflow.TFE_Py_FastPathExecute(
        _ctx._context_handle, _ctx._eager_context.device_name,
        "FunctionBufferingResourceGetNext", name,
        _ctx._post_execution_callbacks, function_buffer_resource,
        "output_types", output_types)
      return _result
    except _core._FallbackException:
      return function_buffering_resource_get_next_eager_fallback(
          function_buffer_resource, output_types=output_types, name=name,
          ctx=_ctx)
    except _core._NotOkStatusException as e:
      if name is not None:
        message = e.message + " name: " + name
      else:
        message = e.message
      _six.raise_from(_core._status_to_exception(e.code, message), None)


def function_buffering_resource_get_next_eager_fallback(function_buffer_resource, output_types, name=None, ctx=None):
  r"""This is the slowpath function for Eager mode.
  This is for function function_buffering_resource_get_next
  """
  _ctx = ctx if ctx else _context.context()
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'function_buffering_resource_get_next' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  function_buffer_resource = _ops.convert_to_tensor(function_buffer_resource, _dtypes.resource)
  _inputs_flat = [function_buffer_resource]
  _attrs = ("output_types", output_types)
  _result = _execute.execute(b"FunctionBufferingResourceGetNext",
                             len(output_types), inputs=_inputs_flat,
                             attrs=_attrs, ctx=_ctx, name=name)
  _execute.record_gradient(
      "FunctionBufferingResourceGetNext", _inputs_flat, _attrs, _result, name)
  return _result

_ops.RegisterShape("FunctionBufferingResourceGetNext")(None)


@tf_export('function_buffering_resource_reset')
def function_buffering_resource_reset(function_buffer_resource, name=None):
  r"""Resets the FunctionBufferingResource.

  Args:
    function_buffer_resource: A `Tensor` of type `resource`.
      The FunctionBufferingResource handle.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context._context
  if _ctx is None or not _ctx._eager_context.is_eager:
    _, _, _op = _op_def_lib._apply_op_helper(
        "FunctionBufferingResourceReset",
        function_buffer_resource=function_buffer_resource, name=name)
    return _op
    _result = None
    return _result

  else:
    try:
      _result = _pywrap_tensorflow.TFE_Py_FastPathExecute(
        _ctx._context_handle, _ctx._eager_context.device_name,
        "FunctionBufferingResourceReset", name,
        _ctx._post_execution_callbacks, function_buffer_resource)
      return _result
    except _core._FallbackException:
      return function_buffering_resource_reset_eager_fallback(
          function_buffer_resource, name=name, ctx=_ctx)
    except _core._NotOkStatusException as e:
      if name is not None:
        message = e.message + " name: " + name
      else:
        message = e.message
      _six.raise_from(_core._status_to_exception(e.code, message), None)


def function_buffering_resource_reset_eager_fallback(function_buffer_resource, name=None, ctx=None):
  r"""This is the slowpath function for Eager mode.
  This is for function function_buffering_resource_reset
  """
  _ctx = ctx if ctx else _context.context()
  function_buffer_resource = _ops.convert_to_tensor(function_buffer_resource, _dtypes.resource)
  _inputs_flat = [function_buffer_resource]
  _attrs = None
  _result = _execute.execute(b"FunctionBufferingResourceReset", 0,
                             inputs=_inputs_flat, attrs=_attrs, ctx=_ctx,
                             name=name)
  _result = None
  return _result

_ops.RegisterShape("FunctionBufferingResourceReset")(None)


@tf_export('ignore_errors_dataset')
def ignore_errors_dataset(input_dataset, output_types, output_shapes, name=None):
  r"""Creates a dataset that contains the elements of `input_dataset` ignoring errors.

  Args:
    input_dataset: A `Tensor` of type `variant`.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context
  if _ctx is None or not _ctx._eager_context.is_eager:
    if not isinstance(output_types, (list, tuple)):
      raise TypeError(
          "Expected list for 'output_types' argument to "
          "'ignore_errors_dataset' Op, not %r." % output_types)
    output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
    if not isinstance(output_shapes, (list, tuple)):
      raise TypeError(
          "Expected list for 'output_shapes' argument to "
          "'ignore_errors_dataset' Op, not %r." % output_shapes)
    output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
    _, _, _op = _op_def_lib._apply_op_helper(
        "IgnoreErrorsDataset", input_dataset=input_dataset,
        output_types=output_types, output_shapes=output_shapes, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("output_types", _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"))
    _execute.record_gradient(
      "IgnoreErrorsDataset", _inputs_flat, _attrs, _result, name)
    _result, = _result
    return _result

  else:
    try:
      _result = _pywrap_tensorflow.TFE_Py_FastPathExecute(
        _ctx._context_handle, _ctx._eager_context.device_name,
        "IgnoreErrorsDataset", name, _ctx._post_execution_callbacks,
        input_dataset, "output_types", output_types, "output_shapes",
        output_shapes)
      return _result
    except _core._FallbackException:
      return ignore_errors_dataset_eager_fallback(
          input_dataset, output_types=output_types,
          output_shapes=output_shapes, name=name, ctx=_ctx)
    except _core._NotOkStatusException as e:
      if name is not None:
        message = e.message + " name: " + name
      else:
        message = e.message
      _six.raise_from(_core._status_to_exception(e.code, message), None)


def ignore_errors_dataset_eager_fallback(input_dataset, output_types, output_shapes, name=None, ctx=None):
  r"""This is the slowpath function for Eager mode.
  This is for function ignore_errors_dataset
  """
  _ctx = ctx if ctx else _context.context()
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'ignore_errors_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'ignore_errors_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  input_dataset = _ops.convert_to_tensor(input_dataset, _dtypes.variant)
  _inputs_flat = [input_dataset]
  _attrs = ("output_types", output_types, "output_shapes", output_shapes)
  _result = _execute.execute(b"IgnoreErrorsDataset", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=_ctx, name=name)
  _execute.record_gradient(
      "IgnoreErrorsDataset", _inputs_flat, _attrs, _result, name)
  _result, = _result
  return _result

_ops.RegisterShape("IgnoreErrorsDataset")(None)


@tf_export('iterator_get_device')
def iterator_get_device(resource, name=None):
  r"""Returns the name of the device on which `resource` has been placed.

  Args:
    resource: A `Tensor` of type `resource`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`.
  """
  _ctx = _context._context
  if _ctx is None or not _ctx._eager_context.is_eager:
    _, _, _op = _op_def_lib._apply_op_helper(
        "IteratorGetDevice", resource=resource, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = None
    _execute.record_gradient(
      "IteratorGetDevice", _inputs_flat, _attrs, _result, name)
    _result, = _result
    return _result

  else:
    try:
      _result = _pywrap_tensorflow.TFE_Py_FastPathExecute(
        _ctx._context_handle, _ctx._eager_context.device_name,
        "IteratorGetDevice", name, _ctx._post_execution_callbacks, resource)
      return _result
    except _core._FallbackException:
      return iterator_get_device_eager_fallback(
          resource, name=name, ctx=_ctx)
    except _core._NotOkStatusException as e:
      if name is not None:
        message = e.message + " name: " + name
      else:
        message = e.message
      _six.raise_from(_core._status_to_exception(e.code, message), None)


def iterator_get_device_eager_fallback(resource, name=None, ctx=None):
  r"""This is the slowpath function for Eager mode.
  This is for function iterator_get_device
  """
  _ctx = ctx if ctx else _context.context()
  resource = _ops.convert_to_tensor(resource, _dtypes.resource)
  _inputs_flat = [resource]
  _attrs = None
  _result = _execute.execute(b"IteratorGetDevice", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=_ctx, name=name)
  _execute.record_gradient(
      "IteratorGetDevice", _inputs_flat, _attrs, _result, name)
  _result, = _result
  return _result

_ops.RegisterShape("IteratorGetDevice")(None)


@tf_export('thread_pool_dataset')
def thread_pool_dataset(input_dataset, thread_pool, output_types, output_shapes, name=None):
  r"""Creates a dataset that uses a custom thread pool to compute `input_dataset`.

  Args:
    input_dataset: A `Tensor` of type `variant`.
    thread_pool: A `Tensor` of type `resource`.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
    A resource produced by the ThreadPoolHandle op.
  """
  _ctx = _context._context
  if _ctx is None or not _ctx._eager_context.is_eager:
    if not isinstance(output_types, (list, tuple)):
      raise TypeError(
          "Expected list for 'output_types' argument to "
          "'thread_pool_dataset' Op, not %r." % output_types)
    output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
    if not isinstance(output_shapes, (list, tuple)):
      raise TypeError(
          "Expected list for 'output_shapes' argument to "
          "'thread_pool_dataset' Op, not %r." % output_shapes)
    output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
    _, _, _op = _op_def_lib._apply_op_helper(
        "ThreadPoolDataset", input_dataset=input_dataset,
        thread_pool=thread_pool, output_types=output_types,
        output_shapes=output_shapes, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("output_types", _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"))
    _execute.record_gradient(
      "ThreadPoolDataset", _inputs_flat, _attrs, _result, name)
    _result, = _result
    return _result

  else:
    try:
      _result = _pywrap_tensorflow.TFE_Py_FastPathExecute(
        _ctx._context_handle, _ctx._eager_context.device_name,
        "ThreadPoolDataset", name, _ctx._post_execution_callbacks,
        input_dataset, thread_pool, "output_types", output_types,
        "output_shapes", output_shapes)
      return _result
    except _core._FallbackException:
      return thread_pool_dataset_eager_fallback(
          input_dataset, thread_pool, output_types=output_types,
          output_shapes=output_shapes, name=name, ctx=_ctx)
    except _core._NotOkStatusException as e:
      if name is not None:
        message = e.message + " name: " + name
      else:
        message = e.message
      _six.raise_from(_core._status_to_exception(e.code, message), None)


def thread_pool_dataset_eager_fallback(input_dataset, thread_pool, output_types, output_shapes, name=None, ctx=None):
  r"""This is the slowpath function for Eager mode.
  This is for function thread_pool_dataset
  """
  _ctx = ctx if ctx else _context.context()
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'thread_pool_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'thread_pool_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  input_dataset = _ops.convert_to_tensor(input_dataset, _dtypes.variant)
  thread_pool = _ops.convert_to_tensor(thread_pool, _dtypes.resource)
  _inputs_flat = [input_dataset, thread_pool]
  _attrs = ("output_types", output_types, "output_shapes", output_shapes)
  _result = _execute.execute(b"ThreadPoolDataset", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=_ctx, name=name)
  _execute.record_gradient(
      "ThreadPoolDataset", _inputs_flat, _attrs, _result, name)
  _result, = _result
  return _result

_ops.RegisterShape("ThreadPoolDataset")(None)


@tf_export('thread_pool_handle')
def thread_pool_handle(num_threads, display_name, container="", shared_name="", name=None):
  r"""Creates a custom thread pool with the given number of threads.

  Args:
    num_threads: An `int`. The number of threads in the thread pool.
    display_name: A `string`.
      A human-readable name for the threads that may be visible in
      some visualizations.
    container: An optional `string`. Defaults to `""`.
    shared_name: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `resource`.
    A resource that can be consumed by one or more ThreadPoolDataset ops.
  """
  _ctx = _context._context
  if _ctx is None or not _ctx._eager_context.is_eager:
    num_threads = _execute.make_int(num_threads, "num_threads")
    display_name = _execute.make_str(display_name, "display_name")
    if container is None:
      container = ""
    container = _execute.make_str(container, "container")
    if shared_name is None:
      shared_name = ""
    shared_name = _execute.make_str(shared_name, "shared_name")
    _, _, _op = _op_def_lib._apply_op_helper(
        "ThreadPoolHandle", num_threads=num_threads,
        display_name=display_name, container=container,
        shared_name=shared_name, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("num_threads", _op.get_attr("num_threads"), "display_name",
              _op.get_attr("display_name"), "container",
              _op.get_attr("container"), "shared_name",
              _op.get_attr("shared_name"))
    _execute.record_gradient(
      "ThreadPoolHandle", _inputs_flat, _attrs, _result, name)
    _result, = _result
    return _result

  else:
    try:
      _result = _pywrap_tensorflow.TFE_Py_FastPathExecute(
        _ctx._context_handle, _ctx._eager_context.device_name,
        "ThreadPoolHandle", name, _ctx._post_execution_callbacks,
        "num_threads", num_threads, "display_name", display_name, "container",
        container, "shared_name", shared_name)
      return _result
    except _core._FallbackException:
      return thread_pool_handle_eager_fallback(
          num_threads=num_threads, display_name=display_name,
          container=container, shared_name=shared_name, name=name, ctx=_ctx)
    except _core._NotOkStatusException as e:
      if name is not None:
        message = e.message + " name: " + name
      else:
        message = e.message
      _six.raise_from(_core._status_to_exception(e.code, message), None)


def thread_pool_handle_eager_fallback(num_threads, display_name, container="", shared_name="", name=None, ctx=None):
  r"""This is the slowpath function for Eager mode.
  This is for function thread_pool_handle
  """
  _ctx = ctx if ctx else _context.context()
  num_threads = _execute.make_int(num_threads, "num_threads")
  display_name = _execute.make_str(display_name, "display_name")
  if container is None:
    container = ""
  container = _execute.make_str(container, "container")
  if shared_name is None:
    shared_name = ""
  shared_name = _execute.make_str(shared_name, "shared_name")
  _inputs_flat = []
  _attrs = ("num_threads", num_threads, "display_name", display_name,
  "container", container, "shared_name", shared_name)
  _result = _execute.execute(b"ThreadPoolHandle", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=_ctx, name=name)
  _execute.record_gradient(
      "ThreadPoolHandle", _inputs_flat, _attrs, _result, name)
  _result, = _result
  return _result

_ops.RegisterShape("ThreadPoolHandle")(None)


@tf_export('unique_dataset')
def unique_dataset(input_dataset, output_types, output_shapes, name=None):
  r"""Creates a dataset that contains the unique elements of `input_dataset`.

  Args:
    input_dataset: A `Tensor` of type `variant`.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context._context
  if _ctx is None or not _ctx._eager_context.is_eager:
    if not isinstance(output_types, (list, tuple)):
      raise TypeError(
          "Expected list for 'output_types' argument to "
          "'unique_dataset' Op, not %r." % output_types)
    output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
    if not isinstance(output_shapes, (list, tuple)):
      raise TypeError(
          "Expected list for 'output_shapes' argument to "
          "'unique_dataset' Op, not %r." % output_shapes)
    output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
    _, _, _op = _op_def_lib._apply_op_helper(
        "UniqueDataset", input_dataset=input_dataset,
        output_types=output_types, output_shapes=output_shapes, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("output_types", _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"))
    _execute.record_gradient(
      "UniqueDataset", _inputs_flat, _attrs, _result, name)
    _result, = _result
    return _result

  else:
    try:
      _result = _pywrap_tensorflow.TFE_Py_FastPathExecute(
        _ctx._context_handle, _ctx._eager_context.device_name,
        "UniqueDataset", name, _ctx._post_execution_callbacks, input_dataset,
        "output_types", output_types, "output_shapes", output_shapes)
      return _result
    except _core._FallbackException:
      return unique_dataset_eager_fallback(
          input_dataset, output_types=output_types,
          output_shapes=output_shapes, name=name, ctx=_ctx)
    except _core._NotOkStatusException as e:
      if name is not None:
        message = e.message + " name: " + name
      else:
        message = e.message
      _six.raise_from(_core._status_to_exception(e.code, message), None)


def unique_dataset_eager_fallback(input_dataset, output_types, output_shapes, name=None, ctx=None):
  r"""This is the slowpath function for Eager mode.
  This is for function unique_dataset
  """
  _ctx = ctx if ctx else _context.context()
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'unique_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'unique_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  input_dataset = _ops.convert_to_tensor(input_dataset, _dtypes.variant)
  _inputs_flat = [input_dataset]
  _attrs = ("output_types", output_types, "output_shapes", output_shapes)
  _result = _execute.execute(b"UniqueDataset", 1, inputs=_inputs_flat,
                             attrs=_attrs, ctx=_ctx, name=name)
  _execute.record_gradient(
      "UniqueDataset", _inputs_flat, _attrs, _result, name)
  _result, = _result
  return _result

_ops.RegisterShape("UniqueDataset")(None)

def _InitOpDefLibrary(op_list_proto_bytes):
  op_list = _op_def_pb2.OpList()
  op_list.ParseFromString(op_list_proto_bytes)
  _op_def_registry.register_op_list(op_list)
  op_def_lib = _op_def_library.OpDefLibrary()
  op_def_lib.add_op_list(op_list)
  return op_def_lib
# op {
#   name: "FunctionBufferingResource"
#   input_arg {
#     name: "string_arg"
#     type: DT_STRING
#   }
#   input_arg {
#     name: "target_device"
#     type: DT_STRING
#   }
#   output_arg {
#     name: "resource"
#     type: DT_RESOURCE
#   }
#   attr {
#     name: "shared_name"
#     type: "string"
#   }
#   attr {
#     name: "container"
#     type: "string"
#   }
#   attr {
#     name: "f"
#     type: "func"
#   }
#   attr {
#     name: "buffer_size"
#     type: "int"
#   }
#   is_stateful: true
# }
# op {
#   name: "FunctionBufferingResourceGetNext"
#   input_arg {
#     name: "function_buffer_resource"
#     type: DT_RESOURCE
#   }
#   output_arg {
#     name: "output"
#     type_list_attr: "output_types"
#   }
#   attr {
#     name: "output_types"
#     type: "list(type)"
#     has_minimum: true
#     minimum: 1
#   }
#   is_stateful: true
# }
# op {
#   name: "FunctionBufferingResourceReset"
#   input_arg {
#     name: "function_buffer_resource"
#     type: DT_RESOURCE
#   }
#   is_stateful: true
# }
# op {
#   name: "IgnoreErrorsDataset"
#   input_arg {
#     name: "input_dataset"
#     type: DT_VARIANT
#   }
#   output_arg {
#     name: "handle"
#     type: DT_VARIANT
#   }
#   attr {
#     name: "output_types"
#     type: "list(type)"
#     has_minimum: true
#     minimum: 1
#   }
#   attr {
#     name: "output_shapes"
#     type: "list(shape)"
#     has_minimum: true
#     minimum: 1
#   }
# }
# op {
#   name: "IteratorGetDevice"
#   input_arg {
#     name: "resource"
#     type: DT_RESOURCE
#   }
#   output_arg {
#     name: "device"
#     type: DT_STRING
#   }
#   is_stateful: true
# }
# op {
#   name: "ThreadPoolDataset"
#   input_arg {
#     name: "input_dataset"
#     type: DT_VARIANT
#   }
#   input_arg {
#     name: "thread_pool"
#     type: DT_RESOURCE
#   }
#   output_arg {
#     name: "handle"
#     type: DT_VARIANT
#   }
#   attr {
#     name: "output_types"
#     type: "list(type)"
#     has_minimum: true
#     minimum: 1
#   }
#   attr {
#     name: "output_shapes"
#     type: "list(shape)"
#     has_minimum: true
#     minimum: 1
#   }
#   is_stateful: true
# }
# op {
#   name: "ThreadPoolHandle"
#   output_arg {
#     name: "handle"
#     type: DT_RESOURCE
#   }
#   attr {
#     name: "num_threads"
#     type: "int"
#   }
#   attr {
#     name: "display_name"
#     type: "string"
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
#   name: "UniqueDataset"
#   input_arg {
#     name: "input_dataset"
#     type: DT_VARIANT
#   }
#   output_arg {
#     name: "handle"
#     type: DT_VARIANT
#   }
#   attr {
#     name: "output_types"
#     type: "list(type)"
#     has_minimum: true
#     minimum: 1
#   }
#   attr {
#     name: "output_shapes"
#     type: "list(shape)"
#     has_minimum: true
#     minimum: 1
#   }
# }
_op_def_lib = _InitOpDefLibrary(b"\n\232\001\n\031FunctionBufferingResource\022\016\n\nstring_arg\030\007\022\021\n\rtarget_device\030\007\032\014\n\010resource\030\024\"\025\n\013shared_name\022\006string\"\023\n\tcontainer\022\006string\"\t\n\001f\022\004func\"\022\n\013buffer_size\022\003int\210\001\001\n{\n FunctionBufferingResourceGetNext\022\034\n\030function_buffer_resource\030\024\032\026\n\006output2\014output_types\"\036\n\014output_types\022\nlist(type)(\0010\001\210\001\001\nA\n\036FunctionBufferingResourceReset\022\034\n\030function_buffer_resource\030\024\210\001\001\nv\n\023IgnoreErrorsDataset\022\021\n\rinput_dataset\030\025\032\n\n\006handle\030\025\"\036\n\014output_types\022\nlist(type)(\0010\001\" \n\routput_shapes\022\013list(shape)(\0010\001\n0\n\021IteratorGetDevice\022\014\n\010resource\030\024\032\n\n\006device\030\007\210\001\001\n\210\001\n\021ThreadPoolDataset\022\021\n\rinput_dataset\030\025\022\017\n\013thread_pool\030\024\032\n\n\006handle\030\025\"\036\n\014output_types\022\nlist(type)(\0010\001\" \n\routput_shapes\022\013list(shape)(\0010\001\210\001\001\n\201\001\n\020ThreadPoolHandle\032\n\n\006handle\030\024\"\022\n\013num_threads\022\003int\"\026\n\014display_name\022\006string\"\027\n\tcontainer\022\006string\032\002\022\000\"\031\n\013shared_name\022\006string\032\002\022\000\210\001\001\np\n\rUniqueDataset\022\021\n\rinput_dataset\030\025\032\n\n\006handle\030\025\"\036\n\014output_types\022\nlist(type)(\0010\001\" \n\routput_shapes\022\013list(shape)(\0010\001")
