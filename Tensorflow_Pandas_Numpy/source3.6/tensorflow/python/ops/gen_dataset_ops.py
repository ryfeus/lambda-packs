"""Python wrappers around TensorFlow ops.

This file is MACHINE GENERATED! Do not edit.
Original C++ source file: dataset_ops.cc
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


def batch_dataset(input_dataset, batch_size, output_types, output_shapes, name=None):
  r"""Creates a dataset that batches `batch_size` elements from `input_dataset`.

  Args:
    input_dataset: A `Tensor` of type `variant`.
    batch_size: A `Tensor` of type `int64`.
      A scalar representing the number of elements to accumulate in a
      batch.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'batch_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'batch_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "BatchDataset", input_dataset=input_dataset, batch_size=batch_size,
        output_types=output_types, output_shapes=output_shapes, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("output_types", _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"))
  else:
    input_dataset = _ops.convert_to_tensor(input_dataset, _dtypes.variant)
    batch_size = _ops.convert_to_tensor(batch_size, _dtypes.int64)
    _inputs_flat = [input_dataset, batch_size]
    _attrs = ("output_types", output_types, "output_shapes", output_shapes)
    _result = _execute.execute(b"BatchDataset", 1, inputs=_inputs_flat,
                               attrs=_attrs, ctx=_ctx, name=name)
  _execute.record_gradient(
      "BatchDataset", _inputs_flat, _attrs, _result, name)
  _result, = _result
  return _result


def cache_dataset(input_dataset, filename, output_types, output_shapes, name=None):
  r"""Creates a dataset that caches elements from `input_dataset`.

  A CacheDataset will iterate over the input_dataset, and store tensors. If the
  cache already exists, the cache will be used. If the cache is inappropriate
  (e.g. cannot be opened, contains tensors of the wrong shape / size), an error
  will the returned when used.

  Args:
    input_dataset: A `Tensor` of type `variant`.
    filename: A `Tensor` of type `string`.
      A path on the filesystem where we should cache the dataset. Note: this
      will be a directory.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'cache_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'cache_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "CacheDataset", input_dataset=input_dataset, filename=filename,
        output_types=output_types, output_shapes=output_shapes, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("output_types", _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"))
  else:
    input_dataset = _ops.convert_to_tensor(input_dataset, _dtypes.variant)
    filename = _ops.convert_to_tensor(filename, _dtypes.string)
    _inputs_flat = [input_dataset, filename]
    _attrs = ("output_types", output_types, "output_shapes", output_shapes)
    _result = _execute.execute(b"CacheDataset", 1, inputs=_inputs_flat,
                               attrs=_attrs, ctx=_ctx, name=name)
  _execute.record_gradient(
      "CacheDataset", _inputs_flat, _attrs, _result, name)
  _result, = _result
  return _result


def concatenate_dataset(input_dataset, another_dataset, output_types, output_shapes, name=None):
  r"""Creates a dataset that concatenates `input_dataset` with `another_dataset`.

  Args:
    input_dataset: A `Tensor` of type `variant`.
    another_dataset: A `Tensor` of type `variant`.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'concatenate_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'concatenate_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "ConcatenateDataset", input_dataset=input_dataset,
        another_dataset=another_dataset, output_types=output_types,
        output_shapes=output_shapes, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("output_types", _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"))
  else:
    input_dataset = _ops.convert_to_tensor(input_dataset, _dtypes.variant)
    another_dataset = _ops.convert_to_tensor(another_dataset, _dtypes.variant)
    _inputs_flat = [input_dataset, another_dataset]
    _attrs = ("output_types", output_types, "output_shapes", output_shapes)
    _result = _execute.execute(b"ConcatenateDataset", 1, inputs=_inputs_flat,
                               attrs=_attrs, ctx=_ctx, name=name)
  _execute.record_gradient(
      "ConcatenateDataset", _inputs_flat, _attrs, _result, name)
  _result, = _result
  return _result


def dense_to_sparse_batch_dataset(input_dataset, batch_size, row_shape, output_types, output_shapes, name=None):
  r"""Creates a dataset that yields a SparseTensor for each element of the input.

  Args:
    input_dataset: A `Tensor` of type `variant`.
      A handle to an input dataset. Must have a single component.
    batch_size: A `Tensor` of type `int64`.
      A scalar representing the number of elements to accumulate in a
      batch.
    row_shape: A `Tensor` of type `int64`.
      A vector representing the dense shape of each row in the produced
      SparseTensor. The shape may be partially specified, using `-1` to indicate
      that a particular dimension should use the maximum size of all batch elements.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'dense_to_sparse_batch_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'dense_to_sparse_batch_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "DenseToSparseBatchDataset", input_dataset=input_dataset,
        batch_size=batch_size, row_shape=row_shape, output_types=output_types,
        output_shapes=output_shapes, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("output_types", _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"))
  else:
    input_dataset = _ops.convert_to_tensor(input_dataset, _dtypes.variant)
    batch_size = _ops.convert_to_tensor(batch_size, _dtypes.int64)
    row_shape = _ops.convert_to_tensor(row_shape, _dtypes.int64)
    _inputs_flat = [input_dataset, batch_size, row_shape]
    _attrs = ("output_types", output_types, "output_shapes", output_shapes)
    _result = _execute.execute(b"DenseToSparseBatchDataset", 1,
                               inputs=_inputs_flat, attrs=_attrs, ctx=_ctx,
                               name=name)
  _execute.record_gradient(
      "DenseToSparseBatchDataset", _inputs_flat, _attrs, _result, name)
  _result, = _result
  return _result


def filter_dataset(input_dataset, other_arguments, predicate, output_types, output_shapes, name=None):
  r"""Creates a dataset containing elements of `input_dataset` matching `predicate`.

  The `predicate` function must return a scalar boolean and accept the
  following arguments:

  * One tensor for each component of an element of `input_dataset`.
  * One tensor for each value in `other_arguments`.

  Args:
    input_dataset: A `Tensor` of type `variant`.
    other_arguments: A list of `Tensor` objects.
      A list of tensors, typically values that were captured when
      building a closure for `predicate`.
    predicate: A function decorated with @Defun.
      A function returning a scalar boolean.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'filter_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'filter_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "FilterDataset", input_dataset=input_dataset,
        other_arguments=other_arguments, predicate=predicate,
        output_types=output_types, output_shapes=output_shapes, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("predicate", _op.get_attr("predicate"), "Targuments",
              _op.get_attr("Targuments"), "output_types",
              _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"))
  else:
    _attr_Targuments, other_arguments = _execute.convert_to_mixed_eager_tensors(other_arguments, _ctx)
    _attr_Targuments = [_t.as_datatype_enum for _t in _attr_Targuments]
    input_dataset = _ops.convert_to_tensor(input_dataset, _dtypes.variant)
    _inputs_flat = [input_dataset] + list(other_arguments)
    _attrs = ("predicate", predicate, "Targuments", _attr_Targuments,
              "output_types", output_types, "output_shapes", output_shapes)
    _result = _execute.execute(b"FilterDataset", 1, inputs=_inputs_flat,
                               attrs=_attrs, ctx=_ctx, name=name)
  _execute.record_gradient(
      "FilterDataset", _inputs_flat, _attrs, _result, name)
  _result, = _result
  return _result


def fixed_length_record_dataset(filenames, header_bytes, record_bytes, footer_bytes, buffer_size, name=None):
  r"""Creates a dataset that emits the records from one or more binary files.

  Args:
    filenames: A `Tensor` of type `string`.
      A scalar or a vector containing the name(s) of the file(s) to be
      read.
    header_bytes: A `Tensor` of type `int64`.
      A scalar representing the number of bytes to skip at the
      beginning of a file.
    record_bytes: A `Tensor` of type `int64`.
      A scalar representing the number of bytes in each record.
    footer_bytes: A `Tensor` of type `int64`.
      A scalar representing the number of bytes to skip at the end
      of a file.
    buffer_size: A `Tensor` of type `int64`.
      A scalar representing the number of bytes to buffer. Must be > 0.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "FixedLengthRecordDataset", filenames=filenames,
        header_bytes=header_bytes, record_bytes=record_bytes,
        footer_bytes=footer_bytes, buffer_size=buffer_size, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = None
  else:
    filenames = _ops.convert_to_tensor(filenames, _dtypes.string)
    header_bytes = _ops.convert_to_tensor(header_bytes, _dtypes.int64)
    record_bytes = _ops.convert_to_tensor(record_bytes, _dtypes.int64)
    footer_bytes = _ops.convert_to_tensor(footer_bytes, _dtypes.int64)
    buffer_size = _ops.convert_to_tensor(buffer_size, _dtypes.int64)
    _inputs_flat = [filenames, header_bytes, record_bytes, footer_bytes, buffer_size]
    _attrs = None
    _result = _execute.execute(b"FixedLengthRecordDataset", 1,
                               inputs=_inputs_flat, attrs=_attrs, ctx=_ctx,
                               name=name)
  _execute.record_gradient(
      "FixedLengthRecordDataset", _inputs_flat, _attrs, _result, name)
  _result, = _result
  return _result


def flat_map_dataset(input_dataset, other_arguments, f, output_types, output_shapes, name=None):
  r"""Creates a dataset that applies `f` to the outputs of `input_dataset`.

  Unlike MapDataset, the `f` in FlatMapDataset is expected to return a
  Dataset variant, and FlatMapDataset will flatten successive results
  into a single Dataset.

  Args:
    input_dataset: A `Tensor` of type `variant`.
    other_arguments: A list of `Tensor` objects.
    f: A function decorated with @Defun.
      A function mapping elements of `input_dataset`, concatenated with
      `other_arguments`, to a Dataset variant that contains elements matching
      `output_types` and `output_shapes`.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'flat_map_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'flat_map_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "FlatMapDataset", input_dataset=input_dataset,
        other_arguments=other_arguments, f=f, output_types=output_types,
        output_shapes=output_shapes, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("f", _op.get_attr("f"), "Targuments",
              _op.get_attr("Targuments"), "output_types",
              _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"))
  else:
    _attr_Targuments, other_arguments = _execute.convert_to_mixed_eager_tensors(other_arguments, _ctx)
    _attr_Targuments = [_t.as_datatype_enum for _t in _attr_Targuments]
    input_dataset = _ops.convert_to_tensor(input_dataset, _dtypes.variant)
    _inputs_flat = [input_dataset] + list(other_arguments)
    _attrs = ("f", f, "Targuments", _attr_Targuments, "output_types",
              output_types, "output_shapes", output_shapes)
    _result = _execute.execute(b"FlatMapDataset", 1, inputs=_inputs_flat,
                               attrs=_attrs, ctx=_ctx, name=name)
  _execute.record_gradient(
      "FlatMapDataset", _inputs_flat, _attrs, _result, name)
  _result, = _result
  return _result


def group_by_window_dataset(input_dataset, key_func_other_arguments, reduce_func_other_arguments, window_size_func_other_arguments, key_func, reduce_func, window_size_func, output_types, output_shapes, name=None):
  r"""Creates a dataset that computes a windowed group-by on `input_dataset`.

  // TODO(mrry): Support non-int64 keys.

  Args:
    input_dataset: A `Tensor` of type `variant`.
    key_func_other_arguments: A list of `Tensor` objects.
    reduce_func_other_arguments: A list of `Tensor` objects.
    window_size_func_other_arguments: A list of `Tensor` objects.
    key_func: A function decorated with @Defun.
      A function mapping an element of `input_dataset`, concatenated
      with `key_func_other_arguments` to a scalar value of type DT_INT64.
    reduce_func: A function decorated with @Defun.
    window_size_func: A function decorated with @Defun.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'group_by_window_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'group_by_window_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "GroupByWindowDataset", input_dataset=input_dataset,
        key_func_other_arguments=key_func_other_arguments,
        reduce_func_other_arguments=reduce_func_other_arguments,
        window_size_func_other_arguments=window_size_func_other_arguments,
        key_func=key_func, reduce_func=reduce_func,
        window_size_func=window_size_func, output_types=output_types,
        output_shapes=output_shapes, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("key_func", _op.get_attr("key_func"), "reduce_func",
              _op.get_attr("reduce_func"), "window_size_func",
              _op.get_attr("window_size_func"), "Tkey_func_other_arguments",
              _op.get_attr("Tkey_func_other_arguments"),
              "Treduce_func_other_arguments",
              _op.get_attr("Treduce_func_other_arguments"),
              "Twindow_size_func_other_arguments",
              _op.get_attr("Twindow_size_func_other_arguments"),
              "output_types", _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"))
  else:
    _attr_Tkey_func_other_arguments, key_func_other_arguments = _execute.convert_to_mixed_eager_tensors(key_func_other_arguments, _ctx)
    _attr_Tkey_func_other_arguments = [_t.as_datatype_enum for _t in _attr_Tkey_func_other_arguments]
    _attr_Treduce_func_other_arguments, reduce_func_other_arguments = _execute.convert_to_mixed_eager_tensors(reduce_func_other_arguments, _ctx)
    _attr_Treduce_func_other_arguments = [_t.as_datatype_enum for _t in _attr_Treduce_func_other_arguments]
    _attr_Twindow_size_func_other_arguments, window_size_func_other_arguments = _execute.convert_to_mixed_eager_tensors(window_size_func_other_arguments, _ctx)
    _attr_Twindow_size_func_other_arguments = [_t.as_datatype_enum for _t in _attr_Twindow_size_func_other_arguments]
    input_dataset = _ops.convert_to_tensor(input_dataset, _dtypes.variant)
    _inputs_flat = [input_dataset] + list(key_func_other_arguments) + list(reduce_func_other_arguments) + list(window_size_func_other_arguments)
    _attrs = ("key_func", key_func, "reduce_func", reduce_func,
              "window_size_func", window_size_func,
              "Tkey_func_other_arguments", _attr_Tkey_func_other_arguments,
              "Treduce_func_other_arguments",
              _attr_Treduce_func_other_arguments,
              "Twindow_size_func_other_arguments",
              _attr_Twindow_size_func_other_arguments, "output_types",
              output_types, "output_shapes", output_shapes)
    _result = _execute.execute(b"GroupByWindowDataset", 1,
                               inputs=_inputs_flat, attrs=_attrs, ctx=_ctx,
                               name=name)
  _execute.record_gradient(
      "GroupByWindowDataset", _inputs_flat, _attrs, _result, name)
  _result, = _result
  return _result


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
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "IgnoreErrorsDataset", input_dataset=input_dataset,
        output_types=output_types, output_shapes=output_shapes, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("output_types", _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"))
  else:
    input_dataset = _ops.convert_to_tensor(input_dataset, _dtypes.variant)
    _inputs_flat = [input_dataset]
    _attrs = ("output_types", output_types, "output_shapes", output_shapes)
    _result = _execute.execute(b"IgnoreErrorsDataset", 1, inputs=_inputs_flat,
                               attrs=_attrs, ctx=_ctx, name=name)
  _execute.record_gradient(
      "IgnoreErrorsDataset", _inputs_flat, _attrs, _result, name)
  _result, = _result
  return _result


def interleave_dataset(input_dataset, other_arguments, cycle_length, block_length, f, output_types, output_shapes, name=None):
  r"""Creates a dataset that applies `f` to the outputs of `input_dataset`.

  Unlike MapDataset, the `f` in InterleaveDataset is expected to return
  a Dataset variant, and InterleaveDataset will flatten successive
  results into a single Dataset. Unlike FlatMapDataset,
  InterleaveDataset will interleave sequences of up to `block_length`
  consecutive elements from `cycle_length` input elements.

  Args:
    input_dataset: A `Tensor` of type `variant`.
    other_arguments: A list of `Tensor` objects.
    cycle_length: A `Tensor` of type `int64`.
    block_length: A `Tensor` of type `int64`.
    f: A function decorated with @Defun.
      A function mapping elements of `input_dataset`, concatenated with
      `other_arguments`, to a Dataset variant that contains elements matching
      `output_types` and `output_shapes`.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'interleave_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'interleave_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "InterleaveDataset", input_dataset=input_dataset,
        other_arguments=other_arguments, cycle_length=cycle_length,
        block_length=block_length, f=f, output_types=output_types,
        output_shapes=output_shapes, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("f", _op.get_attr("f"), "Targuments",
              _op.get_attr("Targuments"), "output_types",
              _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"))
  else:
    _attr_Targuments, other_arguments = _execute.convert_to_mixed_eager_tensors(other_arguments, _ctx)
    _attr_Targuments = [_t.as_datatype_enum for _t in _attr_Targuments]
    input_dataset = _ops.convert_to_tensor(input_dataset, _dtypes.variant)
    cycle_length = _ops.convert_to_tensor(cycle_length, _dtypes.int64)
    block_length = _ops.convert_to_tensor(block_length, _dtypes.int64)
    _inputs_flat = [input_dataset] + list(other_arguments) + [cycle_length, block_length]
    _attrs = ("f", f, "Targuments", _attr_Targuments, "output_types",
              output_types, "output_shapes", output_shapes)
    _result = _execute.execute(b"InterleaveDataset", 1, inputs=_inputs_flat,
                               attrs=_attrs, ctx=_ctx, name=name)
  _execute.record_gradient(
      "InterleaveDataset", _inputs_flat, _attrs, _result, name)
  _result, = _result
  return _result


def iterator(shared_name, container, output_types, output_shapes, name=None):
  r"""A container for an iterator resource.

  Args:
    shared_name: A `string`.
    container: A `string`.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `resource`.
    A handle to the iterator that can be passed to a "MakeIterator"
    or "IteratorGetNext" op.
  """
  shared_name = _execute.make_str(shared_name, "shared_name")
  container = _execute.make_str(container, "container")
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'iterator' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'iterator' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "Iterator", shared_name=shared_name, container=container,
        output_types=output_types, output_shapes=output_shapes, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("shared_name", _op.get_attr("shared_name"), "container",
              _op.get_attr("container"), "output_types",
              _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"))
  else:
    _inputs_flat = []
    _attrs = ("shared_name", shared_name, "container", container,
              "output_types", output_types, "output_shapes", output_shapes)
    _result = _execute.execute(b"Iterator", 1, inputs=_inputs_flat,
                               attrs=_attrs, ctx=_ctx, name=name)
  _execute.record_gradient(
      "Iterator", _inputs_flat, _attrs, _result, name)
  _result, = _result
  return _result


def iterator_from_string_handle(string_handle, output_types=[], output_shapes=[], name=None):
  r"""Converts the given string representing a handle to an iterator to a resource.

  Args:
    string_handle: A `Tensor` of type `string`.
      A string representation of the given handle.
    output_types: An optional list of `tf.DTypes`. Defaults to `[]`.
      If specified, defines the type of each tuple component in an
      element produced by the resulting iterator.
    output_shapes: An optional list of shapes (each a `tf.TensorShape` or list of `ints`). Defaults to `[]`.
      If specified, defines the shape of each tuple component in an
      element produced by the resulting iterator.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `resource`. A handle to an iterator resource.
  """
  if output_types is None:
    output_types = []
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'iterator_from_string_handle' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if output_shapes is None:
    output_shapes = []
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'iterator_from_string_handle' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "IteratorFromStringHandle", string_handle=string_handle,
        output_types=output_types, output_shapes=output_shapes, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("output_types", _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"))
  else:
    string_handle = _ops.convert_to_tensor(string_handle, _dtypes.string)
    _inputs_flat = [string_handle]
    _attrs = ("output_types", output_types, "output_shapes", output_shapes)
    _result = _execute.execute(b"IteratorFromStringHandle", 1,
                               inputs=_inputs_flat, attrs=_attrs, ctx=_ctx,
                               name=name)
  _execute.record_gradient(
      "IteratorFromStringHandle", _inputs_flat, _attrs, _result, name)
  _result, = _result
  return _result


def iterator_get_next(iterator, output_types, output_shapes, name=None):
  r"""Gets the next output from the given iterator.

  Args:
    iterator: A `Tensor` of type `resource`.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    name: A name for the operation (optional).

  Returns:
    A list of `Tensor` objects of type `output_types`.
  """
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'iterator_get_next' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'iterator_get_next' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "IteratorGetNext", iterator=iterator, output_types=output_types,
        output_shapes=output_shapes, name=name)
    _result = _op.outputs[:]
    if not _result:
      return _op
    _inputs_flat = _op.inputs
    _attrs = ("output_types", _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"))
  else:
    iterator = _ops.convert_to_tensor(iterator, _dtypes.resource)
    _inputs_flat = [iterator]
    _attrs = ("output_types", output_types, "output_shapes", output_shapes)
    _result = _execute.execute(b"IteratorGetNext", len(output_types),
                               inputs=_inputs_flat, attrs=_attrs, ctx=_ctx,
                               name=name)
  _execute.record_gradient(
      "IteratorGetNext", _inputs_flat, _attrs, _result, name)
  return _result


def iterator_to_string_handle(resource_handle, name=None):
  r"""Converts the given `resource_handle` representing an iterator to a string.

  Args:
    resource_handle: A `Tensor` of type `resource`.
      A handle to an iterator resource.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`. A string representation of the given handle.
  """
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "IteratorToStringHandle", resource_handle=resource_handle, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = None
  else:
    resource_handle = _ops.convert_to_tensor(resource_handle, _dtypes.resource)
    _inputs_flat = [resource_handle]
    _attrs = None
    _result = _execute.execute(b"IteratorToStringHandle", 1,
                               inputs=_inputs_flat, attrs=_attrs, ctx=_ctx,
                               name=name)
  _execute.record_gradient(
      "IteratorToStringHandle", _inputs_flat, _attrs, _result, name)
  _result, = _result
  return _result


def make_iterator(dataset, iterator, name=None):
  r"""Makes a new iterator from the given `dataset` and stores it in `iterator`.

  This operation may be executed multiple times. Each execution will reset the
  iterator in `iterator` to the first element of `dataset`.

  Args:
    dataset: A `Tensor` of type `variant`.
    iterator: A `Tensor` of type `resource`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "MakeIterator", dataset=dataset, iterator=iterator, name=name)
    return _op
  else:
    dataset = _ops.convert_to_tensor(dataset, _dtypes.variant)
    iterator = _ops.convert_to_tensor(iterator, _dtypes.resource)
    _inputs_flat = [dataset, iterator]
    _attrs = None
    _result = _execute.execute(b"MakeIterator", 0, inputs=_inputs_flat,
                               attrs=_attrs, ctx=_ctx, name=name)
  return _result


def map_dataset(input_dataset, other_arguments, f, output_types, output_shapes, name=None):
  r"""Creates a dataset that applies `f` to the outputs of `input_dataset`.

  Args:
    input_dataset: A `Tensor` of type `variant`.
    other_arguments: A list of `Tensor` objects.
    f: A function decorated with @Defun.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'map_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'map_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "MapDataset", input_dataset=input_dataset,
        other_arguments=other_arguments, f=f, output_types=output_types,
        output_shapes=output_shapes, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("f", _op.get_attr("f"), "Targuments",
              _op.get_attr("Targuments"), "output_types",
              _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"))
  else:
    _attr_Targuments, other_arguments = _execute.convert_to_mixed_eager_tensors(other_arguments, _ctx)
    _attr_Targuments = [_t.as_datatype_enum for _t in _attr_Targuments]
    input_dataset = _ops.convert_to_tensor(input_dataset, _dtypes.variant)
    _inputs_flat = [input_dataset] + list(other_arguments)
    _attrs = ("f", f, "Targuments", _attr_Targuments, "output_types",
              output_types, "output_shapes", output_shapes)
    _result = _execute.execute(b"MapDataset", 1, inputs=_inputs_flat,
                               attrs=_attrs, ctx=_ctx, name=name)
  _execute.record_gradient(
      "MapDataset", _inputs_flat, _attrs, _result, name)
  _result, = _result
  return _result


def one_shot_iterator(dataset_factory, output_types, output_shapes, container="", shared_name="", name=None):
  r"""Makes a "one-shot" iterator that can be iterated only once.

  A one-shot iterator bundles the logic for defining the dataset and
  the state of the iterator in a single op, which allows simple input
  pipelines to be defined without an additional initialization
  ("MakeIterator") step.

  One-shot iterators have the following limitations:

  * They do not support parameterization: all logic for creating the underlying
    dataset must be bundled in the `dataset_factory` function.
  * They are not resettable. Once a one-shot iterator reaches the end of its
    underlying dataset, subsequent "IteratorGetNext" operations on that
    iterator will always produce an `OutOfRange` error.

  For greater flexibility, use "Iterator" and "MakeIterator" to define
  an iterator using an arbitrary subgraph, which may capture tensors
  (including fed values) as parameters, and which may be reset multiple
  times by rerunning "MakeIterator".

  Args:
    dataset_factory: A function decorated with @Defun.
      A function of type `() -> DT_VARIANT`, where the returned
      DT_VARIANT is a dataset.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    container: An optional `string`. Defaults to `""`.
    shared_name: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `resource`.
    A handle to the iterator that can be passed to an "IteratorGetNext"
    op.
  """
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'one_shot_iterator' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'one_shot_iterator' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if container is None:
    container = ""
  container = _execute.make_str(container, "container")
  if shared_name is None:
    shared_name = ""
  shared_name = _execute.make_str(shared_name, "shared_name")
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "OneShotIterator", dataset_factory=dataset_factory,
        output_types=output_types, output_shapes=output_shapes,
        container=container, shared_name=shared_name, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("dataset_factory", _op.get_attr("dataset_factory"),
              "output_types", _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"), "container",
              _op.get_attr("container"), "shared_name",
              _op.get_attr("shared_name"))
  else:
    _inputs_flat = []
    _attrs = ("dataset_factory", dataset_factory, "output_types",
              output_types, "output_shapes", output_shapes, "container",
              container, "shared_name", shared_name)
    _result = _execute.execute(b"OneShotIterator", 1, inputs=_inputs_flat,
                               attrs=_attrs, ctx=_ctx, name=name)
  _execute.record_gradient(
      "OneShotIterator", _inputs_flat, _attrs, _result, name)
  _result, = _result
  return _result


def padded_batch_dataset(input_dataset, batch_size, padded_shapes, padding_values, output_shapes, name=None):
  r"""Creates a dataset that batches and pads `batch_size` elements from the input.

  Args:
    input_dataset: A `Tensor` of type `variant`.
    batch_size: A `Tensor` of type `int64`.
      A scalar representing the number of elements to accumulate in a
      batch.
    padded_shapes: A list of at least 1 `Tensor` objects with type `int64`.
      A list of int64 tensors representing the desired padded shapes
      of the corresponding output components. These shapes may be partially
      specified, using `-1` to indicate that a particular dimension should be
      padded to the maximum size of all batch elements.
    padding_values: A list of `Tensor` objects.
      A list of scalars containing the padding value to use for
      each of the outputs.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  if not isinstance(padded_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'padded_shapes' argument to "
        "'padded_batch_dataset' Op, not %r." % padded_shapes)
  _attr_N = len(padded_shapes)
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'padded_batch_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "PaddedBatchDataset", input_dataset=input_dataset,
        batch_size=batch_size, padded_shapes=padded_shapes,
        padding_values=padding_values, output_shapes=output_shapes, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("Toutput_types", _op.get_attr("Toutput_types"), "output_shapes",
              _op.get_attr("output_shapes"), "N", _op.get_attr("N"))
  else:
    _attr_Toutput_types, padding_values = _execute.convert_to_mixed_eager_tensors(padding_values, _ctx)
    _attr_Toutput_types = [_t.as_datatype_enum for _t in _attr_Toutput_types]
    input_dataset = _ops.convert_to_tensor(input_dataset, _dtypes.variant)
    batch_size = _ops.convert_to_tensor(batch_size, _dtypes.int64)
    padded_shapes = _ops.convert_n_to_tensor(padded_shapes, _dtypes.int64)
    _inputs_flat = [input_dataset, batch_size] + list(padded_shapes) + list(padding_values)
    _attrs = ("Toutput_types", _attr_Toutput_types, "output_shapes",
              output_shapes, "N", _attr_N)
    _result = _execute.execute(b"PaddedBatchDataset", 1, inputs=_inputs_flat,
                               attrs=_attrs, ctx=_ctx, name=name)
  _execute.record_gradient(
      "PaddedBatchDataset", _inputs_flat, _attrs, _result, name)
  _result, = _result
  return _result


def parallel_map_dataset(input_dataset, other_arguments, num_parallel_calls, f, output_types, output_shapes, name=None):
  r"""Creates a dataset that applies `f` to the outputs of `input_dataset`.

  Unlike a "MapDataset", which applies `f` sequentially, this dataset invokes up
  to `num_parallel_calls` copies of `f` in parallel.

  Args:
    input_dataset: A `Tensor` of type `variant`.
    other_arguments: A list of `Tensor` objects.
    num_parallel_calls: A `Tensor` of type `int32`.
      The number of concurrent invocations of `f` that process
      elements from `input_dataset` in parallel.
    f: A function decorated with @Defun.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'parallel_map_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'parallel_map_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "ParallelMapDataset", input_dataset=input_dataset,
        other_arguments=other_arguments,
        num_parallel_calls=num_parallel_calls, f=f, output_types=output_types,
        output_shapes=output_shapes, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("f", _op.get_attr("f"), "Targuments",
              _op.get_attr("Targuments"), "output_types",
              _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"))
  else:
    _attr_Targuments, other_arguments = _execute.convert_to_mixed_eager_tensors(other_arguments, _ctx)
    _attr_Targuments = [_t.as_datatype_enum for _t in _attr_Targuments]
    input_dataset = _ops.convert_to_tensor(input_dataset, _dtypes.variant)
    num_parallel_calls = _ops.convert_to_tensor(num_parallel_calls, _dtypes.int32)
    _inputs_flat = [input_dataset] + list(other_arguments) + [num_parallel_calls]
    _attrs = ("f", f, "Targuments", _attr_Targuments, "output_types",
              output_types, "output_shapes", output_shapes)
    _result = _execute.execute(b"ParallelMapDataset", 1, inputs=_inputs_flat,
                               attrs=_attrs, ctx=_ctx, name=name)
  _execute.record_gradient(
      "ParallelMapDataset", _inputs_flat, _attrs, _result, name)
  _result, = _result
  return _result


def prefetch_dataset(input_dataset, buffer_size, output_types, output_shapes, name=None):
  r"""Creates a dataset that asynchronously prefetches elements from `input_dataset`.

  Args:
    input_dataset: A `Tensor` of type `variant`.
    buffer_size: A `Tensor` of type `int64`.
      The maximum number of elements to buffer in an iterator over
      this dataset.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'prefetch_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'prefetch_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "PrefetchDataset", input_dataset=input_dataset,
        buffer_size=buffer_size, output_types=output_types,
        output_shapes=output_shapes, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("output_types", _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"))
  else:
    input_dataset = _ops.convert_to_tensor(input_dataset, _dtypes.variant)
    buffer_size = _ops.convert_to_tensor(buffer_size, _dtypes.int64)
    _inputs_flat = [input_dataset, buffer_size]
    _attrs = ("output_types", output_types, "output_shapes", output_shapes)
    _result = _execute.execute(b"PrefetchDataset", 1, inputs=_inputs_flat,
                               attrs=_attrs, ctx=_ctx, name=name)
  _execute.record_gradient(
      "PrefetchDataset", _inputs_flat, _attrs, _result, name)
  _result, = _result
  return _result


def range_dataset(start, stop, step, output_types, output_shapes, name=None):
  r"""Creates a dataset with a range of values. Corresponds to python's xrange.

  Args:
    start: A `Tensor` of type `int64`.
      corresponds to start in python's xrange().
    stop: A `Tensor` of type `int64`.
      corresponds to stop in python's xrange().
    step: A `Tensor` of type `int64`.
      corresponds to step in python's xrange().
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'range_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'range_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "RangeDataset", start=start, stop=stop, step=step,
        output_types=output_types, output_shapes=output_shapes, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("output_types", _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"))
  else:
    start = _ops.convert_to_tensor(start, _dtypes.int64)
    stop = _ops.convert_to_tensor(stop, _dtypes.int64)
    step = _ops.convert_to_tensor(step, _dtypes.int64)
    _inputs_flat = [start, stop, step]
    _attrs = ("output_types", output_types, "output_shapes", output_shapes)
    _result = _execute.execute(b"RangeDataset", 1, inputs=_inputs_flat,
                               attrs=_attrs, ctx=_ctx, name=name)
  _execute.record_gradient(
      "RangeDataset", _inputs_flat, _attrs, _result, name)
  _result, = _result
  return _result


def repeat_dataset(input_dataset, count, output_types, output_shapes, name=None):
  r"""Creates a dataset that emits the outputs of `input_dataset` `count` times.

  Args:
    input_dataset: A `Tensor` of type `variant`.
    count: A `Tensor` of type `int64`.
      A scalar representing the number of times that `input_dataset` should
      be repeated. A value of `-1` indicates that it should be repeated infinitely.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'repeat_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'repeat_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "RepeatDataset", input_dataset=input_dataset, count=count,
        output_types=output_types, output_shapes=output_shapes, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("output_types", _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"))
  else:
    input_dataset = _ops.convert_to_tensor(input_dataset, _dtypes.variant)
    count = _ops.convert_to_tensor(count, _dtypes.int64)
    _inputs_flat = [input_dataset, count]
    _attrs = ("output_types", output_types, "output_shapes", output_shapes)
    _result = _execute.execute(b"RepeatDataset", 1, inputs=_inputs_flat,
                               attrs=_attrs, ctx=_ctx, name=name)
  _execute.record_gradient(
      "RepeatDataset", _inputs_flat, _attrs, _result, name)
  _result, = _result
  return _result


def restore_iterator(iterator, path, name=None):
  r"""Restores the state of the `iterator` from the checkpoint saved at `path` using "SaveIterator".

  Args:
    iterator: A `Tensor` of type `resource`.
    path: A `Tensor` of type `string`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "RestoreIterator", iterator=iterator, path=path, name=name)
    return _op
  else:
    iterator = _ops.convert_to_tensor(iterator, _dtypes.resource)
    path = _ops.convert_to_tensor(path, _dtypes.string)
    _inputs_flat = [iterator, path]
    _attrs = None
    _result = _execute.execute(b"RestoreIterator", 0, inputs=_inputs_flat,
                               attrs=_attrs, ctx=_ctx, name=name)
  return _result


def save_iterator(iterator, path, name=None):
  r"""Saves the state of the `iterator` at `path`.

  This state can be restored using "RestoreIterator".

  Args:
    iterator: A `Tensor` of type `resource`.
    path: A `Tensor` of type `string`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "SaveIterator", iterator=iterator, path=path, name=name)
    return _op
  else:
    iterator = _ops.convert_to_tensor(iterator, _dtypes.resource)
    path = _ops.convert_to_tensor(path, _dtypes.string)
    _inputs_flat = [iterator, path]
    _attrs = None
    _result = _execute.execute(b"SaveIterator", 0, inputs=_inputs_flat,
                               attrs=_attrs, ctx=_ctx, name=name)
  return _result


def shuffle_dataset(input_dataset, buffer_size, seed, seed2, output_types, output_shapes, reshuffle_each_iteration=True, name=None):
  r"""Creates a dataset that shuffles elements from `input_dataset` pseudorandomly.

  Args:
    input_dataset: A `Tensor` of type `variant`.
    buffer_size: A `Tensor` of type `int64`.
      The number of output elements to buffer in an iterator over
      this dataset. Compare with the `min_after_dequeue` attr when creating a
      `RandomShuffleQueue`.
    seed: A `Tensor` of type `int64`.
      A scalar seed for the random number generator. If either seed or
      seed2 is set to be non-zero, the random number generator is seeded
      by the given seed.  Otherwise, a random seed is used.
    seed2: A `Tensor` of type `int64`.
      A second scalar seed to avoid seed collision.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    reshuffle_each_iteration: An optional `bool`. Defaults to `True`.
      If true, each iterator over this dataset will be given
      a different pseudorandomly generated seed, based on a sequence seeded by the
      `seed` and `seed2` inputs. If false, each iterator will be given the same
      seed, and repeated iteration over this dataset will yield the exact same
      sequence of results.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'shuffle_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'shuffle_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  if reshuffle_each_iteration is None:
    reshuffle_each_iteration = True
  reshuffle_each_iteration = _execute.make_bool(reshuffle_each_iteration, "reshuffle_each_iteration")
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "ShuffleDataset", input_dataset=input_dataset,
        buffer_size=buffer_size, seed=seed, seed2=seed2,
        output_types=output_types, output_shapes=output_shapes,
        reshuffle_each_iteration=reshuffle_each_iteration, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("reshuffle_each_iteration",
              _op.get_attr("reshuffle_each_iteration"), "output_types",
              _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"))
  else:
    input_dataset = _ops.convert_to_tensor(input_dataset, _dtypes.variant)
    buffer_size = _ops.convert_to_tensor(buffer_size, _dtypes.int64)
    seed = _ops.convert_to_tensor(seed, _dtypes.int64)
    seed2 = _ops.convert_to_tensor(seed2, _dtypes.int64)
    _inputs_flat = [input_dataset, buffer_size, seed, seed2]
    _attrs = ("reshuffle_each_iteration", reshuffle_each_iteration,
              "output_types", output_types, "output_shapes", output_shapes)
    _result = _execute.execute(b"ShuffleDataset", 1, inputs=_inputs_flat,
                               attrs=_attrs, ctx=_ctx, name=name)
  _execute.record_gradient(
      "ShuffleDataset", _inputs_flat, _attrs, _result, name)
  _result, = _result
  return _result


def skip_dataset(input_dataset, count, output_types, output_shapes, name=None):
  r"""Creates a dataset that skips `count` elements from the `input_dataset`.

  Args:
    input_dataset: A `Tensor` of type `variant`.
    count: A `Tensor` of type `int64`.
      A scalar representing the number of elements from the `input_dataset`
      that should be skipped.  If count is -1, skips everything.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'skip_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'skip_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "SkipDataset", input_dataset=input_dataset, count=count,
        output_types=output_types, output_shapes=output_shapes, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("output_types", _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"))
  else:
    input_dataset = _ops.convert_to_tensor(input_dataset, _dtypes.variant)
    count = _ops.convert_to_tensor(count, _dtypes.int64)
    _inputs_flat = [input_dataset, count]
    _attrs = ("output_types", output_types, "output_shapes", output_shapes)
    _result = _execute.execute(b"SkipDataset", 1, inputs=_inputs_flat,
                               attrs=_attrs, ctx=_ctx, name=name)
  _execute.record_gradient(
      "SkipDataset", _inputs_flat, _attrs, _result, name)
  _result, = _result
  return _result


def sloppy_interleave_dataset(input_dataset, other_arguments, cycle_length, block_length, f, output_types, output_shapes, name=None):
  r"""Creates a dataset that applies `f` to the outputs of `input_dataset`.

  The resulting dataset is similar to the `InterleaveDataset`, with the exception
  that if retrieving the next value from a dataset would cause the requester to
  block, it will skip that input dataset. This dataset is especially useful
  when loading data from a variable-latency datastores (e.g. HDFS, GCS), as it
  allows the training step to proceed so long as some data is available.

  !! WARNING !! This dataset is not deterministic!

  Args:
    input_dataset: A `Tensor` of type `variant`.
    other_arguments: A list of `Tensor` objects.
    cycle_length: A `Tensor` of type `int64`.
    block_length: A `Tensor` of type `int64`.
    f: A function decorated with @Defun.
      A function mapping elements of `input_dataset`, concatenated with
      `other_arguments`, to a Dataset variant that contains elements matching
      `output_types` and `output_shapes`.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'sloppy_interleave_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'sloppy_interleave_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "SloppyInterleaveDataset", input_dataset=input_dataset,
        other_arguments=other_arguments, cycle_length=cycle_length,
        block_length=block_length, f=f, output_types=output_types,
        output_shapes=output_shapes, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("f", _op.get_attr("f"), "Targuments",
              _op.get_attr("Targuments"), "output_types",
              _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"))
  else:
    _attr_Targuments, other_arguments = _execute.convert_to_mixed_eager_tensors(other_arguments, _ctx)
    _attr_Targuments = [_t.as_datatype_enum for _t in _attr_Targuments]
    input_dataset = _ops.convert_to_tensor(input_dataset, _dtypes.variant)
    cycle_length = _ops.convert_to_tensor(cycle_length, _dtypes.int64)
    block_length = _ops.convert_to_tensor(block_length, _dtypes.int64)
    _inputs_flat = [input_dataset] + list(other_arguments) + [cycle_length, block_length]
    _attrs = ("f", f, "Targuments", _attr_Targuments, "output_types",
              output_types, "output_shapes", output_shapes)
    _result = _execute.execute(b"SloppyInterleaveDataset", 1,
                               inputs=_inputs_flat, attrs=_attrs, ctx=_ctx,
                               name=name)
  _execute.record_gradient(
      "SloppyInterleaveDataset", _inputs_flat, _attrs, _result, name)
  _result, = _result
  return _result


def sparse_tensor_slice_dataset(indices, values, dense_shape, name=None):
  r"""Creates a dataset that splits a SparseTensor into elements row-wise.

  Args:
    indices: A `Tensor` of type `int64`.
    values: A `Tensor`.
    dense_shape: A `Tensor` of type `int64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "SparseTensorSliceDataset", indices=indices, values=values,
        dense_shape=dense_shape, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("Tvalues", _op.get_attr("Tvalues"))
  else:
    _attr_Tvalues, (values,) = _execute.args_to_matching_eager([values], _ctx)
    _attr_Tvalues = _attr_Tvalues.as_datatype_enum
    indices = _ops.convert_to_tensor(indices, _dtypes.int64)
    dense_shape = _ops.convert_to_tensor(dense_shape, _dtypes.int64)
    _inputs_flat = [indices, values, dense_shape]
    _attrs = ("Tvalues", _attr_Tvalues)
    _result = _execute.execute(b"SparseTensorSliceDataset", 1,
                               inputs=_inputs_flat, attrs=_attrs, ctx=_ctx,
                               name=name)
  _execute.record_gradient(
      "SparseTensorSliceDataset", _inputs_flat, _attrs, _result, name)
  _result, = _result
  return _result


def sql_dataset(driver_name, data_source_name, query, output_types, output_shapes, name=None):
  r"""Creates a dataset that executes a SQL query and emits rows of the result set.

  Args:
    driver_name: A `Tensor` of type `string`.
      The database type. Currently, the only supported type is 'sqlite'.
    data_source_name: A `Tensor` of type `string`.
      A connection string to connect to the database.
    query: A `Tensor` of type `string`. A SQL query to execute.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'sql_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'sql_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "SqlDataset", driver_name=driver_name,
        data_source_name=data_source_name, query=query,
        output_types=output_types, output_shapes=output_shapes, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("output_types", _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"))
  else:
    driver_name = _ops.convert_to_tensor(driver_name, _dtypes.string)
    data_source_name = _ops.convert_to_tensor(data_source_name, _dtypes.string)
    query = _ops.convert_to_tensor(query, _dtypes.string)
    _inputs_flat = [driver_name, data_source_name, query]
    _attrs = ("output_types", output_types, "output_shapes", output_shapes)
    _result = _execute.execute(b"SqlDataset", 1, inputs=_inputs_flat,
                               attrs=_attrs, ctx=_ctx, name=name)
  _execute.record_gradient(
      "SqlDataset", _inputs_flat, _attrs, _result, name)
  _result, = _result
  return _result


def tf_record_dataset(filenames, compression_type, buffer_size, name=None):
  r"""Creates a dataset that emits the records from one or more TFRecord files.

  Args:
    filenames: A `Tensor` of type `string`.
      A scalar or vector containing the name(s) of the file(s) to be
      read.
    compression_type: A `Tensor` of type `string`.
      A scalar containing either (i) the empty string (no
      compression), (ii) "ZLIB", or (iii) "GZIP".
    buffer_size: A `Tensor` of type `int64`.
      A scalar representing the number of bytes to buffer. A value of
      0 means no buffering will be performed.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "TFRecordDataset", filenames=filenames,
        compression_type=compression_type, buffer_size=buffer_size, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = None
  else:
    filenames = _ops.convert_to_tensor(filenames, _dtypes.string)
    compression_type = _ops.convert_to_tensor(compression_type, _dtypes.string)
    buffer_size = _ops.convert_to_tensor(buffer_size, _dtypes.int64)
    _inputs_flat = [filenames, compression_type, buffer_size]
    _attrs = None
    _result = _execute.execute(b"TFRecordDataset", 1, inputs=_inputs_flat,
                               attrs=_attrs, ctx=_ctx, name=name)
  _execute.record_gradient(
      "TFRecordDataset", _inputs_flat, _attrs, _result, name)
  _result, = _result
  return _result


def take_dataset(input_dataset, count, output_types, output_shapes, name=None):
  r"""Creates a dataset that contains `count` elements from the `input_dataset`.

  Args:
    input_dataset: A `Tensor` of type `variant`.
    count: A `Tensor` of type `int64`.
      A scalar representing the number of elements from the `input_dataset`
      that should be taken. A value of `-1` indicates that all of `input_dataset`
      is taken.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'take_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'take_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "TakeDataset", input_dataset=input_dataset, count=count,
        output_types=output_types, output_shapes=output_shapes, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("output_types", _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"))
  else:
    input_dataset = _ops.convert_to_tensor(input_dataset, _dtypes.variant)
    count = _ops.convert_to_tensor(count, _dtypes.int64)
    _inputs_flat = [input_dataset, count]
    _attrs = ("output_types", output_types, "output_shapes", output_shapes)
    _result = _execute.execute(b"TakeDataset", 1, inputs=_inputs_flat,
                               attrs=_attrs, ctx=_ctx, name=name)
  _execute.record_gradient(
      "TakeDataset", _inputs_flat, _attrs, _result, name)
  _result, = _result
  return _result


def tensor_dataset(components, output_shapes, name=None):
  r"""Creates a dataset that emits `components` as a tuple of tensors once.

  Args:
    components: A list of `Tensor` objects.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'tensor_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "TensorDataset", components=components, output_shapes=output_shapes,
        name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("Toutput_types", _op.get_attr("Toutput_types"), "output_shapes",
              _op.get_attr("output_shapes"))
  else:
    _attr_Toutput_types, components = _execute.convert_to_mixed_eager_tensors(components, _ctx)
    _attr_Toutput_types = [_t.as_datatype_enum for _t in _attr_Toutput_types]
    _inputs_flat = list(components)
    _attrs = ("Toutput_types", _attr_Toutput_types, "output_shapes",
              output_shapes)
    _result = _execute.execute(b"TensorDataset", 1, inputs=_inputs_flat,
                               attrs=_attrs, ctx=_ctx, name=name)
  _execute.record_gradient(
      "TensorDataset", _inputs_flat, _attrs, _result, name)
  _result, = _result
  return _result


def tensor_slice_dataset(components, output_shapes, name=None):
  r"""Creates a dataset that emits each dim-0 slice of `components` once.

  Args:
    components: A list of `Tensor` objects.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'tensor_slice_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "TensorSliceDataset", components=components,
        output_shapes=output_shapes, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("Toutput_types", _op.get_attr("Toutput_types"), "output_shapes",
              _op.get_attr("output_shapes"))
  else:
    _attr_Toutput_types, components = _execute.convert_to_mixed_eager_tensors(components, _ctx)
    _attr_Toutput_types = [_t.as_datatype_enum for _t in _attr_Toutput_types]
    _inputs_flat = list(components)
    _attrs = ("Toutput_types", _attr_Toutput_types, "output_shapes",
              output_shapes)
    _result = _execute.execute(b"TensorSliceDataset", 1, inputs=_inputs_flat,
                               attrs=_attrs, ctx=_ctx, name=name)
  _execute.record_gradient(
      "TensorSliceDataset", _inputs_flat, _attrs, _result, name)
  _result, = _result
  return _result


def text_line_dataset(filenames, compression_type, buffer_size, name=None):
  r"""Creates a dataset that emits the lines of one or more text files.

  Args:
    filenames: A `Tensor` of type `string`.
      A scalar or a vector containing the name(s) of the file(s) to be
      read.
    compression_type: A `Tensor` of type `string`.
      A scalar containing either (i) the empty string (no
      compression), (ii) "ZLIB", or (iii) "GZIP".
    buffer_size: A `Tensor` of type `int64`.
      A scalar containing the number of bytes to buffer.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "TextLineDataset", filenames=filenames,
        compression_type=compression_type, buffer_size=buffer_size, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = None
  else:
    filenames = _ops.convert_to_tensor(filenames, _dtypes.string)
    compression_type = _ops.convert_to_tensor(compression_type, _dtypes.string)
    buffer_size = _ops.convert_to_tensor(buffer_size, _dtypes.int64)
    _inputs_flat = [filenames, compression_type, buffer_size]
    _attrs = None
    _result = _execute.execute(b"TextLineDataset", 1, inputs=_inputs_flat,
                               attrs=_attrs, ctx=_ctx, name=name)
  _execute.record_gradient(
      "TextLineDataset", _inputs_flat, _attrs, _result, name)
  _result, = _result
  return _result


def zip_dataset(input_datasets, output_types, output_shapes, name=None):
  r"""Creates a dataset that zips together `input_datasets`.

  Args:
    input_datasets: A list of at least 1 `Tensor` objects with type `variant`.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
  if not isinstance(input_datasets, (list, tuple)):
    raise TypeError(
        "Expected list for 'input_datasets' argument to "
        "'zip_dataset' Op, not %r." % input_datasets)
  _attr_N = len(input_datasets)
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'zip_dataset' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if not isinstance(output_shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_shapes' argument to "
        "'zip_dataset' Op, not %r." % output_shapes)
  output_shapes = [_execute.make_shape(_s, "output_shapes") for _s in output_shapes]
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "ZipDataset", input_datasets=input_datasets,
        output_types=output_types, output_shapes=output_shapes, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("output_types", _op.get_attr("output_types"), "output_shapes",
              _op.get_attr("output_shapes"), "N", _op.get_attr("N"))
  else:
    input_datasets = _ops.convert_n_to_tensor(input_datasets, _dtypes.variant)
    _inputs_flat = list(input_datasets)
    _attrs = ("output_types", output_types, "output_shapes", output_shapes,
              "N", _attr_N)
    _result = _execute.execute(b"ZipDataset", 1, inputs=_inputs_flat,
                               attrs=_attrs, ctx=_ctx, name=name)
  _execute.record_gradient(
      "ZipDataset", _inputs_flat, _attrs, _result, name)
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
#   name: "BatchDataset"
#   input_arg {
#     name: "input_dataset"
#     type: DT_VARIANT
#   }
#   input_arg {
#     name: "batch_size"
#     type: DT_INT64
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
#   name: "CacheDataset"
#   input_arg {
#     name: "input_dataset"
#     type: DT_VARIANT
#   }
#   input_arg {
#     name: "filename"
#     type: DT_STRING
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
#   name: "ConcatenateDataset"
#   input_arg {
#     name: "input_dataset"
#     type: DT_VARIANT
#   }
#   input_arg {
#     name: "another_dataset"
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
#   name: "DenseToSparseBatchDataset"
#   input_arg {
#     name: "input_dataset"
#     type: DT_VARIANT
#   }
#   input_arg {
#     name: "batch_size"
#     type: DT_INT64
#   }
#   input_arg {
#     name: "row_shape"
#     type: DT_INT64
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
#   name: "FilterDataset"
#   input_arg {
#     name: "input_dataset"
#     type: DT_VARIANT
#   }
#   input_arg {
#     name: "other_arguments"
#     type_list_attr: "Targuments"
#   }
#   output_arg {
#     name: "handle"
#     type: DT_VARIANT
#   }
#   attr {
#     name: "predicate"
#     type: "func"
#   }
#   attr {
#     name: "Targuments"
#     type: "list(type)"
#     has_minimum: true
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
#   name: "FixedLengthRecordDataset"
#   input_arg {
#     name: "filenames"
#     type: DT_STRING
#   }
#   input_arg {
#     name: "header_bytes"
#     type: DT_INT64
#   }
#   input_arg {
#     name: "record_bytes"
#     type: DT_INT64
#   }
#   input_arg {
#     name: "footer_bytes"
#     type: DT_INT64
#   }
#   input_arg {
#     name: "buffer_size"
#     type: DT_INT64
#   }
#   output_arg {
#     name: "handle"
#     type: DT_VARIANT
#   }
#   is_stateful: true
# }
# op {
#   name: "FlatMapDataset"
#   input_arg {
#     name: "input_dataset"
#     type: DT_VARIANT
#   }
#   input_arg {
#     name: "other_arguments"
#     type_list_attr: "Targuments"
#   }
#   output_arg {
#     name: "handle"
#     type: DT_VARIANT
#   }
#   attr {
#     name: "f"
#     type: "func"
#   }
#   attr {
#     name: "Targuments"
#     type: "list(type)"
#     has_minimum: true
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
#   name: "GroupByWindowDataset"
#   input_arg {
#     name: "input_dataset"
#     type: DT_VARIANT
#   }
#   input_arg {
#     name: "key_func_other_arguments"
#     type_list_attr: "Tkey_func_other_arguments"
#   }
#   input_arg {
#     name: "reduce_func_other_arguments"
#     type_list_attr: "Treduce_func_other_arguments"
#   }
#   input_arg {
#     name: "window_size_func_other_arguments"
#     type_list_attr: "Twindow_size_func_other_arguments"
#   }
#   output_arg {
#     name: "handle"
#     type: DT_VARIANT
#   }
#   attr {
#     name: "key_func"
#     type: "func"
#   }
#   attr {
#     name: "reduce_func"
#     type: "func"
#   }
#   attr {
#     name: "window_size_func"
#     type: "func"
#   }
#   attr {
#     name: "Tkey_func_other_arguments"
#     type: "list(type)"
#     has_minimum: true
#   }
#   attr {
#     name: "Treduce_func_other_arguments"
#     type: "list(type)"
#     has_minimum: true
#   }
#   attr {
#     name: "Twindow_size_func_other_arguments"
#     type: "list(type)"
#     has_minimum: true
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
#   name: "InterleaveDataset"
#   input_arg {
#     name: "input_dataset"
#     type: DT_VARIANT
#   }
#   input_arg {
#     name: "other_arguments"
#     type_list_attr: "Targuments"
#   }
#   input_arg {
#     name: "cycle_length"
#     type: DT_INT64
#   }
#   input_arg {
#     name: "block_length"
#     type: DT_INT64
#   }
#   output_arg {
#     name: "handle"
#     type: DT_VARIANT
#   }
#   attr {
#     name: "f"
#     type: "func"
#   }
#   attr {
#     name: "Targuments"
#     type: "list(type)"
#     has_minimum: true
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
#   name: "Iterator"
#   output_arg {
#     name: "handle"
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
#   name: "IteratorFromStringHandle"
#   input_arg {
#     name: "string_handle"
#     type: DT_STRING
#   }
#   output_arg {
#     name: "resource_handle"
#     type: DT_RESOURCE
#   }
#   attr {
#     name: "output_types"
#     type: "list(type)"
#     default_value {
#       list {
#       }
#     }
#     has_minimum: true
#   }
#   attr {
#     name: "output_shapes"
#     type: "list(shape)"
#     default_value {
#       list {
#       }
#     }
#     has_minimum: true
#   }
#   is_stateful: true
# }
# op {
#   name: "IteratorGetNext"
#   input_arg {
#     name: "iterator"
#     type: DT_RESOURCE
#   }
#   output_arg {
#     name: "components"
#     type_list_attr: "output_types"
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
#   name: "IteratorToStringHandle"
#   input_arg {
#     name: "resource_handle"
#     type: DT_RESOURCE
#   }
#   output_arg {
#     name: "string_handle"
#     type: DT_STRING
#   }
#   is_stateful: true
# }
# op {
#   name: "MakeIterator"
#   input_arg {
#     name: "dataset"
#     type: DT_VARIANT
#   }
#   input_arg {
#     name: "iterator"
#     type: DT_RESOURCE
#   }
#   is_stateful: true
# }
# op {
#   name: "MapDataset"
#   input_arg {
#     name: "input_dataset"
#     type: DT_VARIANT
#   }
#   input_arg {
#     name: "other_arguments"
#     type_list_attr: "Targuments"
#   }
#   output_arg {
#     name: "handle"
#     type: DT_VARIANT
#   }
#   attr {
#     name: "f"
#     type: "func"
#   }
#   attr {
#     name: "Targuments"
#     type: "list(type)"
#     has_minimum: true
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
#   name: "OneShotIterator"
#   output_arg {
#     name: "handle"
#     type: DT_RESOURCE
#   }
#   attr {
#     name: "dataset_factory"
#     type: "func"
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
#   name: "PaddedBatchDataset"
#   input_arg {
#     name: "input_dataset"
#     type: DT_VARIANT
#   }
#   input_arg {
#     name: "batch_size"
#     type: DT_INT64
#   }
#   input_arg {
#     name: "padded_shapes"
#     type: DT_INT64
#     number_attr: "N"
#   }
#   input_arg {
#     name: "padding_values"
#     type_list_attr: "Toutput_types"
#   }
#   output_arg {
#     name: "handle"
#     type: DT_VARIANT
#   }
#   attr {
#     name: "Toutput_types"
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
#   attr {
#     name: "N"
#     type: "int"
#     has_minimum: true
#     minimum: 1
#   }
# }
# op {
#   name: "ParallelMapDataset"
#   input_arg {
#     name: "input_dataset"
#     type: DT_VARIANT
#   }
#   input_arg {
#     name: "other_arguments"
#     type_list_attr: "Targuments"
#   }
#   input_arg {
#     name: "num_parallel_calls"
#     type: DT_INT32
#   }
#   output_arg {
#     name: "handle"
#     type: DT_VARIANT
#   }
#   attr {
#     name: "f"
#     type: "func"
#   }
#   attr {
#     name: "Targuments"
#     type: "list(type)"
#     has_minimum: true
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
#   name: "PrefetchDataset"
#   input_arg {
#     name: "input_dataset"
#     type: DT_VARIANT
#   }
#   input_arg {
#     name: "buffer_size"
#     type: DT_INT64
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
#   name: "RangeDataset"
#   input_arg {
#     name: "start"
#     type: DT_INT64
#   }
#   input_arg {
#     name: "stop"
#     type: DT_INT64
#   }
#   input_arg {
#     name: "step"
#     type: DT_INT64
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
#   name: "RepeatDataset"
#   input_arg {
#     name: "input_dataset"
#     type: DT_VARIANT
#   }
#   input_arg {
#     name: "count"
#     type: DT_INT64
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
#   name: "RestoreIterator"
#   input_arg {
#     name: "iterator"
#     type: DT_RESOURCE
#   }
#   input_arg {
#     name: "path"
#     type: DT_STRING
#   }
#   is_stateful: true
# }
# op {
#   name: "SaveIterator"
#   input_arg {
#     name: "iterator"
#     type: DT_RESOURCE
#   }
#   input_arg {
#     name: "path"
#     type: DT_STRING
#   }
#   is_stateful: true
# }
# op {
#   name: "ShuffleDataset"
#   input_arg {
#     name: "input_dataset"
#     type: DT_VARIANT
#   }
#   input_arg {
#     name: "buffer_size"
#     type: DT_INT64
#   }
#   input_arg {
#     name: "seed"
#     type: DT_INT64
#   }
#   input_arg {
#     name: "seed2"
#     type: DT_INT64
#   }
#   output_arg {
#     name: "handle"
#     type: DT_VARIANT
#   }
#   attr {
#     name: "reshuffle_each_iteration"
#     type: "bool"
#     default_value {
#       b: true
#     }
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
#   name: "SkipDataset"
#   input_arg {
#     name: "input_dataset"
#     type: DT_VARIANT
#   }
#   input_arg {
#     name: "count"
#     type: DT_INT64
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
#   name: "SloppyInterleaveDataset"
#   input_arg {
#     name: "input_dataset"
#     type: DT_VARIANT
#   }
#   input_arg {
#     name: "other_arguments"
#     type_list_attr: "Targuments"
#   }
#   input_arg {
#     name: "cycle_length"
#     type: DT_INT64
#   }
#   input_arg {
#     name: "block_length"
#     type: DT_INT64
#   }
#   output_arg {
#     name: "handle"
#     type: DT_VARIANT
#   }
#   attr {
#     name: "f"
#     type: "func"
#   }
#   attr {
#     name: "Targuments"
#     type: "list(type)"
#     has_minimum: true
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
#   name: "SparseTensorSliceDataset"
#   input_arg {
#     name: "indices"
#     type: DT_INT64
#   }
#   input_arg {
#     name: "values"
#     type_attr: "Tvalues"
#   }
#   input_arg {
#     name: "dense_shape"
#     type: DT_INT64
#   }
#   output_arg {
#     name: "handle"
#     type: DT_VARIANT
#   }
#   attr {
#     name: "Tvalues"
#     type: "type"
#   }
#   is_stateful: true
# }
# op {
#   name: "SqlDataset"
#   input_arg {
#     name: "driver_name"
#     type: DT_STRING
#   }
#   input_arg {
#     name: "data_source_name"
#     type: DT_STRING
#   }
#   input_arg {
#     name: "query"
#     type: DT_STRING
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
#   name: "TFRecordDataset"
#   input_arg {
#     name: "filenames"
#     type: DT_STRING
#   }
#   input_arg {
#     name: "compression_type"
#     type: DT_STRING
#   }
#   input_arg {
#     name: "buffer_size"
#     type: DT_INT64
#   }
#   output_arg {
#     name: "handle"
#     type: DT_VARIANT
#   }
#   is_stateful: true
# }
# op {
#   name: "TakeDataset"
#   input_arg {
#     name: "input_dataset"
#     type: DT_VARIANT
#   }
#   input_arg {
#     name: "count"
#     type: DT_INT64
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
#   name: "TensorDataset"
#   input_arg {
#     name: "components"
#     type_list_attr: "Toutput_types"
#   }
#   output_arg {
#     name: "handle"
#     type: DT_VARIANT
#   }
#   attr {
#     name: "Toutput_types"
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
#   name: "TensorSliceDataset"
#   input_arg {
#     name: "components"
#     type_list_attr: "Toutput_types"
#   }
#   output_arg {
#     name: "handle"
#     type: DT_VARIANT
#   }
#   attr {
#     name: "Toutput_types"
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
#   name: "TextLineDataset"
#   input_arg {
#     name: "filenames"
#     type: DT_STRING
#   }
#   input_arg {
#     name: "compression_type"
#     type: DT_STRING
#   }
#   input_arg {
#     name: "buffer_size"
#     type: DT_INT64
#   }
#   output_arg {
#     name: "handle"
#     type: DT_VARIANT
#   }
#   is_stateful: true
# }
# op {
#   name: "ZipDataset"
#   input_arg {
#     name: "input_datasets"
#     type: DT_VARIANT
#     number_attr: "N"
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
#   attr {
#     name: "N"
#     type: "int"
#     has_minimum: true
#     minimum: 1
#   }
# }
_op_def_lib = _InitOpDefLibrary(b"\n\177\n\014BatchDataset\022\021\n\rinput_dataset\030\025\022\016\n\nbatch_size\030\t\032\n\n\006handle\030\025\"\036\n\014output_types\022\nlist(type)(\0010\001\" \n\routput_shapes\022\013list(shape)(\0010\001\n}\n\014CacheDataset\022\021\n\rinput_dataset\030\025\022\014\n\010filename\030\007\032\n\n\006handle\030\025\"\036\n\014output_types\022\nlist(type)(\0010\001\" \n\routput_shapes\022\013list(shape)(\0010\001\n\212\001\n\022ConcatenateDataset\022\021\n\rinput_dataset\030\025\022\023\n\017another_dataset\030\025\032\n\n\006handle\030\025\"\036\n\014output_types\022\nlist(type)(\0010\001\" \n\routput_shapes\022\013list(shape)(\0010\001\n\233\001\n\031DenseToSparseBatchDataset\022\021\n\rinput_dataset\030\025\022\016\n\nbatch_size\030\t\022\r\n\trow_shape\030\t\032\n\n\006handle\030\025\"\036\n\014output_types\022\nlist(type)(\0010\001\" \n\routput_shapes\022\013list(shape)(\0010\001\n\276\001\n\rFilterDataset\022\021\n\rinput_dataset\030\025\022\035\n\017other_arguments2\nTarguments\032\n\n\006handle\030\025\"\021\n\tpredicate\022\004func\"\032\n\nTarguments\022\nlist(type)(\001\"\036\n\014output_types\022\nlist(type)(\0010\001\" \n\routput_shapes\022\013list(shape)(\0010\001\n\177\n\030FixedLengthRecordDataset\022\r\n\tfilenames\030\007\022\020\n\014header_bytes\030\t\022\020\n\014record_bytes\030\t\022\020\n\014footer_bytes\030\t\022\017\n\013buffer_size\030\t\032\n\n\006handle\030\025\210\001\001\n\267\001\n\016FlatMapDataset\022\021\n\rinput_dataset\030\025\022\035\n\017other_arguments2\nTarguments\032\n\n\006handle\030\025\"\t\n\001f\022\004func\"\032\n\nTarguments\022\nlist(type)(\001\"\036\n\014output_types\022\nlist(type)(\0010\001\" \n\routput_shapes\022\013list(shape)(\0010\001\n\377\003\n\024GroupByWindowDataset\022\021\n\rinput_dataset\030\025\0225\n\030key_func_other_arguments2\031Tkey_func_other_arguments\022;\n\033reduce_func_other_arguments2\034Treduce_func_other_arguments\022E\n window_size_func_other_arguments2!Twindow_size_func_other_arguments\032\n\n\006handle\030\025\"\020\n\010key_func\022\004func\"\023\n\013reduce_func\022\004func\"\030\n\020window_size_func\022\004func\")\n\031Tkey_func_other_arguments\022\nlist(type)(\001\",\n\034Treduce_func_other_arguments\022\nlist(type)(\001\"1\n!Twindow_size_func_other_arguments\022\nlist(type)(\001\"\036\n\014output_types\022\nlist(type)(\0010\001\" \n\routput_shapes\022\013list(shape)(\0010\001\nv\n\023IgnoreErrorsDataset\022\021\n\rinput_dataset\030\025\032\n\n\006handle\030\025\"\036\n\014output_types\022\nlist(type)(\0010\001\" \n\routput_shapes\022\013list(shape)(\0010\001\n\336\001\n\021InterleaveDataset\022\021\n\rinput_dataset\030\025\022\035\n\017other_arguments2\nTarguments\022\020\n\014cycle_length\030\t\022\020\n\014block_length\030\t\032\n\n\006handle\030\025\"\t\n\001f\022\004func\"\032\n\nTarguments\022\nlist(type)(\001\"\036\n\014output_types\022\nlist(type)(\0010\001\" \n\routput_shapes\022\013list(shape)(\0010\001\n\207\001\n\010Iterator\032\n\n\006handle\030\024\"\025\n\013shared_name\022\006string\"\023\n\tcontainer\022\006string\"\036\n\014output_types\022\nlist(type)(\0010\001\" \n\routput_shapes\022\013list(shape)(\0010\001\210\001\001\n\213\001\n\030IteratorFromStringHandle\022\021\n\rstring_handle\030\007\032\023\n\017resource_handle\030\024\" \n\014output_types\022\nlist(type)\032\002\n\000(\001\"\"\n\routput_shapes\022\013list(shape)\032\002\n\000(\001\210\001\001\n\200\001\n\017IteratorGetNext\022\014\n\010iterator\030\024\032\032\n\ncomponents2\014output_types\"\036\n\014output_types\022\nlist(type)(\0010\001\" \n\routput_shapes\022\013list(shape)(\0010\001\210\001\001\nC\n\026IteratorToStringHandle\022\023\n\017resource_handle\030\024\032\021\n\rstring_handle\030\007\210\001\001\n,\n\014MakeIterator\022\013\n\007dataset\030\025\022\014\n\010iterator\030\024\210\001\001\n\263\001\n\nMapDataset\022\021\n\rinput_dataset\030\025\022\035\n\017other_arguments2\nTarguments\032\n\n\006handle\030\025\"\t\n\001f\022\004func\"\032\n\nTarguments\022\nlist(type)(\001\"\036\n\014output_types\022\nlist(type)(\0010\001\" \n\routput_shapes\022\013list(shape)(\0010\001\n\257\001\n\017OneShotIterator\032\n\n\006handle\030\024\"\027\n\017dataset_factory\022\004func\"\036\n\014output_types\022\nlist(type)(\0010\001\" \n\routput_shapes\022\013list(shape)(\0010\001\"\027\n\tcontainer\022\006string\032\002\022\000\"\031\n\013shared_name\022\006string\032\002\022\000\210\001\001\n\313\001\n\022PaddedBatchDataset\022\021\n\rinput_dataset\030\025\022\016\n\nbatch_size\030\t\022\024\n\rpadded_shapes\030\t*\001N\022\037\n\016padding_values2\rToutput_types\032\n\n\006handle\030\025\"\037\n\rToutput_types\022\nlist(type)(\0010\001\" \n\routput_shapes\022\013list(shape)(\0010\001\"\014\n\001N\022\003int(\0010\001\n\323\001\n\022ParallelMapDataset\022\021\n\rinput_dataset\030\025\022\035\n\017other_arguments2\nTarguments\022\026\n\022num_parallel_calls\030\003\032\n\n\006handle\030\025\"\t\n\001f\022\004func\"\032\n\nTarguments\022\nlist(type)(\001\"\036\n\014output_types\022\nlist(type)(\0010\001\" \n\routput_shapes\022\013list(shape)(\0010\001\n\203\001\n\017PrefetchDataset\022\021\n\rinput_dataset\030\025\022\017\n\013buffer_size\030\t\032\n\n\006handle\030\025\"\036\n\014output_types\022\nlist(type)(\0010\001\" \n\routput_shapes\022\013list(shape)(\0010\001\n~\n\014RangeDataset\022\t\n\005start\030\t\022\010\n\004stop\030\t\022\010\n\004step\030\t\032\n\n\006handle\030\025\"\036\n\014output_types\022\nlist(type)(\0010\001\" \n\routput_shapes\022\013list(shape)(\0010\001\210\001\001\n{\n\rRepeatDataset\022\021\n\rinput_dataset\030\025\022\t\n\005count\030\t\032\n\n\006handle\030\025\"\036\n\014output_types\022\nlist(type)(\0010\001\" \n\routput_shapes\022\013list(shape)(\0010\001\n,\n\017RestoreIterator\022\014\n\010iterator\030\024\022\010\n\004path\030\007\210\001\001\n)\n\014SaveIterator\022\014\n\010iterator\030\024\022\010\n\004path\030\007\210\001\001\n\275\001\n\016ShuffleDataset\022\021\n\rinput_dataset\030\025\022\017\n\013buffer_size\030\t\022\010\n\004seed\030\t\022\t\n\005seed2\030\t\032\n\n\006handle\030\025\"$\n\030reshuffle_each_iteration\022\004bool\032\002(\001\"\036\n\014output_types\022\nlist(type)(\0010\001\" \n\routput_shapes\022\013list(shape)(\0010\001\ny\n\013SkipDataset\022\021\n\rinput_dataset\030\025\022\t\n\005count\030\t\032\n\n\006handle\030\025\"\036\n\014output_types\022\nlist(type)(\0010\001\" \n\routput_shapes\022\013list(shape)(\0010\001\n\344\001\n\027SloppyInterleaveDataset\022\021\n\rinput_dataset\030\025\022\035\n\017other_arguments2\nTarguments\022\020\n\014cycle_length\030\t\022\020\n\014block_length\030\t\032\n\n\006handle\030\025\"\t\n\001f\022\004func\"\032\n\nTarguments\022\nlist(type)(\001\"\036\n\014output_types\022\nlist(type)(\0010\001\" \n\routput_shapes\022\013list(shape)(\0010\001\nk\n\030SparseTensorSliceDataset\022\013\n\007indices\030\t\022\021\n\006values\"\007Tvalues\022\017\n\013dense_shape\030\t\032\n\n\006handle\030\025\"\017\n\007Tvalues\022\004type\210\001\001\n\217\001\n\nSqlDataset\022\017\n\013driver_name\030\007\022\024\n\020data_source_name\030\007\022\t\n\005query\030\007\032\n\n\006handle\030\025\"\036\n\014output_types\022\nlist(type)(\0010\001\" \n\routput_shapes\022\013list(shape)(\0010\001\210\001\001\nV\n\017TFRecordDataset\022\r\n\tfilenames\030\007\022\024\n\020compression_type\030\007\022\017\n\013buffer_size\030\t\032\n\n\006handle\030\025\210\001\001\ny\n\013TakeDataset\022\021\n\rinput_dataset\030\025\022\t\n\005count\030\t\032\n\n\006handle\030\025\"\036\n\014output_types\022\nlist(type)(\0010\001\" \n\routput_shapes\022\013list(shape)(\0010\001\n~\n\rTensorDataset\022\033\n\ncomponents2\rToutput_types\032\n\n\006handle\030\025\"\037\n\rToutput_types\022\nlist(type)(\0010\001\" \n\routput_shapes\022\013list(shape)(\0010\001\210\001\001\n\203\001\n\022TensorSliceDataset\022\033\n\ncomponents2\rToutput_types\032\n\n\006handle\030\025\"\037\n\rToutput_types\022\nlist(type)(\0010\001\" \n\routput_shapes\022\013list(shape)(\0010\001\210\001\001\nV\n\017TextLineDataset\022\r\n\tfilenames\030\007\022\024\n\020compression_type\030\007\022\017\n\013buffer_size\030\t\032\n\n\006handle\030\025\210\001\001\n\177\n\nZipDataset\022\025\n\016input_datasets\030\025*\001N\032\n\n\006handle\030\025\"\036\n\014output_types\022\nlist(type)(\0010\001\" \n\routput_shapes\022\013list(shape)(\0010\001\"\014\n\001N\022\003int(\0010\001")
