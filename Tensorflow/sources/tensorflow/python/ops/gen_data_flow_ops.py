"""Python wrappers around Brain.

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
_accumulator_apply_gradient_outputs = [""]


def accumulator_apply_gradient(handle, local_step, gradient, name=None):
  r"""Applies a gradient to a given accumulator. Does not add if local_step is lesser

  than the accumulator's global_step.

  Args:
    handle: A `Tensor` of type mutable `string`. The handle to a accumulator.
    local_step: A `Tensor` of type `int64`.
      The local_step value at which the gradient was computed.
    gradient: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`, `complex128`, `qint8`, `quint8`, `qint32`, `half`.
      A tensor of the gradient to be accumulated.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  result = _op_def_lib.apply_op("AccumulatorApplyGradient", handle=handle,
                                local_step=local_step, gradient=gradient,
                                name=name)
  return result


_accumulator_num_accumulated_outputs = ["num_accumulated"]


def accumulator_num_accumulated(handle, name=None):
  r"""Returns the number of gradients aggregated in the given accumulators.

  Args:
    handle: A `Tensor` of type mutable `string`. The handle to an accumulator.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int32`.
    The number of gradients aggregated in the given accumulator.
  """
  result = _op_def_lib.apply_op("AccumulatorNumAccumulated", handle=handle,
                                name=name)
  return result


_accumulator_set_global_step_outputs = [""]


def accumulator_set_global_step(handle, new_global_step, name=None):
  r"""Updates the accumulator with a new value for global_step. Logs warning if the

  accumulator's value is already higher than new_global_step.

  Args:
    handle: A `Tensor` of type mutable `string`. The handle to an accumulator.
    new_global_step: A `Tensor` of type `int64`.
      The new global_step value to set.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  result = _op_def_lib.apply_op("AccumulatorSetGlobalStep", handle=handle,
                                new_global_step=new_global_step, name=name)
  return result


_accumulator_take_gradient_outputs = ["average"]


def accumulator_take_gradient(handle, num_required, dtype, name=None):
  r"""Extracts the average gradient in the given ConditionalAccumulator, provided

  that sufficient (i.e., more than num_required) gradients have been accumulated.
  The op blocks until sufficient gradients have been accumulated.
  If the accumulator has already aggregated more than num_required gradients, it
  returns the average of the accumulated gradients.
  Also automatically increments the recorded global_step in the accumulator by 1,
  and resets the aggregate to 0.

  Args:
    handle: A `Tensor` of type mutable `string`. The handle to an accumulator.
    num_required: A `Tensor` of type `int32`.
      Number of gradients required before we return an aggregate.
    dtype: A `tf.DType` from: `tf.float32, tf.float64, tf.int64, tf.int32, tf.uint8, tf.uint16, tf.int16, tf.int8, tf.complex64, tf.complex128, tf.qint8, tf.quint8, tf.qint32, tf.half`.
      The data type of accumulated gradients. Needs to correspond to the type
      of the accumulator.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `dtype`. The average of the accumulated gradients.
  """
  result = _op_def_lib.apply_op("AccumulatorTakeGradient", handle=handle,
                                num_required=num_required, dtype=dtype,
                                name=name)
  return result


__barrier_outputs = ["handle"]


def _barrier(component_types, shapes=None, capacity=None, container=None,
             shared_name=None, name=None):
  r"""Defines a barrier that persists across different graph executions.

  A barrier represents a key-value map, where each key is a string, and
  each value is a tuple of tensors.

  At runtime, the barrier contains 'complete' and 'incomplete'
  elements. A complete element has defined tensors for all components of
  its value tuple, and may be accessed using BarrierTakeMany. An
  incomplete element has some undefined components in its value tuple,
  and may be updated using BarrierInsertMany.

  Args:
    component_types: A list of `tf.DTypes` that has length `>= 1`.
      The type of each component in a value.
    shapes: An optional list of shapes (each a `tf.TensorShape` or list of `ints`). Defaults to `[]`.
      The shape of each component in a value. Each shape must be 1 in the
      first dimension. The length of this attr must be the same as the length of
      component_types.
    capacity: An optional `int`. Defaults to `-1`.
      The capacity of the barrier.  The default capacity is MAX_INT32,
      which is the largest capacity of the underlying queue.
    container: An optional `string`. Defaults to `""`.
      If non-empty, this barrier is placed in the given container.
      Otherwise, a default container is used.
    shared_name: An optional `string`. Defaults to `""`.
      If non-empty, this barrier will be shared under the given name
      across multiple sessions.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type mutable `string`. The handle to the barrier.
  """
  result = _op_def_lib.apply_op("Barrier", component_types=component_types,
                                shapes=shapes, capacity=capacity,
                                container=container, shared_name=shared_name,
                                name=name)
  return result


__barrier_close_outputs = [""]


def _barrier_close(handle, cancel_pending_enqueues=None, name=None):
  r"""Closes the given barrier.

  This operation signals that no more new elements will be inserted in the
  given barrier. Subsequent InsertMany that try to introduce a new key will fail.
  Subsequent InsertMany operations that just add missing components to already
  existing elements will continue to succeed. Subsequent TakeMany operations will
  continue to succeed if sufficient completed elements remain in the barrier.
  Subsequent TakeMany operations that would block will fail immediately.

  Args:
    handle: A `Tensor` of type mutable `string`. The handle to a barrier.
    cancel_pending_enqueues: An optional `bool`. Defaults to `False`.
      If true, all pending enqueue requests that are
      blocked on the barrier's queue will be cancelled. InsertMany will fail, even
      if no new key is introduced.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  result = _op_def_lib.apply_op("BarrierClose", handle=handle,
                                cancel_pending_enqueues=cancel_pending_enqueues,
                                name=name)
  return result


__barrier_incomplete_size_outputs = ["size"]


def _barrier_incomplete_size(handle, name=None):
  r"""Computes the number of incomplete elements in the given barrier.

  Args:
    handle: A `Tensor` of type mutable `string`. The handle to a barrier.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int32`.
    The number of incomplete elements (i.e. those with some of their value
    components not set) in the barrier.
  """
  result = _op_def_lib.apply_op("BarrierIncompleteSize", handle=handle,
                                name=name)
  return result


__barrier_insert_many_outputs = [""]


def _barrier_insert_many(handle, keys, values, component_index, name=None):
  r"""For each key, assigns the respective value to the specified component.

  If a key is not found in the barrier, this operation will create a new
  incomplete element. If a key is found in the barrier, and the element
  already has a value at component_index, this operation will fail with
  INVALID_ARGUMENT, and leave the barrier in an undefined state.

  Args:
    handle: A `Tensor` of type mutable `string`. The handle to a barrier.
    keys: A `Tensor` of type `string`.
      A one-dimensional tensor of keys, with length n.
    values: A `Tensor`.
      An any-dimensional tensor of values, which are associated with the
      respective keys. The 0th dimension must have length n.
    component_index: An `int`.
      The component of the barrier elements that is being assigned.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  result = _op_def_lib.apply_op("BarrierInsertMany", handle=handle, keys=keys,
                                values=values,
                                component_index=component_index, name=name)
  return result


__barrier_ready_size_outputs = ["size"]


def _barrier_ready_size(handle, name=None):
  r"""Computes the number of complete elements in the given barrier.

  Args:
    handle: A `Tensor` of type mutable `string`. The handle to a barrier.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int32`.
    The number of complete elements (i.e. those with all of their value
    components set) in the barrier.
  """
  result = _op_def_lib.apply_op("BarrierReadySize", handle=handle, name=name)
  return result


__barrier_take_many_outputs = ["indices", "keys", "values"]


_BarrierTakeManyOutput = _collections.namedtuple("BarrierTakeMany",
                                                 __barrier_take_many_outputs)


def _barrier_take_many(handle, num_elements, component_types,
                       allow_small_batch=None, wait_for_incomplete=None,
                       timeout_ms=None, name=None):
  r"""Takes the given number of completed elements from a barrier.

  This operation concatenates completed-element component tensors along
  the 0th dimension to make a single component tensor.

  Elements come out of the barrier when they are complete, and in the order
  in which they were placed into the barrier.  The indices output provides
  information about the batch in which each element was originally inserted
  into the barrier.

  Args:
    handle: A `Tensor` of type mutable `string`. The handle to a barrier.
    num_elements: A `Tensor` of type `int32`.
      A single-element tensor containing the number of elements to
      take.
    component_types: A list of `tf.DTypes` that has length `>= 1`.
      The type of each component in a value.
    allow_small_batch: An optional `bool`. Defaults to `False`.
      Allow to return less than num_elements items if barrier is
      already closed.
    wait_for_incomplete: An optional `bool`. Defaults to `False`.
    timeout_ms: An optional `int`. Defaults to `-1`.
      If the queue is empty, this operation will block for up to
      timeout_ms milliseconds.
      Note: This option is not supported yet.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (indices, keys, values).
    indices: A `Tensor` of type `int64`. A one-dimensional tensor of indices, with length num_elems.
      These indices refer to the batch in which the values were placed into the
      barrier (starting with MIN_LONG and increasing with each BarrierInsertMany).
    keys: A `Tensor` of type `string`. A one-dimensional tensor of keys, with length num_elements.
    values: A list of `Tensor` objects of type `component_types`. One any-dimensional tensor per component in a barrier element. All
      values have length num_elements in the 0th dimension.
  """
  result = _op_def_lib.apply_op("BarrierTakeMany", handle=handle,
                                num_elements=num_elements,
                                component_types=component_types,
                                allow_small_batch=allow_small_batch,
                                wait_for_incomplete=wait_for_incomplete,
                                timeout_ms=timeout_ms, name=name)
  return _BarrierTakeManyOutput._make(result)


_conditional_accumulator_outputs = ["handle"]


def conditional_accumulator(dtype, shape, container=None, shared_name=None,
                            name=None):
  r"""A conditional accumulator for aggregating gradients. The accumulator accepts

  gradients marked with local_step greater or equal to the most recent global_step
  known to the accumulator. The average can be extracted from the accumulator,
  provided sufficient gradients have been accumulated. Extracting the average
  automatically resets the aggregate to 0, and increments the global_step recorded
  by the accumulator.

  Args:
    dtype: A `tf.DType` from: `tf.float32, tf.float64, tf.int64, tf.int32, tf.uint8, tf.uint16, tf.int16, tf.int8, tf.complex64, tf.complex128, tf.qint8, tf.quint8, tf.qint32, tf.half`.
      The type of the value being accumulated.
    shape: A `tf.TensorShape` or list of `ints`.
      The shape of the values, can be [], in which case shape is unknown.
    container: An optional `string`. Defaults to `""`.
      If non-empty, this accumulator is placed in the given container.
      Otherwise, a default container is used.
    shared_name: An optional `string`. Defaults to `""`.
      If non-empty, this accumulator will be shared under the given name
      across multiple sessions.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type mutable `string`. The handle to the accumulator.
  """
  result = _op_def_lib.apply_op("ConditionalAccumulator", dtype=dtype,
                                shape=shape, container=container,
                                shared_name=shared_name, name=name)
  return result


__delete_session_tensor_outputs = [""]


def _delete_session_tensor(handle, name=None):
  r"""Delete the tensor specified by its handle in the session.

  Args:
    handle: A `Tensor` of type `string`.
      The handle for a tensor stored in the session state.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  result = _op_def_lib.apply_op("DeleteSessionTensor", handle=handle,
                                name=name)
  return result


_dynamic_partition_outputs = ["outputs"]


def dynamic_partition(data, partitions, num_partitions, name=None):
  r"""Partitions `data` into `num_partitions` tensors using indices from `partitions`.

  For each index tuple `js` of size `partitions.ndim`, the slice `data[js, ...]`
  becomes part of `outputs[partitions[js]]`.  The slices with `partitions[js] = i`
  are placed in `outputs[i]` in lexicographic order of `js`, and the first
  dimension of `outputs[i]` is the number of entries in `partitions` equal to `i`.
  In detail,

  ```python
      outputs[i].shape = [sum(partitions == i)] + data.shape[partitions.ndim:]

      outputs[i] = pack([data[js, ...] for js if partitions[js] == i])
  ```

  `data.shape` must start with `partitions.shape`.

  For example:

  ```python
      # Scalar partitions.
      partitions = 1
      num_partitions = 2
      data = [10, 20]
      outputs[0] = []  # Empty with shape [0, 2]
      outputs[1] = [[10, 20]]

      # Vector partitions.
      partitions = [0, 0, 1, 1, 0]
      num_partitions = 2
      data = [10, 20, 30, 40, 50]
      outputs[0] = [10, 20, 50]
      outputs[1] = [30, 40]
  ```

  <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
  <img style="width:100%" src="../../images/DynamicPartition.png" alt>
  </div>

  Args:
    data: A `Tensor`.
    partitions: A `Tensor` of type `int32`.
      Any shape.  Indices in the range `[0, num_partitions)`.
    num_partitions: An `int` that is `>= 1`.
      The number of partitions to output.
    name: A name for the operation (optional).

  Returns:
    A list of `num_partitions` `Tensor` objects of the same type as data.
  """
  result = _op_def_lib.apply_op("DynamicPartition", data=data,
                                partitions=partitions,
                                num_partitions=num_partitions, name=name)
  return result


_dynamic_stitch_outputs = ["merged"]


def dynamic_stitch(indices, data, name=None):
  r"""Interleave the values from the `data` tensors into a single tensor.

  Builds a merged tensor such that

  ```python
      merged[indices[m][i, ..., j], ...] = data[m][i, ..., j, ...]
  ```

  For example, if each `indices[m]` is scalar or vector, we have

  ```python
      # Scalar indices:
      merged[indices[m], ...] = data[m][...]

      # Vector indices:
      merged[indices[m][i], ...] = data[m][i, ...]
  ```

  Each `data[i].shape` must start with the corresponding `indices[i].shape`,
  and the rest of `data[i].shape` must be constant w.r.t. `i`.  That is, we
  must have `data[i].shape = indices[i].shape + constant`.  In terms of this
  `constant`, the output shape is

      merged.shape = [max(indices)] + constant

  Values are merged in order, so if an index appears in both `indices[m][i]` and
  `indices[n][j]` for `(m,i) < (n,j)` the slice `data[n][j]` will appear in the
  merged result.

  For example:

  ```python
      indices[0] = 6
      indices[1] = [4, 1]
      indices[2] = [[5, 2], [0, 3]]
      data[0] = [61, 62]
      data[1] = [[41, 42], [11, 12]]
      data[2] = [[[51, 52], [21, 22]], [[1, 2], [31, 32]]]
      merged = [[1, 2], [11, 12], [21, 22], [31, 32], [41, 42],
                [51, 52], [61, 62]]
  ```

  <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
  <img style="width:100%" src="../../images/DynamicStitch.png" alt>
  </div>

  Args:
    indices: A list of at least 1 `Tensor` objects of type `int32`.
    data: A list with the same number of `Tensor` objects as `indices` of `Tensor` objects of the same type.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `data`.
  """
  result = _op_def_lib.apply_op("DynamicStitch", indices=indices, data=data,
                                name=name)
  return result


__fifo_queue_outputs = ["handle"]


def _fifo_queue(component_types, shapes=None, capacity=None, container=None,
                shared_name=None, name=None):
  r"""A queue that produces elements in first-in first-out order.

  Args:
    component_types: A list of `tf.DTypes` that has length `>= 1`.
      The type of each component in a value.
    shapes: An optional list of shapes (each a `tf.TensorShape` or list of `ints`). Defaults to `[]`.
      The shape of each component in a value. The length of this attr must
      be either 0 or the same as the length of component_types. If the length of
      this attr is 0, the shapes of queue elements are not constrained, and
      only one element may be dequeued at a time.
    capacity: An optional `int`. Defaults to `-1`.
      The upper bound on the number of elements in this queue.
      Negative numbers mean no limit.
    container: An optional `string`. Defaults to `""`.
      If non-empty, this queue is placed in the given container.
      Otherwise, a default container is used.
    shared_name: An optional `string`. Defaults to `""`.
      If non-empty, this queue will be shared under the given name
      across multiple sessions.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type mutable `string`. The handle to the queue.
  """
  result = _op_def_lib.apply_op("FIFOQueue", component_types=component_types,
                                shapes=shapes, capacity=capacity,
                                container=container, shared_name=shared_name,
                                name=name)
  return result


__fifo_queue_v2_outputs = ["handle"]


def _fifo_queue_v2(component_types, shapes=None, capacity=None,
                   container=None, shared_name=None, name=None):
  r"""A queue that produces elements in first-in first-out order.

  Args:
    component_types: A list of `tf.DTypes` that has length `>= 1`.
      The type of each component in a value.
    shapes: An optional list of shapes (each a `tf.TensorShape` or list of `ints`). Defaults to `[]`.
      The shape of each component in a value. The length of this attr must
      be either 0 or the same as the length of component_types. If the length of
      this attr is 0, the shapes of queue elements are not constrained, and
      only one element may be dequeued at a time.
    capacity: An optional `int`. Defaults to `-1`.
      The upper bound on the number of elements in this queue.
      Negative numbers mean no limit.
    container: An optional `string`. Defaults to `""`.
      If non-empty, this queue is placed in the given container.
      Otherwise, a default container is used.
    shared_name: An optional `string`. Defaults to `""`.
      If non-empty, this queue will be shared under the given name
      across multiple sessions.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `resource`. The handle to the queue.
  """
  result = _op_def_lib.apply_op("FIFOQueueV2",
                                component_types=component_types,
                                shapes=shapes, capacity=capacity,
                                container=container, shared_name=shared_name,
                                name=name)
  return result


__fake_queue_outputs = ["handle"]


def _fake_queue(resource, name=None):
  r"""Deprecated. Do not use.

  Args:
    resource: A `Tensor` of type `resource`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type mutable `string`.
  """
  result = _op_def_lib.apply_op("FakeQueue", resource=resource, name=name)
  return result


__get_session_handle_outputs = ["handle"]


def _get_session_handle(value, name=None):
  r"""Store the input tensor in the state of the current session.

  Args:
    value: A `Tensor`. The tensor to be stored.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`.
    The handle for the tensor stored in the session state.
  """
  result = _op_def_lib.apply_op("GetSessionHandle", value=value, name=name)
  return result


__get_session_tensor_outputs = ["value"]


def _get_session_tensor(handle, dtype, name=None):
  r"""Get the value of the tensor specified by its handle.

  Args:
    handle: A `Tensor` of type `string`.
      The handle for a tensor stored in the session state.
    dtype: A `tf.DType`. The type of the output value.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `dtype`. The tensor for the given handle.
  """
  result = _op_def_lib.apply_op("GetSessionTensor", handle=handle,
                                dtype=dtype, name=name)
  return result


__hash_table_outputs = ["table_handle"]


def _hash_table(key_dtype, value_dtype, container=None, shared_name=None,
                use_node_name_sharing=None, name=None):
  r"""Creates a non-initialized hash table.

  This op creates a hash table, specifying the type of its keys and values.
  Before using the table you will have to initialize it.  After initialization the
  table will be immutable.

  Args:
    key_dtype: A `tf.DType`. Type of the table keys.
    value_dtype: A `tf.DType`. Type of the table values.
    container: An optional `string`. Defaults to `""`.
      If non-empty, this table is placed in the given container.
      Otherwise, a default container is used.
    shared_name: An optional `string`. Defaults to `""`.
      If non-empty, this table is shared under the given name across
      multiple sessions.
    use_node_name_sharing: An optional `bool`. Defaults to `False`.
      If true and shared_name is empty, the table is shared
      using the node name.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type mutable `string`. Handle to a table.
  """
  result = _op_def_lib.apply_op("HashTable", key_dtype=key_dtype,
                                value_dtype=value_dtype, container=container,
                                shared_name=shared_name,
                                use_node_name_sharing=use_node_name_sharing,
                                name=name)
  return result


__initialize_table_outputs = [""]


def _initialize_table(table_handle, keys, values, name=None):
  r"""Table initializer that takes two tensors for keys and values respectively.

  Args:
    table_handle: A `Tensor` of type mutable `string`.
      Handle to a table which will be initialized.
    keys: A `Tensor`. Keys of type Tkey.
    values: A `Tensor`. Values of type Tval.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  result = _op_def_lib.apply_op("InitializeTable", table_handle=table_handle,
                                keys=keys, values=values, name=name)
  return result


__initialize_table_from_text_file_outputs = [""]


def _initialize_table_from_text_file(table_handle, filename, key_index,
                                     value_index, vocab_size=None,
                                     delimiter=None, name=None):
  r"""Initializes a table from a text file.

  It inserts one key-value pair into the table for each line of the file.
  The key and value is extracted from the whole line content, elements from the
  split line based on `delimiter` or the line number (starting from zero).
  Where to extract the key and value from a line is specified by `key_index` and
  `value_index`.

  - A value of -1 means use the line number(starting from zero), expects `int64`.
  - A value of -2 means use the whole line content, expects `string`.
  - A value >= 0 means use the index (starting at zero) of the split line based
    on `delimiter`.

  Args:
    table_handle: A `Tensor` of type mutable `string`.
      Handle to a table which will be initialized.
    filename: A `Tensor` of type `string`. Filename of a vocabulary text file.
    key_index: An `int` that is `>= -2`.
      Column index in a line to get the table `key` values from.
    value_index: An `int` that is `>= -2`.
      Column index that represents information of a line to get the table
      `value` values from.
    vocab_size: An optional `int` that is `>= -1`. Defaults to `-1`.
      Number of elements of the file, use -1 if unknown.
    delimiter: An optional `string`. Defaults to `"\t"`.
      Delimiter to separate fields in a line.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  result = _op_def_lib.apply_op("InitializeTableFromTextFile",
                                table_handle=table_handle, filename=filename,
                                key_index=key_index, value_index=value_index,
                                vocab_size=vocab_size, delimiter=delimiter,
                                name=name)
  return result


__lookup_table_export_outputs = ["keys", "values"]


_LookupTableExportOutput = _collections.namedtuple("LookupTableExport",
                                                   __lookup_table_export_outputs)


def _lookup_table_export(table_handle, Tkeys, Tvalues, name=None):
  r"""Outputs all keys and values in the table.

  Args:
    table_handle: A `Tensor` of type mutable `string`. Handle to the table.
    Tkeys: A `tf.DType`.
    Tvalues: A `tf.DType`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (keys, values).
    keys: A `Tensor` of type `Tkeys`. Vector of all keys present in the table.
    values: A `Tensor` of type `Tvalues`. Tensor of all values in the table. Indexed in parallel with `keys`.
  """
  result = _op_def_lib.apply_op("LookupTableExport",
                                table_handle=table_handle, Tkeys=Tkeys,
                                Tvalues=Tvalues, name=name)
  return _LookupTableExportOutput._make(result)


__lookup_table_find_outputs = ["values"]


def _lookup_table_find(table_handle, keys, default_value, name=None):
  r"""Looks up keys in a table, outputs the corresponding values.

  The tensor `keys` must of the same type as the keys of the table.
  The output `values` is of the type of the table values.

  The scalar `default_value` is the value output for keys not present in the
  table. It must also be of the same type as the table values.

  Args:
    table_handle: A `Tensor` of type mutable `string`. Handle to the table.
    keys: A `Tensor`. Any shape.  Keys to look up.
    default_value: A `Tensor`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `default_value`.
    Same shape as `keys`.  Values found in the table, or `default_values`
    for missing keys.
  """
  result = _op_def_lib.apply_op("LookupTableFind", table_handle=table_handle,
                                keys=keys, default_value=default_value,
                                name=name)
  return result


__lookup_table_import_outputs = [""]


def _lookup_table_import(table_handle, keys, values, name=None):
  r"""Replaces the contents of the table with the specified keys and values.

  The tensor `keys` must be of the same type as the keys of the table.
  The tensor `values` must be of the type of the table values.

  Args:
    table_handle: A `Tensor` of type mutable `string`. Handle to the table.
    keys: A `Tensor`. Any shape.  Keys to look up.
    values: A `Tensor`. Values to associate with keys.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  result = _op_def_lib.apply_op("LookupTableImport",
                                table_handle=table_handle, keys=keys,
                                values=values, name=name)
  return result


__lookup_table_insert_outputs = [""]


def _lookup_table_insert(table_handle, keys, values, name=None):
  r"""Updates the table to associates keys with values.

  The tensor `keys` must be of the same type as the keys of the table.
  The tensor `values` must be of the type of the table values.

  Args:
    table_handle: A `Tensor` of type mutable `string`. Handle to the table.
    keys: A `Tensor`. Any shape.  Keys to look up.
    values: A `Tensor`. Values to associate with keys.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  result = _op_def_lib.apply_op("LookupTableInsert",
                                table_handle=table_handle, keys=keys,
                                values=values, name=name)
  return result


__lookup_table_size_outputs = ["size"]


def _lookup_table_size(table_handle, name=None):
  r"""Computes the number of elements in the given table.

  Args:
    table_handle: A `Tensor` of type mutable `string`. Handle to the table.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int64`.
    Scalar that contains number of elements in the table.
  """
  result = _op_def_lib.apply_op("LookupTableSize", table_handle=table_handle,
                                name=name)
  return result


__mutable_dense_hash_table_outputs = ["table_handle"]


def _mutable_dense_hash_table(empty_key, value_dtype, container=None,
                              shared_name=None, use_node_name_sharing=None,
                              value_shape=None, initial_num_buckets=None,
                              max_load_factor=None, name=None):
  r"""Creates an empty hash table that uses tensors as the backing store. It uses

  "open addressing" with quadratic reprobing to resolve collisions.

  This op creates a mutable hash table, specifying the type of its keys and
  values. Each value must be a scalar. Data can be inserted into the table using
  the insert operations. It does not support the initialization operation.

  Args:
    empty_key: A `Tensor`.
      The key used to represent empty key buckets internally. Must not
      be used in insert or lookup operations.
    value_dtype: A `tf.DType`. Type of the table values.
    container: An optional `string`. Defaults to `""`.
      If non-empty, this table is placed in the given container.
      Otherwise, a default container is used.
    shared_name: An optional `string`. Defaults to `""`.
      If non-empty, this table is shared under the given name across
      multiple sessions.
    use_node_name_sharing: An optional `bool`. Defaults to `False`.
    value_shape: An optional `tf.TensorShape` or list of `ints`. Defaults to `[]`.
      The shape of each value.
    initial_num_buckets: An optional `int`. Defaults to `131072`.
      The initial number of hash table buckets. Must be a power
      to 2.
    max_load_factor: An optional `float`. Defaults to `0.8`.
      The maximum ratio between number of entries and number of
      buckets before growing the table. Must be between 0 and 1.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type mutable `string`. Handle to a table.
  """
  result = _op_def_lib.apply_op("MutableDenseHashTable", empty_key=empty_key,
                                value_dtype=value_dtype, container=container,
                                shared_name=shared_name,
                                use_node_name_sharing=use_node_name_sharing,
                                value_shape=value_shape,
                                initial_num_buckets=initial_num_buckets,
                                max_load_factor=max_load_factor, name=name)
  return result


__mutable_hash_table_outputs = ["table_handle"]


def _mutable_hash_table(key_dtype, value_dtype, container=None,
                        shared_name=None, use_node_name_sharing=None,
                        name=None):
  r"""Creates an empty hash table.

  This op creates a mutable hash table, specifying the type of its keys and
  values. Each value must be a scalar. Data can be inserted into the table using
  the insert operations. It does not support the initialization operation.

  Args:
    key_dtype: A `tf.DType`. Type of the table keys.
    value_dtype: A `tf.DType`. Type of the table values.
    container: An optional `string`. Defaults to `""`.
      If non-empty, this table is placed in the given container.
      Otherwise, a default container is used.
    shared_name: An optional `string`. Defaults to `""`.
      If non-empty, this table is shared under the given name across
      multiple sessions.
    use_node_name_sharing: An optional `bool`. Defaults to `False`.
      If true and shared_name is empty, the table is shared
      using the node name.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type mutable `string`. Handle to a table.
  """
  result = _op_def_lib.apply_op("MutableHashTable", key_dtype=key_dtype,
                                value_dtype=value_dtype, container=container,
                                shared_name=shared_name,
                                use_node_name_sharing=use_node_name_sharing,
                                name=name)
  return result


__mutable_hash_table_of_tensors_outputs = ["table_handle"]


def _mutable_hash_table_of_tensors(key_dtype, value_dtype, container=None,
                                   shared_name=None,
                                   use_node_name_sharing=None,
                                   value_shape=None, name=None):
  r"""Creates an empty hash table.

  This op creates a mutable hash table, specifying the type of its keys and
  values. Each value must be a vector. Data can be inserted into the table using
  the insert operations. It does not support the initialization operation.

  Args:
    key_dtype: A `tf.DType`. Type of the table keys.
    value_dtype: A `tf.DType`. Type of the table values.
    container: An optional `string`. Defaults to `""`.
      If non-empty, this table is placed in the given container.
      Otherwise, a default container is used.
    shared_name: An optional `string`. Defaults to `""`.
      If non-empty, this table is shared under the given name across
      multiple sessions.
    use_node_name_sharing: An optional `bool`. Defaults to `False`.
    value_shape: An optional `tf.TensorShape` or list of `ints`. Defaults to `[]`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type mutable `string`. Handle to a table.
  """
  result = _op_def_lib.apply_op("MutableHashTableOfTensors",
                                key_dtype=key_dtype, value_dtype=value_dtype,
                                container=container, shared_name=shared_name,
                                use_node_name_sharing=use_node_name_sharing,
                                value_shape=value_shape, name=name)
  return result


__padding_fifo_queue_outputs = ["handle"]


def _padding_fifo_queue(component_types, shapes=None, capacity=None,
                        container=None, shared_name=None, name=None):
  r"""A queue that produces elements in first-in first-out order.

  Variable-size shapes are allowed by setting the corresponding shape dimensions
  to 0 in the shape attr.  In this case DequeueMany will pad up to the maximum
  size of any given element in the minibatch.  See below for details.

  Args:
    component_types: A list of `tf.DTypes` that has length `>= 1`.
      The type of each component in a value.
    shapes: An optional list of shapes (each a `tf.TensorShape` or list of `ints`). Defaults to `[]`.
      The shape of each component in a value. The length of this attr must
      be either 0 or the same as the length of component_types.
      Shapes of fixed rank but variable size are allowed by setting
      any shape dimension to -1.  In this case, the inputs' shape may vary along
      the given dimension, and DequeueMany will pad the given dimension with
      zeros up to the maximum shape of all elements in the given batch.
      If the length of this attr is 0, different queue elements may have
      different ranks and shapes, but only one element may be dequeued at a time.
    capacity: An optional `int`. Defaults to `-1`.
      The upper bound on the number of elements in this queue.
      Negative numbers mean no limit.
    container: An optional `string`. Defaults to `""`.
      If non-empty, this queue is placed in the given container.
      Otherwise, a default container is used.
    shared_name: An optional `string`. Defaults to `""`.
      If non-empty, this queue will be shared under the given name
      across multiple sessions.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type mutable `string`. The handle to the queue.
  """
  result = _op_def_lib.apply_op("PaddingFIFOQueue",
                                component_types=component_types,
                                shapes=shapes, capacity=capacity,
                                container=container, shared_name=shared_name,
                                name=name)
  return result


__padding_fifo_queue_v2_outputs = ["handle"]


def _padding_fifo_queue_v2(component_types, shapes=None, capacity=None,
                           container=None, shared_name=None, name=None):
  r"""A queue that produces elements in first-in first-out order.

  Variable-size shapes are allowed by setting the corresponding shape dimensions
  to 0 in the shape attr.  In this case DequeueMany will pad up to the maximum
  size of any given element in the minibatch.  See below for details.

  Args:
    component_types: A list of `tf.DTypes` that has length `>= 1`.
      The type of each component in a value.
    shapes: An optional list of shapes (each a `tf.TensorShape` or list of `ints`). Defaults to `[]`.
      The shape of each component in a value. The length of this attr must
      be either 0 or the same as the length of component_types.
      Shapes of fixed rank but variable size are allowed by setting
      any shape dimension to -1.  In this case, the inputs' shape may vary along
      the given dimension, and DequeueMany will pad the given dimension with
      zeros up to the maximum shape of all elements in the given batch.
      If the length of this attr is 0, different queue elements may have
      different ranks and shapes, but only one element may be dequeued at a time.
    capacity: An optional `int`. Defaults to `-1`.
      The upper bound on the number of elements in this queue.
      Negative numbers mean no limit.
    container: An optional `string`. Defaults to `""`.
      If non-empty, this queue is placed in the given container.
      Otherwise, a default container is used.
    shared_name: An optional `string`. Defaults to `""`.
      If non-empty, this queue will be shared under the given name
      across multiple sessions.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `resource`. The handle to the queue.
  """
  result = _op_def_lib.apply_op("PaddingFIFOQueueV2",
                                component_types=component_types,
                                shapes=shapes, capacity=capacity,
                                container=container, shared_name=shared_name,
                                name=name)
  return result


__priority_queue_outputs = ["handle"]


def _priority_queue(shapes, component_types=None, capacity=None,
                    container=None, shared_name=None, name=None):
  r"""A queue that produces elements sorted by the first component value.

  Note that the PriorityQueue requires the first component of any element
  to be a scalar int64, in addition to the other elements declared by
  component_types.  Therefore calls to Enqueue and EnqueueMany (resp. Dequeue
  and DequeueMany) on a PriorityQueue will all require (resp. output) one extra
  entry in their input (resp. output) lists.

  Args:
    shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`).
      The shape of each component in a value. The length of this attr must
      be either 0 or the same as the length of component_types. If the length of
      this attr is 0, the shapes of queue elements are not constrained, and
      only one element may be dequeued at a time.
    component_types: An optional list of `tf.DTypes`. Defaults to `[]`.
      The type of each component in a value.
    capacity: An optional `int`. Defaults to `-1`.
      The upper bound on the number of elements in this queue.
      Negative numbers mean no limit.
    container: An optional `string`. Defaults to `""`.
      If non-empty, this queue is placed in the given container.
      Otherwise, a default container is used.
    shared_name: An optional `string`. Defaults to `""`.
      If non-empty, this queue will be shared under the given name
      across multiple sessions.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type mutable `string`. The handle to the queue.
  """
  result = _op_def_lib.apply_op("PriorityQueue", shapes=shapes,
                                component_types=component_types,
                                capacity=capacity, container=container,
                                shared_name=shared_name, name=name)
  return result


__priority_queue_v2_outputs = ["handle"]


def _priority_queue_v2(shapes, component_types=None, capacity=None,
                       container=None, shared_name=None, name=None):
  r"""A queue that produces elements sorted by the first component value.

  Note that the PriorityQueue requires the first component of any element
  to be a scalar int64, in addition to the other elements declared by
  component_types.  Therefore calls to Enqueue and EnqueueMany (resp. Dequeue
  and DequeueMany) on a PriorityQueue will all require (resp. output) one extra
  entry in their input (resp. output) lists.

  Args:
    shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`).
      The shape of each component in a value. The length of this attr must
      be either 0 or the same as the length of component_types. If the length of
      this attr is 0, the shapes of queue elements are not constrained, and
      only one element may be dequeued at a time.
    component_types: An optional list of `tf.DTypes`. Defaults to `[]`.
      The type of each component in a value.
    capacity: An optional `int`. Defaults to `-1`.
      The upper bound on the number of elements in this queue.
      Negative numbers mean no limit.
    container: An optional `string`. Defaults to `""`.
      If non-empty, this queue is placed in the given container.
      Otherwise, a default container is used.
    shared_name: An optional `string`. Defaults to `""`.
      If non-empty, this queue will be shared under the given name
      across multiple sessions.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `resource`. The handle to the queue.
  """
  result = _op_def_lib.apply_op("PriorityQueueV2", shapes=shapes,
                                component_types=component_types,
                                capacity=capacity, container=container,
                                shared_name=shared_name, name=name)
  return result


__queue_close_outputs = [""]


def _queue_close(handle, cancel_pending_enqueues=None, name=None):
  r"""Closes the given queue.

  This operation signals that no more elements will be enqueued in the
  given queue. Subsequent Enqueue(Many) operations will fail.
  Subsequent Dequeue(Many) operations will continue to succeed if
  sufficient elements remain in the queue. Subsequent Dequeue(Many)
  operations that would block will fail immediately.

  Args:
    handle: A `Tensor` of type mutable `string`. The handle to a queue.
    cancel_pending_enqueues: An optional `bool`. Defaults to `False`.
      If true, all pending enqueue requests that are
      blocked on the given queue will be cancelled.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  result = _op_def_lib.apply_op("QueueClose", handle=handle,
                                cancel_pending_enqueues=cancel_pending_enqueues,
                                name=name)
  return result


__queue_close_v2_outputs = [""]


def _queue_close_v2(handle, cancel_pending_enqueues=None, name=None):
  r"""Closes the given queue.

  This operation signals that no more elements will be enqueued in the
  given queue. Subsequent Enqueue(Many) operations will fail.
  Subsequent Dequeue(Many) operations will continue to succeed if
  sufficient elements remain in the queue. Subsequent Dequeue(Many)
  operations that would block will fail immediately.

  Args:
    handle: A `Tensor` of type `resource`. The handle to a queue.
    cancel_pending_enqueues: An optional `bool`. Defaults to `False`.
      If true, all pending enqueue requests that are
      blocked on the given queue will be cancelled.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  result = _op_def_lib.apply_op("QueueCloseV2", handle=handle,
                                cancel_pending_enqueues=cancel_pending_enqueues,
                                name=name)
  return result


__queue_dequeue_outputs = ["components"]


def _queue_dequeue(handle, component_types, timeout_ms=None, name=None):
  r"""Dequeues a tuple of one or more tensors from the given queue.

  This operation has k outputs, where k is the number of components
  in the tuples stored in the given queue, and output i is the ith
  component of the dequeued tuple.

  N.B. If the queue is empty, this operation will block until an element
  has been dequeued (or 'timeout_ms' elapses, if specified).

  Args:
    handle: A `Tensor` of type mutable `string`. The handle to a queue.
    component_types: A list of `tf.DTypes` that has length `>= 1`.
      The type of each component in a tuple.
    timeout_ms: An optional `int`. Defaults to `-1`.
      If the queue is empty, this operation will block for up to
      timeout_ms milliseconds.
      Note: This option is not supported yet.
    name: A name for the operation (optional).

  Returns:
    A list of `Tensor` objects of type `component_types`.
    One or more tensors that were dequeued as a tuple.
  """
  result = _op_def_lib.apply_op("QueueDequeue", handle=handle,
                                component_types=component_types,
                                timeout_ms=timeout_ms, name=name)
  return result


__queue_dequeue_many_outputs = ["components"]


def _queue_dequeue_many(handle, n, component_types, timeout_ms=None,
                        name=None):
  r"""Dequeues n tuples of one or more tensors from the given queue.

  If the queue is closed and there are fewer than n elements, then an
  OutOfRange error is returned.

  This operation concatenates queue-element component tensors along the
  0th dimension to make a single component tensor.  All of the components
  in the dequeued tuple will have size n in the 0th dimension.

  This operation has k outputs, where k is the number of components in
  the tuples stored in the given queue, and output i is the ith
  component of the dequeued tuple.

  N.B. If the queue is empty, this operation will block until n elements
  have been dequeued (or 'timeout_ms' elapses, if specified).

  Args:
    handle: A `Tensor` of type mutable `string`. The handle to a queue.
    n: A `Tensor` of type `int32`. The number of tuples to dequeue.
    component_types: A list of `tf.DTypes` that has length `>= 1`.
      The type of each component in a tuple.
    timeout_ms: An optional `int`. Defaults to `-1`.
      If the queue has fewer than n elements, this operation
      will block for up to timeout_ms milliseconds.
      Note: This option is not supported yet.
    name: A name for the operation (optional).

  Returns:
    A list of `Tensor` objects of type `component_types`.
    One or more tensors that were dequeued as a tuple.
  """
  result = _op_def_lib.apply_op("QueueDequeueMany", handle=handle, n=n,
                                component_types=component_types,
                                timeout_ms=timeout_ms, name=name)
  return result


__queue_dequeue_many_v2_outputs = ["components"]


def _queue_dequeue_many_v2(handle, n, component_types, timeout_ms=None,
                           name=None):
  r"""Dequeues n tuples of one or more tensors from the given queue.

  If the queue is closed and there are fewer than n elements, then an
  OutOfRange error is returned.

  This operation concatenates queue-element component tensors along the
  0th dimension to make a single component tensor.  All of the components
  in the dequeued tuple will have size n in the 0th dimension.

  This operation has k outputs, where k is the number of components in
  the tuples stored in the given queue, and output i is the ith
  component of the dequeued tuple.

  N.B. If the queue is empty, this operation will block until n elements
  have been dequeued (or 'timeout_ms' elapses, if specified).

  Args:
    handle: A `Tensor` of type `resource`. The handle to a queue.
    n: A `Tensor` of type `int32`. The number of tuples to dequeue.
    component_types: A list of `tf.DTypes` that has length `>= 1`.
      The type of each component in a tuple.
    timeout_ms: An optional `int`. Defaults to `-1`.
      If the queue has fewer than n elements, this operation
      will block for up to timeout_ms milliseconds.
      Note: This option is not supported yet.
    name: A name for the operation (optional).

  Returns:
    A list of `Tensor` objects of type `component_types`.
    One or more tensors that were dequeued as a tuple.
  """
  result = _op_def_lib.apply_op("QueueDequeueManyV2", handle=handle, n=n,
                                component_types=component_types,
                                timeout_ms=timeout_ms, name=name)
  return result


__queue_dequeue_up_to_outputs = ["components"]


def _queue_dequeue_up_to(handle, n, component_types, timeout_ms=None,
                         name=None):
  r"""Dequeues n tuples of one or more tensors from the given queue.

  This operation is not supported by all queues.  If a queue does not support
  DequeueUpTo, then an Unimplemented error is returned.

  If the queue is closed and there are more than 0 but less than n elements
  remaining, then instead of returning an OutOfRange error like
  QueueDequeueMany, less than `n` elements are returned immediately.  If the queue
  is closed and there are 0 elements left in the queue, then an OutOfRange
  error is returned just like in QueueDequeueMany.  Otherwise the behavior
  is identical to QueueDequeueMany:

  This operation concatenates queue-element component tensors along the
  0th dimension to make a single component tensor.  All of the components
  in the dequeued tuple will have size n in the 0th dimension.

  This operation has k outputs, where k is the number of components in
  the tuples stored in the given queue, and output i is the ith
  component of the dequeued tuple.

  Args:
    handle: A `Tensor` of type mutable `string`. The handle to a queue.
    n: A `Tensor` of type `int32`. The number of tuples to dequeue.
    component_types: A list of `tf.DTypes` that has length `>= 1`.
      The type of each component in a tuple.
    timeout_ms: An optional `int`. Defaults to `-1`.
      If the queue has fewer than n elements, this operation
      will block for up to timeout_ms milliseconds.
      Note: This option is not supported yet.
    name: A name for the operation (optional).

  Returns:
    A list of `Tensor` objects of type `component_types`.
    One or more tensors that were dequeued as a tuple.
  """
  result = _op_def_lib.apply_op("QueueDequeueUpTo", handle=handle, n=n,
                                component_types=component_types,
                                timeout_ms=timeout_ms, name=name)
  return result


__queue_dequeue_up_to_v2_outputs = ["components"]


def _queue_dequeue_up_to_v2(handle, n, component_types, timeout_ms=None,
                            name=None):
  r"""Dequeues n tuples of one or more tensors from the given queue.

  This operation is not supported by all queues.  If a queue does not support
  DequeueUpTo, then an Unimplemented error is returned.

  If the queue is closed and there are more than 0 but less than n elements
  remaining, then instead of returning an OutOfRange error like
  QueueDequeueMany, less than `n` elements are returned immediately.  If the queue
  is closed and there are 0 elements left in the queue, then an OutOfRange
  error is returned just like in QueueDequeueMany.  Otherwise the behavior
  is identical to QueueDequeueMany:

  This operation concatenates queue-element component tensors along the
  0th dimension to make a single component tensor.  All of the components
  in the dequeued tuple will have size n in the 0th dimension.

  This operation has k outputs, where k is the number of components in
  the tuples stored in the given queue, and output i is the ith
  component of the dequeued tuple.

  Args:
    handle: A `Tensor` of type `resource`. The handle to a queue.
    n: A `Tensor` of type `int32`. The number of tuples to dequeue.
    component_types: A list of `tf.DTypes` that has length `>= 1`.
      The type of each component in a tuple.
    timeout_ms: An optional `int`. Defaults to `-1`.
      If the queue has fewer than n elements, this operation
      will block for up to timeout_ms milliseconds.
      Note: This option is not supported yet.
    name: A name for the operation (optional).

  Returns:
    A list of `Tensor` objects of type `component_types`.
    One or more tensors that were dequeued as a tuple.
  """
  result = _op_def_lib.apply_op("QueueDequeueUpToV2", handle=handle, n=n,
                                component_types=component_types,
                                timeout_ms=timeout_ms, name=name)
  return result


__queue_dequeue_v2_outputs = ["components"]


def _queue_dequeue_v2(handle, component_types, timeout_ms=None, name=None):
  r"""Dequeues a tuple of one or more tensors from the given queue.

  This operation has k outputs, where k is the number of components
  in the tuples stored in the given queue, and output i is the ith
  component of the dequeued tuple.

  N.B. If the queue is empty, this operation will block until an element
  has been dequeued (or 'timeout_ms' elapses, if specified).

  Args:
    handle: A `Tensor` of type `resource`. The handle to a queue.
    component_types: A list of `tf.DTypes` that has length `>= 1`.
      The type of each component in a tuple.
    timeout_ms: An optional `int`. Defaults to `-1`.
      If the queue is empty, this operation will block for up to
      timeout_ms milliseconds.
      Note: This option is not supported yet.
    name: A name for the operation (optional).

  Returns:
    A list of `Tensor` objects of type `component_types`.
    One or more tensors that were dequeued as a tuple.
  """
  result = _op_def_lib.apply_op("QueueDequeueV2", handle=handle,
                                component_types=component_types,
                                timeout_ms=timeout_ms, name=name)
  return result


__queue_enqueue_outputs = [""]


def _queue_enqueue(handle, components, timeout_ms=None, name=None):
  r"""Enqueues a tuple of one or more tensors in the given queue.

  The components input has k elements, which correspond to the components of
  tuples stored in the given queue.

  N.B. If the queue is full, this operation will block until the given
  element has been enqueued (or 'timeout_ms' elapses, if specified).

  Args:
    handle: A `Tensor` of type mutable `string`. The handle to a queue.
    components: A list of `Tensor` objects.
      One or more tensors from which the enqueued tensors should be taken.
    timeout_ms: An optional `int`. Defaults to `-1`.
      If the queue is full, this operation will block for up to
      timeout_ms milliseconds.
      Note: This option is not supported yet.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  result = _op_def_lib.apply_op("QueueEnqueue", handle=handle,
                                components=components, timeout_ms=timeout_ms,
                                name=name)
  return result


__queue_enqueue_many_outputs = [""]


def _queue_enqueue_many(handle, components, timeout_ms=None, name=None):
  r"""Enqueues zero or more tuples of one or more tensors in the given queue.

  This operation slices each component tensor along the 0th dimension to
  make multiple queue elements. All of the tuple components must have the
  same size in the 0th dimension.

  The components input has k elements, which correspond to the components of
  tuples stored in the given queue.

  N.B. If the queue is full, this operation will block until the given
  elements have been enqueued (or 'timeout_ms' elapses, if specified).

  Args:
    handle: A `Tensor` of type mutable `string`. The handle to a queue.
    components: A list of `Tensor` objects.
      One or more tensors from which the enqueued tensors should
      be taken.
    timeout_ms: An optional `int`. Defaults to `-1`.
      If the queue is too full, this operation will block for up
      to timeout_ms milliseconds.
      Note: This option is not supported yet.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  result = _op_def_lib.apply_op("QueueEnqueueMany", handle=handle,
                                components=components, timeout_ms=timeout_ms,
                                name=name)
  return result


__queue_enqueue_many_v2_outputs = [""]


def _queue_enqueue_many_v2(handle, components, timeout_ms=None, name=None):
  r"""Enqueues zero or more tuples of one or more tensors in the given queue.

  This operation slices each component tensor along the 0th dimension to
  make multiple queue elements. All of the tuple components must have the
  same size in the 0th dimension.

  The components input has k elements, which correspond to the components of
  tuples stored in the given queue.

  N.B. If the queue is full, this operation will block until the given
  elements have been enqueued (or 'timeout_ms' elapses, if specified).

  Args:
    handle: A `Tensor` of type `resource`. The handle to a queue.
    components: A list of `Tensor` objects.
      One or more tensors from which the enqueued tensors should
      be taken.
    timeout_ms: An optional `int`. Defaults to `-1`.
      If the queue is too full, this operation will block for up
      to timeout_ms milliseconds.
      Note: This option is not supported yet.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  result = _op_def_lib.apply_op("QueueEnqueueManyV2", handle=handle,
                                components=components, timeout_ms=timeout_ms,
                                name=name)
  return result


__queue_enqueue_v2_outputs = [""]


def _queue_enqueue_v2(handle, components, timeout_ms=None, name=None):
  r"""Enqueues a tuple of one or more tensors in the given queue.

  The components input has k elements, which correspond to the components of
  tuples stored in the given queue.

  N.B. If the queue is full, this operation will block until the given
  element has been enqueued (or 'timeout_ms' elapses, if specified).

  Args:
    handle: A `Tensor` of type `resource`. The handle to a queue.
    components: A list of `Tensor` objects.
      One or more tensors from which the enqueued tensors should be taken.
    timeout_ms: An optional `int`. Defaults to `-1`.
      If the queue is full, this operation will block for up to
      timeout_ms milliseconds.
      Note: This option is not supported yet.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  result = _op_def_lib.apply_op("QueueEnqueueV2", handle=handle,
                                components=components, timeout_ms=timeout_ms,
                                name=name)
  return result


__queue_size_outputs = ["size"]


def _queue_size(handle, name=None):
  r"""Computes the number of elements in the given queue.

  Args:
    handle: A `Tensor` of type mutable `string`. The handle to a queue.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int32`. The number of elements in the given queue.
  """
  result = _op_def_lib.apply_op("QueueSize", handle=handle, name=name)
  return result


__queue_size_v2_outputs = ["size"]


def _queue_size_v2(handle, name=None):
  r"""Computes the number of elements in the given queue.

  Args:
    handle: A `Tensor` of type `resource`. The handle to a queue.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int32`. The number of elements in the given queue.
  """
  result = _op_def_lib.apply_op("QueueSizeV2", handle=handle, name=name)
  return result


__random_shuffle_queue_outputs = ["handle"]


def _random_shuffle_queue(component_types, shapes=None, capacity=None,
                          min_after_dequeue=None, seed=None, seed2=None,
                          container=None, shared_name=None, name=None):
  r"""A queue that randomizes the order of elements.

  Args:
    component_types: A list of `tf.DTypes` that has length `>= 1`.
      The type of each component in a value.
    shapes: An optional list of shapes (each a `tf.TensorShape` or list of `ints`). Defaults to `[]`.
      The shape of each component in a value. The length of this attr must
      be either 0 or the same as the length of component_types. If the length of
      this attr is 0, the shapes of queue elements are not constrained, and
      only one element may be dequeued at a time.
    capacity: An optional `int`. Defaults to `-1`.
      The upper bound on the number of elements in this queue.
      Negative numbers mean no limit.
    min_after_dequeue: An optional `int`. Defaults to `0`.
      Dequeue will block unless there would be this
      many elements after the dequeue or the queue is closed. This
      ensures a minimum level of mixing of elements.
    seed: An optional `int`. Defaults to `0`.
      If either seed or seed2 is set to be non-zero, the random number
      generator is seeded by the given seed.  Otherwise, a random seed is used.
    seed2: An optional `int`. Defaults to `0`.
      A second seed to avoid seed collision.
    container: An optional `string`. Defaults to `""`.
      If non-empty, this queue is placed in the given container.
      Otherwise, a default container is used.
    shared_name: An optional `string`. Defaults to `""`.
      If non-empty, this queue will be shared under the given name
      across multiple sessions.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type mutable `string`. The handle to the queue.
  """
  result = _op_def_lib.apply_op("RandomShuffleQueue",
                                component_types=component_types,
                                shapes=shapes, capacity=capacity,
                                min_after_dequeue=min_after_dequeue,
                                seed=seed, seed2=seed2, container=container,
                                shared_name=shared_name, name=name)
  return result


__random_shuffle_queue_v2_outputs = ["handle"]


def _random_shuffle_queue_v2(component_types, shapes=None, capacity=None,
                             min_after_dequeue=None, seed=None, seed2=None,
                             container=None, shared_name=None, name=None):
  r"""A queue that randomizes the order of elements.

  Args:
    component_types: A list of `tf.DTypes` that has length `>= 1`.
      The type of each component in a value.
    shapes: An optional list of shapes (each a `tf.TensorShape` or list of `ints`). Defaults to `[]`.
      The shape of each component in a value. The length of this attr must
      be either 0 or the same as the length of component_types. If the length of
      this attr is 0, the shapes of queue elements are not constrained, and
      only one element may be dequeued at a time.
    capacity: An optional `int`. Defaults to `-1`.
      The upper bound on the number of elements in this queue.
      Negative numbers mean no limit.
    min_after_dequeue: An optional `int`. Defaults to `0`.
      Dequeue will block unless there would be this
      many elements after the dequeue or the queue is closed. This
      ensures a minimum level of mixing of elements.
    seed: An optional `int`. Defaults to `0`.
      If either seed or seed2 is set to be non-zero, the random number
      generator is seeded by the given seed.  Otherwise, a random seed is used.
    seed2: An optional `int`. Defaults to `0`.
      A second seed to avoid seed collision.
    container: An optional `string`. Defaults to `""`.
      If non-empty, this queue is placed in the given container.
      Otherwise, a default container is used.
    shared_name: An optional `string`. Defaults to `""`.
      If non-empty, this queue will be shared under the given name
      across multiple sessions.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `resource`. The handle to the queue.
  """
  result = _op_def_lib.apply_op("RandomShuffleQueueV2",
                                component_types=component_types,
                                shapes=shapes, capacity=capacity,
                                min_after_dequeue=min_after_dequeue,
                                seed=seed, seed2=seed2, container=container,
                                shared_name=shared_name, name=name)
  return result


_record_input_outputs = ["records"]


def record_input(file_pattern, file_random_seed=None,
                 file_shuffle_shift_ratio=None, file_buffer_size=None,
                 file_parallelism=None, batch_size=None, name=None):
  r"""Emits randomized records.

  Args:
    file_pattern: A `string`. Glob pattern for the data files.
    file_random_seed: An optional `int`. Defaults to `301`.
      Random seeds used to produce randomized records.
    file_shuffle_shift_ratio: An optional `float`. Defaults to `0`.
      Shifts the list of files after the list is randomly
      shuffled.
    file_buffer_size: An optional `int`. Defaults to `10000`.
      The randomization shuffling buffer.
    file_parallelism: An optional `int`. Defaults to `16`.
      How many sstables are opened and concurrently iterated over.
    batch_size: An optional `int`. Defaults to `32`. The batch size.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`. A tensor of shape [batch_size].
  """
  result = _op_def_lib.apply_op("RecordInput", file_pattern=file_pattern,
                                file_random_seed=file_random_seed,
                                file_shuffle_shift_ratio=file_shuffle_shift_ratio,
                                file_buffer_size=file_buffer_size,
                                file_parallelism=file_parallelism,
                                batch_size=batch_size, name=name)
  return result


_sparse_accumulator_apply_gradient_outputs = [""]


def sparse_accumulator_apply_gradient(handle, local_step, gradient_indices,
                                      gradient_values, gradient_shape,
                                      has_known_shape, name=None):
  r"""Applies a sparse gradient to a given accumulator. Does not add if local_step is

  lesser than the accumulator's global_step.

  Args:
    handle: A `Tensor` of type mutable `string`. The handle to a accumulator.
    local_step: A `Tensor` of type `int64`.
      The local_step value at which the sparse gradient was computed.
    gradient_indices: A `Tensor` of type `int64`.
      Indices of the sparse gradient to be accumulated. Must be a
      vector.
    gradient_values: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`, `complex128`, `qint8`, `quint8`, `qint32`, `half`.
      Values are the non-zero slices of the gradient, and must have
      the same first dimension as indices, i.e., the nnz represented by indices and
      values must be consistent.
    gradient_shape: A `Tensor` of type `int64`.
      Shape of the sparse gradient to be accumulated.
    has_known_shape: A `bool`.
      Boolean indicating whether gradient_shape is unknown, in which
      case the input is ignored during validation.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  result = _op_def_lib.apply_op("SparseAccumulatorApplyGradient",
                                handle=handle, local_step=local_step,
                                gradient_indices=gradient_indices,
                                gradient_values=gradient_values,
                                gradient_shape=gradient_shape,
                                has_known_shape=has_known_shape, name=name)
  return result


_sparse_accumulator_take_gradient_outputs = ["indices", "values", "shape"]


_SparseAccumulatorTakeGradientOutput = _collections.namedtuple("SparseAccumulatorTakeGradient",
                                                               _sparse_accumulator_take_gradient_outputs)


def sparse_accumulator_take_gradient(handle, num_required, dtype, name=None):
  r"""Extracts the average sparse gradient in the given SparseConditionalAccumulator,

  provided that sufficient (i.e., more than num_required) gradients have been
  accumulated. The op will blocks until sufficient gradients have been
  accumulated. If the accumulator has already aggregated more than num_required
  gradients, it will return its average of the accumulated gradients.
  Also automatically increments the recorded global_step in the accumulator by 1,
  and resets the aggregate to 0.

  Args:
    handle: A `Tensor` of type mutable `string`.
      The handle to a SparseConditionalAccumulator.
    num_required: A `Tensor` of type `int32`.
      Number of gradients required before we return an aggregate.
    dtype: A `tf.DType` from: `tf.float32, tf.float64, tf.int64, tf.int32, tf.uint8, tf.uint16, tf.int16, tf.int8, tf.complex64, tf.complex128, tf.qint8, tf.quint8, tf.qint32, tf.half`.
      The data type of accumulated gradients. Needs to correspond to the type
      of the accumulator.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (indices, values, shape).
    indices: A `Tensor` of type `int64`. Indices of the average of the accumulated sparse gradients.
    values: A `Tensor` of type `dtype`. Values of the average of the accumulated sparse gradients.
    shape: A `Tensor` of type `int64`. Shape of the average of the accumulated sparse gradients.
  """
  result = _op_def_lib.apply_op("SparseAccumulatorTakeGradient",
                                handle=handle, num_required=num_required,
                                dtype=dtype, name=name)
  return _SparseAccumulatorTakeGradientOutput._make(result)


_sparse_conditional_accumulator_outputs = ["handle"]


def sparse_conditional_accumulator(dtype, shape, container=None,
                                   shared_name=None, name=None):
  r"""A conditional accumulator for aggregating sparse gradients. The accumulator

  accepts gradients marked with local_step greater or equal to the most recent
  global_step known to the accumulator. The average can be extracted from the
  accumulator, provided sufficient gradients have been accumulated. Extracting the
  average automatically resets the aggregate to 0, and increments the global_step
  recorded by the accumulator.

  Args:
    dtype: A `tf.DType` from: `tf.float32, tf.float64, tf.int64, tf.int32, tf.uint8, tf.uint16, tf.int16, tf.int8, tf.complex64, tf.complex128, tf.qint8, tf.quint8, tf.qint32, tf.half`.
      The type of the value being accumulated.
    shape: A `tf.TensorShape` or list of `ints`. The shape of the values.
    container: An optional `string`. Defaults to `""`.
      If non-empty, this accumulator is placed in the given container.
      Otherwise, a default container is used.
    shared_name: An optional `string`. Defaults to `""`.
      If non-empty, this accumulator will be shared under the given name
      across multiple sessions.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type mutable `string`. The handle to the accumulator.
  """
  result = _op_def_lib.apply_op("SparseConditionalAccumulator", dtype=dtype,
                                shape=shape, container=container,
                                shared_name=shared_name, name=name)
  return result


__stack_outputs = ["handle"]


def _stack(elem_type, stack_name=None, name=None):
  r"""A stack that produces elements in first-in last-out order.

  Args:
    elem_type: A `tf.DType`. The type of the elements on the stack.
    stack_name: An optional `string`. Defaults to `""`.
      Overrides the name used for the temporary stack resource. Default
      value is the name of the 'Stack' op (which is guaranteed unique).
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type mutable `string`. The handle to the stack.
  """
  result = _op_def_lib.apply_op("Stack", elem_type=elem_type,
                                stack_name=stack_name, name=name)
  return result


__stack_close_outputs = [""]


def _stack_close(handle, name=None):
  r"""Delete the stack from its resource container.

  Args:
    handle: A `Tensor` of type mutable `string`. The handle to a stack.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  result = _op_def_lib.apply_op("StackClose", handle=handle, name=name)
  return result


__stack_pop_outputs = ["elem"]


def _stack_pop(handle, elem_type, name=None):
  r"""Pop the element at the top of the stack.

  Args:
    handle: A `Tensor` of type mutable `string`. The handle to a stack.
    elem_type: A `tf.DType`. The type of the elem that is popped.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `elem_type`.
    The tensor that is popped from the top of the stack.
  """
  result = _op_def_lib.apply_op("StackPop", handle=handle,
                                elem_type=elem_type, name=name)
  return result


__stack_push_outputs = ["output"]


def _stack_push(handle, elem, swap_memory=None, name=None):
  r"""Push an element onto the stack.

  Args:
    handle: A `Tensor` of type mutable `string`. The handle to a stack.
    elem: A `Tensor`. The tensor to be pushed onto the stack.
    swap_memory: An optional `bool`. Defaults to `False`.
      Swap `elem` to CPU. Default to false.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `elem`.
    The same tensor as the input 'elem'.
  """
  result = _op_def_lib.apply_op("StackPush", handle=handle, elem=elem,
                                swap_memory=swap_memory, name=name)
  return result


_stage_outputs = [""]


def stage(values, container=None, shared_name=None, name=None):
  r"""Stage values similar to a lightweight Enqueue.  The basic functionality of this

  Op is similar to a queue with many fewer capabilities and options.  This Op is
  optimized for performance.

  Args:
    values: A list of `Tensor` objects. a list of tensors
    container: An optional `string`. Defaults to `""`.
      If non-empty, this queue is placed in the given container. Otherwise,
      a default container is used.
    shared_name: An optional `string`. Defaults to `""`.
      It is necessary to match this name to the matching Unstage Op.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  result = _op_def_lib.apply_op("Stage", values=values, container=container,
                                shared_name=shared_name, name=name)
  return result


__tensor_array_outputs = ["handle"]


def _tensor_array(size, dtype, dynamic_size=None, clear_after_read=None,
                  tensor_array_name=None, element_shape=None, name=None):
  r"""TODO: add doc.

  Args:
    size: A `Tensor` of type `int32`.
    dtype: A `tf.DType`.
    dynamic_size: An optional `bool`. Defaults to `False`.
    clear_after_read: An optional `bool`. Defaults to `True`.
    tensor_array_name: An optional `string`. Defaults to `""`.
    element_shape: An optional `tf.TensorShape` or list of `ints`. Defaults to `[]`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type mutable `string`.
  """
  result = _op_def_lib.apply_op("TensorArray", size=size, dtype=dtype,
                                dynamic_size=dynamic_size,
                                clear_after_read=clear_after_read,
                                tensor_array_name=tensor_array_name,
                                element_shape=element_shape, name=name)
  return result


__tensor_array_close_outputs = [""]


def _tensor_array_close(handle, name=None):
  r"""TODO: add doc.

  Args:
    handle: A `Tensor` of type mutable `string`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  result = _op_def_lib.apply_op("TensorArrayClose", handle=handle, name=name)
  return result


__tensor_array_close_v2_outputs = [""]


def _tensor_array_close_v2(handle, name=None):
  r"""Deprecated. Use TensorArrayCloseV3

  Args:
    handle: A `Tensor` of type `string`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  result = _op_def_lib.apply_op("TensorArrayCloseV2", handle=handle,
                                name=name)
  return result


__tensor_array_close_v3_outputs = [""]


def _tensor_array_close_v3(handle, name=None):
  r"""Delete the TensorArray from its resource container.  This enables

  the user to close and release the resource in the middle of a step/run.

  Args:
    handle: A `Tensor` of type `resource`.
      The handle to a TensorArray (output of TensorArray or TensorArrayGrad).
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  result = _op_def_lib.apply_op("TensorArrayCloseV3", handle=handle,
                                name=name)
  return result


__tensor_array_concat_outputs = ["value", "lengths"]


_TensorArrayConcatOutput = _collections.namedtuple("TensorArrayConcat",
                                                   __tensor_array_concat_outputs)


def _tensor_array_concat(handle, flow_in, dtype, element_shape_except0=None,
                         name=None):
  r"""TODO: add doc.

  Args:
    handle: A `Tensor` of type mutable `string`.
    flow_in: A `Tensor` of type `float32`.
    dtype: A `tf.DType`.
    element_shape_except0: An optional `tf.TensorShape` or list of `ints`. Defaults to `[]`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (value, lengths).
    value: A `Tensor` of type `dtype`.
    lengths: A `Tensor` of type `int64`.
  """
  result = _op_def_lib.apply_op("TensorArrayConcat", handle=handle,
                                flow_in=flow_in, dtype=dtype,
                                element_shape_except0=element_shape_except0,
                                name=name)
  return _TensorArrayConcatOutput._make(result)


__tensor_array_concat_v2_outputs = ["value", "lengths"]


_TensorArrayConcatV2Output = _collections.namedtuple("TensorArrayConcatV2",
                                                     __tensor_array_concat_v2_outputs)


def _tensor_array_concat_v2(handle, flow_in, dtype,
                            element_shape_except0=None, name=None):
  r"""Deprecated. Use TensorArrayConcatV3

  Args:
    handle: A `Tensor` of type `string`.
    flow_in: A `Tensor` of type `float32`.
    dtype: A `tf.DType`.
    element_shape_except0: An optional `tf.TensorShape` or list of `ints`. Defaults to `[]`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (value, lengths).
    value: A `Tensor` of type `dtype`.
    lengths: A `Tensor` of type `int64`.
  """
  result = _op_def_lib.apply_op("TensorArrayConcatV2", handle=handle,
                                flow_in=flow_in, dtype=dtype,
                                element_shape_except0=element_shape_except0,
                                name=name)
  return _TensorArrayConcatV2Output._make(result)


__tensor_array_concat_v3_outputs = ["value", "lengths"]


_TensorArrayConcatV3Output = _collections.namedtuple("TensorArrayConcatV3",
                                                     __tensor_array_concat_v3_outputs)


def _tensor_array_concat_v3(handle, flow_in, dtype,
                            element_shape_except0=None, name=None):
  r"""Concat the elements from the TensorArray into value `value`.

  Takes `T` elements of shapes

    ```
    (n0 x d0 x d1 x ...), (n1 x d0 x d1 x ...), ..., (n(T-1) x d0 x d1 x ...)
    ```

  and concatenates them into a Tensor of shape:

    ```(n0 + n1 + ... + n(T-1) x d0 x d1 x ...)```

  All elements must have the same shape (excepting the first dimension).

  Args:
    handle: A `Tensor` of type `resource`. The handle to a TensorArray.
    flow_in: A `Tensor` of type `float32`.
      A float scalar that enforces proper chaining of operations.
    dtype: A `tf.DType`. The type of the elem that is returned.
    element_shape_except0: An optional `tf.TensorShape` or list of `ints`. Defaults to `[]`.
      The expected shape of an element, if known,
      excluding the first dimension. Used to validate the shapes of
      TensorArray elements. If this shape is not fully specified, concatenating
      zero-size TensorArrays is an error.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (value, lengths).
    value: A `Tensor` of type `dtype`. All of the elements in the TensorArray, concatenated along the first
      axis.
    lengths: A `Tensor` of type `int64`. A vector of the row sizes of the original T elements in the
      value output.  In the example above, this would be the values:
      `(n1, n2, ..., n(T-1))`.
  """
  result = _op_def_lib.apply_op("TensorArrayConcatV3", handle=handle,
                                flow_in=flow_in, dtype=dtype,
                                element_shape_except0=element_shape_except0,
                                name=name)
  return _TensorArrayConcatV3Output._make(result)


__tensor_array_gather_outputs = ["value"]


def _tensor_array_gather(handle, indices, flow_in, dtype, element_shape=None,
                         name=None):
  r"""TODO: add doc.

  Args:
    handle: A `Tensor` of type mutable `string`.
    indices: A `Tensor` of type `int32`.
    flow_in: A `Tensor` of type `float32`.
    dtype: A `tf.DType`.
    element_shape: An optional `tf.TensorShape` or list of `ints`. Defaults to `[]`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `dtype`.
  """
  result = _op_def_lib.apply_op("TensorArrayGather", handle=handle,
                                indices=indices, flow_in=flow_in, dtype=dtype,
                                element_shape=element_shape, name=name)
  return result


__tensor_array_gather_v2_outputs = ["value"]


def _tensor_array_gather_v2(handle, indices, flow_in, dtype,
                            element_shape=None, name=None):
  r"""Deprecated. Use TensorArrayGatherV3

  Args:
    handle: A `Tensor` of type `string`.
    indices: A `Tensor` of type `int32`.
    flow_in: A `Tensor` of type `float32`.
    dtype: A `tf.DType`.
    element_shape: An optional `tf.TensorShape` or list of `ints`. Defaults to `[]`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `dtype`.
  """
  result = _op_def_lib.apply_op("TensorArrayGatherV2", handle=handle,
                                indices=indices, flow_in=flow_in, dtype=dtype,
                                element_shape=element_shape, name=name)
  return result


__tensor_array_gather_v3_outputs = ["value"]


def _tensor_array_gather_v3(handle, indices, flow_in, dtype,
                            element_shape=None, name=None):
  r"""Gather specific elements from the TensorArray into output `value`.

  All elements selected by `indices` must have the same shape.

  Args:
    handle: A `Tensor` of type `resource`. The handle to a TensorArray.
    indices: A `Tensor` of type `int32`.
      The locations in the TensorArray from which to read tensor elements.
    flow_in: A `Tensor` of type `float32`.
      A float scalar that enforces proper chaining of operations.
    dtype: A `tf.DType`. The type of the elem that is returned.
    element_shape: An optional `tf.TensorShape` or list of `ints`. Defaults to `[]`.
      The expected shape of an element, if known. Used to
      validate the shapes of TensorArray elements. If this shape is not
      fully specified, gathering zero-size TensorArrays is an error.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `dtype`.
    All of the elements in the TensorArray, concatenated along a new
    axis (the new dimension 0).
  """
  result = _op_def_lib.apply_op("TensorArrayGatherV3", handle=handle,
                                indices=indices, flow_in=flow_in, dtype=dtype,
                                element_shape=element_shape, name=name)
  return result


__tensor_array_grad_outputs = ["grad_handle"]


def _tensor_array_grad(handle, flow_in, source, name=None):
  r"""TODO: add doc.

  Args:
    handle: A `Tensor` of type `string`.
    flow_in: A `Tensor` of type `float32`.
    source: A `string`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type mutable `string`.
  """
  result = _op_def_lib.apply_op("TensorArrayGrad", handle=handle,
                                flow_in=flow_in, source=source, name=name)
  return result


__tensor_array_grad_v2_outputs = ["grad_handle"]


def _tensor_array_grad_v2(handle, flow_in, source, name=None):
  r"""Deprecated. Use TensorArrayGradV3

  Args:
    handle: A `Tensor` of type `string`.
    flow_in: A `Tensor` of type `float32`.
    source: A `string`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`.
  """
  result = _op_def_lib.apply_op("TensorArrayGradV2", handle=handle,
                                flow_in=flow_in, source=source, name=name)
  return result


__tensor_array_grad_v3_outputs = ["grad_handle", "flow_out"]


_TensorArrayGradV3Output = _collections.namedtuple("TensorArrayGradV3",
                                                   __tensor_array_grad_v3_outputs)


def _tensor_array_grad_v3(handle, flow_in, source, name=None):
  r"""Creates a TensorArray for storing the gradients of values in the given handle.

  If the given TensorArray gradient already exists, returns a reference to it.

  Locks the size of the original TensorArray by disabling its dynamic size flag.

  **A note about the input flow_in:**

  The handle flow_in forces the execution of the gradient lookup to occur
  only after certain other operations have occurred.  For example, when
  the forward TensorArray is dynamically sized, writes to this TensorArray
  may resize the object.  The gradient TensorArray is statically sized based
  on the size of the forward TensorArray when this operation executes.
  Furthermore, the size of the forward TensorArray is frozen by this call.
  As a result, the flow is used to ensure that the call to generate the gradient
  TensorArray only happens after all writes are executed.

  In the case of dynamically sized TensorArrays, gradient computation should
  only be performed on read operations that have themselves been chained via
  flow to occur only after all writes have executed. That way the final size
  of the forward TensorArray is known when this operation is called.

  **A note about the source attribute:**

  TensorArray gradient calls use an accumulator TensorArray object.  If
  multiple gradients are calculated and run in the same session, the multiple
  gradient nodes may accidentally flow throuth the same accumulator TensorArray.
  This double counts and generally breaks the TensorArray gradient flow.

  The solution is to identify which gradient call this particular
  TensorArray gradient is being called in.  This is performed by identifying
  a unique string (e.g. "gradients", "gradients_1", ...) from the input
  gradient Tensor's name.  This string is used as a suffix when creating
  the TensorArray gradient object here (the attribute `source`).

  The attribute `source` is added as a suffix to the forward TensorArray's
  name when performing the creation / lookup, so that each separate gradient
  calculation gets its own TensorArray accumulator.

  Args:
    handle: A `Tensor` of type `resource`.
      The handle to the forward TensorArray.
    flow_in: A `Tensor` of type `float32`.
      A float scalar that enforces proper chaining of operations.
    source: A `string`.
      The gradient source string, used to decide which gradient TensorArray
      to return.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (grad_handle, flow_out).
    grad_handle: A `Tensor` of type `resource`.
    flow_out: A `Tensor` of type `float32`.
  """
  result = _op_def_lib.apply_op("TensorArrayGradV3", handle=handle,
                                flow_in=flow_in, source=source, name=name)
  return _TensorArrayGradV3Output._make(result)


__tensor_array_pack_outputs = ["value"]


def _tensor_array_pack(handle, flow_in, dtype, element_shape=None, name=None):
  r"""TODO: add doc.

  Args:
    handle: A `Tensor` of type mutable `string`.
    flow_in: A `Tensor` of type `float32`.
    dtype: A `tf.DType`.
    element_shape: An optional `tf.TensorShape` or list of `ints`. Defaults to `[]`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `dtype`.
  """
  result = _op_def_lib.apply_op("TensorArrayPack", handle=handle,
                                flow_in=flow_in, dtype=dtype,
                                element_shape=element_shape, name=name)
  return result


__tensor_array_read_outputs = ["value"]


def _tensor_array_read(handle, index, flow_in, dtype, name=None):
  r"""TODO: add doc.

  Args:
    handle: A `Tensor` of type mutable `string`.
    index: A `Tensor` of type `int32`.
    flow_in: A `Tensor` of type `float32`.
    dtype: A `tf.DType`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `dtype`.
  """
  result = _op_def_lib.apply_op("TensorArrayRead", handle=handle, index=index,
                                flow_in=flow_in, dtype=dtype, name=name)
  return result


__tensor_array_read_v2_outputs = ["value"]


def _tensor_array_read_v2(handle, index, flow_in, dtype, name=None):
  r"""Deprecated. Use TensorArrayReadV3

  Args:
    handle: A `Tensor` of type `string`.
    index: A `Tensor` of type `int32`.
    flow_in: A `Tensor` of type `float32`.
    dtype: A `tf.DType`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `dtype`.
  """
  result = _op_def_lib.apply_op("TensorArrayReadV2", handle=handle,
                                index=index, flow_in=flow_in, dtype=dtype,
                                name=name)
  return result


__tensor_array_read_v3_outputs = ["value"]


def _tensor_array_read_v3(handle, index, flow_in, dtype, name=None):
  r"""Read an element from the TensorArray into output `value`.

  Args:
    handle: A `Tensor` of type `resource`. The handle to a TensorArray.
    index: A `Tensor` of type `int32`.
    flow_in: A `Tensor` of type `float32`.
      A float scalar that enforces proper chaining of operations.
    dtype: A `tf.DType`. The type of the elem that is returned.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `dtype`. The tensor that is read from the TensorArray.
  """
  result = _op_def_lib.apply_op("TensorArrayReadV3", handle=handle,
                                index=index, flow_in=flow_in, dtype=dtype,
                                name=name)
  return result


__tensor_array_scatter_outputs = ["flow_out"]


def _tensor_array_scatter(handle, indices, value, flow_in, name=None):
  r"""TODO: add doc.

  Args:
    handle: A `Tensor` of type mutable `string`.
    indices: A `Tensor` of type `int32`.
    value: A `Tensor`.
    flow_in: A `Tensor` of type `float32`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32`.
  """
  result = _op_def_lib.apply_op("TensorArrayScatter", handle=handle,
                                indices=indices, value=value, flow_in=flow_in,
                                name=name)
  return result


__tensor_array_scatter_v2_outputs = ["flow_out"]


def _tensor_array_scatter_v2(handle, indices, value, flow_in, name=None):
  r"""Deprecated. Use TensorArrayScatterV3

  Args:
    handle: A `Tensor` of type `string`.
    indices: A `Tensor` of type `int32`.
    value: A `Tensor`.
    flow_in: A `Tensor` of type `float32`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32`.
  """
  result = _op_def_lib.apply_op("TensorArrayScatterV2", handle=handle,
                                indices=indices, value=value, flow_in=flow_in,
                                name=name)
  return result


__tensor_array_scatter_v3_outputs = ["flow_out"]


def _tensor_array_scatter_v3(handle, indices, value, flow_in, name=None):
  r"""Scatter the data from the input value into specific TensorArray elements.

  `indices` must be a vector, its length must match the first dim of `value`.

  Args:
    handle: A `Tensor` of type `resource`. The handle to a TensorArray.
    indices: A `Tensor` of type `int32`.
      The locations at which to write the tensor elements.
    value: A `Tensor`. The concatenated tensor to write to the TensorArray.
    flow_in: A `Tensor` of type `float32`.
      A float scalar that enforces proper chaining of operations.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32`.
    A float scalar that enforces proper chaining of operations.
  """
  result = _op_def_lib.apply_op("TensorArrayScatterV3", handle=handle,
                                indices=indices, value=value, flow_in=flow_in,
                                name=name)
  return result


__tensor_array_size_outputs = ["size"]


def _tensor_array_size(handle, flow_in, name=None):
  r"""TODO: add doc.

  Args:
    handle: A `Tensor` of type mutable `string`.
    flow_in: A `Tensor` of type `float32`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int32`.
  """
  result = _op_def_lib.apply_op("TensorArraySize", handle=handle,
                                flow_in=flow_in, name=name)
  return result


__tensor_array_size_v2_outputs = ["size"]


def _tensor_array_size_v2(handle, flow_in, name=None):
  r"""Deprecated. Use TensorArraySizeV3

  Args:
    handle: A `Tensor` of type `string`.
    flow_in: A `Tensor` of type `float32`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int32`.
  """
  result = _op_def_lib.apply_op("TensorArraySizeV2", handle=handle,
                                flow_in=flow_in, name=name)
  return result


__tensor_array_size_v3_outputs = ["size"]


def _tensor_array_size_v3(handle, flow_in, name=None):
  r"""Get the current size of the TensorArray.

  Args:
    handle: A `Tensor` of type `resource`.
      The handle to a TensorArray (output of TensorArray or TensorArrayGrad).
    flow_in: A `Tensor` of type `float32`.
      A float scalar that enforces proper chaining of operations.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int32`. The current size of the TensorArray.
  """
  result = _op_def_lib.apply_op("TensorArraySizeV3", handle=handle,
                                flow_in=flow_in, name=name)
  return result


__tensor_array_split_outputs = ["flow_out"]


def _tensor_array_split(handle, value, lengths, flow_in, name=None):
  r"""TODO: add doc.

  Args:
    handle: A `Tensor` of type mutable `string`.
    value: A `Tensor`.
    lengths: A `Tensor` of type `int64`.
    flow_in: A `Tensor` of type `float32`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32`.
  """
  result = _op_def_lib.apply_op("TensorArraySplit", handle=handle,
                                value=value, lengths=lengths, flow_in=flow_in,
                                name=name)
  return result


__tensor_array_split_v2_outputs = ["flow_out"]


def _tensor_array_split_v2(handle, value, lengths, flow_in, name=None):
  r"""Deprecated. Use TensorArraySplitV3

  Args:
    handle: A `Tensor` of type `string`.
    value: A `Tensor`.
    lengths: A `Tensor` of type `int64`.
    flow_in: A `Tensor` of type `float32`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32`.
  """
  result = _op_def_lib.apply_op("TensorArraySplitV2", handle=handle,
                                value=value, lengths=lengths, flow_in=flow_in,
                                name=name)
  return result


__tensor_array_split_v3_outputs = ["flow_out"]


def _tensor_array_split_v3(handle, value, lengths, flow_in, name=None):
  r"""Split the data from the input value into TensorArray elements.

  Assuming that `lengths` takes on values

    ```(n0, n1, ..., n(T-1))```

  and that `value` has shape

    ```(n0 + n1 + ... + n(T-1) x d0 x d1 x ...)```,

  this splits values into a TensorArray with T tensors.

  TensorArray index t will be the subtensor of values with starting position

    ```(n0 + n1 + ... + n(t-1), 0, 0, ...)```

  and having size

    ```nt x d0 x d1 x ...```

  Args:
    handle: A `Tensor` of type `resource`. The handle to a TensorArray.
    value: A `Tensor`. The concatenated tensor to write to the TensorArray.
    lengths: A `Tensor` of type `int64`.
      The vector of lengths, how to split the rows of value into the
      TensorArray.
    flow_in: A `Tensor` of type `float32`.
      A float scalar that enforces proper chaining of operations.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32`.
    A float scalar that enforces proper chaining of operations.
  """
  result = _op_def_lib.apply_op("TensorArraySplitV3", handle=handle,
                                value=value, lengths=lengths, flow_in=flow_in,
                                name=name)
  return result


__tensor_array_unpack_outputs = ["flow_out"]


def _tensor_array_unpack(handle, value, flow_in, name=None):
  r"""TODO: add doc.

  Args:
    handle: A `Tensor` of type mutable `string`.
    value: A `Tensor`.
    flow_in: A `Tensor` of type `float32`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32`.
  """
  result = _op_def_lib.apply_op("TensorArrayUnpack", handle=handle,
                                value=value, flow_in=flow_in, name=name)
  return result


__tensor_array_v2_outputs = ["handle"]


def _tensor_array_v2(size, dtype, element_shape=None, dynamic_size=None,
                     clear_after_read=None, tensor_array_name=None,
                     name=None):
  r"""Deprecated. Use TensorArrayV3

  Args:
    size: A `Tensor` of type `int32`.
    dtype: A `tf.DType`.
    element_shape: An optional `tf.TensorShape` or list of `ints`. Defaults to `[]`.
    dynamic_size: An optional `bool`. Defaults to `False`.
    clear_after_read: An optional `bool`. Defaults to `True`.
    tensor_array_name: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`.
  """
  result = _op_def_lib.apply_op("TensorArrayV2", size=size, dtype=dtype,
                                element_shape=element_shape,
                                dynamic_size=dynamic_size,
                                clear_after_read=clear_after_read,
                                tensor_array_name=tensor_array_name,
                                name=name)
  return result


__tensor_array_v3_outputs = ["handle", "flow"]


_TensorArrayV3Output = _collections.namedtuple("TensorArrayV3",
                                               __tensor_array_v3_outputs)


def _tensor_array_v3(size, dtype, element_shape=None, dynamic_size=None,
                     clear_after_read=None, tensor_array_name=None,
                     name=None):
  r"""An array of Tensors of given size, with data written via Write and read

  via Read or Pack.

  Args:
    size: A `Tensor` of type `int32`. The size of the array.
    dtype: A `tf.DType`. The type of the elements on the tensor_array.
    element_shape: An optional `tf.TensorShape` or list of `ints`. Defaults to `[]`.
      The expected shape of an element, if known. Used to
      validate the shapes of TensorArray elements. If this shape is not
      fully specified, gathering zero-size TensorArrays is an error.
    dynamic_size: An optional `bool`. Defaults to `False`.
      A boolean that determines whether writes to the TensorArray
      are allowed to grow the size.  By default, this is not allowed.
    clear_after_read: An optional `bool`. Defaults to `True`.
      If true (default), Tensors in the TensorArray are cleared
      after being read.  This disables multiple read semantics but allows early
      release of memory.
    tensor_array_name: An optional `string`. Defaults to `""`.
      Overrides the name used for the temporary tensor_array
      resource. Default value is the name of the 'TensorArray' op (which
      is guaranteed unique).
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (handle, flow).
    handle: A `Tensor` of type `resource`. The handle to the TensorArray.
    flow: A `Tensor` of type `float32`. A scalar used to control gradient flow.
  """
  result = _op_def_lib.apply_op("TensorArrayV3", size=size, dtype=dtype,
                                element_shape=element_shape,
                                dynamic_size=dynamic_size,
                                clear_after_read=clear_after_read,
                                tensor_array_name=tensor_array_name,
                                name=name)
  return _TensorArrayV3Output._make(result)


__tensor_array_write_outputs = ["flow_out"]


def _tensor_array_write(handle, index, value, flow_in, name=None):
  r"""TODO: add doc.

  Args:
    handle: A `Tensor` of type mutable `string`.
    index: A `Tensor` of type `int32`.
    value: A `Tensor`.
    flow_in: A `Tensor` of type `float32`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32`.
  """
  result = _op_def_lib.apply_op("TensorArrayWrite", handle=handle,
                                index=index, value=value, flow_in=flow_in,
                                name=name)
  return result


__tensor_array_write_v2_outputs = ["flow_out"]


def _tensor_array_write_v2(handle, index, value, flow_in, name=None):
  r"""Deprecated. Use TensorArrayGradV3

  Args:
    handle: A `Tensor` of type `string`.
    index: A `Tensor` of type `int32`.
    value: A `Tensor`.
    flow_in: A `Tensor` of type `float32`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32`.
  """
  result = _op_def_lib.apply_op("TensorArrayWriteV2", handle=handle,
                                index=index, value=value, flow_in=flow_in,
                                name=name)
  return result


__tensor_array_write_v3_outputs = ["flow_out"]


def _tensor_array_write_v3(handle, index, value, flow_in, name=None):
  r"""Push an element onto the tensor_array.

  Args:
    handle: A `Tensor` of type `resource`. The handle to a TensorArray.
    index: A `Tensor` of type `int32`.
      The position to write to inside the TensorArray.
    value: A `Tensor`. The tensor to write to the TensorArray.
    flow_in: A `Tensor` of type `float32`.
      A float scalar that enforces proper chaining of operations.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32`.
    A float scalar that enforces proper chaining of operations.
  """
  result = _op_def_lib.apply_op("TensorArrayWriteV3", handle=handle,
                                index=index, value=value, flow_in=flow_in,
                                name=name)
  return result


_unstage_outputs = ["values"]


def unstage(dtypes, container=None, shared_name=None, name=None):
  r"""Op is similar to a lightweight Dequeue.  The basic funtionality is similar to

  dequeue with many fewer capabilities and options.  This Op is optimized for
  performance.

  Args:
    dtypes: A list of `tf.DTypes` that has length `>= 1`.
    container: An optional `string`. Defaults to `""`.
    shared_name: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A list of `Tensor` objects of type `dtypes`.
  """
  result = _op_def_lib.apply_op("Unstage", dtypes=dtypes, container=container,
                                shared_name=shared_name, name=name)
  return result


def _InitOpDefLibrary():
  op_list = _op_def_pb2.OpList()
  _text_format.Merge(_InitOpDefLibrary.op_list_ascii, op_list)
  _op_def_registry.register_op_list(op_list)
  op_def_lib = _op_def_library.OpDefLibrary()
  op_def_lib.add_op_list(op_list)
  return op_def_lib


_InitOpDefLibrary.op_list_ascii = """op {
  name: "AccumulatorApplyGradient"
  input_arg {
    name: "handle"
    type: DT_STRING
    is_ref: true
  }
  input_arg {
    name: "local_step"
    type: DT_INT64
  }
  input_arg {
    name: "gradient"
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
}
op {
  name: "AccumulatorNumAccumulated"
  input_arg {
    name: "handle"
    type: DT_STRING
    is_ref: true
  }
  output_arg {
    name: "num_accumulated"
    type: DT_INT32
  }
}
op {
  name: "AccumulatorSetGlobalStep"
  input_arg {
    name: "handle"
    type: DT_STRING
    is_ref: true
  }
  input_arg {
    name: "new_global_step"
    type: DT_INT64
  }
}
op {
  name: "AccumulatorTakeGradient"
  input_arg {
    name: "handle"
    type: DT_STRING
    is_ref: true
  }
  input_arg {
    name: "num_required"
    type: DT_INT32
  }
  output_arg {
    name: "average"
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
}
op {
  name: "Barrier"
  output_arg {
    name: "handle"
    type: DT_STRING
    is_ref: true
  }
  attr {
    name: "component_types"
    type: "list(type)"
    has_minimum: true
    minimum: 1
  }
  attr {
    name: "shapes"
    type: "list(shape)"
    default_value {
      list {
      }
    }
    has_minimum: true
  }
  attr {
    name: "capacity"
    type: "int"
    default_value {
      i: -1
    }
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
  name: "BarrierClose"
  input_arg {
    name: "handle"
    type: DT_STRING
    is_ref: true
  }
  attr {
    name: "cancel_pending_enqueues"
    type: "bool"
    default_value {
      b: false
    }
  }
}
op {
  name: "BarrierIncompleteSize"
  input_arg {
    name: "handle"
    type: DT_STRING
    is_ref: true
  }
  output_arg {
    name: "size"
    type: DT_INT32
  }
}
op {
  name: "BarrierInsertMany"
  input_arg {
    name: "handle"
    type: DT_STRING
    is_ref: true
  }
  input_arg {
    name: "keys"
    type: DT_STRING
  }
  input_arg {
    name: "values"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
  }
  attr {
    name: "component_index"
    type: "int"
  }
}
op {
  name: "BarrierReadySize"
  input_arg {
    name: "handle"
    type: DT_STRING
    is_ref: true
  }
  output_arg {
    name: "size"
    type: DT_INT32
  }
}
op {
  name: "BarrierTakeMany"
  input_arg {
    name: "handle"
    type: DT_STRING
    is_ref: true
  }
  input_arg {
    name: "num_elements"
    type: DT_INT32
  }
  output_arg {
    name: "indices"
    type: DT_INT64
  }
  output_arg {
    name: "keys"
    type: DT_STRING
  }
  output_arg {
    name: "values"
    type_list_attr: "component_types"
  }
  attr {
    name: "component_types"
    type: "list(type)"
    has_minimum: true
    minimum: 1
  }
  attr {
    name: "allow_small_batch"
    type: "bool"
    default_value {
      b: false
    }
  }
  attr {
    name: "wait_for_incomplete"
    type: "bool"
    default_value {
      b: false
    }
  }
  attr {
    name: "timeout_ms"
    type: "int"
    default_value {
      i: -1
    }
  }
}
op {
  name: "ConditionalAccumulator"
  output_arg {
    name: "handle"
    type: DT_STRING
    is_ref: true
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
    name: "shape"
    type: "shape"
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
  name: "DeleteSessionTensor"
  input_arg {
    name: "handle"
    type: DT_STRING
  }
}
op {
  name: "DynamicPartition"
  input_arg {
    name: "data"
    type_attr: "T"
  }
  input_arg {
    name: "partitions"
    type: DT_INT32
  }
  output_arg {
    name: "outputs"
    type_attr: "T"
    number_attr: "num_partitions"
  }
  attr {
    name: "num_partitions"
    type: "int"
    has_minimum: true
    minimum: 1
  }
  attr {
    name: "T"
    type: "type"
  }
}
op {
  name: "DynamicStitch"
  input_arg {
    name: "indices"
    type: DT_INT32
    number_attr: "N"
  }
  input_arg {
    name: "data"
    type_attr: "T"
    number_attr: "N"
  }
  output_arg {
    name: "merged"
    type_attr: "T"
  }
  attr {
    name: "N"
    type: "int"
    has_minimum: true
    minimum: 1
  }
  attr {
    name: "T"
    type: "type"
  }
}
op {
  name: "FIFOQueue"
  output_arg {
    name: "handle"
    type: DT_STRING
    is_ref: true
  }
  attr {
    name: "component_types"
    type: "list(type)"
    has_minimum: true
    minimum: 1
  }
  attr {
    name: "shapes"
    type: "list(shape)"
    default_value {
      list {
      }
    }
    has_minimum: true
  }
  attr {
    name: "capacity"
    type: "int"
    default_value {
      i: -1
    }
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
  name: "FIFOQueueV2"
  output_arg {
    name: "handle"
    type: DT_RESOURCE
  }
  attr {
    name: "component_types"
    type: "list(type)"
    has_minimum: true
    minimum: 1
  }
  attr {
    name: "shapes"
    type: "list(shape)"
    default_value {
      list {
      }
    }
    has_minimum: true
  }
  attr {
    name: "capacity"
    type: "int"
    default_value {
      i: -1
    }
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
  name: "FakeQueue"
  input_arg {
    name: "resource"
    type: DT_RESOURCE
  }
  output_arg {
    name: "handle"
    type: DT_STRING
    is_ref: true
  }
  is_stateful: true
}
op {
  name: "GetSessionHandle"
  input_arg {
    name: "value"
    type_attr: "T"
  }
  output_arg {
    name: "handle"
    type: DT_STRING
  }
  attr {
    name: "T"
    type: "type"
  }
}
op {
  name: "GetSessionTensor"
  input_arg {
    name: "handle"
    type: DT_STRING
  }
  output_arg {
    name: "value"
    type_attr: "dtype"
  }
  attr {
    name: "dtype"
    type: "type"
  }
}
op {
  name: "HashTable"
  output_arg {
    name: "table_handle"
    type: DT_STRING
    is_ref: true
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
    name: "use_node_name_sharing"
    type: "bool"
    default_value {
      b: false
    }
  }
  attr {
    name: "key_dtype"
    type: "type"
  }
  attr {
    name: "value_dtype"
    type: "type"
  }
  is_stateful: true
}
op {
  name: "InitializeTable"
  input_arg {
    name: "table_handle"
    type: DT_STRING
    is_ref: true
  }
  input_arg {
    name: "keys"
    type_attr: "Tkey"
  }
  input_arg {
    name: "values"
    type_attr: "Tval"
  }
  attr {
    name: "Tkey"
    type: "type"
  }
  attr {
    name: "Tval"
    type: "type"
  }
}
op {
  name: "InitializeTableFromTextFile"
  input_arg {
    name: "table_handle"
    type: DT_STRING
    is_ref: true
  }
  input_arg {
    name: "filename"
    type: DT_STRING
  }
  attr {
    name: "key_index"
    type: "int"
    has_minimum: true
    minimum: -2
  }
  attr {
    name: "value_index"
    type: "int"
    has_minimum: true
    minimum: -2
  }
  attr {
    name: "vocab_size"
    type: "int"
    default_value {
      i: -1
    }
    has_minimum: true
    minimum: -1
  }
  attr {
    name: "delimiter"
    type: "string"
    default_value {
      s: "\t"
    }
  }
}
op {
  name: "LookupTableExport"
  input_arg {
    name: "table_handle"
    type: DT_STRING
    is_ref: true
  }
  output_arg {
    name: "keys"
    type_attr: "Tkeys"
  }
  output_arg {
    name: "values"
    type_attr: "Tvalues"
  }
  attr {
    name: "Tkeys"
    type: "type"
  }
  attr {
    name: "Tvalues"
    type: "type"
  }
}
op {
  name: "LookupTableFind"
  input_arg {
    name: "table_handle"
    type: DT_STRING
    is_ref: true
  }
  input_arg {
    name: "keys"
    type_attr: "Tin"
  }
  input_arg {
    name: "default_value"
    type_attr: "Tout"
  }
  output_arg {
    name: "values"
    type_attr: "Tout"
  }
  attr {
    name: "Tin"
    type: "type"
  }
  attr {
    name: "Tout"
    type: "type"
  }
}
op {
  name: "LookupTableImport"
  input_arg {
    name: "table_handle"
    type: DT_STRING
    is_ref: true
  }
  input_arg {
    name: "keys"
    type_attr: "Tin"
  }
  input_arg {
    name: "values"
    type_attr: "Tout"
  }
  attr {
    name: "Tin"
    type: "type"
  }
  attr {
    name: "Tout"
    type: "type"
  }
}
op {
  name: "LookupTableInsert"
  input_arg {
    name: "table_handle"
    type: DT_STRING
    is_ref: true
  }
  input_arg {
    name: "keys"
    type_attr: "Tin"
  }
  input_arg {
    name: "values"
    type_attr: "Tout"
  }
  attr {
    name: "Tin"
    type: "type"
  }
  attr {
    name: "Tout"
    type: "type"
  }
}
op {
  name: "LookupTableSize"
  input_arg {
    name: "table_handle"
    type: DT_STRING
    is_ref: true
  }
  output_arg {
    name: "size"
    type: DT_INT64
  }
}
op {
  name: "MutableDenseHashTable"
  input_arg {
    name: "empty_key"
    type_attr: "key_dtype"
  }
  output_arg {
    name: "table_handle"
    type: DT_STRING
    is_ref: true
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
    name: "use_node_name_sharing"
    type: "bool"
    default_value {
      b: false
    }
  }
  attr {
    name: "key_dtype"
    type: "type"
  }
  attr {
    name: "value_dtype"
    type: "type"
  }
  attr {
    name: "value_shape"
    type: "shape"
    default_value {
      shape {
      }
    }
  }
  attr {
    name: "initial_num_buckets"
    type: "int"
    default_value {
      i: 131072
    }
  }
  attr {
    name: "max_load_factor"
    type: "float"
    default_value {
      f: 0.8
    }
  }
  is_stateful: true
}
op {
  name: "MutableHashTable"
  output_arg {
    name: "table_handle"
    type: DT_STRING
    is_ref: true
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
    name: "use_node_name_sharing"
    type: "bool"
    default_value {
      b: false
    }
  }
  attr {
    name: "key_dtype"
    type: "type"
  }
  attr {
    name: "value_dtype"
    type: "type"
  }
  is_stateful: true
}
op {
  name: "MutableHashTableOfTensors"
  output_arg {
    name: "table_handle"
    type: DT_STRING
    is_ref: true
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
    name: "use_node_name_sharing"
    type: "bool"
    default_value {
      b: false
    }
  }
  attr {
    name: "key_dtype"
    type: "type"
  }
  attr {
    name: "value_dtype"
    type: "type"
  }
  attr {
    name: "value_shape"
    type: "shape"
    default_value {
      shape {
      }
    }
  }
  is_stateful: true
}
op {
  name: "PaddingFIFOQueue"
  output_arg {
    name: "handle"
    type: DT_STRING
    is_ref: true
  }
  attr {
    name: "component_types"
    type: "list(type)"
    has_minimum: true
    minimum: 1
  }
  attr {
    name: "shapes"
    type: "list(shape)"
    default_value {
      list {
      }
    }
    has_minimum: true
  }
  attr {
    name: "capacity"
    type: "int"
    default_value {
      i: -1
    }
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
  name: "PaddingFIFOQueueV2"
  output_arg {
    name: "handle"
    type: DT_RESOURCE
  }
  attr {
    name: "component_types"
    type: "list(type)"
    has_minimum: true
    minimum: 1
  }
  attr {
    name: "shapes"
    type: "list(shape)"
    default_value {
      list {
      }
    }
    has_minimum: true
  }
  attr {
    name: "capacity"
    type: "int"
    default_value {
      i: -1
    }
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
  name: "PriorityQueue"
  output_arg {
    name: "handle"
    type: DT_STRING
    is_ref: true
  }
  attr {
    name: "component_types"
    type: "list(type)"
    default_value {
      list {
      }
    }
    has_minimum: true
  }
  attr {
    name: "shapes"
    type: "list(shape)"
    has_minimum: true
  }
  attr {
    name: "capacity"
    type: "int"
    default_value {
      i: -1
    }
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
  name: "PriorityQueueV2"
  output_arg {
    name: "handle"
    type: DT_RESOURCE
  }
  attr {
    name: "component_types"
    type: "list(type)"
    default_value {
      list {
      }
    }
    has_minimum: true
  }
  attr {
    name: "shapes"
    type: "list(shape)"
    has_minimum: true
  }
  attr {
    name: "capacity"
    type: "int"
    default_value {
      i: -1
    }
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
  name: "QueueClose"
  input_arg {
    name: "handle"
    type: DT_STRING
    is_ref: true
  }
  attr {
    name: "cancel_pending_enqueues"
    type: "bool"
    default_value {
      b: false
    }
  }
}
op {
  name: "QueueCloseV2"
  input_arg {
    name: "handle"
    type: DT_RESOURCE
  }
  attr {
    name: "cancel_pending_enqueues"
    type: "bool"
    default_value {
      b: false
    }
  }
}
op {
  name: "QueueDequeue"
  input_arg {
    name: "handle"
    type: DT_STRING
    is_ref: true
  }
  output_arg {
    name: "components"
    type_list_attr: "component_types"
  }
  attr {
    name: "component_types"
    type: "list(type)"
    has_minimum: true
    minimum: 1
  }
  attr {
    name: "timeout_ms"
    type: "int"
    default_value {
      i: -1
    }
  }
}
op {
  name: "QueueDequeueMany"
  input_arg {
    name: "handle"
    type: DT_STRING
    is_ref: true
  }
  input_arg {
    name: "n"
    type: DT_INT32
  }
  output_arg {
    name: "components"
    type_list_attr: "component_types"
  }
  attr {
    name: "component_types"
    type: "list(type)"
    has_minimum: true
    minimum: 1
  }
  attr {
    name: "timeout_ms"
    type: "int"
    default_value {
      i: -1
    }
  }
}
op {
  name: "QueueDequeueManyV2"
  input_arg {
    name: "handle"
    type: DT_RESOURCE
  }
  input_arg {
    name: "n"
    type: DT_INT32
  }
  output_arg {
    name: "components"
    type_list_attr: "component_types"
  }
  attr {
    name: "component_types"
    type: "list(type)"
    has_minimum: true
    minimum: 1
  }
  attr {
    name: "timeout_ms"
    type: "int"
    default_value {
      i: -1
    }
  }
}
op {
  name: "QueueDequeueUpTo"
  input_arg {
    name: "handle"
    type: DT_STRING
    is_ref: true
  }
  input_arg {
    name: "n"
    type: DT_INT32
  }
  output_arg {
    name: "components"
    type_list_attr: "component_types"
  }
  attr {
    name: "component_types"
    type: "list(type)"
    has_minimum: true
    minimum: 1
  }
  attr {
    name: "timeout_ms"
    type: "int"
    default_value {
      i: -1
    }
  }
}
op {
  name: "QueueDequeueUpToV2"
  input_arg {
    name: "handle"
    type: DT_RESOURCE
  }
  input_arg {
    name: "n"
    type: DT_INT32
  }
  output_arg {
    name: "components"
    type_list_attr: "component_types"
  }
  attr {
    name: "component_types"
    type: "list(type)"
    has_minimum: true
    minimum: 1
  }
  attr {
    name: "timeout_ms"
    type: "int"
    default_value {
      i: -1
    }
  }
}
op {
  name: "QueueDequeueV2"
  input_arg {
    name: "handle"
    type: DT_RESOURCE
  }
  output_arg {
    name: "components"
    type_list_attr: "component_types"
  }
  attr {
    name: "component_types"
    type: "list(type)"
    has_minimum: true
    minimum: 1
  }
  attr {
    name: "timeout_ms"
    type: "int"
    default_value {
      i: -1
    }
  }
}
op {
  name: "QueueEnqueue"
  input_arg {
    name: "handle"
    type: DT_STRING
    is_ref: true
  }
  input_arg {
    name: "components"
    type_list_attr: "Tcomponents"
  }
  attr {
    name: "Tcomponents"
    type: "list(type)"
    has_minimum: true
    minimum: 1
  }
  attr {
    name: "timeout_ms"
    type: "int"
    default_value {
      i: -1
    }
  }
}
op {
  name: "QueueEnqueueMany"
  input_arg {
    name: "handle"
    type: DT_STRING
    is_ref: true
  }
  input_arg {
    name: "components"
    type_list_attr: "Tcomponents"
  }
  attr {
    name: "Tcomponents"
    type: "list(type)"
    has_minimum: true
    minimum: 1
  }
  attr {
    name: "timeout_ms"
    type: "int"
    default_value {
      i: -1
    }
  }
}
op {
  name: "QueueEnqueueManyV2"
  input_arg {
    name: "handle"
    type: DT_RESOURCE
  }
  input_arg {
    name: "components"
    type_list_attr: "Tcomponents"
  }
  attr {
    name: "Tcomponents"
    type: "list(type)"
    has_minimum: true
    minimum: 1
  }
  attr {
    name: "timeout_ms"
    type: "int"
    default_value {
      i: -1
    }
  }
}
op {
  name: "QueueEnqueueV2"
  input_arg {
    name: "handle"
    type: DT_RESOURCE
  }
  input_arg {
    name: "components"
    type_list_attr: "Tcomponents"
  }
  attr {
    name: "Tcomponents"
    type: "list(type)"
    has_minimum: true
    minimum: 1
  }
  attr {
    name: "timeout_ms"
    type: "int"
    default_value {
      i: -1
    }
  }
}
op {
  name: "QueueSize"
  input_arg {
    name: "handle"
    type: DT_STRING
    is_ref: true
  }
  output_arg {
    name: "size"
    type: DT_INT32
  }
}
op {
  name: "QueueSizeV2"
  input_arg {
    name: "handle"
    type: DT_RESOURCE
  }
  output_arg {
    name: "size"
    type: DT_INT32
  }
}
op {
  name: "RandomShuffleQueue"
  output_arg {
    name: "handle"
    type: DT_STRING
    is_ref: true
  }
  attr {
    name: "component_types"
    type: "list(type)"
    has_minimum: true
    minimum: 1
  }
  attr {
    name: "shapes"
    type: "list(shape)"
    default_value {
      list {
      }
    }
    has_minimum: true
  }
  attr {
    name: "capacity"
    type: "int"
    default_value {
      i: -1
    }
  }
  attr {
    name: "min_after_dequeue"
    type: "int"
    default_value {
      i: 0
    }
  }
  attr {
    name: "seed"
    type: "int"
    default_value {
      i: 0
    }
  }
  attr {
    name: "seed2"
    type: "int"
    default_value {
      i: 0
    }
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
  name: "RandomShuffleQueueV2"
  output_arg {
    name: "handle"
    type: DT_RESOURCE
  }
  attr {
    name: "component_types"
    type: "list(type)"
    has_minimum: true
    minimum: 1
  }
  attr {
    name: "shapes"
    type: "list(shape)"
    default_value {
      list {
      }
    }
    has_minimum: true
  }
  attr {
    name: "capacity"
    type: "int"
    default_value {
      i: -1
    }
  }
  attr {
    name: "min_after_dequeue"
    type: "int"
    default_value {
      i: 0
    }
  }
  attr {
    name: "seed"
    type: "int"
    default_value {
      i: 0
    }
  }
  attr {
    name: "seed2"
    type: "int"
    default_value {
      i: 0
    }
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
  name: "RecordInput"
  output_arg {
    name: "records"
    type: DT_STRING
  }
  attr {
    name: "file_pattern"
    type: "string"
  }
  attr {
    name: "file_random_seed"
    type: "int"
    default_value {
      i: 301
    }
  }
  attr {
    name: "file_shuffle_shift_ratio"
    type: "float"
    default_value {
      f: 0
    }
  }
  attr {
    name: "file_buffer_size"
    type: "int"
    default_value {
      i: 10000
    }
  }
  attr {
    name: "file_parallelism"
    type: "int"
    default_value {
      i: 16
    }
  }
  attr {
    name: "batch_size"
    type: "int"
    default_value {
      i: 32
    }
  }
  is_stateful: true
}
op {
  name: "SparseAccumulatorApplyGradient"
  input_arg {
    name: "handle"
    type: DT_STRING
    is_ref: true
  }
  input_arg {
    name: "local_step"
    type: DT_INT64
  }
  input_arg {
    name: "gradient_indices"
    type: DT_INT64
  }
  input_arg {
    name: "gradient_values"
    type_attr: "dtype"
  }
  input_arg {
    name: "gradient_shape"
    type: DT_INT64
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
    name: "has_known_shape"
    type: "bool"
  }
}
op {
  name: "SparseAccumulatorTakeGradient"
  input_arg {
    name: "handle"
    type: DT_STRING
    is_ref: true
  }
  input_arg {
    name: "num_required"
    type: DT_INT32
  }
  output_arg {
    name: "indices"
    type: DT_INT64
  }
  output_arg {
    name: "values"
    type_attr: "dtype"
  }
  output_arg {
    name: "shape"
    type: DT_INT64
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
}
op {
  name: "SparseConditionalAccumulator"
  output_arg {
    name: "handle"
    type: DT_STRING
    is_ref: true
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
    name: "shape"
    type: "shape"
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
  name: "Stack"
  output_arg {
    name: "handle"
    type: DT_STRING
    is_ref: true
  }
  attr {
    name: "elem_type"
    type: "type"
  }
  attr {
    name: "stack_name"
    type: "string"
    default_value {
      s: ""
    }
  }
  is_stateful: true
}
op {
  name: "StackClose"
  input_arg {
    name: "handle"
    type: DT_STRING
    is_ref: true
  }
}
op {
  name: "StackPop"
  input_arg {
    name: "handle"
    type: DT_STRING
    is_ref: true
  }
  output_arg {
    name: "elem"
    type_attr: "elem_type"
  }
  attr {
    name: "elem_type"
    type: "type"
  }
}
op {
  name: "StackPush"
  input_arg {
    name: "handle"
    type: DT_STRING
    is_ref: true
  }
  input_arg {
    name: "elem"
    type_attr: "T"
  }
  output_arg {
    name: "output"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
  }
  attr {
    name: "swap_memory"
    type: "bool"
    default_value {
      b: false
    }
  }
}
op {
  name: "Stage"
  input_arg {
    name: "values"
    type_list_attr: "dtypes"
  }
  attr {
    name: "dtypes"
    type: "list(type)"
    has_minimum: true
    minimum: 1
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
  name: "TensorArray"
  input_arg {
    name: "size"
    type: DT_INT32
  }
  output_arg {
    name: "handle"
    type: DT_STRING
    is_ref: true
  }
  attr {
    name: "dtype"
    type: "type"
  }
  attr {
    name: "dynamic_size"
    type: "bool"
    default_value {
      b: false
    }
  }
  attr {
    name: "clear_after_read"
    type: "bool"
    default_value {
      b: true
    }
  }
  attr {
    name: "tensor_array_name"
    type: "string"
    default_value {
      s: ""
    }
  }
  attr {
    name: "element_shape"
    type: "shape"
    default_value {
      shape {
        unknown_rank: true
      }
    }
  }
  deprecation {
    version: 16
    explanation: "Use TensorArrayV3"
  }
  is_stateful: true
}
op {
  name: "TensorArrayClose"
  input_arg {
    name: "handle"
    type: DT_STRING
    is_ref: true
  }
  deprecation {
    version: 16
    explanation: "Use TensorArrayCloseV3"
  }
}
op {
  name: "TensorArrayCloseV2"
  input_arg {
    name: "handle"
    type: DT_STRING
  }
}
op {
  name: "TensorArrayCloseV3"
  input_arg {
    name: "handle"
    type: DT_RESOURCE
  }
}
op {
  name: "TensorArrayConcat"
  input_arg {
    name: "handle"
    type: DT_STRING
    is_ref: true
  }
  input_arg {
    name: "flow_in"
    type: DT_FLOAT
  }
  output_arg {
    name: "value"
    type_attr: "dtype"
  }
  output_arg {
    name: "lengths"
    type: DT_INT64
  }
  attr {
    name: "dtype"
    type: "type"
  }
  attr {
    name: "element_shape_except0"
    type: "shape"
    default_value {
      shape {
        unknown_rank: true
      }
    }
  }
  deprecation {
    version: 16
    explanation: "Use TensorArrayGradV3"
  }
}
op {
  name: "TensorArrayConcatV2"
  input_arg {
    name: "handle"
    type: DT_STRING
  }
  input_arg {
    name: "flow_in"
    type: DT_FLOAT
  }
  output_arg {
    name: "value"
    type_attr: "dtype"
  }
  output_arg {
    name: "lengths"
    type: DT_INT64
  }
  attr {
    name: "dtype"
    type: "type"
  }
  attr {
    name: "element_shape_except0"
    type: "shape"
    default_value {
      shape {
        unknown_rank: true
      }
    }
  }
}
op {
  name: "TensorArrayConcatV3"
  input_arg {
    name: "handle"
    type: DT_RESOURCE
  }
  input_arg {
    name: "flow_in"
    type: DT_FLOAT
  }
  output_arg {
    name: "value"
    type_attr: "dtype"
  }
  output_arg {
    name: "lengths"
    type: DT_INT64
  }
  attr {
    name: "dtype"
    type: "type"
  }
  attr {
    name: "element_shape_except0"
    type: "shape"
    default_value {
      shape {
        unknown_rank: true
      }
    }
  }
}
op {
  name: "TensorArrayGather"
  input_arg {
    name: "handle"
    type: DT_STRING
    is_ref: true
  }
  input_arg {
    name: "indices"
    type: DT_INT32
  }
  input_arg {
    name: "flow_in"
    type: DT_FLOAT
  }
  output_arg {
    name: "value"
    type_attr: "dtype"
  }
  attr {
    name: "dtype"
    type: "type"
  }
  attr {
    name: "element_shape"
    type: "shape"
    default_value {
      shape {
        unknown_rank: true
      }
    }
  }
  deprecation {
    version: 16
    explanation: "Use TensorArrayGatherV3"
  }
}
op {
  name: "TensorArrayGatherV2"
  input_arg {
    name: "handle"
    type: DT_STRING
  }
  input_arg {
    name: "indices"
    type: DT_INT32
  }
  input_arg {
    name: "flow_in"
    type: DT_FLOAT
  }
  output_arg {
    name: "value"
    type_attr: "dtype"
  }
  attr {
    name: "dtype"
    type: "type"
  }
  attr {
    name: "element_shape"
    type: "shape"
    default_value {
      shape {
        unknown_rank: true
      }
    }
  }
}
op {
  name: "TensorArrayGatherV3"
  input_arg {
    name: "handle"
    type: DT_RESOURCE
  }
  input_arg {
    name: "indices"
    type: DT_INT32
  }
  input_arg {
    name: "flow_in"
    type: DT_FLOAT
  }
  output_arg {
    name: "value"
    type_attr: "dtype"
  }
  attr {
    name: "dtype"
    type: "type"
  }
  attr {
    name: "element_shape"
    type: "shape"
    default_value {
      shape {
        unknown_rank: true
      }
    }
  }
}
op {
  name: "TensorArrayGrad"
  input_arg {
    name: "handle"
    type: DT_STRING
  }
  input_arg {
    name: "flow_in"
    type: DT_FLOAT
  }
  output_arg {
    name: "grad_handle"
    type: DT_STRING
    is_ref: true
  }
  attr {
    name: "source"
    type: "string"
  }
  deprecation {
    version: 16
    explanation: "Use TensorArrayGradV3"
  }
  is_stateful: true
}
op {
  name: "TensorArrayGradV2"
  input_arg {
    name: "handle"
    type: DT_STRING
  }
  input_arg {
    name: "flow_in"
    type: DT_FLOAT
  }
  output_arg {
    name: "grad_handle"
    type: DT_STRING
  }
  attr {
    name: "source"
    type: "string"
  }
  is_stateful: true
}
op {
  name: "TensorArrayGradV3"
  input_arg {
    name: "handle"
    type: DT_RESOURCE
  }
  input_arg {
    name: "flow_in"
    type: DT_FLOAT
  }
  output_arg {
    name: "grad_handle"
    type: DT_RESOURCE
  }
  output_arg {
    name: "flow_out"
    type: DT_FLOAT
  }
  attr {
    name: "source"
    type: "string"
  }
  is_stateful: true
}
op {
  name: "TensorArrayPack"
  input_arg {
    name: "handle"
    type: DT_STRING
    is_ref: true
  }
  input_arg {
    name: "flow_in"
    type: DT_FLOAT
  }
  output_arg {
    name: "value"
    type_attr: "dtype"
  }
  attr {
    name: "dtype"
    type: "type"
  }
  attr {
    name: "element_shape"
    type: "shape"
    default_value {
      shape {
        unknown_rank: true
      }
    }
  }
  deprecation {
    version: 16
    explanation: "Use TensorArrayGatherV3 with RangeOp"
  }
}
op {
  name: "TensorArrayRead"
  input_arg {
    name: "handle"
    type: DT_STRING
    is_ref: true
  }
  input_arg {
    name: "index"
    type: DT_INT32
  }
  input_arg {
    name: "flow_in"
    type: DT_FLOAT
  }
  output_arg {
    name: "value"
    type_attr: "dtype"
  }
  attr {
    name: "dtype"
    type: "type"
  }
  deprecation {
    version: 16
    explanation: "Use TensorArrayReadV3"
  }
}
op {
  name: "TensorArrayReadV2"
  input_arg {
    name: "handle"
    type: DT_STRING
  }
  input_arg {
    name: "index"
    type: DT_INT32
  }
  input_arg {
    name: "flow_in"
    type: DT_FLOAT
  }
  output_arg {
    name: "value"
    type_attr: "dtype"
  }
  attr {
    name: "dtype"
    type: "type"
  }
}
op {
  name: "TensorArrayReadV3"
  input_arg {
    name: "handle"
    type: DT_RESOURCE
  }
  input_arg {
    name: "index"
    type: DT_INT32
  }
  input_arg {
    name: "flow_in"
    type: DT_FLOAT
  }
  output_arg {
    name: "value"
    type_attr: "dtype"
  }
  attr {
    name: "dtype"
    type: "type"
  }
}
op {
  name: "TensorArrayScatter"
  input_arg {
    name: "handle"
    type: DT_STRING
    is_ref: true
  }
  input_arg {
    name: "indices"
    type: DT_INT32
  }
  input_arg {
    name: "value"
    type_attr: "T"
  }
  input_arg {
    name: "flow_in"
    type: DT_FLOAT
  }
  output_arg {
    name: "flow_out"
    type: DT_FLOAT
  }
  attr {
    name: "T"
    type: "type"
  }
  deprecation {
    version: 19
    explanation: "Use TensorArrayGradV3"
  }
}
op {
  name: "TensorArrayScatterV2"
  input_arg {
    name: "handle"
    type: DT_STRING
  }
  input_arg {
    name: "indices"
    type: DT_INT32
  }
  input_arg {
    name: "value"
    type_attr: "T"
  }
  input_arg {
    name: "flow_in"
    type: DT_FLOAT
  }
  output_arg {
    name: "flow_out"
    type: DT_FLOAT
  }
  attr {
    name: "T"
    type: "type"
  }
}
op {
  name: "TensorArrayScatterV3"
  input_arg {
    name: "handle"
    type: DT_RESOURCE
  }
  input_arg {
    name: "indices"
    type: DT_INT32
  }
  input_arg {
    name: "value"
    type_attr: "T"
  }
  input_arg {
    name: "flow_in"
    type: DT_FLOAT
  }
  output_arg {
    name: "flow_out"
    type: DT_FLOAT
  }
  attr {
    name: "T"
    type: "type"
  }
}
op {
  name: "TensorArraySize"
  input_arg {
    name: "handle"
    type: DT_STRING
    is_ref: true
  }
  input_arg {
    name: "flow_in"
    type: DT_FLOAT
  }
  output_arg {
    name: "size"
    type: DT_INT32
  }
  deprecation {
    version: 16
    explanation: "Use TensorArraySizeV3"
  }
}
op {
  name: "TensorArraySizeV2"
  input_arg {
    name: "handle"
    type: DT_STRING
  }
  input_arg {
    name: "flow_in"
    type: DT_FLOAT
  }
  output_arg {
    name: "size"
    type: DT_INT32
  }
}
op {
  name: "TensorArraySizeV3"
  input_arg {
    name: "handle"
    type: DT_RESOURCE
  }
  input_arg {
    name: "flow_in"
    type: DT_FLOAT
  }
  output_arg {
    name: "size"
    type: DT_INT32
  }
}
op {
  name: "TensorArraySplit"
  input_arg {
    name: "handle"
    type: DT_STRING
    is_ref: true
  }
  input_arg {
    name: "value"
    type_attr: "T"
  }
  input_arg {
    name: "lengths"
    type: DT_INT64
  }
  input_arg {
    name: "flow_in"
    type: DT_FLOAT
  }
  output_arg {
    name: "flow_out"
    type: DT_FLOAT
  }
  attr {
    name: "T"
    type: "type"
  }
  deprecation {
    version: 16
    explanation: "Use TensorArraySplitV3"
  }
}
op {
  name: "TensorArraySplitV2"
  input_arg {
    name: "handle"
    type: DT_STRING
  }
  input_arg {
    name: "value"
    type_attr: "T"
  }
  input_arg {
    name: "lengths"
    type: DT_INT64
  }
  input_arg {
    name: "flow_in"
    type: DT_FLOAT
  }
  output_arg {
    name: "flow_out"
    type: DT_FLOAT
  }
  attr {
    name: "T"
    type: "type"
  }
}
op {
  name: "TensorArraySplitV3"
  input_arg {
    name: "handle"
    type: DT_RESOURCE
  }
  input_arg {
    name: "value"
    type_attr: "T"
  }
  input_arg {
    name: "lengths"
    type: DT_INT64
  }
  input_arg {
    name: "flow_in"
    type: DT_FLOAT
  }
  output_arg {
    name: "flow_out"
    type: DT_FLOAT
  }
  attr {
    name: "T"
    type: "type"
  }
}
op {
  name: "TensorArrayUnpack"
  input_arg {
    name: "handle"
    type: DT_STRING
    is_ref: true
  }
  input_arg {
    name: "value"
    type_attr: "T"
  }
  input_arg {
    name: "flow_in"
    type: DT_FLOAT
  }
  output_arg {
    name: "flow_out"
    type: DT_FLOAT
  }
  attr {
    name: "T"
    type: "type"
  }
  deprecation {
    version: 20
    explanation: "Use TensorArrayScatterV3 with RangeOp"
  }
}
op {
  name: "TensorArrayV2"
  input_arg {
    name: "size"
    type: DT_INT32
  }
  output_arg {
    name: "handle"
    type: DT_STRING
  }
  attr {
    name: "dtype"
    type: "type"
  }
  attr {
    name: "element_shape"
    type: "shape"
    default_value {
      shape {
        unknown_rank: true
      }
    }
  }
  attr {
    name: "dynamic_size"
    type: "bool"
    default_value {
      b: false
    }
  }
  attr {
    name: "clear_after_read"
    type: "bool"
    default_value {
      b: true
    }
  }
  attr {
    name: "tensor_array_name"
    type: "string"
    default_value {
      s: ""
    }
  }
  is_stateful: true
}
op {
  name: "TensorArrayV3"
  input_arg {
    name: "size"
    type: DT_INT32
  }
  output_arg {
    name: "handle"
    type: DT_RESOURCE
  }
  output_arg {
    name: "flow"
    type: DT_FLOAT
  }
  attr {
    name: "dtype"
    type: "type"
  }
  attr {
    name: "element_shape"
    type: "shape"
    default_value {
      shape {
        unknown_rank: true
      }
    }
  }
  attr {
    name: "dynamic_size"
    type: "bool"
    default_value {
      b: false
    }
  }
  attr {
    name: "clear_after_read"
    type: "bool"
    default_value {
      b: true
    }
  }
  attr {
    name: "tensor_array_name"
    type: "string"
    default_value {
      s: ""
    }
  }
  is_stateful: true
}
op {
  name: "TensorArrayWrite"
  input_arg {
    name: "handle"
    type: DT_STRING
    is_ref: true
  }
  input_arg {
    name: "index"
    type: DT_INT32
  }
  input_arg {
    name: "value"
    type_attr: "T"
  }
  input_arg {
    name: "flow_in"
    type: DT_FLOAT
  }
  output_arg {
    name: "flow_out"
    type: DT_FLOAT
  }
  attr {
    name: "T"
    type: "type"
  }
  deprecation {
    version: 16
    explanation: "Use TensorArrayWriteV3"
  }
}
op {
  name: "TensorArrayWriteV2"
  input_arg {
    name: "handle"
    type: DT_STRING
  }
  input_arg {
    name: "index"
    type: DT_INT32
  }
  input_arg {
    name: "value"
    type_attr: "T"
  }
  input_arg {
    name: "flow_in"
    type: DT_FLOAT
  }
  output_arg {
    name: "flow_out"
    type: DT_FLOAT
  }
  attr {
    name: "T"
    type: "type"
  }
}
op {
  name: "TensorArrayWriteV3"
  input_arg {
    name: "handle"
    type: DT_RESOURCE
  }
  input_arg {
    name: "index"
    type: DT_INT32
  }
  input_arg {
    name: "value"
    type_attr: "T"
  }
  input_arg {
    name: "flow_in"
    type: DT_FLOAT
  }
  output_arg {
    name: "flow_out"
    type: DT_FLOAT
  }
  attr {
    name: "T"
    type: "type"
  }
}
op {
  name: "Unstage"
  output_arg {
    name: "values"
    type_list_attr: "dtypes"
  }
  attr {
    name: "dtypes"
    type: "list(type)"
    has_minimum: true
    minimum: 1
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
"""


_op_def_lib = _InitOpDefLibrary()
