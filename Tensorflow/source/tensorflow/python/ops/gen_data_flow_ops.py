"""Python wrappers around Brain.

This file is MACHINE GENERATED! Do not edit.
"""

import collections

from google.protobuf import text_format

from tensorflow.core.framework import op_def_pb2
from tensorflow.python.framework import op_def_registry
from tensorflow.python.framework import ops
from tensorflow.python.ops import op_def_library


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

      outputs[i].shape = [sum(partitions == i)] + data.shape[partitions.ndim:]

      outputs[i] = pack([data[js, ...] for js if partitions[js] == i])

  `data.shape` must start with `partitions.shape`.

  For example:

      # Scalar partitions
      partitions = 1
      num_partitions = 2
      data = [10, 20]
      outputs[0] = []  # Empty with shape [0, 2]
      outputs[1] = [[10, 20]]

      # Vector partitions
      partitions = [0, 0, 1, 1, 0]
      num_partitions = 2
      data = [10, 20, 30, 40, 50]
      outputs[0] = [10, 20, 50]
      outputs[1] = [30, 40]

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

      merged[indices[m][i, ..., j], ...] = data[m][i, ..., j, ...]

  For example, if each `indices[m]` is scalar or vector, we have

      # Scalar indices
      merged[indices[m], ...] = data[m][...]

      # Vector indices
      merged[indices[m][i], ...] = data[m][i, ...]

  Each `data[i].shape` must start with the corresponding `indices[i].shape`,
  and the rest of `data[i].shape` must be constant w.r.t. `i`.  That is, we
  must have `data[i].shape = indices[i].shape + constant`.  In terms of this
  `constant`, the output shape is

      merged.shape = [max(indices)] + constant

  Values are merged in order, so if an index appears in both `indices[m][i]` and
  `indices[n][j]` for `(m,i) < (n,j)` the slice `data[n][j]` will appear in the
  merged result.

  For example:

      indices[0] = 6
      indices[1] = [4, 1]
      indices[2] = [[5, 2], [0, 3]]
      data[0] = [61, 62]
      data[1] = [[41, 42], [11, 12]]
      data[2] = [[[51, 52], [21, 22]], [[1, 2], [31, 32]]]
      merged = [[1, 2], [11, 12], [21, 22], [31, 32], [41, 42],
                [51, 52], [61, 62]]

  <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
  <img style="width:100%" src="../../images/DynamicStitch.png" alt>
  </div>

  Args:
    indices: A list of at least 2 `Tensor` objects of type `int32`.
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
                name=None):
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
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type mutable `string`. Handle to a table.
  """
  result = _op_def_lib.apply_op("HashTable", key_dtype=key_dtype,
                                value_dtype=value_dtype, container=container,
                                shared_name=shared_name, name=name)
  return result


__initialize_table_outputs = [""]


def _initialize_table(table_handle, keys, values, name=None):
  r"""Table initializer that takes two tensors for keys and values respectively.

  Args:
    table_handle: A `Tensor` of type mutable `string`.
      Handle to a table which will be initialized.
    keys: A `Tensor`. Keys of type Tkey.
    values: A `Tensor`. Values of type Tval. Same shape as `keys`.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  result = _op_def_lib.apply_op("InitializeTable", table_handle=table_handle,
                                keys=keys, values=values, name=name)
  return result


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


__queue_dequeue_up_to_outputs = ["components"]


def _queue_dequeue_up_to(handle, n, component_types, timeout_ms=None,
                         name=None):
  r"""Dequeues n tuples of one or more tensors from the given queue.

  This operation is not supported by all queues.  If a queue does not support
  DequeueUpTo, then an Unimplemented error is returned.

  If the queue is closed and there are more than 0 but less than n elements
  remaining, then instead of returning an OutOfRange error like
  QueueDequeueMany, the remaining elements are returned immediately.  If the queue
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


__tensor_array_outputs = ["handle"]


def _tensor_array(size, dtype, dynamic_size=None, clear_after_read=None,
                  tensor_array_name=None, name=None):
  r"""An array of Tensors of given size, with data written via Write and read

  via Read or Pack.

  Args:
    size: A `Tensor` of type `int32`. The size of the array.
    dtype: A `tf.DType`. The type of the elements on the tensor_array.
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
    A `Tensor` of type mutable `string`. The handle to the TensorArray.
  """
  result = _op_def_lib.apply_op("TensorArray", size=size, dtype=dtype,
                                dynamic_size=dynamic_size,
                                clear_after_read=clear_after_read,
                                tensor_array_name=tensor_array_name,
                                name=name)
  return result


__tensor_array_close_outputs = [""]


def _tensor_array_close(handle, name=None):
  r"""Delete the TensorArray from its resource container.  This enables

  the user to close and release the resource in the middle of a step/run.

  Args:
    handle: A `Tensor` of type mutable `string`.
      The handle to a TensorArray (output of TensorArray or TensorArrayGrad).
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  result = _op_def_lib.apply_op("TensorArrayClose", handle=handle, name=name)
  return result


__tensor_array_concat_outputs = ["value", "lengths"]


_TensorArrayConcatOutput = collections.namedtuple("TensorArrayConcat",
                                                  __tensor_array_concat_outputs)


def _tensor_array_concat(handle, flow_in, dtype, name=None):
  r"""Concat the elements from the TensorArray into value `value`.

  Takes `T` elements of shapes

    ```
    (n0 x d0 x d1 x ...), (n1 x d0 x d1 x ...), ..., (n(T-1) x d0 x d1 x ...)
    ```

  and concatenates them into a Tensor of shape:

    ```(n0 + n1 + ... + n(T-1) x d0 x d1 x ...)```

  All elements must have the same shape (excepting the first dimension).

  Args:
    handle: A `Tensor` of type mutable `string`. The handle to a TensorArray.
    flow_in: A `Tensor` of type `float32`.
      A float scalar that enforces proper chaining of operations.
    dtype: A `tf.DType`. The type of the elem that is returned.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (value, lengths).
    value: A `Tensor` of type `dtype`. All of the elements in the TensorArray, concatenated along the first
      axis.
    lengths: A `Tensor` of type `int64`. A vector of the row sizes of the original T elements in the
      value output.  In the example above, this would be the values:
      `(n1, n2, ..., n(T-1))`.
  """
  result = _op_def_lib.apply_op("TensorArrayConcat", handle=handle,
                                flow_in=flow_in, dtype=dtype, name=name)
  return _TensorArrayConcatOutput._make(result)


__tensor_array_grad_outputs = ["grad_handle"]


def _tensor_array_grad(handle, flow_in, source, name=None):
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

  In terms of e.g. python TensorArray sugar wrappers when using dynamically sized

  Args:
    handle: A `Tensor` of type mutable `string`.
    flow_in: A `Tensor` of type `float32`.
    source: A `string`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type mutable `string`.
  """
  result = _op_def_lib.apply_op("TensorArrayGrad", handle=handle,
                                flow_in=flow_in, source=source, name=name)
  return result


__tensor_array_pack_outputs = ["value"]


def _tensor_array_pack(handle, flow_in, dtype, name=None):
  r"""Pack the elements from the TensorArray into output `value`.

  All elements must have the same shape.

  Args:
    handle: A `Tensor` of type mutable `string`. The handle to a TensorArray.
    flow_in: A `Tensor` of type `float32`.
      A float scalar that enforces proper chaining of operations.
    dtype: A `tf.DType`. The type of the elem that is returned.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `dtype`.
    All of the elements in the TensorArray, concatenated along a new
    axis (the new dimension 0).
  """
  result = _op_def_lib.apply_op("TensorArrayPack", handle=handle,
                                flow_in=flow_in, dtype=dtype, name=name)
  return result


__tensor_array_read_outputs = ["value"]


def _tensor_array_read(handle, index, flow_in, dtype, name=None):
  r"""Read an element from the TensorArray into output `value`.

  Args:
    handle: A `Tensor` of type mutable `string`. The handle to a TensorArray.
    index: A `Tensor` of type `int32`.
    flow_in: A `Tensor` of type `float32`.
      A float scalar that enforces proper chaining of operations.
    dtype: A `tf.DType`. The type of the elem that is returned.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `dtype`. The tensor that is read from the TensorArray.
  """
  result = _op_def_lib.apply_op("TensorArrayRead", handle=handle, index=index,
                                flow_in=flow_in, dtype=dtype, name=name)
  return result


__tensor_array_size_outputs = ["size"]


def _tensor_array_size(handle, flow_in, name=None):
  r"""Get the current size of the TensorArray.

  Args:
    handle: A `Tensor` of type mutable `string`.
      The handle to a TensorArray (output of TensorArray or TensorArrayGrad).
    flow_in: A `Tensor` of type `float32`.
      A float scalar that enforces proper chaining of operations.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int32`. The current size of the TensorArray.
  """
  result = _op_def_lib.apply_op("TensorArraySize", handle=handle,
                                flow_in=flow_in, name=name)
  return result


__tensor_array_split_outputs = ["flow_out"]


def _tensor_array_split(handle, value, lengths, flow_in, name=None):
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
    handle: A `Tensor` of type mutable `string`. The handle to a TensorArray.
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
  result = _op_def_lib.apply_op("TensorArraySplit", handle=handle,
                                value=value, lengths=lengths, flow_in=flow_in,
                                name=name)
  return result


__tensor_array_unpack_outputs = ["flow_out"]


def _tensor_array_unpack(handle, value, flow_in, name=None):
  r"""Unpack the data from the input value into TensorArray elements.

  Args:
    handle: A `Tensor` of type mutable `string`. The handle to a TensorArray.
    value: A `Tensor`. The concatenated tensor to write to the TensorArray.
    flow_in: A `Tensor` of type `float32`.
      A float scalar that enforces proper chaining of operations.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32`.
    A float scalar that enforces proper chaining of operations.
  """
  result = _op_def_lib.apply_op("TensorArrayUnpack", handle=handle,
                                value=value, flow_in=flow_in, name=name)
  return result


__tensor_array_write_outputs = ["flow_out"]


def _tensor_array_write(handle, index, value, flow_in, name=None):
  r"""Push an element onto the tensor_array.

  Args:
    handle: A `Tensor` of type mutable `string`. The handle to a TensorArray.
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
  result = _op_def_lib.apply_op("TensorArrayWrite", handle=handle,
                                index=index, value=value, flow_in=flow_in,
                                name=name)
  return result


def _InitOpDefLibrary():
  op_list = op_def_pb2.OpList()
  text_format.Merge(_InitOpDefLibrary.op_list_ascii, op_list)
  op_def_registry.register_op_list(op_list)
  op_def_lib = op_def_library.OpDefLibrary()
  op_def_lib.add_op_list(op_list)
  return op_def_lib


_InitOpDefLibrary.op_list_ascii = """op {
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
    minimum: 2
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
  is_stateful: true
}
op {
  name: "TensorArrayClose"
  input_arg {
    name: "handle"
    type: DT_STRING
    is_ref: true
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
}
op {
  name: "TensorArrayGrad"
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
    name: "grad_handle"
    type: DT_STRING
    is_ref: true
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
}
"""


_op_def_lib = _InitOpDefLibrary()
