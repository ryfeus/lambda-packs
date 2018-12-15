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

_batch_outputs = ["batched_tensors", "batch_index", "id"]
_BatchOutput = _collections.namedtuple(
    "Batch", _batch_outputs)


def batch(in_tensors, num_batch_threads, max_batch_size, batch_timeout_micros,
          grad_timeout_micros, allowed_batch_sizes=None, container=None,
          shared_name=None, batching_queue=None, name=None):
  r"""Batches all input tensors nondeterministically.

  When many instances of this Op are being run concurrently with the same
  container/shared_name in the same device, some will output zero-shaped Tensors
  and others will output Tensors of size up to max_batch_size.

  All Tensors in in_tensors are batched together (so, for example, labels and
  features should be batched with a single instance of this operation.

  Each invocation of batch emits an `id` scalar which will be used to identify
  this particular invocation when doing unbatch or its gradient.

  Each op which emits a non-empty batch will also emit a non-empty batch_index
  Tensor, which, is a [K, 3] matrix where each row contains the invocation's id,
  start, and length of elements of each set of Tensors present in batched_tensors.

  Batched tensors are concatenated along the first dimension, and all tensors in
  in_tensors must have the first dimension of the same size.

  Args:
    in_tensors: A list of `Tensor` objects. The tensors to be batched.
    num_batch_threads: An `int`.
      Number of scheduling threads for processing batches of work.
      Determines the number of batches processed in parallel.
    max_batch_size: An `int`. Batch sizes will never be bigger than this.
    batch_timeout_micros: An `int`.
      Maximum number of microseconds to wait before outputting
      an incomplete batch.
    grad_timeout_micros: An `int`.
      The timeout to use for the gradient. See Unbatch.
    allowed_batch_sizes: An optional list of `ints`. Defaults to `[]`.
      Optional list of allowed batch sizes. If left empty, does
      nothing. Otherwise, supplies a list of batch sizes, causing the op to pad
      batches up to one of those sizes. The entries must increase monotonically, and
      the final entry must equal max_batch_size.
    container: An optional `string`. Defaults to `""`.
      Controls the scope of sharing of this batch.
    shared_name: An optional `string`. Defaults to `""`.
      Concurrently running instances of batch in the same device with the
      same container and shared_name will batch their elements together. If left
      empty, the op name will be used as the shared name.
    batching_queue: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (batched_tensors, batch_index, id).

    batched_tensors: A list of `Tensor` objects. Has the same type as `in_tensors`. Either empty tensors or a batch of concatenated Tensors.
    batch_index: A `Tensor` of type `int64`. If out_tensors is non-empty, has information to invert it.
    id: A `Tensor` of type `int64`. always contains a scalar with a unique ID for this invocation of Batch.
  """
  result = _op_def_lib.apply_op("Batch", in_tensors=in_tensors,
                                num_batch_threads=num_batch_threads,
                                max_batch_size=max_batch_size,
                                batch_timeout_micros=batch_timeout_micros,
                                grad_timeout_micros=grad_timeout_micros,
                                allowed_batch_sizes=allowed_batch_sizes,
                                container=container, shared_name=shared_name,
                                batching_queue=batching_queue, name=name)
  return _BatchOutput._make(result)


_ops.RegisterShape("Batch")(None)

def unbatch(batched_tensor, batch_index, id, timeout_micros, container=None,
            shared_name=None, name=None):
  r"""Reverses the operation of Batch for a single output Tensor.

  An instance of Unbatch either receives an empty batched_tensor, in which case it
  asynchronously waits until the values become available from a concurrently
  running instance of Unbatch with the same container and shared_name, or receives
  a non-empty batched_tensor in which case it finalizes all other concurrently
  running instances and outputs its own element from the batch.

  Args:
    batched_tensor: A `Tensor`.
      The possibly transformed output of Batch. The size of the first
      dimension should remain unchanged by the transformations for the operation to
      work.
    batch_index: A `Tensor` of type `int64`.
      The matching batch_index obtained from Batch.
    id: A `Tensor` of type `int64`. The id scalar emitted by Batch.
    timeout_micros: An `int`.
      Maximum amount of time (in microseconds) to wait to receive the
      batched input tensor associated with a given invocation of the op.
    container: An optional `string`. Defaults to `""`.
      Container to control resource sharing.
    shared_name: An optional `string`. Defaults to `""`.
      Instances of Unbatch with the same container and shared_name are
      assumed to possibly belong to the same batch. If left empty, the op name will
      be used as the shared name.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `batched_tensor`.
    The Tensor corresponding to this execution.
  """
  result = _op_def_lib.apply_op("Unbatch", batched_tensor=batched_tensor,
                                batch_index=batch_index, id=id,
                                timeout_micros=timeout_micros,
                                container=container, shared_name=shared_name,
                                name=name)
  return result


_ops.RegisterShape("Unbatch")(None)

def unbatch_grad(original_input, batch_index, grad, id, container=None,
                 shared_name=None, name=None):
  r"""Gradient of Unbatch.

  Acts like Batch but using the given batch_index index of batching things as they
  become available. This ensures that the gradients are propagated back in the
  same session which did the forward pass.

  Args:
    original_input: A `Tensor`.
      The input to the Unbatch operation this is the gradient of.
    batch_index: A `Tensor` of type `int64`.
      The batch_index given to the Unbatch operation this is the gradient
      of.
    grad: A `Tensor`. Must have the same type as `original_input`.
      The downstream gradient.
    id: A `Tensor` of type `int64`. The id scalar emitted by Batch.
    container: An optional `string`. Defaults to `""`.
      Container to control resource sharing.
    shared_name: An optional `string`. Defaults to `""`.
      Instances of UnbatchGrad with the same container and shared_name
      are assumed to possibly belong to the same batch. If left empty, the op name
      will be used as the shared name.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `original_input`.
    The return value, either an empty tensor or the batched gradient.
  """
  result = _op_def_lib.apply_op("UnbatchGrad", original_input=original_input,
                                batch_index=batch_index, grad=grad, id=id,
                                container=container, shared_name=shared_name,
                                name=name)
  return result


_ops.RegisterShape("UnbatchGrad")(None)
def _InitOpDefLibrary():
  op_list = _op_def_pb2.OpList()
  _text_format.Merge(_InitOpDefLibrary.op_list_ascii, op_list)
  _op_def_registry.register_op_list(op_list)
  op_def_lib = _op_def_library.OpDefLibrary()
  op_def_lib.add_op_list(op_list)
  return op_def_lib


_InitOpDefLibrary.op_list_ascii = """op {
  name: "Batch"
  input_arg {
    name: "in_tensors"
    type_list_attr: "T"
  }
  output_arg {
    name: "batched_tensors"
    type_list_attr: "T"
  }
  output_arg {
    name: "batch_index"
    type: DT_INT64
  }
  output_arg {
    name: "id"
    type: DT_INT64
  }
  attr {
    name: "num_batch_threads"
    type: "int"
  }
  attr {
    name: "max_batch_size"
    type: "int"
  }
  attr {
    name: "batch_timeout_micros"
    type: "int"
  }
  attr {
    name: "allowed_batch_sizes"
    type: "list(int)"
    default_value {
      list {
      }
    }
  }
  attr {
    name: "grad_timeout_micros"
    type: "int"
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
    name: "batching_queue"
    type: "string"
    default_value {
      s: ""
    }
  }
  attr {
    name: "T"
    type: "list(type)"
    has_minimum: true
    minimum: 1
  }
}
op {
  name: "Unbatch"
  input_arg {
    name: "batched_tensor"
    type_attr: "T"
  }
  input_arg {
    name: "batch_index"
    type: DT_INT64
  }
  input_arg {
    name: "id"
    type: DT_INT64
  }
  output_arg {
    name: "unbatched_tensor"
    type_attr: "T"
  }
  attr {
    name: "timeout_micros"
    type: "int"
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
    name: "T"
    type: "type"
  }
}
op {
  name: "UnbatchGrad"
  input_arg {
    name: "original_input"
    type_attr: "T"
  }
  input_arg {
    name: "batch_index"
    type: DT_INT64
  }
  input_arg {
    name: "grad"
    type_attr: "T"
  }
  input_arg {
    name: "id"
    type: DT_INT64
  }
  output_arg {
    name: "batched_grad"
    type_attr: "T"
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
    name: "T"
    type: "type"
  }
}
"""


_op_def_lib = _InitOpDefLibrary()
