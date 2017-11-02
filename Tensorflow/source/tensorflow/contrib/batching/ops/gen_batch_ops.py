"""Python wrappers around TensorFlow ops.

This file is MACHINE GENERATED! Do not edit.
Original C++ source file: batch_ops.cc
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


_batch_outputs = ["batched_tensors", "batch_index", "id"]
_BatchOutput = _collections.namedtuple(
    "Batch", _batch_outputs)


def batch(in_tensors, num_batch_threads, max_batch_size, batch_timeout_micros, grad_timeout_micros, allowed_batch_sizes=[], container="", shared_name="", batching_queue="", name=None):
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
  num_batch_threads = _execute.make_int(num_batch_threads, "num_batch_threads")
  max_batch_size = _execute.make_int(max_batch_size, "max_batch_size")
  batch_timeout_micros = _execute.make_int(batch_timeout_micros, "batch_timeout_micros")
  grad_timeout_micros = _execute.make_int(grad_timeout_micros, "grad_timeout_micros")
  if allowed_batch_sizes is None:
    allowed_batch_sizes = []
  if not isinstance(allowed_batch_sizes, (list, tuple)):
    raise TypeError(
        "Expected list for 'allowed_batch_sizes' argument to "
        "'batch' Op, not %r." % allowed_batch_sizes)
  allowed_batch_sizes = [_execute.make_int(_i, "allowed_batch_sizes") for _i in allowed_batch_sizes]
  if container is None:
    container = ""
  container = _execute.make_str(container, "container")
  if shared_name is None:
    shared_name = ""
  shared_name = _execute.make_str(shared_name, "shared_name")
  if batching_queue is None:
    batching_queue = ""
  batching_queue = _execute.make_str(batching_queue, "batching_queue")
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "Batch", in_tensors=in_tensors, num_batch_threads=num_batch_threads,
        max_batch_size=max_batch_size,
        batch_timeout_micros=batch_timeout_micros,
        grad_timeout_micros=grad_timeout_micros,
        allowed_batch_sizes=allowed_batch_sizes, container=container,
        shared_name=shared_name, batching_queue=batching_queue, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("num_batch_threads", _op.get_attr("num_batch_threads"),
              "max_batch_size", _op.get_attr("max_batch_size"),
              "batch_timeout_micros", _op.get_attr("batch_timeout_micros"),
              "allowed_batch_sizes", _op.get_attr("allowed_batch_sizes"),
              "grad_timeout_micros", _op.get_attr("grad_timeout_micros"),
              "container", _op.get_attr("container"), "shared_name",
              _op.get_attr("shared_name"), "batching_queue",
              _op.get_attr("batching_queue"), "T", _op.get_attr("T"))
  else:
    _attr_T, in_tensors = _execute.convert_to_mixed_eager_tensors(in_tensors, _ctx)
    _attr_T = [_t.as_datatype_enum for _t in _attr_T]
    _inputs_flat = list(in_tensors)
    _attrs = ("num_batch_threads", num_batch_threads, "max_batch_size",
              max_batch_size, "batch_timeout_micros", batch_timeout_micros,
              "allowed_batch_sizes", allowed_batch_sizes,
              "grad_timeout_micros", grad_timeout_micros, "container",
              container, "shared_name", shared_name, "batching_queue",
              batching_queue, "T", _attr_T)
    _result = _execute.execute(b"Batch", len(in_tensors) + 2,
                               inputs=_inputs_flat, attrs=_attrs, ctx=_ctx,
                               name=name)
  _execute.record_gradient(
      "Batch", _inputs_flat, _attrs, _result, name)
  _result = [_result[:len(in_tensors)]] + _result[len(in_tensors):]
  _result = _BatchOutput._make(_result)
  return _result

_ops.RegisterShape("Batch")(None)


def unbatch(batched_tensor, batch_index, id, timeout_micros, container="", shared_name="", name=None):
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
  timeout_micros = _execute.make_int(timeout_micros, "timeout_micros")
  if container is None:
    container = ""
  container = _execute.make_str(container, "container")
  if shared_name is None:
    shared_name = ""
  shared_name = _execute.make_str(shared_name, "shared_name")
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "Unbatch", batched_tensor=batched_tensor, batch_index=batch_index,
        id=id, timeout_micros=timeout_micros, container=container,
        shared_name=shared_name, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("timeout_micros", _op.get_attr("timeout_micros"), "container",
              _op.get_attr("container"), "shared_name",
              _op.get_attr("shared_name"), "T", _op.get_attr("T"))
  else:
    _attr_T, (batched_tensor,) = _execute.args_to_matching_eager([batched_tensor], _ctx)
    _attr_T = _attr_T.as_datatype_enum
    batch_index = _ops.convert_to_tensor(batch_index, _dtypes.int64)
    id = _ops.convert_to_tensor(id, _dtypes.int64)
    _inputs_flat = [batched_tensor, batch_index, id]
    _attrs = ("timeout_micros", timeout_micros, "container", container,
              "shared_name", shared_name, "T", _attr_T)
    _result = _execute.execute(b"Unbatch", 1, inputs=_inputs_flat,
                               attrs=_attrs, ctx=_ctx, name=name)
  _execute.record_gradient(
      "Unbatch", _inputs_flat, _attrs, _result, name)
  _result, = _result
  return _result

_ops.RegisterShape("Unbatch")(None)


def unbatch_grad(original_input, batch_index, grad, id, container="", shared_name="", name=None):
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
  if container is None:
    container = ""
  container = _execute.make_str(container, "container")
  if shared_name is None:
    shared_name = ""
  shared_name = _execute.make_str(shared_name, "shared_name")
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "UnbatchGrad", original_input=original_input, batch_index=batch_index,
        grad=grad, id=id, container=container, shared_name=shared_name,
        name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("container", _op.get_attr("container"), "shared_name",
              _op.get_attr("shared_name"), "T", _op.get_attr("T"))
  else:
    _attr_T, _inputs_T = _execute.args_to_matching_eager([original_input, grad], _ctx)
    (original_input, grad) = _inputs_T
    _attr_T = _attr_T.as_datatype_enum
    batch_index = _ops.convert_to_tensor(batch_index, _dtypes.int64)
    id = _ops.convert_to_tensor(id, _dtypes.int64)
    _inputs_flat = [original_input, batch_index, grad, id]
    _attrs = ("container", container, "shared_name", shared_name, "T",
              _attr_T)
    _result = _execute.execute(b"UnbatchGrad", 1, inputs=_inputs_flat,
                               attrs=_attrs, ctx=_ctx, name=name)
  _execute.record_gradient(
      "UnbatchGrad", _inputs_flat, _attrs, _result, name)
  _result, = _result
  return _result

_ops.RegisterShape("UnbatchGrad")(None)

def _InitOpDefLibrary(op_list_proto_bytes):
  op_list = _op_def_pb2.OpList()
  op_list.ParseFromString(op_list_proto_bytes)
  _op_def_registry.register_op_list(op_list)
  op_def_lib = _op_def_library.OpDefLibrary()
  op_def_lib.add_op_list(op_list)
  return op_def_lib
# op {
#   name: "Batch"
#   input_arg {
#     name: "in_tensors"
#     type_list_attr: "T"
#   }
#   output_arg {
#     name: "batched_tensors"
#     type_list_attr: "T"
#   }
#   output_arg {
#     name: "batch_index"
#     type: DT_INT64
#   }
#   output_arg {
#     name: "id"
#     type: DT_INT64
#   }
#   attr {
#     name: "num_batch_threads"
#     type: "int"
#   }
#   attr {
#     name: "max_batch_size"
#     type: "int"
#   }
#   attr {
#     name: "batch_timeout_micros"
#     type: "int"
#   }
#   attr {
#     name: "allowed_batch_sizes"
#     type: "list(int)"
#     default_value {
#       list {
#       }
#     }
#   }
#   attr {
#     name: "grad_timeout_micros"
#     type: "int"
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
#     name: "batching_queue"
#     type: "string"
#     default_value {
#       s: ""
#     }
#   }
#   attr {
#     name: "T"
#     type: "list(type)"
#     has_minimum: true
#     minimum: 1
#   }
# }
# op {
#   name: "Unbatch"
#   input_arg {
#     name: "batched_tensor"
#     type_attr: "T"
#   }
#   input_arg {
#     name: "batch_index"
#     type: DT_INT64
#   }
#   input_arg {
#     name: "id"
#     type: DT_INT64
#   }
#   output_arg {
#     name: "unbatched_tensor"
#     type_attr: "T"
#   }
#   attr {
#     name: "timeout_micros"
#     type: "int"
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
#     name: "T"
#     type: "type"
#   }
# }
# op {
#   name: "UnbatchGrad"
#   input_arg {
#     name: "original_input"
#     type_attr: "T"
#   }
#   input_arg {
#     name: "batch_index"
#     type: DT_INT64
#   }
#   input_arg {
#     name: "grad"
#     type_attr: "T"
#   }
#   input_arg {
#     name: "id"
#     type: DT_INT64
#   }
#   output_arg {
#     name: "batched_grad"
#     type_attr: "T"
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
#     name: "T"
#     type: "type"
#   }
# }
_op_def_lib = _InitOpDefLibrary(b"\n\276\002\n\005Batch\022\017\n\nin_tensors2\001T\032\024\n\017batched_tensors2\001T\032\017\n\013batch_index\030\t\032\006\n\002id\030\t\"\030\n\021num_batch_threads\022\003int\"\025\n\016max_batch_size\022\003int\"\033\n\024batch_timeout_micros\022\003int\"$\n\023allowed_batch_sizes\022\tlist(int)\032\002\n\000\"\032\n\023grad_timeout_micros\022\003int\"\027\n\tcontainer\022\006string\032\002\022\000\"\031\n\013shared_name\022\006string\032\002\022\000\"\034\n\016batching_queue\022\006string\032\002\022\000\"\023\n\001T\022\nlist(type)(\0010\001\n\244\001\n\007Unbatch\022\023\n\016batched_tensor\"\001T\022\017\n\013batch_index\030\t\022\006\n\002id\030\t\032\025\n\020unbatched_tensor\"\001T\"\025\n\016timeout_micros\022\003int\"\027\n\tcontainer\022\006string\032\002\022\000\"\031\n\013shared_name\022\006string\032\002\022\000\"\t\n\001T\022\004type\n\230\001\n\013UnbatchGrad\022\023\n\016original_input\"\001T\022\017\n\013batch_index\030\t\022\t\n\004grad\"\001T\022\006\n\002id\030\t\032\021\n\014batched_grad\"\001T\"\027\n\tcontainer\022\006string\032\002\022\000\"\031\n\013shared_name\022\006string\032\002\022\000\"\t\n\001T\022\004type")
