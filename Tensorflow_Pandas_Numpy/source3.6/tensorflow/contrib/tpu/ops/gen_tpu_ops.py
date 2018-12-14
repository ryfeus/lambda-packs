"""Python wrappers around TensorFlow ops.

This file is MACHINE GENERATED! Do not edit.
Original C++ source file: tpu_ops.cc
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


def configure_distributed_tpu(embedding_config="", name=None):
  r"""An op that sets up the centralized structures for a distributed TPU

  system.

  Args:
    embedding_config: An optional `string`. Defaults to `""`. Internal use.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int32`.
    A two-dimensional array. For each host (the outer
    dimension) the array lists the global ids of the TPUs on that host.
  """
  if embedding_config is None:
    embedding_config = ""
  embedding_config = _execute.make_str(embedding_config, "embedding_config")
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "ConfigureDistributedTPU", embedding_config=embedding_config,
        name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("embedding_config", _op.get_attr("embedding_config"))
  else:
    _inputs_flat = []
    _attrs = ("embedding_config", embedding_config)
    _result = _execute.execute(b"ConfigureDistributedTPU", 1,
                               inputs=_inputs_flat, attrs=_attrs, ctx=_ctx,
                               name=name)
  _execute.record_gradient(
      "ConfigureDistributedTPU", _inputs_flat, _attrs, _result, name)
  _result, = _result
  return _result

_ops.RegisterShape("ConfigureDistributedTPU")(None)


def cross_replica_sum(input, name=None):
  r"""An Op to sum inputs across replicated TPU instances. Each

  instance supplies its own input, and the output of each is the sum of
  all the inputs.

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`.
      The local input to the sum.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
    The sum of all the distributed inputs.
  """
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "CrossReplicaSum", input=input, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("T", _op.get_attr("T"))
  else:
    _attr_T, (input,) = _execute.args_to_matching_eager([input], _ctx)
    _attr_T = _attr_T.as_datatype_enum
    _inputs_flat = [input]
    _attrs = ("T", _attr_T)
    _result = _execute.execute(b"CrossReplicaSum", 1, inputs=_inputs_flat,
                               attrs=_attrs, ctx=_ctx, name=name)
  _execute.record_gradient(
      "CrossReplicaSum", _inputs_flat, _attrs, _result, name)
  _result, = _result
  return _result

_ops.RegisterShape("CrossReplicaSum")(None)


def infeed_dequeue(dtype, shape, name=None):
  r"""A placeholder op for a value that will be fed into the computation.

  Args:
    dtype: A `tf.DType`. The type of elements in the tensor.
    shape: A `tf.TensorShape` or list of `ints`. The shape of the tensor.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `dtype`.
    A tensor that will be provided using the infeed mechanism.
  """
  dtype = _execute.make_type(dtype, "dtype")
  shape = _execute.make_shape(shape, "shape")
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "InfeedDequeue", dtype=dtype, shape=shape, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("dtype", _op.get_attr("dtype"), "shape", _op.get_attr("shape"))
  else:
    _inputs_flat = []
    _attrs = ("dtype", dtype, "shape", shape)
    _result = _execute.execute(b"InfeedDequeue", 1, inputs=_inputs_flat,
                               attrs=_attrs, ctx=_ctx, name=name)
  _execute.record_gradient(
      "InfeedDequeue", _inputs_flat, _attrs, _result, name)
  _result, = _result
  return _result

_ops.RegisterShape("InfeedDequeue")(None)


def infeed_dequeue_tuple(dtypes, shapes, name=None):
  r"""A placeholder op for multiple values that will be fed into the computation

  simultaneously as an XLA tuple.

  Args:
    dtypes: A list of `tf.DTypes` that has length `>= 1`.
      The element types of each element in `outputs`.
    shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`).
      The shapes of each tensor in `outputs`.
    name: A name for the operation (optional).

  Returns:
    A list of `Tensor` objects of type `dtypes`.
    A list of tensors that will be provided using the infeed mechanism.
  """
  if not isinstance(dtypes, (list, tuple)):
    raise TypeError(
        "Expected list for 'dtypes' argument to "
        "'infeed_dequeue_tuple' Op, not %r." % dtypes)
  dtypes = [_execute.make_type(_t, "dtypes") for _t in dtypes]
  if not isinstance(shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'shapes' argument to "
        "'infeed_dequeue_tuple' Op, not %r." % shapes)
  shapes = [_execute.make_shape(_s, "shapes") for _s in shapes]
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "InfeedDequeueTuple", dtypes=dtypes, shapes=shapes, name=name)
    _result = _op.outputs[:]
    if not _result:
      return _op
    _inputs_flat = _op.inputs
    _attrs = ("dtypes", _op.get_attr("dtypes"), "shapes",
              _op.get_attr("shapes"))
  else:
    _inputs_flat = []
    _attrs = ("dtypes", dtypes, "shapes", shapes)
    _result = _execute.execute(b"InfeedDequeueTuple", len(dtypes),
                               inputs=_inputs_flat, attrs=_attrs, ctx=_ctx,
                               name=name)
  _execute.record_gradient(
      "InfeedDequeueTuple", _inputs_flat, _attrs, _result, name)
  return _result

_ops.RegisterShape("InfeedDequeueTuple")(None)


def infeed_enqueue(input, shape=[], device_ordinal=-1, name=None):
  r"""An op which feeds a single Tensor value into the computation.

  Args:
    input: A `Tensor`.
      A tensor that will be provided using the infeed mechanism.
    shape: An optional `tf.TensorShape` or list of `ints`. Defaults to `[]`.
      The shape of the tensor.
    device_ordinal: An optional `int`. Defaults to `-1`.
      The TPU device to use. This should be -1 when the Op
      is running on a TPU device, and >= 0 when the Op is running on the CPU
      device.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  if shape is None:
    shape = []
  shape = _execute.make_shape(shape, "shape")
  if device_ordinal is None:
    device_ordinal = -1
  device_ordinal = _execute.make_int(device_ordinal, "device_ordinal")
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "InfeedEnqueue", input=input, shape=shape,
        device_ordinal=device_ordinal, name=name)
    return _op
  else:
    _attr_dtype, (input,) = _execute.args_to_matching_eager([input], _ctx)
    _attr_dtype = _attr_dtype.as_datatype_enum
    _inputs_flat = [input]
    _attrs = ("dtype", _attr_dtype, "shape", shape, "device_ordinal",
              device_ordinal)
    _result = _execute.execute(b"InfeedEnqueue", 0, inputs=_inputs_flat,
                               attrs=_attrs, ctx=_ctx, name=name)
  return _result

_ops.RegisterShape("InfeedEnqueue")(None)


def infeed_enqueue_tuple(inputs, shapes, device_ordinal=-1, name=None):
  r"""An op which feeds multiple Tensor values into the computation as an XLA tuple.

  Args:
    inputs: A list of `Tensor` objects.
      A list of tensors that will be provided using the infeed mechanism.
    shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`).
      The shapes of each tensor in `inputs`.
    device_ordinal: An optional `int`. Defaults to `-1`.
      The TPU device to use. This should be -1 when the Op
      is running on a TPU device, and >= 0 when the Op is running on the CPU
      device.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  if not isinstance(shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'shapes' argument to "
        "'infeed_enqueue_tuple' Op, not %r." % shapes)
  shapes = [_execute.make_shape(_s, "shapes") for _s in shapes]
  if device_ordinal is None:
    device_ordinal = -1
  device_ordinal = _execute.make_int(device_ordinal, "device_ordinal")
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "InfeedEnqueueTuple", inputs=inputs, shapes=shapes,
        device_ordinal=device_ordinal, name=name)
    return _op
  else:
    _attr_dtypes, inputs = _execute.convert_to_mixed_eager_tensors(inputs, _ctx)
    _attr_dtypes = [_t.as_datatype_enum for _t in _attr_dtypes]
    _inputs_flat = list(inputs)
    _attrs = ("dtypes", _attr_dtypes, "shapes", shapes, "device_ordinal",
              device_ordinal)
    _result = _execute.execute(b"InfeedEnqueueTuple", 0, inputs=_inputs_flat,
                               attrs=_attrs, ctx=_ctx, name=name)
  return _result

_ops.RegisterShape("InfeedEnqueueTuple")(None)


def outfeed_dequeue(dtype, shape, device_ordinal=-1, name=None):
  r"""Retrieves a single tensor from the computation outfeed.  This operation will

  block indefinitely until data is available.

  Args:
    dtype: A `tf.DType`. The type of elements in the tensor.
    shape: A `tf.TensorShape` or list of `ints`. The shape of the tensor.
    device_ordinal: An optional `int`. Defaults to `-1`.
      The TPU device to use. This should be -1 when the Op
      is running on a TPU device, and >= 0 when the Op is running on the CPU
      device.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `dtype`.
    A tensor that will be read from the device outfeed.
  """
  dtype = _execute.make_type(dtype, "dtype")
  shape = _execute.make_shape(shape, "shape")
  if device_ordinal is None:
    device_ordinal = -1
  device_ordinal = _execute.make_int(device_ordinal, "device_ordinal")
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "OutfeedDequeue", dtype=dtype, shape=shape,
        device_ordinal=device_ordinal, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("dtype", _op.get_attr("dtype"), "shape", _op.get_attr("shape"),
              "device_ordinal", _op.get_attr("device_ordinal"))
  else:
    _inputs_flat = []
    _attrs = ("dtype", dtype, "shape", shape, "device_ordinal",
              device_ordinal)
    _result = _execute.execute(b"OutfeedDequeue", 1, inputs=_inputs_flat,
                               attrs=_attrs, ctx=_ctx, name=name)
  _execute.record_gradient(
      "OutfeedDequeue", _inputs_flat, _attrs, _result, name)
  _result, = _result
  return _result

_ops.RegisterShape("OutfeedDequeue")(None)


def outfeed_dequeue_tuple(dtypes, shapes, device_ordinal=-1, name=None):
  r"""Retrieve multiple values that will be emitted by the computation as an XLA

  tuple.  This operations will block indefinitely until data is available.
  Output `i` corresponds to XLA tuple element `i`.

  Args:
    dtypes: A list of `tf.DTypes` that has length `>= 1`.
      The element types of each element in `outputs`.
    shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`).
      The shapes of each tensor in `outputs`.
    device_ordinal: An optional `int`. Defaults to `-1`.
      The TPU device to use. This should be -1 when the Op
      is running on a TPU device, and >= 0 when the Op is running on the CPU
      device.
    name: A name for the operation (optional).

  Returns:
    A list of `Tensor` objects of type `dtypes`.
    A list of tensors that will be read from the outfeed.
  """
  if not isinstance(dtypes, (list, tuple)):
    raise TypeError(
        "Expected list for 'dtypes' argument to "
        "'outfeed_dequeue_tuple' Op, not %r." % dtypes)
  dtypes = [_execute.make_type(_t, "dtypes") for _t in dtypes]
  if not isinstance(shapes, (list, tuple)):
    raise TypeError(
        "Expected list for 'shapes' argument to "
        "'outfeed_dequeue_tuple' Op, not %r." % shapes)
  shapes = [_execute.make_shape(_s, "shapes") for _s in shapes]
  if device_ordinal is None:
    device_ordinal = -1
  device_ordinal = _execute.make_int(device_ordinal, "device_ordinal")
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "OutfeedDequeueTuple", dtypes=dtypes, shapes=shapes,
        device_ordinal=device_ordinal, name=name)
    _result = _op.outputs[:]
    if not _result:
      return _op
    _inputs_flat = _op.inputs
    _attrs = ("dtypes", _op.get_attr("dtypes"), "shapes",
              _op.get_attr("shapes"), "device_ordinal",
              _op.get_attr("device_ordinal"))
  else:
    _inputs_flat = []
    _attrs = ("dtypes", dtypes, "shapes", shapes, "device_ordinal",
              device_ordinal)
    _result = _execute.execute(b"OutfeedDequeueTuple", len(dtypes),
                               inputs=_inputs_flat, attrs=_attrs, ctx=_ctx,
                               name=name)
  _execute.record_gradient(
      "OutfeedDequeueTuple", _inputs_flat, _attrs, _result, name)
  return _result

_ops.RegisterShape("OutfeedDequeueTuple")(None)


def outfeed_enqueue(input, name=None):
  r"""An op which emits a single Tensor value from an XLA computation.

  Args:
    input: A `Tensor`. A tensor that will be inserted into the outfeed queue.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "OutfeedEnqueue", input=input, name=name)
    return _op
  else:
    _attr_dtype, (input,) = _execute.args_to_matching_eager([input], _ctx)
    _attr_dtype = _attr_dtype.as_datatype_enum
    _inputs_flat = [input]
    _attrs = ("dtype", _attr_dtype)
    _result = _execute.execute(b"OutfeedEnqueue", 0, inputs=_inputs_flat,
                               attrs=_attrs, ctx=_ctx, name=name)
  return _result

_ops.RegisterShape("OutfeedEnqueue")(None)


def outfeed_enqueue_tuple(inputs, name=None):
  r"""An op which emits multiple Tensor values from an XLA computation.

  Args:
    inputs: A list of `Tensor` objects.
      A list of tensors that will be inserted into the outfeed queue as an
      XLA tuple.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "OutfeedEnqueueTuple", inputs=inputs, name=name)
    return _op
  else:
    _attr_dtypes, inputs = _execute.convert_to_mixed_eager_tensors(inputs, _ctx)
    _attr_dtypes = [_t.as_datatype_enum for _t in _attr_dtypes]
    _inputs_flat = list(inputs)
    _attrs = ("dtypes", _attr_dtypes)
    _result = _execute.execute(b"OutfeedEnqueueTuple", 0, inputs=_inputs_flat,
                               attrs=_attrs, ctx=_ctx, name=name)
  return _result

_ops.RegisterShape("OutfeedEnqueueTuple")(None)


def shutdown_distributed_tpu(name=None):
  r"""An op that shuts down a running distributed TPU system. The Op returns

  an error if no system is running.

  Args:
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "ShutdownDistributedTPU", name=name)
    return _op
  else:
    _inputs_flat = []
    _attrs = None
    _result = _execute.execute(b"ShutdownDistributedTPU", 0,
                               inputs=_inputs_flat, attrs=_attrs, ctx=_ctx,
                               name=name)
  return _result

_ops.RegisterShape("ShutdownDistributedTPU")(None)


def tpu_replicate(inputs, broadcast_inputs, variables, computation, num_replicas, output_types, global_tpu_id=[], name=None):
  r"""Runs replicated computations on a distributed TPU system.

  Args:
    inputs: A list of `Tensor` objects.
      the inputs to 'computation', flattened, in replica-major order.
    broadcast_inputs: A list of `Tensor` objects.
      additional arguments to broadcast to all replicas. The
      broadcast inputs are appended to the per-replica inputs when calling
      computation.
    variables: A list of `Tensor` objects with type `resource`.
    computation: A function decorated with @Defun.
      a function containing the computation to run.
    num_replicas: An `int` that is `>= 1`.
      the number of replicas of the computation to run.
    output_types: A list of `tf.DTypes`.
      the types of the outputs of 'computation'.
    global_tpu_id: An optional list of `ints`. Defaults to `[]`.
      map from device to global tpu id.
    name: A name for the operation (optional).

  Returns:
    A list of `Tensor` objects of type `output_types`.
    the outputs of 'computation'.
  """
  if not isinstance(variables, (list, tuple)):
    raise TypeError(
        "Expected list for 'variables' argument to "
        "'tpu_replicate' Op, not %r." % variables)
  _attr_NumVariables = len(variables)
  num_replicas = _execute.make_int(num_replicas, "num_replicas")
  if not isinstance(output_types, (list, tuple)):
    raise TypeError(
        "Expected list for 'output_types' argument to "
        "'tpu_replicate' Op, not %r." % output_types)
  output_types = [_execute.make_type(_t, "output_types") for _t in output_types]
  if global_tpu_id is None:
    global_tpu_id = []
  if not isinstance(global_tpu_id, (list, tuple)):
    raise TypeError(
        "Expected list for 'global_tpu_id' argument to "
        "'tpu_replicate' Op, not %r." % global_tpu_id)
  global_tpu_id = [_execute.make_int(_i, "global_tpu_id") for _i in global_tpu_id]
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "TPUReplicate", inputs=inputs, broadcast_inputs=broadcast_inputs,
        variables=variables, computation=computation,
        num_replicas=num_replicas, output_types=output_types,
        global_tpu_id=global_tpu_id, name=name)
    _result = _op.outputs[:]
    if not _result:
      return _op
    _inputs_flat = _op.inputs
    _attrs = ("computation", _op.get_attr("computation"), "num_replicas",
              _op.get_attr("num_replicas"), "global_tpu_id",
              _op.get_attr("global_tpu_id"), "Tinputs",
              _op.get_attr("Tinputs"), "Tbroadcast_inputs",
              _op.get_attr("Tbroadcast_inputs"), "NumVariables",
              _op.get_attr("NumVariables"), "output_types",
              _op.get_attr("output_types"))
  else:
    _attr_Tinputs, inputs = _execute.convert_to_mixed_eager_tensors(inputs, _ctx)
    _attr_Tinputs = [_t.as_datatype_enum for _t in _attr_Tinputs]
    _attr_Tbroadcast_inputs, broadcast_inputs = _execute.convert_to_mixed_eager_tensors(broadcast_inputs, _ctx)
    _attr_Tbroadcast_inputs = [_t.as_datatype_enum for _t in _attr_Tbroadcast_inputs]
    variables = _ops.convert_n_to_tensor(variables, _dtypes.resource)
    _inputs_flat = list(inputs) + list(broadcast_inputs) + list(variables)
    _attrs = ("computation", computation, "num_replicas", num_replicas,
              "global_tpu_id", global_tpu_id, "Tinputs", _attr_Tinputs,
              "Tbroadcast_inputs", _attr_Tbroadcast_inputs, "NumVariables",
              _attr_NumVariables, "output_types", output_types)
    _result = _execute.execute(b"TPUReplicate", len(output_types),
                               inputs=_inputs_flat, attrs=_attrs, ctx=_ctx,
                               name=name)
  _execute.record_gradient(
      "TPUReplicate", _inputs_flat, _attrs, _result, name)
  return _result

_ops.RegisterShape("TPUReplicate")(None)


def tpu_replicated_input(inputs, name=None):
  r"""Operator that connects N unreplicated inputs to an N-way replicated TPU computation.

  Args:
    inputs: A list of at least 1 `Tensor` objects with the same type.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `inputs`.
  """
  if not isinstance(inputs, (list, tuple)):
    raise TypeError(
        "Expected list for 'inputs' argument to "
        "'tpu_replicated_input' Op, not %r." % inputs)
  _attr_N = len(inputs)
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "TPUReplicatedInput", inputs=inputs, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("N", _op.get_attr("N"), "T", _op.get_attr("T"))
  else:
    _attr_T, inputs = _execute.args_to_matching_eager(list(inputs), _ctx)
    _attr_T = _attr_T.as_datatype_enum
    _inputs_flat = list(inputs)
    _attrs = ("N", _attr_N, "T", _attr_T)
    _result = _execute.execute(b"TPUReplicatedInput", 1, inputs=_inputs_flat,
                               attrs=_attrs, ctx=_ctx, name=name)
  _execute.record_gradient(
      "TPUReplicatedInput", _inputs_flat, _attrs, _result, name)
  _result, = _result
  return _result

_ops.RegisterShape("TPUReplicatedInput")(None)


def tpu_replicated_output(input, num_replicas, name=None):
  r"""Operator that connects the output of an N-way replicated TPU computation to N separate outputs.

  Args:
    input: A `Tensor`.
    num_replicas: An `int` that is `>= 1`.
    name: A name for the operation (optional).

  Returns:
    A list of `num_replicas` `Tensor` objects with the same type as `input`.
  """
  num_replicas = _execute.make_int(num_replicas, "num_replicas")
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "TPUReplicatedOutput", input=input, num_replicas=num_replicas,
        name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("num_replicas", _op.get_attr("num_replicas"), "T",
              _op.get_attr("T"))
  else:
    _attr_T, (input,) = _execute.args_to_matching_eager([input], _ctx)
    _attr_T = _attr_T.as_datatype_enum
    _inputs_flat = [input]
    _attrs = ("num_replicas", num_replicas, "T", _attr_T)
    _result = _execute.execute(b"TPUReplicatedOutput", num_replicas,
                               inputs=_inputs_flat, attrs=_attrs, ctx=_ctx,
                               name=name)
  _execute.record_gradient(
      "TPUReplicatedOutput", _inputs_flat, _attrs, _result, name)
  return _result

_ops.RegisterShape("TPUReplicatedOutput")(None)

def _InitOpDefLibrary(op_list_proto_bytes):
  op_list = _op_def_pb2.OpList()
  op_list.ParseFromString(op_list_proto_bytes)
  _op_def_registry.register_op_list(op_list)
  op_def_lib = _op_def_library.OpDefLibrary()
  op_def_lib.add_op_list(op_list)
  return op_def_lib
# op {
#   name: "ConfigureDistributedTPU"
#   output_arg {
#     name: "global_tpu_array"
#     type: DT_INT32
#   }
#   attr {
#     name: "embedding_config"
#     type: "string"
#     default_value {
#       s: ""
#     }
#   }
#   is_stateful: true
# }
# op {
#   name: "CrossReplicaSum"
#   input_arg {
#     name: "input"
#     type_attr: "T"
#   }
#   output_arg {
#     name: "output"
#     type_attr: "T"
#   }
#   attr {
#     name: "T"
#     type: "type"
#     allowed_values {
#       list {
#         type: DT_FLOAT
#       }
#     }
#   }
# }
# op {
#   name: "InfeedDequeue"
#   output_arg {
#     name: "output"
#     type_attr: "dtype"
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
#   name: "InfeedDequeueTuple"
#   output_arg {
#     name: "outputs"
#     type_list_attr: "dtypes"
#   }
#   attr {
#     name: "dtypes"
#     type: "list(type)"
#     has_minimum: true
#     minimum: 1
#   }
#   attr {
#     name: "shapes"
#     type: "list(shape)"
#   }
#   is_stateful: true
# }
# op {
#   name: "InfeedEnqueue"
#   input_arg {
#     name: "input"
#     type_attr: "dtype"
#   }
#   attr {
#     name: "dtype"
#     type: "type"
#   }
#   attr {
#     name: "shape"
#     type: "shape"
#     default_value {
#       shape {
#       }
#     }
#   }
#   attr {
#     name: "device_ordinal"
#     type: "int"
#     default_value {
#       i: -1
#     }
#   }
#   is_stateful: true
# }
# op {
#   name: "InfeedEnqueueTuple"
#   input_arg {
#     name: "inputs"
#     type_list_attr: "dtypes"
#   }
#   attr {
#     name: "dtypes"
#     type: "list(type)"
#     has_minimum: true
#     minimum: 1
#   }
#   attr {
#     name: "shapes"
#     type: "list(shape)"
#   }
#   attr {
#     name: "device_ordinal"
#     type: "int"
#     default_value {
#       i: -1
#     }
#   }
#   is_stateful: true
# }
# op {
#   name: "OutfeedDequeue"
#   output_arg {
#     name: "output"
#     type_attr: "dtype"
#   }
#   attr {
#     name: "dtype"
#     type: "type"
#   }
#   attr {
#     name: "shape"
#     type: "shape"
#   }
#   attr {
#     name: "device_ordinal"
#     type: "int"
#     default_value {
#       i: -1
#     }
#   }
#   is_stateful: true
# }
# op {
#   name: "OutfeedDequeueTuple"
#   output_arg {
#     name: "outputs"
#     type_list_attr: "dtypes"
#   }
#   attr {
#     name: "dtypes"
#     type: "list(type)"
#     has_minimum: true
#     minimum: 1
#   }
#   attr {
#     name: "shapes"
#     type: "list(shape)"
#   }
#   attr {
#     name: "device_ordinal"
#     type: "int"
#     default_value {
#       i: -1
#     }
#   }
#   is_stateful: true
# }
# op {
#   name: "OutfeedEnqueue"
#   input_arg {
#     name: "input"
#     type_attr: "dtype"
#   }
#   attr {
#     name: "dtype"
#     type: "type"
#   }
#   is_stateful: true
# }
# op {
#   name: "OutfeedEnqueueTuple"
#   input_arg {
#     name: "inputs"
#     type_list_attr: "dtypes"
#   }
#   attr {
#     name: "dtypes"
#     type: "list(type)"
#     has_minimum: true
#     minimum: 1
#   }
#   is_stateful: true
# }
# op {
#   name: "ShutdownDistributedTPU"
#   is_stateful: true
# }
# op {
#   name: "TPUReplicate"
#   input_arg {
#     name: "inputs"
#     type_list_attr: "Tinputs"
#   }
#   input_arg {
#     name: "broadcast_inputs"
#     type_list_attr: "Tbroadcast_inputs"
#   }
#   input_arg {
#     name: "variables"
#     type: DT_RESOURCE
#     number_attr: "NumVariables"
#   }
#   output_arg {
#     name: "outputs"
#     type_list_attr: "output_types"
#   }
#   attr {
#     name: "computation"
#     type: "func"
#   }
#   attr {
#     name: "num_replicas"
#     type: "int"
#     has_minimum: true
#     minimum: 1
#   }
#   attr {
#     name: "global_tpu_id"
#     type: "list(int)"
#     default_value {
#       list {
#       }
#     }
#   }
#   attr {
#     name: "Tinputs"
#     type: "list(type)"
#     has_minimum: true
#   }
#   attr {
#     name: "Tbroadcast_inputs"
#     type: "list(type)"
#     has_minimum: true
#   }
#   attr {
#     name: "NumVariables"
#     type: "int"
#     has_minimum: true
#   }
#   attr {
#     name: "output_types"
#     type: "list(type)"
#     has_minimum: true
#   }
#   is_stateful: true
# }
# op {
#   name: "TPUReplicatedInput"
#   input_arg {
#     name: "inputs"
#     type_attr: "T"
#     number_attr: "N"
#   }
#   output_arg {
#     name: "output"
#     type_attr: "T"
#   }
#   attr {
#     name: "N"
#     type: "int"
#     has_minimum: true
#     minimum: 1
#   }
#   attr {
#     name: "T"
#     type: "type"
#   }
# }
# op {
#   name: "TPUReplicatedOutput"
#   input_arg {
#     name: "input"
#     type_attr: "T"
#   }
#   output_arg {
#     name: "outputs"
#     type_attr: "T"
#     number_attr: "num_replicas"
#   }
#   attr {
#     name: "num_replicas"
#     type: "int"
#     has_minimum: true
#     minimum: 1
#   }
#   attr {
#     name: "T"
#     type: "type"
#   }
# }
_op_def_lib = _InitOpDefLibrary(b"\nR\n\027ConfigureDistributedTPU\032\024\n\020global_tpu_array\030\003\"\036\n\020embedding_config\022\006string\032\002\022\000\210\001\001\n<\n\017CrossReplicaSum\022\n\n\005input\"\001T\032\013\n\006output\"\001T\"\020\n\001T\022\004type:\005\n\0032\001\001\nB\n\rInfeedDequeue\032\017\n\006output\"\005dtype\"\r\n\005dtype\022\004type\"\016\n\005shape\022\005shape\210\001\001\n[\n\022InfeedDequeueTuple\032\021\n\007outputs2\006dtypes\"\030\n\006dtypes\022\nlist(type)(\0010\001\"\025\n\006shapes\022\013list(shape)\210\001\001\ni\n\rInfeedEnqueue\022\016\n\005input\"\005dtype\"\r\n\005dtype\022\004type\"\022\n\005shape\022\005shape\032\002:\000\"\"\n\016device_ordinal\022\003int\032\013\030\377\377\377\377\377\377\377\377\377\001\210\001\001\n~\n\022InfeedEnqueueTuple\022\020\n\006inputs2\006dtypes\"\030\n\006dtypes\022\nlist(type)(\0010\001\"\025\n\006shapes\022\013list(shape)\"\"\n\016device_ordinal\022\003int\032\013\030\377\377\377\377\377\377\377\377\377\001\210\001\001\ng\n\016OutfeedDequeue\032\017\n\006output\"\005dtype\"\r\n\005dtype\022\004type\"\016\n\005shape\022\005shape\"\"\n\016device_ordinal\022\003int\032\013\030\377\377\377\377\377\377\377\377\377\001\210\001\001\n\200\001\n\023OutfeedDequeueTuple\032\021\n\007outputs2\006dtypes\"\030\n\006dtypes\022\nlist(type)(\0010\001\"\025\n\006shapes\022\013list(shape)\"\"\n\016device_ordinal\022\003int\032\013\030\377\377\377\377\377\377\377\377\377\001\210\001\001\n2\n\016OutfeedEnqueue\022\016\n\005input\"\005dtype\"\r\n\005dtype\022\004type\210\001\001\nD\n\023OutfeedEnqueueTuple\022\020\n\006inputs2\006dtypes\"\030\n\006dtypes\022\nlist(type)(\0010\001\210\001\001\n\033\n\026ShutdownDistributedTPU\210\001\001\n\300\002\n\014TPUReplicate\022\021\n\006inputs2\007Tinputs\022%\n\020broadcast_inputs2\021Tbroadcast_inputs\022\033\n\tvariables\030\024*\014NumVariables\032\027\n\007outputs2\014output_types\"\023\n\013computation\022\004func\"\027\n\014num_replicas\022\003int(\0010\001\"\036\n\rglobal_tpu_id\022\tlist(int)\032\002\n\000\"\027\n\007Tinputs\022\nlist(type)(\001\"!\n\021Tbroadcast_inputs\022\nlist(type)(\001\"\025\n\014NumVariables\022\003int(\001\"\034\n\014output_types\022\nlist(type)(\001\210\001\001\nJ\n\022TPUReplicatedInput\022\016\n\006inputs\"\001T*\001N\032\013\n\006output\"\001T\"\014\n\001N\022\003int(\0010\001\"\t\n\001T\022\004type\na\n\023TPUReplicatedOutput\022\n\n\005input\"\001T\032\032\n\007outputs\"\001T*\014num_replicas\"\027\n\014num_replicas\022\003int(\0010\001\"\t\n\001T\022\004type")
