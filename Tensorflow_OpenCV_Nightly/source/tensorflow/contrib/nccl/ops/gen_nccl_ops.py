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

def nccl_all_reduce(input, reduction, num_devices, shared_name, name=None):
  r"""Outputs a tensor containing the reduction across all input tensors passed to ops

  within the same `shared_name.

  The graph should be constructed so if one op runs with shared_name value `c`,
  then `num_devices` ops will run with shared_name value `c`.  Failure to do so
  will cause the graph execution to fail to complete.

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`.
      the input to the reduction
    reduction: A `string` from: `"min", "max", "prod", "sum"`.
      the reduction operation to perform.
    num_devices: An `int`.
      The number of devices participating in this reduction.
    shared_name: A `string`.
      Identifier that shared between ops of the same reduction.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
    the value of the reduction across all `num_devices` devices.
  """
  result = _op_def_lib.apply_op("NcclAllReduce", input=input,
                                reduction=reduction, num_devices=num_devices,
                                shared_name=shared_name, name=name)
  return result


_ops.RegisterShape("NcclAllReduce")(None)

def nccl_broadcast_recv(shape, T, num_devices, shared_name, name=None):
  r"""Sends data of shape `shape` from the NcclBroadcastSend op registered in the

  same `shared_name`.

  The graph should be constructed so that one device runs `NcclBroadcastSend` and
  `num_devices-1` devices run NcclBroadcastRecv ops with shared_name value `c`.
  Failure to do so will cause the graph execution to fail to complete.

  Args:
    shape: A `Tensor` of type `int64`. The shape of the output.
    T: A `tf.DType` from: `tf.float32, tf.float64, tf.int32, tf.int64`.
    num_devices: An `int`.
      The number of devices participating in this reduction.
    shared_name: A `string`.
      Identifier that is shared between ops of the same broadcast.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `T`.
    The broadcast data received from the NcclBroadcastSend op.
  """
  result = _op_def_lib.apply_op("NcclBroadcastRecv", shape=shape, T=T,
                                num_devices=num_devices,
                                shared_name=shared_name, name=name)
  return result


_ops.RegisterShape("NcclBroadcastRecv")(None)

def nccl_broadcast_send(input, num_devices, shared_name, name=None):
  r"""Sends `input` to the NcclBroadcastRecv ops registered in the same `shared_name`.

  The graph should be constructed so that one device runs `NcclBroadcastSend` and
  `num_devices-1` devices run NcclBroadcastRecv ops with shared_name value `c`.
  Failure to do so will cause the graph execution to fail to complete.

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`.
      The input to the broadcast
    num_devices: An `int`.
      The number of devices participating in this reduction.
    shared_name: A `string`.
      Identifier that is shared between ops of the same broadcast.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  result = _op_def_lib.apply_op("NcclBroadcastSend", input=input,
                                num_devices=num_devices,
                                shared_name=shared_name, name=name)
  return result


_ops.RegisterShape("NcclBroadcastSend")(None)
def _InitOpDefLibrary():
  op_list = _op_def_pb2.OpList()
  _text_format.Merge(_InitOpDefLibrary.op_list_ascii, op_list)
  _op_def_registry.register_op_list(op_list)
  op_def_lib = _op_def_library.OpDefLibrary()
  op_def_lib.add_op_list(op_list)
  return op_def_lib


_InitOpDefLibrary.op_list_ascii = """op {
  name: "NcclAllReduce"
  input_arg {
    name: "input"
    type_attr: "T"
  }
  output_arg {
    name: "data"
    type_attr: "T"
  }
  attr {
    name: "reduction"
    type: "string"
    allowed_values {
      list {
        s: "min"
        s: "max"
        s: "prod"
        s: "sum"
      }
    }
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT32
        type: DT_INT64
      }
    }
  }
  attr {
    name: "num_devices"
    type: "int"
  }
  attr {
    name: "shared_name"
    type: "string"
  }
  is_stateful: true
}
op {
  name: "NcclBroadcastRecv"
  input_arg {
    name: "shape"
    type: DT_INT64
  }
  output_arg {
    name: "output"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT32
        type: DT_INT64
      }
    }
  }
  attr {
    name: "num_devices"
    type: "int"
  }
  attr {
    name: "shared_name"
    type: "string"
  }
  is_stateful: true
}
op {
  name: "NcclBroadcastSend"
  input_arg {
    name: "input"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT32
        type: DT_INT64
      }
    }
  }
  attr {
    name: "num_devices"
    type: "int"
  }
  attr {
    name: "shared_name"
    type: "string"
  }
  is_stateful: true
}
"""


_op_def_lib = _InitOpDefLibrary()
