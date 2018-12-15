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

def stateless_random_normal(shape, seed, dtype=None, name=None):
  r"""Outputs deterministic pseudorandom values from a normal distribution.

  The generated values will have mean 0 and standard deviation 1.

  The outputs are a deterministic function of `shape` and `seed`.

  Args:
    shape: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      The shape of the output tensor.
    seed: A `Tensor` of type `int64`. 2 seeds (shape [2]).
    dtype: An optional `tf.DType` from: `tf.half, tf.float32, tf.float64`. Defaults to `tf.float32`.
      The type of the output.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `dtype`. Random values with specified shape.
  """
  result = _op_def_lib.apply_op("StatelessRandomNormal", shape=shape,
                                seed=seed, dtype=dtype, name=name)
  return result


_ops.RegisterShape("StatelessRandomNormal")(None)

def stateless_random_uniform(shape, seed, dtype=None, name=None):
  r"""Outputs deterministic pseudorandom random values from a uniform distribution.

  The generated values follow a uniform distribution in the range `[0, 1)`. The
  lower bound 0 is included in the range, while the upper bound 1 is excluded.

  The outputs are a deterministic function of `shape` and `seed`.

  Args:
    shape: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      The shape of the output tensor.
    seed: A `Tensor` of type `int64`. 2 seeds (shape [2]).
    dtype: An optional `tf.DType` from: `tf.half, tf.float32, tf.float64`. Defaults to `tf.float32`.
      The type of the output.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `dtype`. Random values with specified shape.
  """
  result = _op_def_lib.apply_op("StatelessRandomUniform", shape=shape,
                                seed=seed, dtype=dtype, name=name)
  return result


_ops.RegisterShape("StatelessRandomUniform")(None)

def stateless_truncated_normal(shape, seed, dtype=None, name=None):
  r"""Outputs deterministic pseudorandom values from a truncated normal distribution.

  The generated values follow a normal distribution with mean 0 and standard
  deviation 1, except that values whose magnitude is more than 2 standard
  deviations from the mean are dropped and re-picked.

  The outputs are a deterministic function of `shape` and `seed`.

  Args:
    shape: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      The shape of the output tensor.
    seed: A `Tensor` of type `int64`. 2 seeds (shape [2]).
    dtype: An optional `tf.DType` from: `tf.half, tf.float32, tf.float64`. Defaults to `tf.float32`.
      The type of the output.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `dtype`. Random values with specified shape.
  """
  result = _op_def_lib.apply_op("StatelessTruncatedNormal", shape=shape,
                                seed=seed, dtype=dtype, name=name)
  return result


_ops.RegisterShape("StatelessTruncatedNormal")(None)
def _InitOpDefLibrary():
  op_list = _op_def_pb2.OpList()
  _text_format.Merge(_InitOpDefLibrary.op_list_ascii, op_list)
  _op_def_registry.register_op_list(op_list)
  op_def_lib = _op_def_library.OpDefLibrary()
  op_def_lib.add_op_list(op_list)
  return op_def_lib


_InitOpDefLibrary.op_list_ascii = """op {
  name: "StatelessRandomNormal"
  input_arg {
    name: "shape"
    type_attr: "T"
  }
  input_arg {
    name: "seed"
    type: DT_INT64
  }
  output_arg {
    name: "output"
    type_attr: "dtype"
  }
  attr {
    name: "dtype"
    type: "type"
    default_value {
      type: DT_FLOAT
    }
    allowed_values {
      list {
        type: DT_HALF
        type: DT_FLOAT
        type: DT_DOUBLE
      }
    }
  }
  attr {
    name: "T"
    type: "type"
    default_value {
      type: DT_INT32
    }
    allowed_values {
      list {
        type: DT_INT32
        type: DT_INT64
      }
    }
  }
}
op {
  name: "StatelessRandomUniform"
  input_arg {
    name: "shape"
    type_attr: "T"
  }
  input_arg {
    name: "seed"
    type: DT_INT64
  }
  output_arg {
    name: "output"
    type_attr: "dtype"
  }
  attr {
    name: "dtype"
    type: "type"
    default_value {
      type: DT_FLOAT
    }
    allowed_values {
      list {
        type: DT_HALF
        type: DT_FLOAT
        type: DT_DOUBLE
      }
    }
  }
  attr {
    name: "T"
    type: "type"
    default_value {
      type: DT_INT32
    }
    allowed_values {
      list {
        type: DT_INT32
        type: DT_INT64
      }
    }
  }
}
op {
  name: "StatelessTruncatedNormal"
  input_arg {
    name: "shape"
    type_attr: "T"
  }
  input_arg {
    name: "seed"
    type: DT_INT64
  }
  output_arg {
    name: "output"
    type_attr: "dtype"
  }
  attr {
    name: "dtype"
    type: "type"
    default_value {
      type: DT_FLOAT
    }
    allowed_values {
      list {
        type: DT_HALF
        type: DT_FLOAT
        type: DT_DOUBLE
      }
    }
  }
  attr {
    name: "T"
    type: "type"
    default_value {
      type: DT_INT32
    }
    allowed_values {
      list {
        type: DT_INT32
        type: DT_INT64
      }
    }
  }
}
"""


_op_def_lib = _InitOpDefLibrary()
