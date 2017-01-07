"""Python wrappers around Brain.

This file is MACHINE GENERATED! Do not edit.
"""

import collections

from google.protobuf import text_format

from tensorflow.core.framework import op_def_pb2
from tensorflow.python.framework import op_def_registry
from tensorflow.python.framework import ops
from tensorflow.python.ops import op_def_library


_multinomial_outputs = ["output"]


def multinomial(logits, num_samples, seed=None, seed2=None, name=None):
  r"""Draws samples from a multinomial distribution.

  Args:
    logits: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`, `uint8`, `int16`, `int8`, `uint16`, `half`.
      2-D Tensor with shape `[batch_size, num_classes]`.  Each slice `[i, :]`
      represents the unnormalized log probabilities for all classes.
    num_samples: A `Tensor` of type `int32`.
      0-D.  Number of independent samples to draw for each row slice.
    seed: An optional `int`. Defaults to `0`.
      If either seed or seed2 is set to be non-zero, the internal random number
      generator is seeded by the given seed.  Otherwise, a random seed is used.
    seed2: An optional `int`. Defaults to `0`.
      A second seed to avoid seed collision.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int64`.
    2-D Tensor with shape `[batch_size, num_samples]`.  Each slice `[i, :]`
    contains the drawn class labels with range `[0, num_classes)`.
  """
  result = _op_def_lib.apply_op("Multinomial", logits=logits,
                                num_samples=num_samples, seed=seed,
                                seed2=seed2, name=name)
  return result


__random_shuffle_outputs = ["output"]


def _random_shuffle(value, seed=None, seed2=None, name=None):
  r"""Randomly shuffles a tensor along its first dimension.

    The tensor is shuffled along dimension 0, such that each `value[j]` is mapped
    to one and only one `output[i]`. For example, a mapping that might occur for a
    3x2 tensor is:

  ```prettyprint
  [[1, 2],       [[5, 6],
   [3, 4],  ==>   [1, 2],
   [5, 6]]        [3, 4]]
  ```

  Args:
    value: A `Tensor`. The tensor to be shuffled.
    seed: An optional `int`. Defaults to `0`.
      If either `seed` or `seed2` are set to be non-zero, the random number
      generator is seeded by the given seed.  Otherwise, it is seeded by a
      random seed.
    seed2: An optional `int`. Defaults to `0`.
      A second seed to avoid seed collision.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `value`.
    A tensor of same shape and type as `value`, shuffled along its first
    dimension.
  """
  result = _op_def_lib.apply_op("RandomShuffle", value=value, seed=seed,
                                seed2=seed2, name=name)
  return result


__random_standard_normal_outputs = ["output"]


def _random_standard_normal(shape, dtype, seed=None, seed2=None, name=None):
  r"""Outputs random values from a normal distribution.

  The generated values will have mean 0 and standard deviation 1.

  Args:
    shape: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      The shape of the output tensor.
    dtype: A `tf.DType` from: `tf.half, tf.float32, tf.float64`.
      The type of the output.
    seed: An optional `int`. Defaults to `0`.
      If either `seed` or `seed2` are set to be non-zero, the random number
      generator is seeded by the given seed.  Otherwise, it is seeded by a
      random seed.
    seed2: An optional `int`. Defaults to `0`.
      A second seed to avoid seed collision.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `dtype`.
    A tensor of the specified shape filled with random normal values.
  """
  result = _op_def_lib.apply_op("RandomStandardNormal", shape=shape,
                                dtype=dtype, seed=seed, seed2=seed2,
                                name=name)
  return result


__random_uniform_outputs = ["output"]


def _random_uniform(shape, dtype, seed=None, seed2=None, name=None):
  r"""Outputs random values from a uniform distribution.

  The generated values follow a uniform distribution in the range `[0, 1)`. The
  lower bound 0 is included in the range, while the upper bound 1 is excluded.

  Args:
    shape: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      The shape of the output tensor.
    dtype: A `tf.DType` from: `tf.half, tf.float32, tf.float64`.
      The type of the output.
    seed: An optional `int`. Defaults to `0`.
      If either `seed` or `seed2` are set to be non-zero, the random number
      generator is seeded by the given seed.  Otherwise, it is seeded by a
      random seed.
    seed2: An optional `int`. Defaults to `0`.
      A second seed to avoid seed collision.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `dtype`.
    A tensor of the specified shape filled with uniform random values.
  """
  result = _op_def_lib.apply_op("RandomUniform", shape=shape, dtype=dtype,
                                seed=seed, seed2=seed2, name=name)
  return result


__random_uniform_int_outputs = ["output"]


def _random_uniform_int(shape, minval, maxval, seed=None, seed2=None,
                        name=None):
  r"""Outputs random integers from a uniform distribution.

  The generated values are uniform integers in the range `[minval, maxval)`.
  The lower bound `minval` is included in the range, while the upper bound
  `maxval` is excluded.

  The random integers are slightly biased unless `maxval - minval` is an exact
  power of two.  The bias is small for values of `maxval - minval` significantly
  smaller than the range of the output (either `2^32` or `2^64`).

  Args:
    shape: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      The shape of the output tensor.
    minval: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      0-D.  Inclusive lower bound on the generated integers.
    maxval: A `Tensor`. Must have the same type as `minval`.
      0-D.  Exclusive upper bound on the generated integers.
    seed: An optional `int`. Defaults to `0`.
      If either `seed` or `seed2` are set to be non-zero, the random number
      generator is seeded by the given seed.  Otherwise, it is seeded by a
      random seed.
    seed2: An optional `int`. Defaults to `0`.
      A second seed to avoid seed collision.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `minval`.
    A tensor of the specified shape filled with uniform random integers.
  """
  result = _op_def_lib.apply_op("RandomUniformInt", shape=shape,
                                minval=minval, maxval=maxval, seed=seed,
                                seed2=seed2, name=name)
  return result


__truncated_normal_outputs = ["output"]


def _truncated_normal(shape, dtype, seed=None, seed2=None, name=None):
  r"""Outputs random values from a truncated normal distribution.

  The generated values follow a normal distribution with mean 0 and standard
  deviation 1, except that values whose magnitude is more than 2 standard
  deviations from the mean are dropped and re-picked.

  Args:
    shape: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      The shape of the output tensor.
    dtype: A `tf.DType` from: `tf.half, tf.float32, tf.float64`.
      The type of the output.
    seed: An optional `int`. Defaults to `0`.
      If either `seed` or `seed2` are set to be non-zero, the random number
      generator is seeded by the given seed.  Otherwise, it is seeded by a
      random seed.
    seed2: An optional `int`. Defaults to `0`.
      A second seed to avoid seed collision.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `dtype`.
    A tensor of the specified shape filled with random truncated normal
    values.
  """
  result = _op_def_lib.apply_op("TruncatedNormal", shape=shape, dtype=dtype,
                                seed=seed, seed2=seed2, name=name)
  return result


def _InitOpDefLibrary():
  op_list = op_def_pb2.OpList()
  text_format.Merge(_InitOpDefLibrary.op_list_ascii, op_list)
  op_def_registry.register_op_list(op_list)
  op_def_lib = op_def_library.OpDefLibrary()
  op_def_lib.add_op_list(op_list)
  return op_def_lib


_InitOpDefLibrary.op_list_ascii = """op {
  name: "Multinomial"
  input_arg {
    name: "logits"
    type_attr: "T"
  }
  input_arg {
    name: "num_samples"
    type: DT_INT32
  }
  output_arg {
    name: "output"
    type: DT_INT64
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
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT32
        type: DT_INT64
        type: DT_UINT8
        type: DT_INT16
        type: DT_INT8
        type: DT_UINT16
        type: DT_HALF
      }
    }
  }
  is_stateful: true
}
op {
  name: "RandomShuffle"
  input_arg {
    name: "value"
    type_attr: "T"
  }
  output_arg {
    name: "output"
    type_attr: "T"
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
    name: "T"
    type: "type"
  }
  is_stateful: true
}
op {
  name: "RandomStandardNormal"
  input_arg {
    name: "shape"
    type_attr: "T"
  }
  output_arg {
    name: "output"
    type_attr: "dtype"
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
    name: "dtype"
    type: "type"
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
  name: "RandomUniform"
  input_arg {
    name: "shape"
    type_attr: "T"
  }
  output_arg {
    name: "output"
    type_attr: "dtype"
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
    name: "dtype"
    type: "type"
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
  name: "RandomUniformInt"
  input_arg {
    name: "shape"
    type_attr: "T"
  }
  input_arg {
    name: "minval"
    type_attr: "Tout"
  }
  input_arg {
    name: "maxval"
    type_attr: "Tout"
  }
  output_arg {
    name: "output"
    type_attr: "Tout"
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
    name: "Tout"
    type: "type"
    allowed_values {
      list {
        type: DT_INT32
        type: DT_INT64
      }
    }
  }
  attr {
    name: "T"
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
  name: "TruncatedNormal"
  input_arg {
    name: "shape"
    type_attr: "T"
  }
  output_arg {
    name: "output"
    type_attr: "dtype"
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
    name: "dtype"
    type: "type"
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
    allowed_values {
      list {
        type: DT_INT32
        type: DT_INT64
      }
    }
  }
  is_stateful: true
}
"""


_op_def_lib = _InitOpDefLibrary()
