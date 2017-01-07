"""Python wrappers around Brain.

This file is MACHINE GENERATED! Do not edit.
"""

import collections

from google.protobuf import text_format

from tensorflow.core.framework import op_def_pb2
from tensorflow.python.framework import op_def_registry
from tensorflow.python.framework import ops
from tensorflow.python.ops import op_def_library


__assert_outputs = [""]


def _assert(condition, data, summarize=None, name=None):
  r"""Asserts that the given condition is true.

  If `condition` evaluates to false, print the list of tensors in `data`.
  `summarize` determines how many entries of the tensors to print.

  Args:
    condition: A `Tensor` of type `bool`. The condition to evaluate.
    data: A list of `Tensor` objects.
      The tensors to print out when condition is false.
    summarize: An optional `int`. Defaults to `3`.
      Print this many entries of each tensor.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  result = _op_def_lib.apply_op("Assert", condition=condition, data=data,
                                summarize=summarize, name=name)
  return result


__audio_summary_outputs = ["summary"]


def _audio_summary(tag, tensor, sample_rate, max_outputs=None, name=None):
  r"""Outputs a `Summary` protocol buffer with audio.

  The summary has up to `max_outputs` summary values containing audio. The
  audio is built from `tensor` which must be 3-D with shape `[batch_size,
  frames, channels]` or 2-D with shape `[batch_size, frames]`. The values are
  assumed to be in the range of `[-1.0, 1.0]` with a sample rate of `sample_rate`.

  The `tag` argument is a scalar `Tensor` of type `string`.  It is used to
  build the `tag` of the summary values:

  *  If `max_outputs` is 1, the summary value tag is '*tag*/audio'.
  *  If `max_outputs` is greater than 1, the summary value tags are
     generated sequentially as '*tag*/audio/0', '*tag*/audio/1', etc.

  Args:
    tag: A `Tensor` of type `string`.
      Scalar. Used to build the `tag` attribute of the summary values.
    tensor: A `Tensor` of type `float32`. 2-D of shape `[batch_size, frames]`.
    sample_rate: A `float`. The sample rate of the signal in hertz.
    max_outputs: An optional `int` that is `>= 1`. Defaults to `3`.
      Max number of batch elements to generate audio for.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`. Scalar. Serialized `Summary` protocol buffer.
  """
  result = _op_def_lib.apply_op("AudioSummary", tag=tag, tensor=tensor,
                                sample_rate=sample_rate,
                                max_outputs=max_outputs, name=name)
  return result


__histogram_summary_outputs = ["summary"]


def _histogram_summary(tag, values, name=None):
  r"""Outputs a `Summary` protocol buffer with a histogram.

  The generated
  [`Summary`](https://www.tensorflow.org/code/tensorflow/core/framework/summary.proto)
  has one summary value containing a histogram for `values`.

  This op reports an `InvalidArgument` error if any value is not finite.

  Args:
    tag: A `Tensor` of type `string`.
      Scalar.  Tag to use for the `Summary.Value`.
    values: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`, `uint8`, `int16`, `int8`, `uint16`, `half`.
      Any shape. Values to use to build the histogram.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`. Scalar. Serialized `Summary` protocol buffer.
  """
  result = _op_def_lib.apply_op("HistogramSummary", tag=tag, values=values,
                                name=name)
  return result


__image_summary_outputs = ["summary"]


def _image_summary(tag, tensor, max_images=None, bad_color=None, name=None):
  r"""Outputs a `Summary` protocol buffer with images.

  The summary has up to `max_images` summary values containing images. The
  images are built from `tensor` which must be 4-D with shape `[batch_size,
  height, width, channels]` and where `channels` can be:

  *  1: `tensor` is interpreted as Grayscale.
  *  3: `tensor` is interpreted as RGB.
  *  4: `tensor` is interpreted as RGBA.

  The images have the same number of channels as the input tensor. For float
  input, the values are normalized one image at a time to fit in the range
  `[0, 255]`.  `uint8` values are unchanged.  The op uses two different
  normalization algorithms:

  *  If the input values are all positive, they are rescaled so the largest one
     is 255.

  *  If any input value is negative, the values are shifted so input value 0.0
     is at 127.  They are then rescaled so that either the smallest value is 0,
     or the largest one is 255.

  The `tag` argument is a scalar `Tensor` of type `string`.  It is used to
  build the `tag` of the summary values:

  *  If `max_images` is 1, the summary value tag is '*tag*/image'.
  *  If `max_images` is greater than 1, the summary value tags are
     generated sequentially as '*tag*/image/0', '*tag*/image/1', etc.

  The `bad_color` argument is the color to use in the generated images for
  non-finite input values.  It is a `unit8` 1-D tensor of length `channels`.
  Each element must be in the range `[0, 255]` (It represents the value of a
  pixel in the output image).  Non-finite values in the input tensor are
  replaced by this tensor in the output image.  The default value is the color
  red.

  Args:
    tag: A `Tensor` of type `string`.
      Scalar. Used to build the `tag` attribute of the summary values.
    tensor: A `Tensor`. Must be one of the following types: `uint8`, `float32`, `half`.
      4-D of shape `[batch_size, height, width, channels]` where
      `channels` is 1, 3, or 4.
    max_images: An optional `int` that is `>= 1`. Defaults to `3`.
      Max number of batch elements to generate images for.
    bad_color: . Defaults to `[]`.
      Color to use for pixels with non-finite values.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`. Scalar. Serialized `Summary` protocol buffer.
  """
  result = _op_def_lib.apply_op("ImageSummary", tag=tag, tensor=tensor,
                                max_images=max_images, bad_color=bad_color,
                                name=name)
  return result


__merge_summary_outputs = ["summary"]


def _merge_summary(inputs, name=None):
  r"""Merges summaries.

  This op creates a
  [`Summary`](https://www.tensorflow.org/code/tensorflow/core/framework/summary.proto)
  protocol buffer that contains the union of all the values in the input
  summaries.

  When the Op is run, it reports an `InvalidArgument` error if multiple values
  in the summaries to merge use the same tag.

  Args:
    inputs: A list of at least 1 `Tensor` objects of type `string`.
      Can be of any shape.  Each must contain serialized `Summary` protocol
      buffers.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`. Scalar. Serialized `Summary` protocol buffer.
  """
  result = _op_def_lib.apply_op("MergeSummary", inputs=inputs, name=name)
  return result


__print_outputs = ["output"]


def _print(input, data, message=None, first_n=None, summarize=None,
           name=None):
  r"""Prints a list of tensors.

  Passes `input` through to `output` and prints `data` when evaluating.

  Args:
    input: A `Tensor`. The tensor passed to `output`
    data: A list of `Tensor` objects.
      A list of tensors to print out when op is evaluated.
    message: An optional `string`. Defaults to `""`.
      A string, prefix of the error message.
    first_n: An optional `int`. Defaults to `-1`.
      Only log `first_n` number of times. -1 disables logging.
    summarize: An optional `int`. Defaults to `3`.
      Only print this many entries of each tensor.
    name: A name for the operation (optional).

  Returns:
    The unmodified `input` tensor
  """
  result = _op_def_lib.apply_op("Print", input=input, data=data,
                                message=message, first_n=first_n,
                                summarize=summarize, name=name)
  return result


__scalar_summary_outputs = ["summary"]


def _scalar_summary(tags, values, name=None):
  r"""Outputs a `Summary` protocol buffer with scalar values.

  The input `tags` and `values` must have the same shape.  The generated summary
  has a summary value for each tag-value pair in `tags` and `values`.

  Args:
    tags: A `Tensor` of type `string`. Tags for the summary.
    values: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`, `uint8`, `int16`, `int8`, `uint16`, `half`.
      Same shape as `tags.  Values for the summary.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`.
    Scalar.  Serialized `Summary` protocol buffer.
  """
  result = _op_def_lib.apply_op("ScalarSummary", tags=tags, values=values,
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
  name: "Assert"
  input_arg {
    name: "condition"
    type: DT_BOOL
  }
  input_arg {
    name: "data"
    type_list_attr: "T"
  }
  attr {
    name: "T"
    type: "list(type)"
    has_minimum: true
    minimum: 1
  }
  attr {
    name: "summarize"
    type: "int"
    default_value {
      i: 3
    }
  }
  is_stateful: true
}
op {
  name: "AudioSummary"
  input_arg {
    name: "tag"
    type: DT_STRING
  }
  input_arg {
    name: "tensor"
    type: DT_FLOAT
  }
  output_arg {
    name: "summary"
    type: DT_STRING
  }
  attr {
    name: "sample_rate"
    type: "float"
  }
  attr {
    name: "max_outputs"
    type: "int"
    default_value {
      i: 3
    }
    has_minimum: true
    minimum: 1
  }
}
op {
  name: "HistogramSummary"
  input_arg {
    name: "tag"
    type: DT_STRING
  }
  input_arg {
    name: "values"
    type_attr: "T"
  }
  output_arg {
    name: "summary"
    type: DT_STRING
  }
  attr {
    name: "T"
    type: "type"
    default_value {
      type: DT_FLOAT
    }
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
}
op {
  name: "ImageSummary"
  input_arg {
    name: "tag"
    type: DT_STRING
  }
  input_arg {
    name: "tensor"
    type_attr: "T"
  }
  output_arg {
    name: "summary"
    type: DT_STRING
  }
  attr {
    name: "max_images"
    type: "int"
    default_value {
      i: 3
    }
    has_minimum: true
    minimum: 1
  }
  attr {
    name: "T"
    type: "type"
    default_value {
      type: DT_FLOAT
    }
    allowed_values {
      list {
        type: DT_UINT8
        type: DT_FLOAT
        type: DT_HALF
      }
    }
  }
  attr {
    name: "bad_color"
    type: "tensor"
    default_value {
      tensor {
        dtype: DT_UINT8
        tensor_shape {
          dim {
            size: 4
          }
        }
        int_val: 255
        int_val: 0
        int_val: 0
        int_val: 255
      }
    }
  }
}
op {
  name: "MergeSummary"
  input_arg {
    name: "inputs"
    type: DT_STRING
    number_attr: "N"
  }
  output_arg {
    name: "summary"
    type: DT_STRING
  }
  attr {
    name: "N"
    type: "int"
    has_minimum: true
    minimum: 1
  }
}
op {
  name: "Print"
  input_arg {
    name: "input"
    type_attr: "T"
  }
  input_arg {
    name: "data"
    type_list_attr: "U"
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
    name: "U"
    type: "list(type)"
    has_minimum: true
    minimum: 1
  }
  attr {
    name: "message"
    type: "string"
    default_value {
      s: ""
    }
  }
  attr {
    name: "first_n"
    type: "int"
    default_value {
      i: -1
    }
  }
  attr {
    name: "summarize"
    type: "int"
    default_value {
      i: 3
    }
  }
  is_stateful: true
}
op {
  name: "ScalarSummary"
  input_arg {
    name: "tags"
    type: DT_STRING
  }
  input_arg {
    name: "values"
    type_attr: "T"
  }
  output_arg {
    name: "summary"
    type: DT_STRING
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
}
"""


_op_def_lib = _InitOpDefLibrary()
