"""Python wrappers around TensorFlow ops.

This file is MACHINE GENERATED! Do not edit.
Original C++ source file: gen_summary_ops.cc
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


def close_summary_writer(writer, name=None):
  r"""Flushes and closes the summary writer.

  Also removes it from the resource manager. To reopen, use another
  CreateSummaryFileWriter op.

  Args:
    writer: A `Tensor` of type `resource`.
      A handle to the summary writer resource.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "CloseSummaryWriter", writer=writer, name=name)
    return _op
  else:
    writer = _ops.convert_to_tensor(writer, _dtypes.resource)
    _inputs_flat = [writer]
    _attrs = None
    _result = _execute.execute(b"CloseSummaryWriter", 0, inputs=_inputs_flat,
                               attrs=_attrs, ctx=_ctx, name=name)
  return _result

_ops.RegisterShape("CloseSummaryWriter")(None)


def create_summary_file_writer(writer, logdir, max_queue, flush_millis, filename_suffix, name=None):
  r"""Creates a summary file writer accessible by the given resource handle.

  Args:
    writer: A `Tensor` of type `resource`.
      A handle to the summary writer resource
    logdir: A `Tensor` of type `string`.
      Directory where the event file will be written.
    max_queue: A `Tensor` of type `int32`.
      Size of the queue of pending events and summaries.
    flush_millis: A `Tensor` of type `int32`.
      How often, in milliseconds, to flush the pending events and
      summaries to disk.
    filename_suffix: A `Tensor` of type `string`.
      Every event file's name is suffixed with this suffix.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "CreateSummaryFileWriter", writer=writer, logdir=logdir,
        max_queue=max_queue, flush_millis=flush_millis,
        filename_suffix=filename_suffix, name=name)
    return _op
  else:
    writer = _ops.convert_to_tensor(writer, _dtypes.resource)
    logdir = _ops.convert_to_tensor(logdir, _dtypes.string)
    max_queue = _ops.convert_to_tensor(max_queue, _dtypes.int32)
    flush_millis = _ops.convert_to_tensor(flush_millis, _dtypes.int32)
    filename_suffix = _ops.convert_to_tensor(filename_suffix, _dtypes.string)
    _inputs_flat = [writer, logdir, max_queue, flush_millis, filename_suffix]
    _attrs = None
    _result = _execute.execute(b"CreateSummaryFileWriter", 0,
                               inputs=_inputs_flat, attrs=_attrs, ctx=_ctx,
                               name=name)
  return _result

_ops.RegisterShape("CreateSummaryFileWriter")(None)


def flush_summary_writer(writer, name=None):
  r"""Flushes the writer's unwritten events.

  Args:
    writer: A `Tensor` of type `resource`.
      A handle to the summary writer resource.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "FlushSummaryWriter", writer=writer, name=name)
    return _op
  else:
    writer = _ops.convert_to_tensor(writer, _dtypes.resource)
    _inputs_flat = [writer]
    _attrs = None
    _result = _execute.execute(b"FlushSummaryWriter", 0, inputs=_inputs_flat,
                               attrs=_attrs, ctx=_ctx, name=name)
  return _result

_ops.RegisterShape("FlushSummaryWriter")(None)


def summary_writer(shared_name="", container="", name=None):
  r"""Returns a handle to be used to access a summary writer.

  The summary writer is an in-graph resource which can be used by ops to write
  summaries to event files.

  Args:
    shared_name: An optional `string`. Defaults to `""`.
    container: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `resource`. the summary writer resource. Scalar handle.
  """
  if shared_name is None:
    shared_name = ""
  shared_name = _execute.make_str(shared_name, "shared_name")
  if container is None:
    container = ""
  container = _execute.make_str(container, "container")
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "SummaryWriter", shared_name=shared_name, container=container,
        name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("shared_name", _op.get_attr("shared_name"), "container",
              _op.get_attr("container"))
  else:
    _inputs_flat = []
    _attrs = ("shared_name", shared_name, "container", container)
    _result = _execute.execute(b"SummaryWriter", 1, inputs=_inputs_flat,
                               attrs=_attrs, ctx=_ctx, name=name)
  _execute.record_gradient(
      "SummaryWriter", _inputs_flat, _attrs, _result, name)
  _result, = _result
  return _result

_ops.RegisterShape("SummaryWriter")(None)


def write_audio_summary(writer, global_step, tag, tensor, sample_rate, max_outputs=3, name=None):
  r"""Writes a `Summary` protocol buffer with audio.

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
    writer: A `Tensor` of type `resource`. A handle to a summary writer.
    global_step: A `Tensor` of type `int64`.
      The step to write the summary for.
    tag: A `Tensor` of type `string`.
      Scalar. Used to build the `tag` attribute of the summary values.
    tensor: A `Tensor` of type `float32`. 2-D of shape `[batch_size, frames]`.
    sample_rate: A `Tensor` of type `float32`.
      The sample rate of the signal in hertz.
    max_outputs: An optional `int` that is `>= 1`. Defaults to `3`.
      Max number of batch elements to generate audio for.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  if max_outputs is None:
    max_outputs = 3
  max_outputs = _execute.make_int(max_outputs, "max_outputs")
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "WriteAudioSummary", writer=writer, global_step=global_step, tag=tag,
        tensor=tensor, sample_rate=sample_rate, max_outputs=max_outputs,
        name=name)
    return _op
  else:
    writer = _ops.convert_to_tensor(writer, _dtypes.resource)
    global_step = _ops.convert_to_tensor(global_step, _dtypes.int64)
    tag = _ops.convert_to_tensor(tag, _dtypes.string)
    tensor = _ops.convert_to_tensor(tensor, _dtypes.float32)
    sample_rate = _ops.convert_to_tensor(sample_rate, _dtypes.float32)
    _inputs_flat = [writer, global_step, tag, tensor, sample_rate]
    _attrs = ("max_outputs", max_outputs)
    _result = _execute.execute(b"WriteAudioSummary", 0, inputs=_inputs_flat,
                               attrs=_attrs, ctx=_ctx, name=name)
  return _result

_ops.RegisterShape("WriteAudioSummary")(None)


def write_histogram_summary(writer, global_step, tag, values, name=None):
  r"""Writes a `Summary` protocol buffer with a histogram.

  The generated
  [`Summary`](https://www.tensorflow.org/code/tensorflow/core/framework/summary.proto)
  has one summary value containing a histogram for `values`.

  This op reports an `InvalidArgument` error if any value is not finite.

  Args:
    writer: A `Tensor` of type `resource`. A handle to a summary writer.
    global_step: A `Tensor` of type `int64`.
      The step to write the summary for.
    tag: A `Tensor` of type `string`.
      Scalar.  Tag to use for the `Summary.Value`.
    values: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`, `uint8`, `int16`, `int8`, `uint16`, `half`.
      Any shape. Values to use to build the histogram.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "WriteHistogramSummary", writer=writer, global_step=global_step,
        tag=tag, values=values, name=name)
    return _op
  else:
    _attr_T, (values,) = _execute.args_to_matching_eager([values], _ctx, _dtypes.float32)
    _attr_T = _attr_T.as_datatype_enum
    writer = _ops.convert_to_tensor(writer, _dtypes.resource)
    global_step = _ops.convert_to_tensor(global_step, _dtypes.int64)
    tag = _ops.convert_to_tensor(tag, _dtypes.string)
    _inputs_flat = [writer, global_step, tag, values]
    _attrs = ("T", _attr_T)
    _result = _execute.execute(b"WriteHistogramSummary", 0,
                               inputs=_inputs_flat, attrs=_attrs, ctx=_ctx,
                               name=name)
  return _result

_ops.RegisterShape("WriteHistogramSummary")(None)


def write_image_summary(writer, global_step, tag, tensor, bad_color, max_images=3, name=None):
  r"""Writes a `Summary` protocol buffer with images.

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
    writer: A `Tensor` of type `resource`. A handle to a summary writer.
    global_step: A `Tensor` of type `int64`.
      The step to write the summary for.
    tag: A `Tensor` of type `string`.
      Scalar. Used to build the `tag` attribute of the summary values.
    tensor: A `Tensor`. Must be one of the following types: `uint8`, `float32`, `half`.
      4-D of shape `[batch_size, height, width, channels]` where
      `channels` is 1, 3, or 4.
    bad_color: A `Tensor` of type `uint8`.
      Color to use for pixels with non-finite values.
    max_images: An optional `int` that is `>= 1`. Defaults to `3`.
      Max number of batch elements to generate images for.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  if max_images is None:
    max_images = 3
  max_images = _execute.make_int(max_images, "max_images")
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "WriteImageSummary", writer=writer, global_step=global_step, tag=tag,
        tensor=tensor, bad_color=bad_color, max_images=max_images, name=name)
    return _op
  else:
    _attr_T, (tensor,) = _execute.args_to_matching_eager([tensor], _ctx, _dtypes.float32)
    _attr_T = _attr_T.as_datatype_enum
    writer = _ops.convert_to_tensor(writer, _dtypes.resource)
    global_step = _ops.convert_to_tensor(global_step, _dtypes.int64)
    tag = _ops.convert_to_tensor(tag, _dtypes.string)
    bad_color = _ops.convert_to_tensor(bad_color, _dtypes.uint8)
    _inputs_flat = [writer, global_step, tag, tensor, bad_color]
    _attrs = ("max_images", max_images, "T", _attr_T)
    _result = _execute.execute(b"WriteImageSummary", 0, inputs=_inputs_flat,
                               attrs=_attrs, ctx=_ctx, name=name)
  return _result

_ops.RegisterShape("WriteImageSummary")(None)


def write_scalar_summary(writer, global_step, tag, value, name=None):
  r"""Writes a `Summary` protocol buffer with scalar values.

  The input `tag` and `value` must have the scalars.

  Args:
    writer: A `Tensor` of type `resource`. A handle to a summary writer.
    global_step: A `Tensor` of type `int64`.
      The step to write the summary for.
    tag: A `Tensor` of type `string`. Tag for the summary.
    value: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`, `uint8`, `int16`, `int8`, `uint16`, `half`.
      Value for the summary.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "WriteScalarSummary", writer=writer, global_step=global_step, tag=tag,
        value=value, name=name)
    return _op
  else:
    _attr_T, (value,) = _execute.args_to_matching_eager([value], _ctx)
    _attr_T = _attr_T.as_datatype_enum
    writer = _ops.convert_to_tensor(writer, _dtypes.resource)
    global_step = _ops.convert_to_tensor(global_step, _dtypes.int64)
    tag = _ops.convert_to_tensor(tag, _dtypes.string)
    _inputs_flat = [writer, global_step, tag, value]
    _attrs = ("T", _attr_T)
    _result = _execute.execute(b"WriteScalarSummary", 0, inputs=_inputs_flat,
                               attrs=_attrs, ctx=_ctx, name=name)
  return _result

_ops.RegisterShape("WriteScalarSummary")(None)


def write_summary(writer, global_step, tensor, tag, summary_metadata, name=None):
  r"""Outputs a `Summary` protocol buffer with a tensor.

  Args:
    writer: A `Tensor` of type `resource`. A handle to a summary writer.
    global_step: A `Tensor` of type `int64`.
      The step to write the summary for.
    tensor: A `Tensor`. A tensor to serialize.
    tag: A `Tensor` of type `string`. The summary's tag.
    summary_metadata: A `Tensor` of type `string`.
      Serialized SummaryMetadata protocol buffer containing
      plugin-related metadata for this summary.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "WriteSummary", writer=writer, global_step=global_step, tensor=tensor,
        tag=tag, summary_metadata=summary_metadata, name=name)
    return _op
  else:
    _attr_T, (tensor,) = _execute.args_to_matching_eager([tensor], _ctx)
    _attr_T = _attr_T.as_datatype_enum
    writer = _ops.convert_to_tensor(writer, _dtypes.resource)
    global_step = _ops.convert_to_tensor(global_step, _dtypes.int64)
    tag = _ops.convert_to_tensor(tag, _dtypes.string)
    summary_metadata = _ops.convert_to_tensor(summary_metadata, _dtypes.string)
    _inputs_flat = [writer, global_step, tensor, tag, summary_metadata]
    _attrs = ("T", _attr_T)
    _result = _execute.execute(b"WriteSummary", 0, inputs=_inputs_flat,
                               attrs=_attrs, ctx=_ctx, name=name)
  return _result

_ops.RegisterShape("WriteSummary")(None)

def _InitOpDefLibrary(op_list_proto_bytes):
  op_list = _op_def_pb2.OpList()
  op_list.ParseFromString(op_list_proto_bytes)
  _op_def_registry.register_op_list(op_list)
  op_def_lib = _op_def_library.OpDefLibrary()
  op_def_lib.add_op_list(op_list)
  return op_def_lib
# op {
#   name: "CloseSummaryWriter"
#   input_arg {
#     name: "writer"
#     type: DT_RESOURCE
#   }
#   is_stateful: true
# }
# op {
#   name: "CreateSummaryFileWriter"
#   input_arg {
#     name: "writer"
#     type: DT_RESOURCE
#   }
#   input_arg {
#     name: "logdir"
#     type: DT_STRING
#   }
#   input_arg {
#     name: "max_queue"
#     type: DT_INT32
#   }
#   input_arg {
#     name: "flush_millis"
#     type: DT_INT32
#   }
#   input_arg {
#     name: "filename_suffix"
#     type: DT_STRING
#   }
#   is_stateful: true
# }
# op {
#   name: "FlushSummaryWriter"
#   input_arg {
#     name: "writer"
#     type: DT_RESOURCE
#   }
#   is_stateful: true
# }
# op {
#   name: "SummaryWriter"
#   output_arg {
#     name: "writer"
#     type: DT_RESOURCE
#   }
#   attr {
#     name: "shared_name"
#     type: "string"
#     default_value {
#       s: ""
#     }
#   }
#   attr {
#     name: "container"
#     type: "string"
#     default_value {
#       s: ""
#     }
#   }
#   is_stateful: true
# }
# op {
#   name: "WriteAudioSummary"
#   input_arg {
#     name: "writer"
#     type: DT_RESOURCE
#   }
#   input_arg {
#     name: "global_step"
#     type: DT_INT64
#   }
#   input_arg {
#     name: "tag"
#     type: DT_STRING
#   }
#   input_arg {
#     name: "tensor"
#     type: DT_FLOAT
#   }
#   input_arg {
#     name: "sample_rate"
#     type: DT_FLOAT
#   }
#   attr {
#     name: "max_outputs"
#     type: "int"
#     default_value {
#       i: 3
#     }
#     has_minimum: true
#     minimum: 1
#   }
#   is_stateful: true
# }
# op {
#   name: "WriteHistogramSummary"
#   input_arg {
#     name: "writer"
#     type: DT_RESOURCE
#   }
#   input_arg {
#     name: "global_step"
#     type: DT_INT64
#   }
#   input_arg {
#     name: "tag"
#     type: DT_STRING
#   }
#   input_arg {
#     name: "values"
#     type_attr: "T"
#   }
#   attr {
#     name: "T"
#     type: "type"
#     default_value {
#       type: DT_FLOAT
#     }
#     allowed_values {
#       list {
#         type: DT_FLOAT
#         type: DT_DOUBLE
#         type: DT_INT32
#         type: DT_INT64
#         type: DT_UINT8
#         type: DT_INT16
#         type: DT_INT8
#         type: DT_UINT16
#         type: DT_HALF
#       }
#     }
#   }
#   is_stateful: true
# }
# op {
#   name: "WriteImageSummary"
#   input_arg {
#     name: "writer"
#     type: DT_RESOURCE
#   }
#   input_arg {
#     name: "global_step"
#     type: DT_INT64
#   }
#   input_arg {
#     name: "tag"
#     type: DT_STRING
#   }
#   input_arg {
#     name: "tensor"
#     type_attr: "T"
#   }
#   input_arg {
#     name: "bad_color"
#     type: DT_UINT8
#   }
#   attr {
#     name: "max_images"
#     type: "int"
#     default_value {
#       i: 3
#     }
#     has_minimum: true
#     minimum: 1
#   }
#   attr {
#     name: "T"
#     type: "type"
#     default_value {
#       type: DT_FLOAT
#     }
#     allowed_values {
#       list {
#         type: DT_UINT8
#         type: DT_FLOAT
#         type: DT_HALF
#       }
#     }
#   }
#   is_stateful: true
# }
# op {
#   name: "WriteScalarSummary"
#   input_arg {
#     name: "writer"
#     type: DT_RESOURCE
#   }
#   input_arg {
#     name: "global_step"
#     type: DT_INT64
#   }
#   input_arg {
#     name: "tag"
#     type: DT_STRING
#   }
#   input_arg {
#     name: "value"
#     type_attr: "T"
#   }
#   attr {
#     name: "T"
#     type: "type"
#     allowed_values {
#       list {
#         type: DT_FLOAT
#         type: DT_DOUBLE
#         type: DT_INT32
#         type: DT_INT64
#         type: DT_UINT8
#         type: DT_INT16
#         type: DT_INT8
#         type: DT_UINT16
#         type: DT_HALF
#       }
#     }
#   }
#   is_stateful: true
# }
# op {
#   name: "WriteSummary"
#   input_arg {
#     name: "writer"
#     type: DT_RESOURCE
#   }
#   input_arg {
#     name: "global_step"
#     type: DT_INT64
#   }
#   input_arg {
#     name: "tensor"
#     type_attr: "T"
#   }
#   input_arg {
#     name: "tag"
#     type: DT_STRING
#   }
#   input_arg {
#     name: "summary_metadata"
#     type: DT_STRING
#   }
#   attr {
#     name: "T"
#     type: "type"
#   }
#   is_stateful: true
# }
_op_def_lib = _InitOpDefLibrary(b"\n#\n\022CloseSummaryWriter\022\n\n\006writer\030\024\210\001\001\nj\n\027CreateSummaryFileWriter\022\n\n\006writer\030\024\022\n\n\006logdir\030\007\022\r\n\tmax_queue\030\003\022\020\n\014flush_millis\030\003\022\023\n\017filename_suffix\030\007\210\001\001\n#\n\022FlushSummaryWriter\022\n\n\006writer\030\024\210\001\001\nR\n\rSummaryWriter\032\n\n\006writer\030\024\"\031\n\013shared_name\022\006string\032\002\022\000\"\027\n\tcontainer\022\006string\032\002\022\000\210\001\001\nu\n\021WriteAudioSummary\022\n\n\006writer\030\024\022\017\n\013global_step\030\t\022\007\n\003tag\030\007\022\n\n\006tensor\030\001\022\017\n\013sample_rate\030\001\"\032\n\013max_outputs\022\003int\032\002\030\003(\0010\001\210\001\001\nk\n\025WriteHistogramSummary\022\n\n\006writer\030\024\022\017\n\013global_step\030\t\022\007\n\003tag\030\007\022\013\n\006values\"\001T\"\034\n\001T\022\004type\032\0020\001:\r\n\0132\t\001\002\003\t\004\005\006\021\023\210\001\001\n\213\001\n\021WriteImageSummary\022\n\n\006writer\030\024\022\017\n\013global_step\030\t\022\007\n\003tag\030\007\022\013\n\006tensor\"\001T\022\r\n\tbad_color\030\004\"\031\n\nmax_images\022\003int\032\002\030\003(\0010\001\"\026\n\001T\022\004type\032\0020\001:\007\n\0052\003\004\001\023\210\001\001\nc\n\022WriteScalarSummary\022\n\n\006writer\030\024\022\017\n\013global_step\030\t\022\007\n\003tag\030\007\022\n\n\005value\"\001T\"\030\n\001T\022\004type:\r\n\0132\t\001\002\003\t\004\005\006\021\023\210\001\001\ne\n\014WriteSummary\022\n\n\006writer\030\024\022\017\n\013global_step\030\t\022\013\n\006tensor\"\001T\022\007\n\003tag\030\007\022\024\n\020summary_metadata\030\007\"\t\n\001T\022\004type\210\001\001")
