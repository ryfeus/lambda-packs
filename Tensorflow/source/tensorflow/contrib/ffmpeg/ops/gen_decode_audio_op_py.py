"""Python wrappers around TensorFlow ops.

This file is MACHINE GENERATED! Do not edit.
Original C++ source file: decode_audio_op_py.cc
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


def decode_audio(contents, file_format, samples_per_second, channel_count, name=None):
  r"""Processes the contents of an audio file into a tensor using FFmpeg to decode

  the file.

  One row of the tensor is created for each channel in the audio file. Each
  channel contains audio samples starting at the beginning of the audio and
  having `1/samples_per_second` time between them. If the `channel_count` is
  different from the contents of the file, channels will be merged or created.

  Args:
    contents: A `Tensor` of type `string`. The binary audio file contents.
    file_format: A `string`.
      A string describing the audio file format. This can be "mp3", "mp4", "ogg", or "wav".
    samples_per_second: An `int`.
      The number of samples per second that the audio should have.
    channel_count: An `int`. The number of channels of audio to read.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32`.
    A rank 2 tensor containing all tracks of the audio. Dimension 0
    is time and dimension 1 is the channel. If ffmpeg fails to decode the audio
    then an empty tensor will be returned.
  """
  file_format = _execute.make_str(file_format, "file_format")
  samples_per_second = _execute.make_int(samples_per_second, "samples_per_second")
  channel_count = _execute.make_int(channel_count, "channel_count")
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "DecodeAudio", contents=contents, file_format=file_format,
        samples_per_second=samples_per_second, channel_count=channel_count,
        name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("file_format", _op.get_attr("file_format"),
              "samples_per_second", _op.get_attr("samples_per_second"),
              "channel_count", _op.get_attr("channel_count"))
  else:
    contents = _ops.convert_to_tensor(contents, _dtypes.string)
    _inputs_flat = [contents]
    _attrs = ("file_format", file_format, "samples_per_second",
              samples_per_second, "channel_count", channel_count)
    _result = _execute.execute(b"DecodeAudio", 1, inputs=_inputs_flat,
                               attrs=_attrs, ctx=_ctx, name=name)
  _execute.record_gradient(
      "DecodeAudio", _inputs_flat, _attrs, _result, name)
  _result, = _result
  return _result


def decode_audio_v2(contents, file_format, samples_per_second, channel_count, name=None):
  r"""Processes the contents of an audio file into a tensor using FFmpeg to decode

  the file.

  One row of the tensor is created for each channel in the audio file. Each
  channel contains audio samples starting at the beginning of the audio and
  having `1/samples_per_second` time between them. If the `channel_count` is
  different from the contents of the file, channels will be merged or created.

  Args:
    contents: A `Tensor` of type `string`.
      The binary audio file contents, as a string or rank-0 string
      tensor.
    file_format: A `Tensor` of type `string`.
      A string or rank-0 string tensor describing the audio file
      format. This must be one of: "mp3", "mp4", "ogg", "wav".
    samples_per_second: A `Tensor` of type `int32`.
      The number of samples per second that the audio
      should have, as an `int` or rank-0 `int32` tensor. This value must
      be positive.
    channel_count: A `Tensor` of type `int32`.
      The number of channels of audio to read, as an int rank-0
      int32 tensor. Must be a positive integer.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32`.
    A rank-2 tensor containing all tracks of the audio.
    Dimension 0 is time and dimension 1 is the channel. If ffmpeg fails
    to decode the audio then an empty tensor will be returned.
  """
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "DecodeAudioV2", contents=contents, file_format=file_format,
        samples_per_second=samples_per_second, channel_count=channel_count,
        name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = None
  else:
    contents = _ops.convert_to_tensor(contents, _dtypes.string)
    file_format = _ops.convert_to_tensor(file_format, _dtypes.string)
    samples_per_second = _ops.convert_to_tensor(samples_per_second, _dtypes.int32)
    channel_count = _ops.convert_to_tensor(channel_count, _dtypes.int32)
    _inputs_flat = [contents, file_format, samples_per_second, channel_count]
    _attrs = None
    _result = _execute.execute(b"DecodeAudioV2", 1, inputs=_inputs_flat,
                               attrs=_attrs, ctx=_ctx, name=name)
  _execute.record_gradient(
      "DecodeAudioV2", _inputs_flat, _attrs, _result, name)
  _result, = _result
  return _result

def _InitOpDefLibrary(op_list_proto_bytes):
  op_list = _op_def_pb2.OpList()
  op_list.ParseFromString(op_list_proto_bytes)
  _op_def_registry.register_op_list(op_list)
  op_def_lib = _op_def_library.OpDefLibrary()
  op_def_lib.add_op_list(op_list)
  return op_def_lib
# op {
#   name: "DecodeAudio"
#   input_arg {
#     name: "contents"
#     type: DT_STRING
#   }
#   output_arg {
#     name: "sampled_audio"
#     type: DT_FLOAT
#   }
#   attr {
#     name: "file_format"
#     type: "string"
#   }
#   attr {
#     name: "samples_per_second"
#     type: "int"
#   }
#   attr {
#     name: "channel_count"
#     type: "int"
#   }
# }
# op {
#   name: "DecodeAudioV2"
#   input_arg {
#     name: "contents"
#     type: DT_STRING
#   }
#   input_arg {
#     name: "file_format"
#     type: DT_STRING
#   }
#   input_arg {
#     name: "samples_per_second"
#     type: DT_INT32
#   }
#   input_arg {
#     name: "channel_count"
#     type: DT_INT32
#   }
#   output_arg {
#     name: "sampled_audio"
#     type: DT_FLOAT
#   }
# }
_op_def_lib = _InitOpDefLibrary(b"\nv\n\013DecodeAudio\022\014\n\010contents\030\007\032\021\n\rsampled_audio\030\001\"\025\n\013file_format\022\006string\"\031\n\022samples_per_second\022\003int\"\024\n\rchannel_count\022\003int\nl\n\rDecodeAudioV2\022\014\n\010contents\030\007\022\017\n\013file_format\030\007\022\026\n\022samples_per_second\030\003\022\021\n\rchannel_count\030\003\032\021\n\rsampled_audio\030\001")
