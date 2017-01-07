"""Python wrappers around Brain.

This file is MACHINE GENERATED! Do not edit.
"""

import collections

from google.protobuf import text_format

from tensorflow.core.framework import op_def_pb2
from tensorflow.python.framework import op_def_registry
from tensorflow.python.framework import ops
from tensorflow.python.ops import op_def_library


__fixed_length_record_reader_outputs = ["reader_handle"]


def _fixed_length_record_reader(record_bytes, header_bytes=None,
                                footer_bytes=None, container=None,
                                shared_name=None, name=None):
  r"""A Reader that outputs fixed-length records from a file.

  Args:
    record_bytes: An `int`.
    header_bytes: An optional `int`. Defaults to `0`.
    footer_bytes: An optional `int`. Defaults to `0`.
    container: An optional `string`. Defaults to `""`.
      If non-empty, this reader is placed in the given container.
      Otherwise, a default container is used.
    shared_name: An optional `string`. Defaults to `""`.
      If non-empty, this reader is named in the given bucket
      with this shared_name. Otherwise, the node name is used instead.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type mutable `string`. The handle to reference the Reader.
  """
  result = _op_def_lib.apply_op("FixedLengthRecordReader",
                                record_bytes=record_bytes,
                                header_bytes=header_bytes,
                                footer_bytes=footer_bytes,
                                container=container, shared_name=shared_name,
                                name=name)
  return result


__identity_reader_outputs = ["reader_handle"]


def _identity_reader(container=None, shared_name=None, name=None):
  r"""A Reader that outputs the queued work as both the key and value.

  To use, enqueue strings in a Queue.  ReaderRead will take the front
  work string and output (work, work).

  Args:
    container: An optional `string`. Defaults to `""`.
      If non-empty, this reader is placed in the given container.
      Otherwise, a default container is used.
    shared_name: An optional `string`. Defaults to `""`.
      If non-empty, this reader is named in the given bucket
      with this shared_name. Otherwise, the node name is used instead.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type mutable `string`. The handle to reference the Reader.
  """
  result = _op_def_lib.apply_op("IdentityReader", container=container,
                                shared_name=shared_name, name=name)
  return result


_matching_files_outputs = ["filenames"]


def matching_files(pattern, name=None):
  r"""Returns the set of files matching a pattern.

  Note that this routine only supports wildcard characters in the
  basename portion of the pattern, not in the directory portion.

  Args:
    pattern: A `Tensor` of type `string`. A (scalar) shell wildcard pattern.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`. A vector of matching filenames.
  """
  result = _op_def_lib.apply_op("MatchingFiles", pattern=pattern, name=name)
  return result


_read_file_outputs = ["contents"]


def read_file(filename, name=None):
  r"""Reads and outputs the entire contents of the input filename.

  Args:
    filename: A `Tensor` of type `string`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`.
  """
  result = _op_def_lib.apply_op("ReadFile", filename=filename, name=name)
  return result


__reader_num_records_produced_outputs = ["records_produced"]


def _reader_num_records_produced(reader_handle, name=None):
  r"""Returns the number of records this Reader has produced.

  This is the same as the number of ReaderRead executions that have
  succeeded.

  Args:
    reader_handle: A `Tensor` of type mutable `string`. Handle to a Reader.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int64`.
  """
  result = _op_def_lib.apply_op("ReaderNumRecordsProduced",
                                reader_handle=reader_handle, name=name)
  return result


__reader_num_work_units_completed_outputs = ["units_completed"]


def _reader_num_work_units_completed(reader_handle, name=None):
  r"""Returns the number of work units this Reader has finished processing.

  Args:
    reader_handle: A `Tensor` of type mutable `string`. Handle to a Reader.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int64`.
  """
  result = _op_def_lib.apply_op("ReaderNumWorkUnitsCompleted",
                                reader_handle=reader_handle, name=name)
  return result


__reader_read_outputs = ["key", "value"]


_ReaderReadOutput = collections.namedtuple("ReaderRead",
                                           __reader_read_outputs)


def _reader_read(reader_handle, queue_handle, name=None):
  r"""Returns the next record (key, value pair) produced by a Reader.

  Will dequeue from the input queue if necessary (e.g. when the
  Reader needs to start reading from a new file since it has finished
  with the previous file).

  Args:
    reader_handle: A `Tensor` of type mutable `string`. Handle to a Reader.
    queue_handle: A `Tensor` of type mutable `string`.
      Handle to a Queue, with string work items.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (key, value).
    key: A `Tensor` of type `string`. A scalar.
    value: A `Tensor` of type `string`. A scalar.
  """
  result = _op_def_lib.apply_op("ReaderRead", reader_handle=reader_handle,
                                queue_handle=queue_handle, name=name)
  return _ReaderReadOutput._make(result)


__reader_read_up_to_outputs = ["keys", "values"]


_ReaderReadUpToOutput = collections.namedtuple("ReaderReadUpTo",
                                               __reader_read_up_to_outputs)


def _reader_read_up_to(reader_handle, queue_handle, num_records, name=None):
  r"""Returns up to `num_records` (key, value) pairs produced by a Reader.

  Will dequeue from the input queue if necessary (e.g. when the
  Reader needs to start reading from a new file since it has finished
  with the previous file).

  Args:
    reader_handle: A `Tensor` of type mutable `string`. Handle to a `Reader`.
    queue_handle: A `Tensor` of type mutable `string`.
      Handle to a `Queue`, with string work items.
    num_records: A `Tensor` of type `int64`.
      number of records to read from `Reader`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (keys, values).
    keys: A `Tensor` of type `string`. A 1-D tensor.
    values: A `Tensor` of type `string`. A 1-D tensor.
  """
  result = _op_def_lib.apply_op("ReaderReadUpTo", reader_handle=reader_handle,
                                queue_handle=queue_handle,
                                num_records=num_records, name=name)
  return _ReaderReadUpToOutput._make(result)


__reader_reset_outputs = [""]


def _reader_reset(reader_handle, name=None):
  r"""Restore a Reader to its initial clean state.

  Args:
    reader_handle: A `Tensor` of type mutable `string`. Handle to a Reader.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  result = _op_def_lib.apply_op("ReaderReset", reader_handle=reader_handle,
                                name=name)
  return result


__reader_restore_state_outputs = [""]


def _reader_restore_state(reader_handle, state, name=None):
  r"""Restore a reader to a previously saved state.

  Not all Readers support being restored, so this can produce an
  Unimplemented error.

  Args:
    reader_handle: A `Tensor` of type mutable `string`. Handle to a Reader.
    state: A `Tensor` of type `string`.
      Result of a ReaderSerializeState of a Reader with type
      matching reader_handle.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  result = _op_def_lib.apply_op("ReaderRestoreState",
                                reader_handle=reader_handle, state=state,
                                name=name)
  return result


__reader_serialize_state_outputs = ["state"]


def _reader_serialize_state(reader_handle, name=None):
  r"""Produce a string tensor that encodes the state of a Reader.

  Not all Readers support being serialized, so this can produce an
  Unimplemented error.

  Args:
    reader_handle: A `Tensor` of type mutable `string`. Handle to a Reader.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`.
  """
  result = _op_def_lib.apply_op("ReaderSerializeState",
                                reader_handle=reader_handle, name=name)
  return result


__restore_outputs = ["tensor"]


def _restore(file_pattern, tensor_name, dt, preferred_shard=None, name=None):
  r"""Restores a tensor from checkpoint files.

  Reads a tensor stored in one or several files. If there are several files (for
  instance because a tensor was saved as slices), `file_pattern` may contain
  wildcard symbols (`*` and `?`) in the filename portion only, not in the
  directory portion.

  If a `file_pattern` matches several files, `preferred_shard` can be used to hint
  in which file the requested tensor is likely to be found. This op will first
  open the file at index `preferred_shard` in the list of matching files and try
  to restore tensors from that file.  Only if some tensors or tensor slices are
  not found in that first file, then the Op opens all the files. Setting
  `preferred_shard` to match the value passed as the `shard` input
  of a matching `Save` Op may speed up Restore.  This attribute only affects
  performance, not correctness.  The default value -1 means files are processed in
  order.

  See also `RestoreSlice`.

  Args:
    file_pattern: A `Tensor` of type `string`.
      Must have a single element. The pattern of the files from
      which we read the tensor.
    tensor_name: A `Tensor` of type `string`.
      Must have a single element. The name of the tensor to be
      restored.
    dt: A `tf.DType`. The type of the tensor to be restored.
    preferred_shard: An optional `int`. Defaults to `-1`.
      Index of file to open first if multiple files match
      `file_pattern`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `dt`. The restored tensor.
  """
  result = _op_def_lib.apply_op("Restore", file_pattern=file_pattern,
                                tensor_name=tensor_name, dt=dt,
                                preferred_shard=preferred_shard, name=name)
  return result


__restore_slice_outputs = ["tensor"]


def _restore_slice(file_pattern, tensor_name, shape_and_slice, dt,
                   preferred_shard=None, name=None):
  r"""Restores a tensor from checkpoint files.

  This is like `Restore` except that restored tensor can be listed as filling
  only a slice of a larger tensor.  `shape_and_slice` specifies the shape of the
  larger tensor and the slice that the restored tensor covers.

  The `shape_and_slice` input has the same format as the
  elements of the `shapes_and_slices` input of the `SaveSlices` op.

  Args:
    file_pattern: A `Tensor` of type `string`.
      Must have a single element. The pattern of the files from
      which we read the tensor.
    tensor_name: A `Tensor` of type `string`.
      Must have a single element. The name of the tensor to be
      restored.
    shape_and_slice: A `Tensor` of type `string`.
      Scalar. The shapes and slice specifications to use when
      restoring a tensors.
    dt: A `tf.DType`. The type of the tensor to be restored.
    preferred_shard: An optional `int`. Defaults to `-1`.
      Index of file to open first if multiple files match
      `file_pattern`. See the documentation for `Restore`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `dt`. The restored tensor.
  """
  result = _op_def_lib.apply_op("RestoreSlice", file_pattern=file_pattern,
                                tensor_name=tensor_name,
                                shape_and_slice=shape_and_slice, dt=dt,
                                preferred_shard=preferred_shard, name=name)
  return result


__save_outputs = [""]


def _save(filename, tensor_names, data, name=None):
  r"""Saves the input tensors to disk.

  The size of `tensor_names` must match the number of tensors in `data`. `data[i]`
  is written to `filename` with name `tensor_names[i]`.

  See also `SaveSlices`.

  Args:
    filename: A `Tensor` of type `string`.
      Must have a single element. The name of the file to which we write
      the tensor.
    tensor_names: A `Tensor` of type `string`.
      Shape `[N]`. The names of the tensors to be saved.
    data: A list of `Tensor` objects. `N` tensors to save.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  result = _op_def_lib.apply_op("Save", filename=filename,
                                tensor_names=tensor_names, data=data,
                                name=name)
  return result


__save_slices_outputs = [""]


def _save_slices(filename, tensor_names, shapes_and_slices, data, name=None):
  r"""Saves input tensors slices to disk.

  This is like `Save` except that tensors can be listed in the saved file as being
  a slice of a larger tensor.  `shapes_and_slices` specifies the shape of the
  larger tensor and the slice that this tensor covers. `shapes_and_slices` must
  have as many elements as `tensor_names`.

  Elements of the `shapes_and_slices` input must either be:

  *  The empty string, in which case the corresponding tensor is
     saved normally.
  *  A string of the form `dim0 dim1 ... dimN-1 slice-spec` where the
     `dimI` are the dimensions of the larger tensor and `slice-spec`
     specifies what part is covered by the tensor to save.

  `slice-spec` itself is a `:`-separated list: `slice0:slice1:...:sliceN-1`
  where each `sliceI` is either:

  *  The string `-` meaning that the slice covers all indices of this dimension
  *  `start,length` where `start` and `length` are integers.  In that
     case the slice covers `length` indices starting at `start`.

  See also `Save`.

  Args:
    filename: A `Tensor` of type `string`.
      Must have a single element. The name of the file to which we write the
      tensor.
    tensor_names: A `Tensor` of type `string`.
      Shape `[N]`. The names of the tensors to be saved.
    shapes_and_slices: A `Tensor` of type `string`.
      Shape `[N]`.  The shapes and slice specifications to use when
      saving the tensors.
    data: A list of `Tensor` objects. `N` tensors to save.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  result = _op_def_lib.apply_op("SaveSlices", filename=filename,
                                tensor_names=tensor_names,
                                shapes_and_slices=shapes_and_slices,
                                data=data, name=name)
  return result


__sharded_filename_outputs = ["filename"]


def _sharded_filename(basename, shard, num_shards, name=None):
  r"""Generate a sharded filename. The filename is printf formatted as

     %s-%05d-of-%05d, basename, shard, num_shards.

  Args:
    basename: A `Tensor` of type `string`.
    shard: A `Tensor` of type `int32`.
    num_shards: A `Tensor` of type `int32`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`.
  """
  result = _op_def_lib.apply_op("ShardedFilename", basename=basename,
                                shard=shard, num_shards=num_shards, name=name)
  return result


__sharded_filespec_outputs = ["filename"]


def _sharded_filespec(basename, num_shards, name=None):
  r"""Generate a glob pattern matching all sharded file names.

  Args:
    basename: A `Tensor` of type `string`.
    num_shards: A `Tensor` of type `int32`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`.
  """
  result = _op_def_lib.apply_op("ShardedFilespec", basename=basename,
                                num_shards=num_shards, name=name)
  return result


__tf_record_reader_outputs = ["reader_handle"]


def _tf_record_reader(container=None, shared_name=None, name=None):
  r"""A Reader that outputs the records from a TensorFlow Records file.

  Args:
    container: An optional `string`. Defaults to `""`.
      If non-empty, this reader is placed in the given container.
      Otherwise, a default container is used.
    shared_name: An optional `string`. Defaults to `""`.
      If non-empty, this reader is named in the given bucket
      with this shared_name. Otherwise, the node name is used instead.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type mutable `string`. The handle to reference the Reader.
  """
  result = _op_def_lib.apply_op("TFRecordReader", container=container,
                                shared_name=shared_name, name=name)
  return result


__text_line_reader_outputs = ["reader_handle"]


def _text_line_reader(skip_header_lines=None, container=None,
                      shared_name=None, name=None):
  r"""A Reader that outputs the lines of a file delimited by '\n'.

  Args:
    skip_header_lines: An optional `int`. Defaults to `0`.
      Number of lines to skip from the beginning of every file.
    container: An optional `string`. Defaults to `""`.
      If non-empty, this reader is placed in the given container.
      Otherwise, a default container is used.
    shared_name: An optional `string`. Defaults to `""`.
      If non-empty, this reader is named in the given bucket
      with this shared_name. Otherwise, the node name is used instead.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type mutable `string`. The handle to reference the Reader.
  """
  result = _op_def_lib.apply_op("TextLineReader",
                                skip_header_lines=skip_header_lines,
                                container=container, shared_name=shared_name,
                                name=name)
  return result


__whole_file_reader_outputs = ["reader_handle"]


def _whole_file_reader(container=None, shared_name=None, name=None):
  r"""A Reader that outputs the entire contents of a file as a value.

  To use, enqueue filenames in a Queue.  The output of ReaderRead will
  be a filename (key) and the contents of that file (value).

  Args:
    container: An optional `string`. Defaults to `""`.
      If non-empty, this reader is placed in the given container.
      Otherwise, a default container is used.
    shared_name: An optional `string`. Defaults to `""`.
      If non-empty, this reader is named in the given bucket
      with this shared_name. Otherwise, the node name is used instead.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type mutable `string`. The handle to reference the Reader.
  """
  result = _op_def_lib.apply_op("WholeFileReader", container=container,
                                shared_name=shared_name, name=name)
  return result


def _InitOpDefLibrary():
  op_list = op_def_pb2.OpList()
  text_format.Merge(_InitOpDefLibrary.op_list_ascii, op_list)
  op_def_registry.register_op_list(op_list)
  op_def_lib = op_def_library.OpDefLibrary()
  op_def_lib.add_op_list(op_list)
  return op_def_lib


_InitOpDefLibrary.op_list_ascii = """op {
  name: "FixedLengthRecordReader"
  output_arg {
    name: "reader_handle"
    type: DT_STRING
    is_ref: true
  }
  attr {
    name: "header_bytes"
    type: "int"
    default_value {
      i: 0
    }
  }
  attr {
    name: "record_bytes"
    type: "int"
  }
  attr {
    name: "footer_bytes"
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
  name: "IdentityReader"
  output_arg {
    name: "reader_handle"
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
  is_stateful: true
}
op {
  name: "MatchingFiles"
  input_arg {
    name: "pattern"
    type: DT_STRING
  }
  output_arg {
    name: "filenames"
    type: DT_STRING
  }
}
op {
  name: "ReadFile"
  input_arg {
    name: "filename"
    type: DT_STRING
  }
  output_arg {
    name: "contents"
    type: DT_STRING
  }
}
op {
  name: "ReaderNumRecordsProduced"
  input_arg {
    name: "reader_handle"
    type: DT_STRING
    is_ref: true
  }
  output_arg {
    name: "records_produced"
    type: DT_INT64
  }
}
op {
  name: "ReaderNumWorkUnitsCompleted"
  input_arg {
    name: "reader_handle"
    type: DT_STRING
    is_ref: true
  }
  output_arg {
    name: "units_completed"
    type: DT_INT64
  }
}
op {
  name: "ReaderRead"
  input_arg {
    name: "reader_handle"
    type: DT_STRING
    is_ref: true
  }
  input_arg {
    name: "queue_handle"
    type: DT_STRING
    is_ref: true
  }
  output_arg {
    name: "key"
    type: DT_STRING
  }
  output_arg {
    name: "value"
    type: DT_STRING
  }
}
op {
  name: "ReaderReadUpTo"
  input_arg {
    name: "reader_handle"
    type: DT_STRING
    is_ref: true
  }
  input_arg {
    name: "queue_handle"
    type: DT_STRING
    is_ref: true
  }
  input_arg {
    name: "num_records"
    type: DT_INT64
  }
  output_arg {
    name: "keys"
    type: DT_STRING
  }
  output_arg {
    name: "values"
    type: DT_STRING
  }
}
op {
  name: "ReaderReset"
  input_arg {
    name: "reader_handle"
    type: DT_STRING
    is_ref: true
  }
}
op {
  name: "ReaderRestoreState"
  input_arg {
    name: "reader_handle"
    type: DT_STRING
    is_ref: true
  }
  input_arg {
    name: "state"
    type: DT_STRING
  }
}
op {
  name: "ReaderSerializeState"
  input_arg {
    name: "reader_handle"
    type: DT_STRING
    is_ref: true
  }
  output_arg {
    name: "state"
    type: DT_STRING
  }
}
op {
  name: "Restore"
  input_arg {
    name: "file_pattern"
    type: DT_STRING
  }
  input_arg {
    name: "tensor_name"
    type: DT_STRING
  }
  output_arg {
    name: "tensor"
    type_attr: "dt"
  }
  attr {
    name: "dt"
    type: "type"
  }
  attr {
    name: "preferred_shard"
    type: "int"
    default_value {
      i: -1
    }
  }
}
op {
  name: "RestoreSlice"
  input_arg {
    name: "file_pattern"
    type: DT_STRING
  }
  input_arg {
    name: "tensor_name"
    type: DT_STRING
  }
  input_arg {
    name: "shape_and_slice"
    type: DT_STRING
  }
  output_arg {
    name: "tensor"
    type_attr: "dt"
  }
  attr {
    name: "dt"
    type: "type"
  }
  attr {
    name: "preferred_shard"
    type: "int"
    default_value {
      i: -1
    }
  }
}
op {
  name: "Save"
  input_arg {
    name: "filename"
    type: DT_STRING
  }
  input_arg {
    name: "tensor_names"
    type: DT_STRING
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
}
op {
  name: "SaveSlices"
  input_arg {
    name: "filename"
    type: DT_STRING
  }
  input_arg {
    name: "tensor_names"
    type: DT_STRING
  }
  input_arg {
    name: "shapes_and_slices"
    type: DT_STRING
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
}
op {
  name: "ShardedFilename"
  input_arg {
    name: "basename"
    type: DT_STRING
  }
  input_arg {
    name: "shard"
    type: DT_INT32
  }
  input_arg {
    name: "num_shards"
    type: DT_INT32
  }
  output_arg {
    name: "filename"
    type: DT_STRING
  }
}
op {
  name: "ShardedFilespec"
  input_arg {
    name: "basename"
    type: DT_STRING
  }
  input_arg {
    name: "num_shards"
    type: DT_INT32
  }
  output_arg {
    name: "filename"
    type: DT_STRING
  }
}
op {
  name: "TFRecordReader"
  output_arg {
    name: "reader_handle"
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
  is_stateful: true
}
op {
  name: "TextLineReader"
  output_arg {
    name: "reader_handle"
    type: DT_STRING
    is_ref: true
  }
  attr {
    name: "skip_header_lines"
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
  name: "WholeFileReader"
  output_arg {
    name: "reader_handle"
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
  is_stateful: true
}
"""


_op_def_lib = _InitOpDefLibrary()
