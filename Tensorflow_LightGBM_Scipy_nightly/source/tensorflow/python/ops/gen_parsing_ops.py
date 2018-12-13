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

def decode_csv(records, record_defaults, field_delim=None, name=None):
  r"""Convert CSV records to tensors. Each column maps to one tensor.

  RFC 4180 format is expected for the CSV records.
  (https://tools.ietf.org/html/rfc4180)
  Note that we allow leading and trailing spaces with int or float field.

  Args:
    records: A `Tensor` of type `string`.
      Each string is a record/row in the csv and all records should have
      the same format.
    record_defaults: A list of `Tensor` objects with types from: `float32`, `int32`, `int64`, `string`.
      One tensor per column of the input record, with either a
      scalar default value for that column or empty if the column is required.
    field_delim: An optional `string`. Defaults to `","`.
      delimiter to separate fields in a record.
    name: A name for the operation (optional).

  Returns:
    A list of `Tensor` objects. Has the same type as `record_defaults`.
    Each tensor will have the same shape as records.
  """
  result = _op_def_lib.apply_op("DecodeCSV", records=records,
                                record_defaults=record_defaults,
                                field_delim=field_delim, name=name)
  return result



def decode_json_example(json_examples, name=None):
  r"""Convert JSON-encoded Example records to binary protocol buffer strings.

  This op translates a tensor containing Example records, encoded using
  the [standard JSON
  mapping](https://developers.google.com/protocol-buffers/docs/proto3#json),
  into a tensor containing the same records encoded as binary protocol
  buffers. The resulting tensor can then be fed to any of the other
  Example-parsing ops.

  Args:
    json_examples: A `Tensor` of type `string`.
      Each string is a JSON object serialized according to the JSON
      mapping of the Example proto.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`.
    Each string is a binary Example protocol buffer corresponding
    to the respective element of `json_examples`.
  """
  result = _op_def_lib.apply_op("DecodeJSONExample",
                                json_examples=json_examples, name=name)
  return result



def decode_raw(bytes, out_type, little_endian=None, name=None):
  r"""Reinterpret the bytes of a string as a vector of numbers.

  Args:
    bytes: A `Tensor` of type `string`.
      All the elements must have the same length.
    out_type: A `tf.DType` from: `tf.half, tf.float32, tf.float64, tf.int32, tf.uint8, tf.int16, tf.int8, tf.int64`.
    little_endian: An optional `bool`. Defaults to `True`.
      Whether the input `bytes` are in little-endian order.
      Ignored for `out_type` values that are stored in a single byte like
      `uint8`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `out_type`.
    A Tensor with one more dimension than the input `bytes`.  The
    added dimension will have size equal to the length of the elements
    of `bytes` divided by the number of bytes to represent `out_type`.
  """
  result = _op_def_lib.apply_op("DecodeRaw", bytes=bytes, out_type=out_type,
                                little_endian=little_endian, name=name)
  return result



__parse_example_outputs = ["sparse_indices", "sparse_values", "sparse_shapes",
                          "dense_values"]
_ParseExampleOutput = _collections.namedtuple(
    "ParseExample", __parse_example_outputs)


def _parse_example(serialized, names, sparse_keys, dense_keys, dense_defaults,
                   sparse_types, dense_shapes, name=None):
  r"""Transforms a vector of brain.Example protos (as strings) into typed tensors.

  Args:
    serialized: A `Tensor` of type `string`.
      A vector containing a batch of binary serialized Example protos.
    names: A `Tensor` of type `string`.
      A vector containing the names of the serialized protos.
      May contain, for example, table key (descriptive) names for the
      corresponding serialized protos.  These are purely useful for debugging
      purposes, and the presence of values here has no effect on the output.
      May also be an empty vector if no names are available.
      If non-empty, this vector must be the same length as "serialized".
    sparse_keys: A list of `Tensor` objects with type `string`.
      A list of Nsparse string Tensors (scalars).
      The keys expected in the Examples' features associated with sparse values.
    dense_keys: A list of `Tensor` objects with type `string`.
      A list of Ndense string Tensors (scalars).
      The keys expected in the Examples' features associated with dense values.
    dense_defaults: A list of `Tensor` objects with types from: `float32`, `int64`, `string`.
      A list of Ndense Tensors (some may be empty).
      dense_defaults[j] provides default values
      when the example's feature_map lacks dense_key[j].  If an empty Tensor is
      provided for dense_defaults[j], then the Feature dense_keys[j] is required.
      The input type is inferred from dense_defaults[j], even when it's empty.
      If dense_defaults[j] is not empty, and dense_shapes[j] is fully defined,
      then the shape of dense_defaults[j] must match that of dense_shapes[j].
      If dense_shapes[j] has an undefined major dimension (variable strides dense
      feature), dense_defaults[j] must contain a single element:
      the padding element.
    sparse_types: A list of `tf.DTypes` from: `tf.float32, tf.int64, tf.string`.
      A list of Nsparse types; the data types of data in each Feature
      given in sparse_keys.
      Currently the ParseExample supports DT_FLOAT (FloatList),
      DT_INT64 (Int64List), and DT_STRING (BytesList).
    dense_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`).
      A list of Ndense shapes; the shapes of data in each Feature
      given in dense_keys.
      The number of elements in the Feature corresponding to dense_key[j]
      must always equal dense_shapes[j].NumEntries().
      If dense_shapes[j] == (D0, D1, ..., DN) then the shape of output
      Tensor dense_values[j] will be (|serialized|, D0, D1, ..., DN):
      The dense outputs are just the inputs row-stacked by batch.
      This works for dense_shapes[j] = (-1, D1, ..., DN).  In this case
      the shape of the output Tensor dense_values[j] will be
      (|serialized|, M, D1, .., DN), where M is the maximum number of blocks
      of elements of length D1 * .... * DN, across all minibatch entries
      in the input.  Any minibatch entry with less than M blocks of elements of
      length D1 * ... * DN will be padded with the corresponding default_value
      scalar element along the second dimension.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (sparse_indices, sparse_values, sparse_shapes, dense_values).

    sparse_indices: A list with the same length as `sparse_keys` of `Tensor` objects with type `int64`.
    sparse_values: A list of `Tensor` objects of type `sparse_types`.
    sparse_shapes: A list with the same length as `sparse_keys` of `Tensor` objects with type `int64`.
    dense_values: A list of `Tensor` objects. Has the same type as `dense_defaults`.
  """
  result = _op_def_lib.apply_op("ParseExample", serialized=serialized,
                                names=names, sparse_keys=sparse_keys,
                                dense_keys=dense_keys,
                                dense_defaults=dense_defaults,
                                sparse_types=sparse_types,
                                dense_shapes=dense_shapes, name=name)
  return _ParseExampleOutput._make(result)



__parse_single_sequence_example_outputs = ["context_sparse_indices",
                                          "context_sparse_values",
                                          "context_sparse_shapes",
                                          "context_dense_values",
                                          "feature_list_sparse_indices",
                                          "feature_list_sparse_values",
                                          "feature_list_sparse_shapes",
                                          "feature_list_dense_values"]
_ParseSingleSequenceExampleOutput = _collections.namedtuple(
    "ParseSingleSequenceExample", __parse_single_sequence_example_outputs)


def _parse_single_sequence_example(serialized,
                                   feature_list_dense_missing_assumed_empty,
                                   context_sparse_keys, context_dense_keys,
                                   feature_list_sparse_keys,
                                   feature_list_dense_keys,
                                   context_dense_defaults, debug_name,
                                   context_sparse_types=None,
                                   feature_list_dense_types=None,
                                   context_dense_shapes=None,
                                   feature_list_sparse_types=None,
                                   feature_list_dense_shapes=None, name=None):
  r"""Transforms a scalar brain.SequenceExample proto (as strings) into typed tensors.

  Args:
    serialized: A `Tensor` of type `string`.
      A scalar containing a binary serialized SequenceExample proto.
    feature_list_dense_missing_assumed_empty: A `Tensor` of type `string`.
      A vector listing the
      FeatureList keys which may be missing from the SequenceExample.  If the
      associated FeatureList is missing, it is treated as empty.  By default,
      any FeatureList not listed in this vector must exist in the SequenceExample.
    context_sparse_keys: A list of `Tensor` objects with type `string`.
      A list of Ncontext_sparse string Tensors (scalars).
      The keys expected in the Examples' features associated with context_sparse
      values.
    context_dense_keys: A list of `Tensor` objects with type `string`.
      A list of Ncontext_dense string Tensors (scalars).
      The keys expected in the SequenceExamples' context features associated with
      dense values.
    feature_list_sparse_keys: A list of `Tensor` objects with type `string`.
      A list of Nfeature_list_sparse string Tensors
      (scalars).  The keys expected in the FeatureLists associated with sparse
      values.
    feature_list_dense_keys: A list of `Tensor` objects with type `string`.
      A list of Nfeature_list_dense string Tensors (scalars).
      The keys expected in the SequenceExamples' feature_lists associated
      with lists of dense values.
    context_dense_defaults: A list of `Tensor` objects with types from: `float32`, `int64`, `string`.
      A list of Ncontext_dense Tensors (some may be empty).
      context_dense_defaults[j] provides default values
      when the SequenceExample's context map lacks context_dense_key[j].
      If an empty Tensor is provided for context_dense_defaults[j],
      then the Feature context_dense_keys[j] is required.
      The input type is inferred from context_dense_defaults[j], even when it's
      empty.  If context_dense_defaults[j] is not empty, its shape must match
      context_dense_shapes[j].
    debug_name: A `Tensor` of type `string`.
      A scalar containing the name of the serialized proto.
      May contain, for example, table key (descriptive) name for the
      corresponding serialized proto.  This is purely useful for debugging
      purposes, and the presence of values here has no effect on the output.
      May also be an empty scalar if no name is available.
    context_sparse_types: An optional list of `tf.DTypes` from: `tf.float32, tf.int64, tf.string`. Defaults to `[]`.
      A list of Ncontext_sparse types; the data types of data in
      each context Feature given in context_sparse_keys.
      Currently the ParseSingleSequenceExample supports DT_FLOAT (FloatList),
      DT_INT64 (Int64List), and DT_STRING (BytesList).
    feature_list_dense_types: An optional list of `tf.DTypes` from: `tf.float32, tf.int64, tf.string`. Defaults to `[]`.
    context_dense_shapes: An optional list of shapes (each a `tf.TensorShape` or list of `ints`). Defaults to `[]`.
      A list of Ncontext_dense shapes; the shapes of data in
      each context Feature given in context_dense_keys.
      The number of elements in the Feature corresponding to context_dense_key[j]
      must always equal context_dense_shapes[j].NumEntries().
      The shape of context_dense_values[j] will match context_dense_shapes[j].
    feature_list_sparse_types: An optional list of `tf.DTypes` from: `tf.float32, tf.int64, tf.string`. Defaults to `[]`.
      A list of Nfeature_list_sparse types; the data types
      of data in each FeatureList given in feature_list_sparse_keys.
      Currently the ParseSingleSequenceExample supports DT_FLOAT (FloatList),
      DT_INT64 (Int64List), and DT_STRING (BytesList).
    feature_list_dense_shapes: An optional list of shapes (each a `tf.TensorShape` or list of `ints`). Defaults to `[]`.
      A list of Nfeature_list_dense shapes; the shapes of
      data in each FeatureList given in feature_list_dense_keys.
      The shape of each Feature in the FeatureList corresponding to
      feature_list_dense_key[j] must always equal
      feature_list_dense_shapes[j].NumEntries().
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (context_sparse_indices, context_sparse_values, context_sparse_shapes, context_dense_values, feature_list_sparse_indices, feature_list_sparse_values, feature_list_sparse_shapes, feature_list_dense_values).

    context_sparse_indices: A list with the same length as `context_sparse_keys` of `Tensor` objects with type `int64`.
    context_sparse_values: A list of `Tensor` objects of type `context_sparse_types`.
    context_sparse_shapes: A list with the same length as `context_sparse_keys` of `Tensor` objects with type `int64`.
    context_dense_values: A list of `Tensor` objects. Has the same type as `context_dense_defaults`.
    feature_list_sparse_indices: A list with the same length as `feature_list_sparse_keys` of `Tensor` objects with type `int64`.
    feature_list_sparse_values: A list of `Tensor` objects of type `feature_list_sparse_types`.
    feature_list_sparse_shapes: A list with the same length as `feature_list_sparse_keys` of `Tensor` objects with type `int64`.
    feature_list_dense_values: A list of `Tensor` objects of type `feature_list_dense_types`.
  """
  result = _op_def_lib.apply_op("ParseSingleSequenceExample",
                                serialized=serialized,
                                feature_list_dense_missing_assumed_empty=feature_list_dense_missing_assumed_empty,
                                context_sparse_keys=context_sparse_keys,
                                context_dense_keys=context_dense_keys,
                                feature_list_sparse_keys=feature_list_sparse_keys,
                                feature_list_dense_keys=feature_list_dense_keys,
                                context_dense_defaults=context_dense_defaults,
                                debug_name=debug_name,
                                context_sparse_types=context_sparse_types,
                                feature_list_dense_types=feature_list_dense_types,
                                context_dense_shapes=context_dense_shapes,
                                feature_list_sparse_types=feature_list_sparse_types,
                                feature_list_dense_shapes=feature_list_dense_shapes,
                                name=name)
  return _ParseSingleSequenceExampleOutput._make(result)



def parse_tensor(serialized, out_type, name=None):
  r"""Transforms a serialized tensorflow.TensorProto proto into a Tensor.

  Args:
    serialized: A `Tensor` of type `string`.
      A scalar string containing a serialized TensorProto proto.
    out_type: A `tf.DType`.
      The type of the serialized tensor.  The provided type must match the
      type of the serialized tensor and no implicit conversion will take place.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `out_type`. A Tensor of type `out_type`.
  """
  result = _op_def_lib.apply_op("ParseTensor", serialized=serialized,
                                out_type=out_type, name=name)
  return result



def string_to_number(string_tensor, out_type=None, name=None):
  r"""Converts each string in the input Tensor to the specified numeric type.

  (Note that int32 overflow results in an error while float overflow
  results in a rounded value.)

  Args:
    string_tensor: A `Tensor` of type `string`.
    out_type: An optional `tf.DType` from: `tf.float32, tf.float64, tf.int32, tf.int64`. Defaults to `tf.float32`.
      The numeric type to interpret each string in `string_tensor` as.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `out_type`.
    A Tensor of the same shape as the input `string_tensor`.
  """
  result = _op_def_lib.apply_op("StringToNumber", string_tensor=string_tensor,
                                out_type=out_type, name=name)
  return result


def _InitOpDefLibrary():
  op_list = _op_def_pb2.OpList()
  _text_format.Merge(_InitOpDefLibrary.op_list_ascii, op_list)
  _op_def_registry.register_op_list(op_list)
  op_def_lib = _op_def_library.OpDefLibrary()
  op_def_lib.add_op_list(op_list)
  return op_def_lib


_InitOpDefLibrary.op_list_ascii = """op {
  name: "DecodeCSV"
  input_arg {
    name: "records"
    type: DT_STRING
  }
  input_arg {
    name: "record_defaults"
    type_list_attr: "OUT_TYPE"
  }
  output_arg {
    name: "output"
    type_list_attr: "OUT_TYPE"
  }
  attr {
    name: "OUT_TYPE"
    type: "list(type)"
    has_minimum: true
    minimum: 1
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_INT32
        type: DT_INT64
        type: DT_STRING
      }
    }
  }
  attr {
    name: "field_delim"
    type: "string"
    default_value {
      s: ","
    }
  }
}
op {
  name: "DecodeJSONExample"
  input_arg {
    name: "json_examples"
    type: DT_STRING
  }
  output_arg {
    name: "binary_examples"
    type: DT_STRING
  }
}
op {
  name: "DecodeRaw"
  input_arg {
    name: "bytes"
    type: DT_STRING
  }
  output_arg {
    name: "output"
    type_attr: "out_type"
  }
  attr {
    name: "out_type"
    type: "type"
    allowed_values {
      list {
        type: DT_HALF
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT32
        type: DT_UINT8
        type: DT_INT16
        type: DT_INT8
        type: DT_INT64
      }
    }
  }
  attr {
    name: "little_endian"
    type: "bool"
    default_value {
      b: true
    }
  }
}
op {
  name: "ParseExample"
  input_arg {
    name: "serialized"
    type: DT_STRING
  }
  input_arg {
    name: "names"
    type: DT_STRING
  }
  input_arg {
    name: "sparse_keys"
    type: DT_STRING
    number_attr: "Nsparse"
  }
  input_arg {
    name: "dense_keys"
    type: DT_STRING
    number_attr: "Ndense"
  }
  input_arg {
    name: "dense_defaults"
    type_list_attr: "Tdense"
  }
  output_arg {
    name: "sparse_indices"
    type: DT_INT64
    number_attr: "Nsparse"
  }
  output_arg {
    name: "sparse_values"
    type_list_attr: "sparse_types"
  }
  output_arg {
    name: "sparse_shapes"
    type: DT_INT64
    number_attr: "Nsparse"
  }
  output_arg {
    name: "dense_values"
    type_list_attr: "Tdense"
  }
  attr {
    name: "Nsparse"
    type: "int"
    has_minimum: true
  }
  attr {
    name: "Ndense"
    type: "int"
    has_minimum: true
  }
  attr {
    name: "sparse_types"
    type: "list(type)"
    has_minimum: true
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_INT64
        type: DT_STRING
      }
    }
  }
  attr {
    name: "Tdense"
    type: "list(type)"
    has_minimum: true
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_INT64
        type: DT_STRING
      }
    }
  }
  attr {
    name: "dense_shapes"
    type: "list(shape)"
    has_minimum: true
  }
}
op {
  name: "ParseSingleSequenceExample"
  input_arg {
    name: "serialized"
    type: DT_STRING
  }
  input_arg {
    name: "feature_list_dense_missing_assumed_empty"
    type: DT_STRING
  }
  input_arg {
    name: "context_sparse_keys"
    type: DT_STRING
    number_attr: "Ncontext_sparse"
  }
  input_arg {
    name: "context_dense_keys"
    type: DT_STRING
    number_attr: "Ncontext_dense"
  }
  input_arg {
    name: "feature_list_sparse_keys"
    type: DT_STRING
    number_attr: "Nfeature_list_sparse"
  }
  input_arg {
    name: "feature_list_dense_keys"
    type: DT_STRING
    number_attr: "Nfeature_list_dense"
  }
  input_arg {
    name: "context_dense_defaults"
    type_list_attr: "Tcontext_dense"
  }
  input_arg {
    name: "debug_name"
    type: DT_STRING
  }
  output_arg {
    name: "context_sparse_indices"
    type: DT_INT64
    number_attr: "Ncontext_sparse"
  }
  output_arg {
    name: "context_sparse_values"
    type_list_attr: "context_sparse_types"
  }
  output_arg {
    name: "context_sparse_shapes"
    type: DT_INT64
    number_attr: "Ncontext_sparse"
  }
  output_arg {
    name: "context_dense_values"
    type_list_attr: "Tcontext_dense"
  }
  output_arg {
    name: "feature_list_sparse_indices"
    type: DT_INT64
    number_attr: "Nfeature_list_sparse"
  }
  output_arg {
    name: "feature_list_sparse_values"
    type_list_attr: "feature_list_sparse_types"
  }
  output_arg {
    name: "feature_list_sparse_shapes"
    type: DT_INT64
    number_attr: "Nfeature_list_sparse"
  }
  output_arg {
    name: "feature_list_dense_values"
    type_list_attr: "feature_list_dense_types"
  }
  attr {
    name: "Ncontext_sparse"
    type: "int"
    default_value {
      i: 0
    }
    has_minimum: true
  }
  attr {
    name: "Ncontext_dense"
    type: "int"
    default_value {
      i: 0
    }
    has_minimum: true
  }
  attr {
    name: "Nfeature_list_sparse"
    type: "int"
    default_value {
      i: 0
    }
    has_minimum: true
  }
  attr {
    name: "Nfeature_list_dense"
    type: "int"
    default_value {
      i: 0
    }
    has_minimum: true
  }
  attr {
    name: "context_sparse_types"
    type: "list(type)"
    default_value {
      list {
      }
    }
    has_minimum: true
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_INT64
        type: DT_STRING
      }
    }
  }
  attr {
    name: "Tcontext_dense"
    type: "list(type)"
    default_value {
      list {
      }
    }
    has_minimum: true
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_INT64
        type: DT_STRING
      }
    }
  }
  attr {
    name: "feature_list_dense_types"
    type: "list(type)"
    default_value {
      list {
      }
    }
    has_minimum: true
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_INT64
        type: DT_STRING
      }
    }
  }
  attr {
    name: "context_dense_shapes"
    type: "list(shape)"
    default_value {
      list {
      }
    }
    has_minimum: true
  }
  attr {
    name: "feature_list_sparse_types"
    type: "list(type)"
    default_value {
      list {
      }
    }
    has_minimum: true
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_INT64
        type: DT_STRING
      }
    }
  }
  attr {
    name: "feature_list_dense_shapes"
    type: "list(shape)"
    default_value {
      list {
      }
    }
    has_minimum: true
  }
}
op {
  name: "ParseTensor"
  input_arg {
    name: "serialized"
    type: DT_STRING
  }
  output_arg {
    name: "output"
    type_attr: "out_type"
  }
  attr {
    name: "out_type"
    type: "type"
  }
}
op {
  name: "StringToNumber"
  input_arg {
    name: "string_tensor"
    type: DT_STRING
  }
  output_arg {
    name: "output"
    type_attr: "out_type"
  }
  attr {
    name: "out_type"
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
      }
    }
  }
}
"""


_op_def_lib = _InitOpDefLibrary()
