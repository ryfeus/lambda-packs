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

def _hash_table(key_dtype, value_dtype, container=None, shared_name=None,
                use_node_name_sharing=None, name=None):
  r"""Creates a non-initialized hash table.

  This op creates a hash table, specifying the type of its keys and values.
  Before using the table you will have to initialize it.  After initialization the
  table will be immutable.

  Args:
    key_dtype: A `tf.DType`. Type of the table keys.
    value_dtype: A `tf.DType`. Type of the table values.
    container: An optional `string`. Defaults to `""`.
      If non-empty, this table is placed in the given container.
      Otherwise, a default container is used.
    shared_name: An optional `string`. Defaults to `""`.
      If non-empty, this table is shared under the given name across
      multiple sessions.
    use_node_name_sharing: An optional `bool`. Defaults to `False`.
      If true and shared_name is empty, the table is shared
      using the node name.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type mutable `string`. Handle to a table.
  """
  result = _op_def_lib.apply_op("HashTable", key_dtype=key_dtype,
                                value_dtype=value_dtype, container=container,
                                shared_name=shared_name,
                                use_node_name_sharing=use_node_name_sharing,
                                name=name)
  return result



def _hash_table_v2(key_dtype, value_dtype, container=None, shared_name=None,
                   use_node_name_sharing=None, name=None):
  r"""Creates a non-initialized hash table.

  This op creates a hash table, specifying the type of its keys and values.
  Before using the table you will have to initialize it.  After initialization the
  table will be immutable.

  Args:
    key_dtype: A `tf.DType`. Type of the table keys.
    value_dtype: A `tf.DType`. Type of the table values.
    container: An optional `string`. Defaults to `""`.
      If non-empty, this table is placed in the given container.
      Otherwise, a default container is used.
    shared_name: An optional `string`. Defaults to `""`.
      If non-empty, this table is shared under the given name across
      multiple sessions.
    use_node_name_sharing: An optional `bool`. Defaults to `False`.
      If true and shared_name is empty, the table is shared
      using the node name.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `resource`. Handle to a table.
  """
  result = _op_def_lib.apply_op("HashTableV2", key_dtype=key_dtype,
                                value_dtype=value_dtype, container=container,
                                shared_name=shared_name,
                                use_node_name_sharing=use_node_name_sharing,
                                name=name)
  return result



def _initialize_table(table_handle, keys, values, name=None):
  r"""Table initializer that takes two tensors for keys and values respectively.

  Args:
    table_handle: A `Tensor` of type mutable `string`.
      Handle to a table which will be initialized.
    keys: A `Tensor`. Keys of type Tkey.
    values: A `Tensor`. Values of type Tval.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  result = _op_def_lib.apply_op("InitializeTable", table_handle=table_handle,
                                keys=keys, values=values, name=name)
  return result



def _initialize_table_from_text_file(table_handle, filename, key_index,
                                     value_index, vocab_size=None,
                                     delimiter=None, name=None):
  r"""Initializes a table from a text file.

  It inserts one key-value pair into the table for each line of the file.
  The key and value is extracted from the whole line content, elements from the
  split line based on `delimiter` or the line number (starting from zero).
  Where to extract the key and value from a line is specified by `key_index` and
  `value_index`.

  - A value of -1 means use the line number(starting from zero), expects `int64`.
  - A value of -2 means use the whole line content, expects `string`.
  - A value >= 0 means use the index (starting at zero) of the split line based
    on `delimiter`.

  Args:
    table_handle: A `Tensor` of type mutable `string`.
      Handle to a table which will be initialized.
    filename: A `Tensor` of type `string`. Filename of a vocabulary text file.
    key_index: An `int` that is `>= -2`.
      Column index in a line to get the table `key` values from.
    value_index: An `int` that is `>= -2`.
      Column index that represents information of a line to get the table
      `value` values from.
    vocab_size: An optional `int` that is `>= -1`. Defaults to `-1`.
      Number of elements of the file, use -1 if unknown.
    delimiter: An optional `string`. Defaults to `"\t"`.
      Delimiter to separate fields in a line.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  result = _op_def_lib.apply_op("InitializeTableFromTextFile",
                                table_handle=table_handle, filename=filename,
                                key_index=key_index, value_index=value_index,
                                vocab_size=vocab_size, delimiter=delimiter,
                                name=name)
  return result



def _initialize_table_from_text_file_v2(table_handle, filename, key_index,
                                        value_index, vocab_size=None,
                                        delimiter=None, name=None):
  r"""Initializes a table from a text file.

  It inserts one key-value pair into the table for each line of the file.
  The key and value is extracted from the whole line content, elements from the
  split line based on `delimiter` or the line number (starting from zero).
  Where to extract the key and value from a line is specified by `key_index` and
  `value_index`.

  - A value of -1 means use the line number(starting from zero), expects `int64`.
  - A value of -2 means use the whole line content, expects `string`.
  - A value >= 0 means use the index (starting at zero) of the split line based
    on `delimiter`.

  Args:
    table_handle: A `Tensor` of type `resource`.
      Handle to a table which will be initialized.
    filename: A `Tensor` of type `string`. Filename of a vocabulary text file.
    key_index: An `int` that is `>= -2`.
      Column index in a line to get the table `key` values from.
    value_index: An `int` that is `>= -2`.
      Column index that represents information of a line to get the table
      `value` values from.
    vocab_size: An optional `int` that is `>= -1`. Defaults to `-1`.
      Number of elements of the file, use -1 if unknown.
    delimiter: An optional `string`. Defaults to `"\t"`.
      Delimiter to separate fields in a line.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  result = _op_def_lib.apply_op("InitializeTableFromTextFileV2",
                                table_handle=table_handle, filename=filename,
                                key_index=key_index, value_index=value_index,
                                vocab_size=vocab_size, delimiter=delimiter,
                                name=name)
  return result



def _initialize_table_v2(table_handle, keys, values, name=None):
  r"""Table initializer that takes two tensors for keys and values respectively.

  Args:
    table_handle: A `Tensor` of type `resource`.
      Handle to a table which will be initialized.
    keys: A `Tensor`. Keys of type Tkey.
    values: A `Tensor`. Values of type Tval.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  result = _op_def_lib.apply_op("InitializeTableV2",
                                table_handle=table_handle, keys=keys,
                                values=values, name=name)
  return result



__lookup_table_export_outputs = ["keys", "values"]
_LookupTableExportOutput = _collections.namedtuple(
    "LookupTableExport", __lookup_table_export_outputs)


def _lookup_table_export(table_handle, Tkeys, Tvalues, name=None):
  r"""Outputs all keys and values in the table.

  Args:
    table_handle: A `Tensor` of type mutable `string`. Handle to the table.
    Tkeys: A `tf.DType`.
    Tvalues: A `tf.DType`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (keys, values).

    keys: A `Tensor` of type `Tkeys`. Vector of all keys present in the table.
    values: A `Tensor` of type `Tvalues`. Tensor of all values in the table. Indexed in parallel with `keys`.
  """
  result = _op_def_lib.apply_op("LookupTableExport",
                                table_handle=table_handle, Tkeys=Tkeys,
                                Tvalues=Tvalues, name=name)
  return _LookupTableExportOutput._make(result)



__lookup_table_export_v2_outputs = ["keys", "values"]
_LookupTableExportV2Output = _collections.namedtuple(
    "LookupTableExportV2", __lookup_table_export_v2_outputs)


def _lookup_table_export_v2(table_handle, Tkeys, Tvalues, name=None):
  r"""Outputs all keys and values in the table.

  Args:
    table_handle: A `Tensor` of type `resource`. Handle to the table.
    Tkeys: A `tf.DType`.
    Tvalues: A `tf.DType`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (keys, values).

    keys: A `Tensor` of type `Tkeys`. Vector of all keys present in the table.
    values: A `Tensor` of type `Tvalues`. Tensor of all values in the table. Indexed in parallel with `keys`.
  """
  result = _op_def_lib.apply_op("LookupTableExportV2",
                                table_handle=table_handle, Tkeys=Tkeys,
                                Tvalues=Tvalues, name=name)
  return _LookupTableExportV2Output._make(result)



def _lookup_table_find(table_handle, keys, default_value, name=None):
  r"""Looks up keys in a table, outputs the corresponding values.

  The tensor `keys` must of the same type as the keys of the table.
  The output `values` is of the type of the table values.

  The scalar `default_value` is the value output for keys not present in the
  table. It must also be of the same type as the table values.

  Args:
    table_handle: A `Tensor` of type mutable `string`. Handle to the table.
    keys: A `Tensor`. Any shape.  Keys to look up.
    default_value: A `Tensor`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `default_value`.
    Same shape as `keys`.  Values found in the table, or `default_values`
    for missing keys.
  """
  result = _op_def_lib.apply_op("LookupTableFind", table_handle=table_handle,
                                keys=keys, default_value=default_value,
                                name=name)
  return result



def _lookup_table_find_v2(table_handle, keys, default_value, name=None):
  r"""Looks up keys in a table, outputs the corresponding values.

  The tensor `keys` must of the same type as the keys of the table.
  The output `values` is of the type of the table values.

  The scalar `default_value` is the value output for keys not present in the
  table. It must also be of the same type as the table values.

  Args:
    table_handle: A `Tensor` of type `resource`. Handle to the table.
    keys: A `Tensor`. Any shape.  Keys to look up.
    default_value: A `Tensor`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `default_value`.
    Same shape as `keys`.  Values found in the table, or `default_values`
    for missing keys.
  """
  result = _op_def_lib.apply_op("LookupTableFindV2",
                                table_handle=table_handle, keys=keys,
                                default_value=default_value, name=name)
  return result



def _lookup_table_import(table_handle, keys, values, name=None):
  r"""Replaces the contents of the table with the specified keys and values.

  The tensor `keys` must be of the same type as the keys of the table.
  The tensor `values` must be of the type of the table values.

  Args:
    table_handle: A `Tensor` of type mutable `string`. Handle to the table.
    keys: A `Tensor`. Any shape.  Keys to look up.
    values: A `Tensor`. Values to associate with keys.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  result = _op_def_lib.apply_op("LookupTableImport",
                                table_handle=table_handle, keys=keys,
                                values=values, name=name)
  return result



def _lookup_table_import_v2(table_handle, keys, values, name=None):
  r"""Replaces the contents of the table with the specified keys and values.

  The tensor `keys` must be of the same type as the keys of the table.
  The tensor `values` must be of the type of the table values.

  Args:
    table_handle: A `Tensor` of type `resource`. Handle to the table.
    keys: A `Tensor`. Any shape.  Keys to look up.
    values: A `Tensor`. Values to associate with keys.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  result = _op_def_lib.apply_op("LookupTableImportV2",
                                table_handle=table_handle, keys=keys,
                                values=values, name=name)
  return result



def _lookup_table_insert(table_handle, keys, values, name=None):
  r"""Updates the table to associates keys with values.

  The tensor `keys` must be of the same type as the keys of the table.
  The tensor `values` must be of the type of the table values.

  Args:
    table_handle: A `Tensor` of type mutable `string`. Handle to the table.
    keys: A `Tensor`. Any shape.  Keys to look up.
    values: A `Tensor`. Values to associate with keys.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  result = _op_def_lib.apply_op("LookupTableInsert",
                                table_handle=table_handle, keys=keys,
                                values=values, name=name)
  return result



def _lookup_table_insert_v2(table_handle, keys, values, name=None):
  r"""Updates the table to associates keys with values.

  The tensor `keys` must be of the same type as the keys of the table.
  The tensor `values` must be of the type of the table values.

  Args:
    table_handle: A `Tensor` of type `resource`. Handle to the table.
    keys: A `Tensor`. Any shape.  Keys to look up.
    values: A `Tensor`. Values to associate with keys.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  result = _op_def_lib.apply_op("LookupTableInsertV2",
                                table_handle=table_handle, keys=keys,
                                values=values, name=name)
  return result



def _lookup_table_size(table_handle, name=None):
  r"""Computes the number of elements in the given table.

  Args:
    table_handle: A `Tensor` of type mutable `string`. Handle to the table.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int64`.
    Scalar that contains number of elements in the table.
  """
  result = _op_def_lib.apply_op("LookupTableSize", table_handle=table_handle,
                                name=name)
  return result



def _lookup_table_size_v2(table_handle, name=None):
  r"""Computes the number of elements in the given table.

  Args:
    table_handle: A `Tensor` of type `resource`. Handle to the table.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int64`.
    Scalar that contains number of elements in the table.
  """
  result = _op_def_lib.apply_op("LookupTableSizeV2",
                                table_handle=table_handle, name=name)
  return result



def _mutable_dense_hash_table(empty_key, value_dtype, container=None,
                              shared_name=None, use_node_name_sharing=None,
                              value_shape=None, initial_num_buckets=None,
                              max_load_factor=None, name=None):
  r"""Creates an empty hash table that uses tensors as the backing store.

  It uses "open addressing" with quadratic reprobing to resolve
  collisions.

  This op creates a mutable hash table, specifying the type of its keys and
  values. Each value must be a scalar. Data can be inserted into the table using
  the insert operations. It does not support the initialization operation.

  Args:
    empty_key: A `Tensor`.
      The key used to represent empty key buckets internally. Must not
      be used in insert or lookup operations.
    value_dtype: A `tf.DType`. Type of the table values.
    container: An optional `string`. Defaults to `""`.
      If non-empty, this table is placed in the given container.
      Otherwise, a default container is used.
    shared_name: An optional `string`. Defaults to `""`.
      If non-empty, this table is shared under the given name across
      multiple sessions.
    use_node_name_sharing: An optional `bool`. Defaults to `False`.
    value_shape: An optional `tf.TensorShape` or list of `ints`. Defaults to `[]`.
      The shape of each value.
    initial_num_buckets: An optional `int`. Defaults to `131072`.
      The initial number of hash table buckets. Must be a power
      to 2.
    max_load_factor: An optional `float`. Defaults to `0.8`.
      The maximum ratio between number of entries and number of
      buckets before growing the table. Must be between 0 and 1.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type mutable `string`. Handle to a table.
  """
  result = _op_def_lib.apply_op("MutableDenseHashTable", empty_key=empty_key,
                                value_dtype=value_dtype, container=container,
                                shared_name=shared_name,
                                use_node_name_sharing=use_node_name_sharing,
                                value_shape=value_shape,
                                initial_num_buckets=initial_num_buckets,
                                max_load_factor=max_load_factor, name=name)
  return result



def _mutable_dense_hash_table_v2(empty_key, value_dtype, container=None,
                                 shared_name=None, use_node_name_sharing=None,
                                 value_shape=None, initial_num_buckets=None,
                                 max_load_factor=None, name=None):
  r"""Creates an empty hash table that uses tensors as the backing store.

  It uses "open addressing" with quadratic reprobing to resolve
  collisions.

  This op creates a mutable hash table, specifying the type of its keys and
  values. Each value must be a scalar. Data can be inserted into the table using
  the insert operations. It does not support the initialization operation.

  Args:
    empty_key: A `Tensor`.
      The key used to represent empty key buckets internally. Must not
      be used in insert or lookup operations.
    value_dtype: A `tf.DType`. Type of the table values.
    container: An optional `string`. Defaults to `""`.
      If non-empty, this table is placed in the given container.
      Otherwise, a default container is used.
    shared_name: An optional `string`. Defaults to `""`.
      If non-empty, this table is shared under the given name across
      multiple sessions.
    use_node_name_sharing: An optional `bool`. Defaults to `False`.
    value_shape: An optional `tf.TensorShape` or list of `ints`. Defaults to `[]`.
      The shape of each value.
    initial_num_buckets: An optional `int`. Defaults to `131072`.
      The initial number of hash table buckets. Must be a power
      to 2.
    max_load_factor: An optional `float`. Defaults to `0.8`.
      The maximum ratio between number of entries and number of
      buckets before growing the table. Must be between 0 and 1.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `resource`. Handle to a table.
  """
  result = _op_def_lib.apply_op("MutableDenseHashTableV2",
                                empty_key=empty_key, value_dtype=value_dtype,
                                container=container, shared_name=shared_name,
                                use_node_name_sharing=use_node_name_sharing,
                                value_shape=value_shape,
                                initial_num_buckets=initial_num_buckets,
                                max_load_factor=max_load_factor, name=name)
  return result



def _mutable_hash_table(key_dtype, value_dtype, container=None,
                        shared_name=None, use_node_name_sharing=None,
                        name=None):
  r"""Creates an empty hash table.

  This op creates a mutable hash table, specifying the type of its keys and
  values. Each value must be a scalar. Data can be inserted into the table using
  the insert operations. It does not support the initialization operation.

  Args:
    key_dtype: A `tf.DType`. Type of the table keys.
    value_dtype: A `tf.DType`. Type of the table values.
    container: An optional `string`. Defaults to `""`.
      If non-empty, this table is placed in the given container.
      Otherwise, a default container is used.
    shared_name: An optional `string`. Defaults to `""`.
      If non-empty, this table is shared under the given name across
      multiple sessions.
    use_node_name_sharing: An optional `bool`. Defaults to `False`.
      If true and shared_name is empty, the table is shared
      using the node name.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type mutable `string`. Handle to a table.
  """
  result = _op_def_lib.apply_op("MutableHashTable", key_dtype=key_dtype,
                                value_dtype=value_dtype, container=container,
                                shared_name=shared_name,
                                use_node_name_sharing=use_node_name_sharing,
                                name=name)
  return result



def _mutable_hash_table_of_tensors(key_dtype, value_dtype, container=None,
                                   shared_name=None,
                                   use_node_name_sharing=None,
                                   value_shape=None, name=None):
  r"""Creates an empty hash table.

  This op creates a mutable hash table, specifying the type of its keys and
  values. Each value must be a vector. Data can be inserted into the table using
  the insert operations. It does not support the initialization operation.

  Args:
    key_dtype: A `tf.DType`. Type of the table keys.
    value_dtype: A `tf.DType`. Type of the table values.
    container: An optional `string`. Defaults to `""`.
      If non-empty, this table is placed in the given container.
      Otherwise, a default container is used.
    shared_name: An optional `string`. Defaults to `""`.
      If non-empty, this table is shared under the given name across
      multiple sessions.
    use_node_name_sharing: An optional `bool`. Defaults to `False`.
    value_shape: An optional `tf.TensorShape` or list of `ints`. Defaults to `[]`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type mutable `string`. Handle to a table.
  """
  result = _op_def_lib.apply_op("MutableHashTableOfTensors",
                                key_dtype=key_dtype, value_dtype=value_dtype,
                                container=container, shared_name=shared_name,
                                use_node_name_sharing=use_node_name_sharing,
                                value_shape=value_shape, name=name)
  return result



def _mutable_hash_table_of_tensors_v2(key_dtype, value_dtype, container=None,
                                      shared_name=None,
                                      use_node_name_sharing=None,
                                      value_shape=None, name=None):
  r"""Creates an empty hash table.

  This op creates a mutable hash table, specifying the type of its keys and
  values. Each value must be a vector. Data can be inserted into the table using
  the insert operations. It does not support the initialization operation.

  Args:
    key_dtype: A `tf.DType`. Type of the table keys.
    value_dtype: A `tf.DType`. Type of the table values.
    container: An optional `string`. Defaults to `""`.
      If non-empty, this table is placed in the given container.
      Otherwise, a default container is used.
    shared_name: An optional `string`. Defaults to `""`.
      If non-empty, this table is shared under the given name across
      multiple sessions.
    use_node_name_sharing: An optional `bool`. Defaults to `False`.
    value_shape: An optional `tf.TensorShape` or list of `ints`. Defaults to `[]`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `resource`. Handle to a table.
  """
  result = _op_def_lib.apply_op("MutableHashTableOfTensorsV2",
                                key_dtype=key_dtype, value_dtype=value_dtype,
                                container=container, shared_name=shared_name,
                                use_node_name_sharing=use_node_name_sharing,
                                value_shape=value_shape, name=name)
  return result



def _mutable_hash_table_v2(key_dtype, value_dtype, container=None,
                           shared_name=None, use_node_name_sharing=None,
                           name=None):
  r"""Creates an empty hash table.

  This op creates a mutable hash table, specifying the type of its keys and
  values. Each value must be a scalar. Data can be inserted into the table using
  the insert operations. It does not support the initialization operation.

  Args:
    key_dtype: A `tf.DType`. Type of the table keys.
    value_dtype: A `tf.DType`. Type of the table values.
    container: An optional `string`. Defaults to `""`.
      If non-empty, this table is placed in the given container.
      Otherwise, a default container is used.
    shared_name: An optional `string`. Defaults to `""`.
      If non-empty, this table is shared under the given name across
      multiple sessions.
    use_node_name_sharing: An optional `bool`. Defaults to `False`.
      If true and shared_name is empty, the table is shared
      using the node name.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `resource`. Handle to a table.
  """
  result = _op_def_lib.apply_op("MutableHashTableV2", key_dtype=key_dtype,
                                value_dtype=value_dtype, container=container,
                                shared_name=shared_name,
                                use_node_name_sharing=use_node_name_sharing,
                                name=name)
  return result


def _InitOpDefLibrary():
  op_list = _op_def_pb2.OpList()
  _text_format.Merge(_InitOpDefLibrary.op_list_ascii, op_list)
  _op_def_registry.register_op_list(op_list)
  op_def_lib = _op_def_library.OpDefLibrary()
  op_def_lib.add_op_list(op_list)
  return op_def_lib


_InitOpDefLibrary.op_list_ascii = """op {
  name: "HashTable"
  output_arg {
    name: "table_handle"
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
  attr {
    name: "use_node_name_sharing"
    type: "bool"
    default_value {
      b: false
    }
  }
  attr {
    name: "key_dtype"
    type: "type"
  }
  attr {
    name: "value_dtype"
    type: "type"
  }
  is_stateful: true
}
op {
  name: "HashTableV2"
  output_arg {
    name: "table_handle"
    type: DT_RESOURCE
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
  attr {
    name: "use_node_name_sharing"
    type: "bool"
    default_value {
      b: false
    }
  }
  attr {
    name: "key_dtype"
    type: "type"
  }
  attr {
    name: "value_dtype"
    type: "type"
  }
  is_stateful: true
}
op {
  name: "InitializeTable"
  input_arg {
    name: "table_handle"
    type: DT_STRING
    is_ref: true
  }
  input_arg {
    name: "keys"
    type_attr: "Tkey"
  }
  input_arg {
    name: "values"
    type_attr: "Tval"
  }
  attr {
    name: "Tkey"
    type: "type"
  }
  attr {
    name: "Tval"
    type: "type"
  }
}
op {
  name: "InitializeTableFromTextFile"
  input_arg {
    name: "table_handle"
    type: DT_STRING
    is_ref: true
  }
  input_arg {
    name: "filename"
    type: DT_STRING
  }
  attr {
    name: "key_index"
    type: "int"
    has_minimum: true
    minimum: -2
  }
  attr {
    name: "value_index"
    type: "int"
    has_minimum: true
    minimum: -2
  }
  attr {
    name: "vocab_size"
    type: "int"
    default_value {
      i: -1
    }
    has_minimum: true
    minimum: -1
  }
  attr {
    name: "delimiter"
    type: "string"
    default_value {
      s: "\t"
    }
  }
}
op {
  name: "InitializeTableFromTextFileV2"
  input_arg {
    name: "table_handle"
    type: DT_RESOURCE
  }
  input_arg {
    name: "filename"
    type: DT_STRING
  }
  attr {
    name: "key_index"
    type: "int"
    has_minimum: true
    minimum: -2
  }
  attr {
    name: "value_index"
    type: "int"
    has_minimum: true
    minimum: -2
  }
  attr {
    name: "vocab_size"
    type: "int"
    default_value {
      i: -1
    }
    has_minimum: true
    minimum: -1
  }
  attr {
    name: "delimiter"
    type: "string"
    default_value {
      s: "\t"
    }
  }
  is_stateful: true
}
op {
  name: "InitializeTableV2"
  input_arg {
    name: "table_handle"
    type: DT_RESOURCE
  }
  input_arg {
    name: "keys"
    type_attr: "Tkey"
  }
  input_arg {
    name: "values"
    type_attr: "Tval"
  }
  attr {
    name: "Tkey"
    type: "type"
  }
  attr {
    name: "Tval"
    type: "type"
  }
  is_stateful: true
}
op {
  name: "LookupTableExport"
  input_arg {
    name: "table_handle"
    type: DT_STRING
    is_ref: true
  }
  output_arg {
    name: "keys"
    type_attr: "Tkeys"
  }
  output_arg {
    name: "values"
    type_attr: "Tvalues"
  }
  attr {
    name: "Tkeys"
    type: "type"
  }
  attr {
    name: "Tvalues"
    type: "type"
  }
}
op {
  name: "LookupTableExportV2"
  input_arg {
    name: "table_handle"
    type: DT_RESOURCE
  }
  output_arg {
    name: "keys"
    type_attr: "Tkeys"
  }
  output_arg {
    name: "values"
    type_attr: "Tvalues"
  }
  attr {
    name: "Tkeys"
    type: "type"
  }
  attr {
    name: "Tvalues"
    type: "type"
  }
  is_stateful: true
}
op {
  name: "LookupTableFind"
  input_arg {
    name: "table_handle"
    type: DT_STRING
    is_ref: true
  }
  input_arg {
    name: "keys"
    type_attr: "Tin"
  }
  input_arg {
    name: "default_value"
    type_attr: "Tout"
  }
  output_arg {
    name: "values"
    type_attr: "Tout"
  }
  attr {
    name: "Tin"
    type: "type"
  }
  attr {
    name: "Tout"
    type: "type"
  }
}
op {
  name: "LookupTableFindV2"
  input_arg {
    name: "table_handle"
    type: DT_RESOURCE
  }
  input_arg {
    name: "keys"
    type_attr: "Tin"
  }
  input_arg {
    name: "default_value"
    type_attr: "Tout"
  }
  output_arg {
    name: "values"
    type_attr: "Tout"
  }
  attr {
    name: "Tin"
    type: "type"
  }
  attr {
    name: "Tout"
    type: "type"
  }
  is_stateful: true
}
op {
  name: "LookupTableImport"
  input_arg {
    name: "table_handle"
    type: DT_STRING
    is_ref: true
  }
  input_arg {
    name: "keys"
    type_attr: "Tin"
  }
  input_arg {
    name: "values"
    type_attr: "Tout"
  }
  attr {
    name: "Tin"
    type: "type"
  }
  attr {
    name: "Tout"
    type: "type"
  }
}
op {
  name: "LookupTableImportV2"
  input_arg {
    name: "table_handle"
    type: DT_RESOURCE
  }
  input_arg {
    name: "keys"
    type_attr: "Tin"
  }
  input_arg {
    name: "values"
    type_attr: "Tout"
  }
  attr {
    name: "Tin"
    type: "type"
  }
  attr {
    name: "Tout"
    type: "type"
  }
  is_stateful: true
}
op {
  name: "LookupTableInsert"
  input_arg {
    name: "table_handle"
    type: DT_STRING
    is_ref: true
  }
  input_arg {
    name: "keys"
    type_attr: "Tin"
  }
  input_arg {
    name: "values"
    type_attr: "Tout"
  }
  attr {
    name: "Tin"
    type: "type"
  }
  attr {
    name: "Tout"
    type: "type"
  }
}
op {
  name: "LookupTableInsertV2"
  input_arg {
    name: "table_handle"
    type: DT_RESOURCE
  }
  input_arg {
    name: "keys"
    type_attr: "Tin"
  }
  input_arg {
    name: "values"
    type_attr: "Tout"
  }
  attr {
    name: "Tin"
    type: "type"
  }
  attr {
    name: "Tout"
    type: "type"
  }
  is_stateful: true
}
op {
  name: "LookupTableSize"
  input_arg {
    name: "table_handle"
    type: DT_STRING
    is_ref: true
  }
  output_arg {
    name: "size"
    type: DT_INT64
  }
}
op {
  name: "LookupTableSizeV2"
  input_arg {
    name: "table_handle"
    type: DT_RESOURCE
  }
  output_arg {
    name: "size"
    type: DT_INT64
  }
  is_stateful: true
}
op {
  name: "MutableDenseHashTable"
  input_arg {
    name: "empty_key"
    type_attr: "key_dtype"
  }
  output_arg {
    name: "table_handle"
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
  attr {
    name: "use_node_name_sharing"
    type: "bool"
    default_value {
      b: false
    }
  }
  attr {
    name: "key_dtype"
    type: "type"
  }
  attr {
    name: "value_dtype"
    type: "type"
  }
  attr {
    name: "value_shape"
    type: "shape"
    default_value {
      shape {
      }
    }
  }
  attr {
    name: "initial_num_buckets"
    type: "int"
    default_value {
      i: 131072
    }
  }
  attr {
    name: "max_load_factor"
    type: "float"
    default_value {
      f: 0.8
    }
  }
  is_stateful: true
}
op {
  name: "MutableDenseHashTableV2"
  input_arg {
    name: "empty_key"
    type_attr: "key_dtype"
  }
  output_arg {
    name: "table_handle"
    type: DT_RESOURCE
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
  attr {
    name: "use_node_name_sharing"
    type: "bool"
    default_value {
      b: false
    }
  }
  attr {
    name: "key_dtype"
    type: "type"
  }
  attr {
    name: "value_dtype"
    type: "type"
  }
  attr {
    name: "value_shape"
    type: "shape"
    default_value {
      shape {
      }
    }
  }
  attr {
    name: "initial_num_buckets"
    type: "int"
    default_value {
      i: 131072
    }
  }
  attr {
    name: "max_load_factor"
    type: "float"
    default_value {
      f: 0.8
    }
  }
  is_stateful: true
}
op {
  name: "MutableHashTable"
  output_arg {
    name: "table_handle"
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
  attr {
    name: "use_node_name_sharing"
    type: "bool"
    default_value {
      b: false
    }
  }
  attr {
    name: "key_dtype"
    type: "type"
  }
  attr {
    name: "value_dtype"
    type: "type"
  }
  is_stateful: true
}
op {
  name: "MutableHashTableOfTensors"
  output_arg {
    name: "table_handle"
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
  attr {
    name: "use_node_name_sharing"
    type: "bool"
    default_value {
      b: false
    }
  }
  attr {
    name: "key_dtype"
    type: "type"
  }
  attr {
    name: "value_dtype"
    type: "type"
  }
  attr {
    name: "value_shape"
    type: "shape"
    default_value {
      shape {
      }
    }
  }
  is_stateful: true
}
op {
  name: "MutableHashTableOfTensorsV2"
  output_arg {
    name: "table_handle"
    type: DT_RESOURCE
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
  attr {
    name: "use_node_name_sharing"
    type: "bool"
    default_value {
      b: false
    }
  }
  attr {
    name: "key_dtype"
    type: "type"
  }
  attr {
    name: "value_dtype"
    type: "type"
  }
  attr {
    name: "value_shape"
    type: "shape"
    default_value {
      shape {
      }
    }
  }
  is_stateful: true
}
op {
  name: "MutableHashTableV2"
  output_arg {
    name: "table_handle"
    type: DT_RESOURCE
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
  attr {
    name: "use_node_name_sharing"
    type: "bool"
    default_value {
      b: false
    }
  }
  attr {
    name: "key_dtype"
    type: "type"
  }
  attr {
    name: "value_dtype"
    type: "type"
  }
  is_stateful: true
}
"""


_op_def_lib = _InitOpDefLibrary()
