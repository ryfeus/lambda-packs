"""Python wrappers around TensorFlow ops.

This file is MACHINE GENERATED! Do not edit.
Original C++ source file: nearest_neighbor_ops_pywrapper.cc
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


_hyperplane_lsh_probes_outputs = ["probes", "table_ids"]
_HyperplaneLSHProbesOutput = _collections.namedtuple(
    "HyperplaneLSHProbes", _hyperplane_lsh_probes_outputs)


def hyperplane_lsh_probes(point_hyperplane_product, num_tables, num_hyperplanes_per_table, num_probes, name=None):
  r"""Computes probes for the hyperplane hash.

  The op supports multiprobing, i.e., the number of requested probes can be
  larger than the number of tables. In that case, the same table can be probed
  multiple times.

  The first `num_tables` probes are always the primary hashes for each table.

  Args:
    point_hyperplane_product: A `Tensor`. Must be one of the following types: `float32`, `float64`.
      a matrix of inner products between the hyperplanes
      and the points to be hashed. These values should not be quantized so that we
      can correctly compute the probing sequence. The expected shape is
      `batch_size` times `num_tables * num_hyperplanes_per_table`, i.e., each
      element of the batch corresponds to one row of the matrix.
    num_tables: A `Tensor` of type `int32`.
      the number of tables to compute probes for.
    num_hyperplanes_per_table: A `Tensor` of type `int32`.
      the number of hyperplanes per table.
    num_probes: A `Tensor` of type `int32`.
      the requested number of probes per table.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (probes, table_ids).

    probes: A `Tensor` of type `int32`. the output matrix of probes. Size `batch_size` times `num_probes`.
    table_ids: A `Tensor` of type `int32`. the output matrix of tables ids. Size `batch_size` times `num_probes`.
  """
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "HyperplaneLSHProbes",
        point_hyperplane_product=point_hyperplane_product,
        num_tables=num_tables,
        num_hyperplanes_per_table=num_hyperplanes_per_table,
        num_probes=num_probes, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("CoordinateType", _op.get_attr("CoordinateType"))
  else:
    _attr_CoordinateType, (point_hyperplane_product,) = _execute.args_to_matching_eager([point_hyperplane_product], _ctx)
    _attr_CoordinateType = _attr_CoordinateType.as_datatype_enum
    num_tables = _ops.convert_to_tensor(num_tables, _dtypes.int32)
    num_hyperplanes_per_table = _ops.convert_to_tensor(num_hyperplanes_per_table, _dtypes.int32)
    num_probes = _ops.convert_to_tensor(num_probes, _dtypes.int32)
    _inputs_flat = [point_hyperplane_product, num_tables, num_hyperplanes_per_table, num_probes]
    _attrs = ("CoordinateType", _attr_CoordinateType)
    _result = _execute.execute(b"HyperplaneLSHProbes", 2, inputs=_inputs_flat,
                               attrs=_attrs, ctx=_ctx, name=name)
  _execute.record_gradient(
      "HyperplaneLSHProbes", _inputs_flat, _attrs, _result, name)
  _result = _HyperplaneLSHProbesOutput._make(_result)
  return _result

_ops.RegisterShape("HyperplaneLSHProbes")(None)

def _InitOpDefLibrary(op_list_proto_bytes):
  op_list = _op_def_pb2.OpList()
  op_list.ParseFromString(op_list_proto_bytes)
  _op_def_registry.register_op_list(op_list)
  op_def_lib = _op_def_library.OpDefLibrary()
  op_def_lib.add_op_list(op_list)
  return op_def_lib
# op {
#   name: "HyperplaneLSHProbes"
#   input_arg {
#     name: "point_hyperplane_product"
#     type_attr: "CoordinateType"
#   }
#   input_arg {
#     name: "num_tables"
#     type: DT_INT32
#   }
#   input_arg {
#     name: "num_hyperplanes_per_table"
#     type: DT_INT32
#   }
#   input_arg {
#     name: "num_probes"
#     type: DT_INT32
#   }
#   output_arg {
#     name: "probes"
#     type: DT_INT32
#   }
#   output_arg {
#     name: "table_ids"
#     type: DT_INT32
#   }
#   attr {
#     name: "CoordinateType"
#     type: "type"
#     allowed_values {
#       list {
#         type: DT_FLOAT
#         type: DT_DOUBLE
#       }
#     }
#   }
# }
_op_def_lib = _InitOpDefLibrary(b"\n\273\001\n\023HyperplaneLSHProbes\022*\n\030point_hyperplane_product\"\016CoordinateType\022\016\n\nnum_tables\030\003\022\035\n\031num_hyperplanes_per_table\030\003\022\016\n\nnum_probes\030\003\032\n\n\006probes\030\003\032\r\n\ttable_ids\030\003\"\036\n\016CoordinateType\022\004type:\006\n\0042\002\001\002")
