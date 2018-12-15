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

def _batch_cholesky(input, name=None):
  r"""TODO: add doc.

  Args:
    input: A `Tensor`. Must be one of the following types: `float64`, `float32`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  result = _op_def_lib.apply_op("BatchCholesky", input=input, name=name)
  return result



def _batch_cholesky_grad(l, grad, name=None):
  r"""TODO: add doc.

  Args:
    l: A `Tensor`. Must be one of the following types: `float32`, `float64`.
    grad: A `Tensor`. Must have the same type as `l`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `l`.
  """
  result = _op_def_lib.apply_op("BatchCholeskyGrad", l=l, grad=grad,
                                name=name)
  return result



def _batch_matrix_determinant(input, name=None):
  r"""TODO: add doc.

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  result = _op_def_lib.apply_op("BatchMatrixDeterminant", input=input,
                                name=name)
  return result



def _batch_matrix_inverse(input, adjoint=None, name=None):
  r"""TODO: add doc.

  Args:
    input: A `Tensor`. Must be one of the following types: `float64`, `float32`.
    adjoint: An optional `bool`. Defaults to `False`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  result = _op_def_lib.apply_op("BatchMatrixInverse", input=input,
                                adjoint=adjoint, name=name)
  return result



def _batch_matrix_solve(matrix, rhs, adjoint=None, name=None):
  r"""TODO: add doc.

  Args:
    matrix: A `Tensor`. Must be one of the following types: `float64`, `float32`.
    rhs: A `Tensor`. Must have the same type as `matrix`.
    adjoint: An optional `bool`. Defaults to `False`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `matrix`.
  """
  result = _op_def_lib.apply_op("BatchMatrixSolve", matrix=matrix, rhs=rhs,
                                adjoint=adjoint, name=name)
  return result



def _batch_matrix_solve_ls(matrix, rhs, l2_regularizer, fast=None, name=None):
  r"""TODO: add doc.

  Args:
    matrix: A `Tensor`. Must be one of the following types: `float64`, `float32`.
    rhs: A `Tensor`. Must have the same type as `matrix`.
    l2_regularizer: A `Tensor` of type `float64`.
    fast: An optional `bool`. Defaults to `True`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `matrix`.
  """
  result = _op_def_lib.apply_op("BatchMatrixSolveLs", matrix=matrix, rhs=rhs,
                                l2_regularizer=l2_regularizer, fast=fast,
                                name=name)
  return result



def _batch_matrix_triangular_solve(matrix, rhs, lower=None, adjoint=None,
                                   name=None):
  r"""TODO: add doc.

  Args:
    matrix: A `Tensor`. Must be one of the following types: `float64`, `float32`.
    rhs: A `Tensor`. Must have the same type as `matrix`.
    lower: An optional `bool`. Defaults to `True`.
    adjoint: An optional `bool`. Defaults to `False`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `matrix`.
  """
  result = _op_def_lib.apply_op("BatchMatrixTriangularSolve", matrix=matrix,
                                rhs=rhs, lower=lower, adjoint=adjoint,
                                name=name)
  return result



def _batch_self_adjoint_eig(input, name=None):
  r"""TODO: add doc.

  Args:
    input: A `Tensor`. Must be one of the following types: `float64`, `float32`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
  result = _op_def_lib.apply_op("BatchSelfAdjointEig", input=input, name=name)
  return result



__batch_self_adjoint_eig_v2_outputs = ["e", "v"]
_BatchSelfAdjointEigV2Output = _collections.namedtuple(
    "BatchSelfAdjointEigV2", __batch_self_adjoint_eig_v2_outputs)


def _batch_self_adjoint_eig_v2(input, compute_v=None, name=None):
  r"""TODO: add doc.

  Args:
    input: A `Tensor`. Must be one of the following types: `float64`, `float32`.
    compute_v: An optional `bool`. Defaults to `True`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (e, v).

    e: A `Tensor`. Has the same type as `input`.
    v: A `Tensor`. Has the same type as `input`.
  """
  result = _op_def_lib.apply_op("BatchSelfAdjointEigV2", input=input,
                                compute_v=compute_v, name=name)
  return _BatchSelfAdjointEigV2Output._make(result)



__batch_svd_outputs = ["s", "u", "v"]
_BatchSvdOutput = _collections.namedtuple(
    "BatchSvd", __batch_svd_outputs)


def _batch_svd(input, compute_uv=None, full_matrices=None, name=None):
  r"""TODO: add doc.

  Args:
    input: A `Tensor`. Must be one of the following types: `float64`, `float32`, `complex64`, `complex128`.
    compute_uv: An optional `bool`. Defaults to `True`.
    full_matrices: An optional `bool`. Defaults to `False`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (s, u, v).

    s: A `Tensor`. Has the same type as `input`.
    u: A `Tensor`. Has the same type as `input`.
    v: A `Tensor`. Has the same type as `input`.
  """
  result = _op_def_lib.apply_op("BatchSvd", input=input,
                                compute_uv=compute_uv,
                                full_matrices=full_matrices, name=name)
  return _BatchSvdOutput._make(result)



def cholesky(input, name=None):
  r"""Computes the Cholesky decomposition of one or more square matrices.

  The input is a tensor of shape `[..., M, M]` whose inner-most 2 dimensions
  form square matrices, with the same constraints as the single matrix Cholesky
  decomposition above. The output is a tensor of the same shape as the input
  containing the Cholesky decompositions for all input submatrices `[..., :, :]`.

  Args:
    input: A `Tensor`. Must be one of the following types: `float64`, `float32`.
      Shape is `[..., M, M]`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`. Shape is `[..., M, M]`.
  """
  result = _op_def_lib.apply_op("Cholesky", input=input, name=name)
  return result



def cholesky_grad(l, grad, name=None):
  r"""Computes the reverse mode backpropagated gradient of the Cholesky algorithm.

  For an explanation see "Differentiation of the Cholesky algorithm" by
  Iain Murray http://arxiv.org/abs/1602.07527.

  Args:
    l: A `Tensor`. Must be one of the following types: `float32`, `float64`.
      Output of batch Cholesky algorithm l = cholesky(A). Shape is `[..., M, M]`.
      Algorithm depends only on lower triangular part of the innermost matrices of
      this tensor.
    grad: A `Tensor`. Must have the same type as `l`.
      df/dl where f is some scalar function. Shape is `[..., M, M]`.
      Algorithm depends only on lower triangular part of the innermost matrices of
      this tensor.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `l`.
    Symmetrized version of df/dA . Shape is `[..., M, M]`
  """
  result = _op_def_lib.apply_op("CholeskyGrad", l=l, grad=grad, name=name)
  return result



def matrix_determinant(input, name=None):
  r"""Computes the determinant of one ore more square matrices.

  The input is a tensor of shape `[..., M, M]` whose inner-most 2 dimensions
  form square matrices. The output is a tensor containing the determinants
  for all input submatrices `[..., :, :]`.

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`.
      Shape is `[..., M, M]`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`. Shape is `[...]`.
  """
  result = _op_def_lib.apply_op("MatrixDeterminant", input=input, name=name)
  return result



def matrix_inverse(input, adjoint=None, name=None):
  r"""Computes the inverse of one or more square invertible matrices or their

  adjoints (conjugate transposes).

  The input is a tensor of shape `[..., M, M]` whose inner-most 2 dimensions
  form square matrices. The output is a tensor of the same shape as the input
  containing the inverse for all input submatrices `[..., :, :]`.

  The op uses LU decomposition with partial pivoting to compute the inverses.

  If a matrix is not invertible there is no guarantee what the op does. It
  may detect the condition and raise an exception or it may simply return a
  garbage result.

  Args:
    input: A `Tensor`. Must be one of the following types: `float64`, `float32`.
      Shape is `[..., M, M]`.
    adjoint: An optional `bool`. Defaults to `False`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`. Shape is `[..., M, M]`.

    @compatibility(numpy)
    Equivalent to np.linalg.inv
    @end_compatibility
  """
  result = _op_def_lib.apply_op("MatrixInverse", input=input, adjoint=adjoint,
                                name=name)
  return result



def matrix_solve(matrix, rhs, adjoint=None, name=None):
  r"""Solves systems of linear equations.

  `Matrix` is a tensor of shape `[..., M, M]` whose inner-most 2 dimensions
  form square matrices. `Rhs` is a tensor of shape `[..., M, K]`. The `output` is
  a tensor shape `[..., M, K]`.  If `adjoint` is `False` then each output matrix
  satisfies `matrix[..., :, :] * output[..., :, :] = rhs[..., :, :]`.
  If `adjoint` is `True` then each output matrix satisfies
  `adjoint(matrix[..., :, :]) * output[..., :, :] = rhs[..., :, :]`.

  Args:
    matrix: A `Tensor`. Must be one of the following types: `float64`, `float32`, `complex64`, `complex128`.
      Shape is `[..., M, M]`.
    rhs: A `Tensor`. Must have the same type as `matrix`.
      Shape is `[..., M, K]`.
    adjoint: An optional `bool`. Defaults to `False`.
      Boolean indicating whether to solve with `matrix` or its (block-wise)
      adjoint.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `matrix`. Shape is `[..., M, K]`.
  """
  result = _op_def_lib.apply_op("MatrixSolve", matrix=matrix, rhs=rhs,
                                adjoint=adjoint, name=name)
  return result



def _matrix_solve_ls(matrix, rhs, l2_regularizer, fast=None, name=None):
  r"""Solves one or more linear least-squares problems.

  `matrix` is a tensor of shape `[..., M, N]` whose inner-most 2 dimensions
  form matrices of size `[M, N]`. Rhs is a tensor of shape `[..., M, K]`.
  The output is a tensor shape `[..., N, K]` where each output matrix solves
  each of the equations matrix[..., :, :] * output[..., :, :] = rhs[..., :, :]
  in the least squares sense.

  matrix and right-hand sides in the batch:

  `matrix`=\\(A \in \Re^{m \times n}\\),
  `rhs`=\\(B  \in \Re^{m \times k}\\),
  `output`=\\(X  \in \Re^{n \times k}\\),
  `l2_regularizer`=\\(\lambda\\).

  If `fast` is `True`, then the solution is computed by solving the normal
  equations using Cholesky decomposition. Specifically, if \\(m \ge n\\) then
  \\(X = (A^T A + \lambda I)^{-1} A^T B\\), which solves the least-squares
  problem \\(X = \mathrm{argmin}_{Z \in \Re^{n \times k} } ||A Z - B||_F^2 +
  \lambda ||Z||_F^2\\). If \\(m \lt n\\) then `output` is computed as
  \\(X = A^T (A A^T + \lambda I)^{-1} B\\), which (for \\(\lambda = 0\\)) is the
  minimum-norm solution to the under-determined linear system, i.e.
  \\(X = \mathrm{argmin}_{Z \in \Re^{n \times k} } ||Z||_F^2 \\), subject to
  \\(A Z = B\\). Notice that the fast path is only numerically stable when
  \\(A\\) is numerically full rank and has a condition number
  \\(\mathrm{cond}(A) \lt \frac{1}{\sqrt{\epsilon_{mach} } }\\) or\\(\lambda\\) is
  sufficiently large.

  If `fast` is `False` an algorithm based on the numerically robust complete
  orthogonal decomposition is used. This computes the minimum-norm
  least-squares solution, even when \\(A\\) is rank deficient. This path is
  typically 6-7 times slower than the fast path. If `fast` is `False` then
  `l2_regularizer` is ignored.

  Args:
    matrix: A `Tensor`. Must be one of the following types: `float64`, `float32`.
      Shape is `[..., M, N]`.
    rhs: A `Tensor`. Must have the same type as `matrix`.
      Shape is `[..., M, K]`.
    l2_regularizer: A `Tensor` of type `float64`. Scalar tensor.

      @compatibility(numpy)
      Equivalent to np.linalg.lstsq
      @end_compatibility
    fast: An optional `bool`. Defaults to `True`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `matrix`. Shape is `[..., N, K]`.
  """
  result = _op_def_lib.apply_op("MatrixSolveLs", matrix=matrix, rhs=rhs,
                                l2_regularizer=l2_regularizer, fast=fast,
                                name=name)
  return result



def matrix_triangular_solve(matrix, rhs, lower=None, adjoint=None, name=None):
  r"""Solves systems of linear equations with upper or lower triangular matrices by

  backsubstitution.

  `matrix` is a tensor of shape `[..., M, M]` whose inner-most 2 dimensions form
  square matrices. If `lower` is `True` then the strictly upper triangular part
  of each inner-most matrix is assumed to be zero and not accessed.
  If `lower` is False then the strictly lower triangular part of each inner-most
  matrix is assumed to be zero and not accessed.
  `rhs` is a tensor of shape `[..., M, K]`.

  The output is a tensor of shape `[..., M, K]`. If `adjoint` is
  `True` then the innermost matrices in output` satisfy matrix equations
  `matrix[..., :, :] * output[..., :, :] = rhs[..., :, :]`.
  If `adjoint` is `False` then the strictly then the  innermost matrices in
  `output` satisfy matrix equations
  `adjoint(matrix[..., i, k]) * output[..., k, j] = rhs[..., i, j]`.

  Args:
    matrix: A `Tensor`. Must be one of the following types: `float64`, `float32`.
      Shape is `[..., M, M]`.
    rhs: A `Tensor`. Must have the same type as `matrix`.
      Shape is `[..., M, K]`.
    lower: An optional `bool`. Defaults to `True`.
      Boolean indicating whether the innermost matrices in `matrix` are
      lower or upper triangular.
    adjoint: An optional `bool`. Defaults to `False`.
      Boolean indicating whether to solve with `matrix` or its (block-wise)
               adjoint.

      @compatibility(numpy)
      Equivalent to np.linalg.triangular_solve
      @end_compatibility
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `matrix`. Shape is `[..., M, K]`.
  """
  result = _op_def_lib.apply_op("MatrixTriangularSolve", matrix=matrix,
                                rhs=rhs, lower=lower, adjoint=adjoint,
                                name=name)
  return result



_qr_outputs = ["q", "r"]
_QrOutput = _collections.namedtuple(
    "Qr", _qr_outputs)


def qr(input, full_matrices=None, name=None):
  r"""Computes the QR decompositions of one or more matrices.

  Computes the QR decomposition of each inner matrix in `tensor` such that
  `tensor[..., :, :] = q[..., :, :] * r[..., :,:])`

  ```prettyprint
  # a is a tensor.
  # q is a tensor of orthonormal matrices.
  # r is a tensor of upper triangular matrices.
  q, r = qr(a)
  q_full, r_full = qr(a, full_matrices=True)
  ```

  Args:
    input: A `Tensor`. Must be one of the following types: `float64`, `float32`, `complex64`, `complex128`.
      A tensor of shape `[..., M, N]` whose inner-most 2 dimensions
      form matrices of size `[M, N]`. Let `P` be the minimum of `M` and `N`.
    full_matrices: An optional `bool`. Defaults to `False`.
      If true, compute full-sized `q` and `r`. If false
      (the default), compute only the leading `P` columns of `q`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (q, r).

    q: A `Tensor`. Has the same type as `input`. Orthonormal basis for range of `a`. If `full_matrices` is `False` then
      shape is `[..., M, P]`; if `full_matrices` is `True` then shape is
      `[..., M, M]`.
    r: A `Tensor`. Has the same type as `input`. Triangular factor. If `full_matrices` is `False` then shape is
      `[..., P, N]`. If `full_matrices` is `True` then shape is `[..., M, N]`.
  """
  result = _op_def_lib.apply_op("Qr", input=input,
                                full_matrices=full_matrices, name=name)
  return _QrOutput._make(result)



def _self_adjoint_eig(input, name=None):
  r"""Computes the Eigen Decomposition of a batch of square self-adjoint matrices.

  The input is a tensor of shape `[..., M, M]` whose inner-most 2 dimensions
  form square matrices, with the same constraints as the single matrix
  SelfAdjointEig.

  The result is a [..., M+1, M] matrix with [..., 0,:] containing the
  eigenvalues, and subsequent [...,1:, :] containing the eigenvectors.

  Args:
    input: A `Tensor`. Must be one of the following types: `float64`, `float32`.
      Shape is `[..., M, M]`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`. Shape is `[..., M+1, M]`.
  """
  result = _op_def_lib.apply_op("SelfAdjointEig", input=input, name=name)
  return result



__self_adjoint_eig_v2_outputs = ["e", "v"]
_SelfAdjointEigV2Output = _collections.namedtuple(
    "SelfAdjointEigV2", __self_adjoint_eig_v2_outputs)


def _self_adjoint_eig_v2(input, compute_v=None, name=None):
  r"""Computes the eigen decomposition of one or more square self-adjoint matrices.

  Computes the eigenvalues and (optionally) eigenvectors of each inner matrix in
  `input` such that `input[..., :, :] = v[..., :, :] * diag(e[..., :])`.

  ```prettyprint
  # a is a tensor.
  # e is a tensor of eigenvalues.
  # v is a tensor of eigenvectors.
  e, v = self_adjoint_eig(a)
  e = self_adjoint_eig(a, compute_v=False)
  ```

  Args:
    input: A `Tensor`. Must be one of the following types: `float64`, `float32`, `complex64`, `complex128`.
      `Tensor` input of shape `[N, N]`.
    compute_v: An optional `bool`. Defaults to `True`.
      If `True` then eigenvectors will be computed and returned in `v`.
      Otherwise, only the eigenvalues will be computed.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (e, v).

    e: A `Tensor`. Has the same type as `input`. Eigenvalues. Shape is `[N]`.
    v: A `Tensor`. Has the same type as `input`. Eigenvectors. Shape is `[N, N]`.
  """
  result = _op_def_lib.apply_op("SelfAdjointEigV2", input=input,
                                compute_v=compute_v, name=name)
  return _SelfAdjointEigV2Output._make(result)



__svd_outputs = ["s", "u", "v"]
_SvdOutput = _collections.namedtuple(
    "Svd", __svd_outputs)


def _svd(input, compute_uv=None, full_matrices=None, name=None):
  r"""Computes the singular value decompositions of one or more matrices.

  Computes the SVD of each inner matrix in `input` such that
  `input[..., :, :] = u[..., :, :] * diag(s[..., :, :]) * transpose(v[..., :, :])`

  ```prettyprint
  # a is a tensor containing a batch of matrices.
  # s is a tensor of singular values for each matrix.
  # u is the tensor containing of left singular vectors for each matrix.
  # v is the tensor containing of right singular vectors for each matrix.
  s, u, v = svd(a)
  s, _, _ = svd(a, compute_uv=False)
  ```

  Args:
    input: A `Tensor`. Must be one of the following types: `float64`, `float32`, `complex64`, `complex128`.
      A tensor of shape `[..., M, N]` whose inner-most 2 dimensions
      form matrices of size `[M, N]`. Let `P` be the minimum of `M` and `N`.
    compute_uv: An optional `bool`. Defaults to `True`.
      If true, left and right singular vectors will be
      computed and returned in `u` and `v`, respectively.
      If false, `u` and `v` are not set and should never referenced.
    full_matrices: An optional `bool`. Defaults to `False`.
      If true, compute full-sized `u` and `v`. If false
      (the default), compute only the leading `P` singular vectors.
      Ignored if `compute_uv` is `False`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (s, u, v).

    s: A `Tensor`. Has the same type as `input`. Singular values. Shape is `[..., P]`.
    u: A `Tensor`. Has the same type as `input`. Left singular vectors. If `full_matrices` is `False` then shape is
      `[..., M, P]`; if `full_matrices` is `True` then shape is
      `[..., M, M]`. Undefined if `compute_uv` is `False`.
    v: A `Tensor`. Has the same type as `input`. Left singular vectors. If `full_matrices` is `False` then shape is
      `[..., N, P]`. If `full_matrices` is `True` then shape is `[..., N, N]`.
      Undefined if `compute_uv` is false.
  """
  result = _op_def_lib.apply_op("Svd", input=input, compute_uv=compute_uv,
                                full_matrices=full_matrices, name=name)
  return _SvdOutput._make(result)


def _InitOpDefLibrary():
  op_list = _op_def_pb2.OpList()
  _text_format.Merge(_InitOpDefLibrary.op_list_ascii, op_list)
  _op_def_registry.register_op_list(op_list)
  op_def_lib = _op_def_library.OpDefLibrary()
  op_def_lib.add_op_list(op_list)
  return op_def_lib


_InitOpDefLibrary.op_list_ascii = """op {
  name: "BatchCholesky"
  input_arg {
    name: "input"
    type_attr: "T"
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
        type: DT_DOUBLE
        type: DT_FLOAT
      }
    }
  }
  deprecation {
    version: 13
    explanation: "Use Cholesky instead."
  }
}
op {
  name: "BatchCholeskyGrad"
  input_arg {
    name: "l"
    type_attr: "T"
  }
  input_arg {
    name: "grad"
    type_attr: "T"
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
      }
    }
  }
  deprecation {
    version: 13
    explanation: "Use CholeskyGrad instead."
  }
}
op {
  name: "BatchMatrixDeterminant"
  input_arg {
    name: "input"
    type_attr: "T"
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
      }
    }
  }
  deprecation {
    version: 13
    explanation: "Use MatrixDeterminant instead."
  }
}
op {
  name: "BatchMatrixInverse"
  input_arg {
    name: "input"
    type_attr: "T"
  }
  output_arg {
    name: "output"
    type_attr: "T"
  }
  attr {
    name: "adjoint"
    type: "bool"
    default_value {
      b: false
    }
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_DOUBLE
        type: DT_FLOAT
      }
    }
  }
  deprecation {
    version: 13
    explanation: "Use MatrixInverse instead."
  }
}
op {
  name: "BatchMatrixSolve"
  input_arg {
    name: "matrix"
    type_attr: "T"
  }
  input_arg {
    name: "rhs"
    type_attr: "T"
  }
  output_arg {
    name: "output"
    type_attr: "T"
  }
  attr {
    name: "adjoint"
    type: "bool"
    default_value {
      b: false
    }
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_DOUBLE
        type: DT_FLOAT
      }
    }
  }
  deprecation {
    version: 13
    explanation: "Use MatrixSolve instead."
  }
}
op {
  name: "BatchMatrixSolveLs"
  input_arg {
    name: "matrix"
    type_attr: "T"
  }
  input_arg {
    name: "rhs"
    type_attr: "T"
  }
  input_arg {
    name: "l2_regularizer"
    type: DT_DOUBLE
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
        type: DT_DOUBLE
        type: DT_FLOAT
      }
    }
  }
  attr {
    name: "fast"
    type: "bool"
    default_value {
      b: true
    }
  }
  deprecation {
    version: 13
    explanation: "Use MatrixSolveLs instead."
  }
}
op {
  name: "BatchMatrixTriangularSolve"
  input_arg {
    name: "matrix"
    type_attr: "T"
  }
  input_arg {
    name: "rhs"
    type_attr: "T"
  }
  output_arg {
    name: "output"
    type_attr: "T"
  }
  attr {
    name: "lower"
    type: "bool"
    default_value {
      b: true
    }
  }
  attr {
    name: "adjoint"
    type: "bool"
    default_value {
      b: false
    }
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_DOUBLE
        type: DT_FLOAT
      }
    }
  }
  deprecation {
    version: 13
    explanation: "Use MatrixTriangularSolve instead."
  }
}
op {
  name: "BatchSelfAdjointEig"
  input_arg {
    name: "input"
    type_attr: "T"
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
        type: DT_DOUBLE
        type: DT_FLOAT
      }
    }
  }
  deprecation {
    version: 11
    explanation: "Use SelfAdjointEigV2 instead."
  }
}
op {
  name: "BatchSelfAdjointEigV2"
  input_arg {
    name: "input"
    type_attr: "T"
  }
  output_arg {
    name: "e"
    type_attr: "T"
  }
  output_arg {
    name: "v"
    type_attr: "T"
  }
  attr {
    name: "compute_v"
    type: "bool"
    default_value {
      b: true
    }
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_DOUBLE
        type: DT_FLOAT
      }
    }
  }
  deprecation {
    version: 13
    explanation: "Use SelfAdjointEigV2 instead."
  }
}
op {
  name: "BatchSvd"
  input_arg {
    name: "input"
    type_attr: "T"
  }
  output_arg {
    name: "s"
    type_attr: "T"
  }
  output_arg {
    name: "u"
    type_attr: "T"
  }
  output_arg {
    name: "v"
    type_attr: "T"
  }
  attr {
    name: "compute_uv"
    type: "bool"
    default_value {
      b: true
    }
  }
  attr {
    name: "full_matrices"
    type: "bool"
    default_value {
      b: false
    }
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_DOUBLE
        type: DT_FLOAT
        type: DT_COMPLEX64
        type: DT_COMPLEX128
      }
    }
  }
  deprecation {
    version: 13
    explanation: "Use Svd instead."
  }
}
op {
  name: "Cholesky"
  input_arg {
    name: "input"
    type_attr: "T"
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
        type: DT_DOUBLE
        type: DT_FLOAT
      }
    }
  }
}
op {
  name: "CholeskyGrad"
  input_arg {
    name: "l"
    type_attr: "T"
  }
  input_arg {
    name: "grad"
    type_attr: "T"
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
      }
    }
  }
}
op {
  name: "MatrixDeterminant"
  input_arg {
    name: "input"
    type_attr: "T"
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
      }
    }
  }
}
op {
  name: "MatrixInverse"
  input_arg {
    name: "input"
    type_attr: "T"
  }
  output_arg {
    name: "output"
    type_attr: "T"
  }
  attr {
    name: "adjoint"
    type: "bool"
    default_value {
      b: false
    }
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_DOUBLE
        type: DT_FLOAT
      }
    }
  }
}
op {
  name: "MatrixSolve"
  input_arg {
    name: "matrix"
    type_attr: "T"
  }
  input_arg {
    name: "rhs"
    type_attr: "T"
  }
  output_arg {
    name: "output"
    type_attr: "T"
  }
  attr {
    name: "adjoint"
    type: "bool"
    default_value {
      b: false
    }
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_DOUBLE
        type: DT_FLOAT
        type: DT_COMPLEX64
        type: DT_COMPLEX128
      }
    }
  }
}
op {
  name: "MatrixSolveLs"
  input_arg {
    name: "matrix"
    type_attr: "T"
  }
  input_arg {
    name: "rhs"
    type_attr: "T"
  }
  input_arg {
    name: "l2_regularizer"
    type: DT_DOUBLE
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
        type: DT_DOUBLE
        type: DT_FLOAT
      }
    }
  }
  attr {
    name: "fast"
    type: "bool"
    default_value {
      b: true
    }
  }
}
op {
  name: "MatrixTriangularSolve"
  input_arg {
    name: "matrix"
    type_attr: "T"
  }
  input_arg {
    name: "rhs"
    type_attr: "T"
  }
  output_arg {
    name: "output"
    type_attr: "T"
  }
  attr {
    name: "lower"
    type: "bool"
    default_value {
      b: true
    }
  }
  attr {
    name: "adjoint"
    type: "bool"
    default_value {
      b: false
    }
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_DOUBLE
        type: DT_FLOAT
      }
    }
  }
}
op {
  name: "Qr"
  input_arg {
    name: "input"
    type_attr: "T"
  }
  output_arg {
    name: "q"
    type_attr: "T"
  }
  output_arg {
    name: "r"
    type_attr: "T"
  }
  attr {
    name: "full_matrices"
    type: "bool"
    default_value {
      b: false
    }
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_DOUBLE
        type: DT_FLOAT
        type: DT_COMPLEX64
        type: DT_COMPLEX128
      }
    }
  }
}
op {
  name: "SelfAdjointEig"
  input_arg {
    name: "input"
    type_attr: "T"
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
        type: DT_DOUBLE
        type: DT_FLOAT
      }
    }
  }
  deprecation {
    version: 11
    explanation: "Use SelfAdjointEigV2 instead."
  }
}
op {
  name: "SelfAdjointEigV2"
  input_arg {
    name: "input"
    type_attr: "T"
  }
  output_arg {
    name: "e"
    type_attr: "T"
  }
  output_arg {
    name: "v"
    type_attr: "T"
  }
  attr {
    name: "compute_v"
    type: "bool"
    default_value {
      b: true
    }
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_DOUBLE
        type: DT_FLOAT
        type: DT_COMPLEX64
        type: DT_COMPLEX128
      }
    }
  }
}
op {
  name: "Svd"
  input_arg {
    name: "input"
    type_attr: "T"
  }
  output_arg {
    name: "s"
    type_attr: "T"
  }
  output_arg {
    name: "u"
    type_attr: "T"
  }
  output_arg {
    name: "v"
    type_attr: "T"
  }
  attr {
    name: "compute_uv"
    type: "bool"
    default_value {
      b: true
    }
  }
  attr {
    name: "full_matrices"
    type: "bool"
    default_value {
      b: false
    }
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_DOUBLE
        type: DT_FLOAT
        type: DT_COMPLEX64
        type: DT_COMPLEX128
      }
    }
  }
}
"""


_op_def_lib = _InitOpDefLibrary()
