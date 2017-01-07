"""Python wrappers around Brain.

This file is MACHINE GENERATED! Do not edit.
"""

import collections

from google.protobuf import text_format

from tensorflow.core.framework import op_def_pb2
from tensorflow.python.framework import op_def_registry
from tensorflow.python.framework import ops
from tensorflow.python.ops import op_def_library


_batch_cholesky_outputs = ["output"]


def batch_cholesky(input, name=None):
  r"""Calculates the Cholesky decomposition of a batch of square matrices.

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
  result = _op_def_lib.apply_op("BatchCholesky", input=input, name=name)
  return result


_batch_cholesky_grad_outputs = ["output"]


def batch_cholesky_grad(l, grad, name=None):
  r"""Calculates the reverse mode backpropagated gradient of the Cholesky algorithm.

  For an explanation see "Differentiation of the Cholesky algorithm" by
  Iain Murray http://arxiv.org/abs/1602.07527.

  Args:
    l: A `Tensor`. Must be one of the following types: `float32`, `float64`.
      Output of batch Cholesky algorithm l = batch_cholesky(A). Shape is `[..., M, M]`.
      Algorithm depends only on lower triangular part of the innermost matrices of
      this tensor.
    grad: A `Tensor`. Must have the same type as `l`.
      df/dl where f is some scalar function. Shape is `[..., M, M]'.
      Algorithm depends only on lower triangular part of the innermost matrices of
      this tensor.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `l`.
    Symmetrized version of df/dA . Shape is `[..., M, M]'
  """
  result = _op_def_lib.apply_op("BatchCholeskyGrad", l=l, grad=grad,
                                name=name)
  return result


_batch_matrix_determinant_outputs = ["output"]


def batch_matrix_determinant(input, name=None):
  r"""Calculates the determinants for a batch of square matrices.

  The input is a tensor of shape `[..., M, M]` whose inner-most 2 dimensions
  form square matrices. The output is a 1-D tensor containing the determinants
  for all input submatrices `[..., :, :]`.

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`.
      Shape is `[..., M, M]`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`. Shape is `[...]`.
  """
  result = _op_def_lib.apply_op("BatchMatrixDeterminant", input=input,
                                name=name)
  return result


_batch_matrix_inverse_outputs = ["output"]


def batch_matrix_inverse(input, adjoint=None, name=None):
  r"""Calculates the inverse of square invertible matrices or their adjoints

  (conjugate transposes).

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
  """
  result = _op_def_lib.apply_op("BatchMatrixInverse", input=input,
                                adjoint=adjoint, name=name)
  return result


_batch_matrix_solve_outputs = ["output"]


def batch_matrix_solve(matrix, rhs, adjoint=None, name=None):
  r"""Solves systems of linear equations. Checks for invertibility.

  Matrix is a tensor of shape `[..., M, M]` whose inner-most 2 dimensions
  form square matrices. Rhs is a tensor of shape
  `[..., M, K]`. The output is a tensor shape `[..., M, K]`.  If `adjoint` is `False` then each output
  matrix satisfies `matrix[..., :, :] * output[..., :, :] = rhs[..., :, :]`.
  If `adjoint` is `True` then each output
  matrix satisfies `adjoint(matrix[..., :, :]) * output[..., :, :] = rhs[..., :, :]`.

  Args:
    matrix: A `Tensor`. Must be one of the following types: `float64`, `float32`.
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
  result = _op_def_lib.apply_op("BatchMatrixSolve", matrix=matrix, rhs=rhs,
                                adjoint=adjoint, name=name)
  return result


_batch_matrix_solve_ls_outputs = ["output"]


def batch_matrix_solve_ls(matrix, rhs, l2_regularizer, fast=None, name=None):
  r"""Solves multiple linear least-squares problems.

  `matrix` is a tensor of shape `[..., M, N]` whose inner-most 2 dimensions
  form square matrices. Rhs is a tensor of shape `[..., M, K]`. The output
  is a tensor shape `[..., N, K]` where each output matrix solves each of
  the equations matrix[..., :, :] * output[..., :, :] = rhs[..., :, :] in the
  least squares sense.

  Below we will use the following notation for each pair of
  matrix and right-hand sides in the batch:

  `matrix`=\\(A \in \Re^{m \times n}\\),
  `rhs`=\\(B  \in \Re^{m \times k}\\),
  `output`=\\(X  \in \Re^{n \times k}\\),
  `l2_regularizer`=\\(\lambda\\).

  If `fast` is `True`, then the solution is computed by solving the normal
  equations using Cholesky decomposition. Specifically, if \\(m \ge n\\) then
  \\(X = (A^T A + \lambda I)^{-1} A^T B\\), which solves the least-squares
  problem \\(X = \mathrm{argmin}_{Z \in \Re^{n \times k}} ||A Z - B||_F^2 +
  \lambda ||Z||_F^2\\). If \\(m \lt n\\) then `output` is computed as
  \\(X = A^T (A A^T + \lambda I)^{-1} B\\), which (for \\(\lambda = 0\\)) is the
  minimum-norm solution to the under-determined linear system, i.e.
  \\(X = \mathrm{argmin}_{Z \in \Re^{n \times k}} ||Z||_F^2 \\), subject to
  \\(A Z = B\\). Notice that the fast path is only numerically stable when
  \\(A\\) is numerically full rank and has a condition number
  \\(\mathrm{cond}(A) \lt \frac{1}{\sqrt{\epsilon_{mach}}}\\) or\\(\lambda\\) is
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
    l2_regularizer: A `Tensor` of type `float64`.
    fast: An optional `bool`. Defaults to `True`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `matrix`. Shape is `[..., N, K]`.
  """
  result = _op_def_lib.apply_op("BatchMatrixSolveLs", matrix=matrix, rhs=rhs,
                                l2_regularizer=l2_regularizer, fast=fast,
                                name=name)
  return result


_batch_matrix_triangular_solve_outputs = ["output"]


def batch_matrix_triangular_solve(matrix, rhs, lower=None, adjoint=None,
                                  name=None):
  r"""Solves systems of linear equations with upper or lower triangular matrices by

  backsubstitution.

  `matrix` is a tensor of shape `[..., M, M]` whose inner-most 2 dimensions form
  square matrices. If `lower` is `True` then the strictly upper triangular part
  of each inner-most matrix is assumed to be zero and not accessed.
  If `lower` is False then the strictly lower triangular part of each inner-most
  matrix is assumed to be zero and not accessed.
  `rhs` is a tensor of shape [..., M, K]`.

  The output is a tensor of shape `[..., M, K]`. If `adjoint` is `True` then the
  innermost matrices in output` satisfy matrix equations
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
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `matrix`. Shape is `[..., M, K]`.
  """
  result = _op_def_lib.apply_op("BatchMatrixTriangularSolve", matrix=matrix,
                                rhs=rhs, lower=lower, adjoint=adjoint,
                                name=name)
  return result


_batch_self_adjoint_eig_outputs = ["output"]


def batch_self_adjoint_eig(input, name=None):
  r"""Calculates the Eigen Decomposition of a batch of square self-adjoint matrices.

  The input is a tensor of shape `[..., M, M]` whose inner-most 2 dimensions
  form square matrices, with the same constraints as the single matrix
  SelfAdjointEig.

  The result is a '[..., M+1, M] matrix with [..., 0,:] containing the
  eigenvalues, and subsequent [...,1:, :] containing the eigenvectors.

  Args:
    input: A `Tensor`. Must be one of the following types: `float64`, `float32`.
      Shape is `[..., M, M]`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`. Shape is `[..., M+1, M]`.
  """
  result = _op_def_lib.apply_op("BatchSelfAdjointEig", input=input, name=name)
  return result


_cholesky_outputs = ["output"]


def cholesky(input, name=None):
  r"""Calculates the Cholesky decomposition of a square matrix.

  The input has to be symmetric and positive definite. Only the lower-triangular
  part of the input will be used for this operation. The upper-triangular part
  will not be read.

  The result is the lower-triangular matrix of the Cholesky decomposition of the
  input, `L`, so that `input = L L^*`.

  Args:
    input: A `Tensor`. Must be one of the following types: `float64`, `float32`.
      Shape is `[M, M]`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`. Shape is `[M, M]`.
  """
  result = _op_def_lib.apply_op("Cholesky", input=input, name=name)
  return result


_cholesky_grad_outputs = ["output"]


def cholesky_grad(l, grad, name=None):
  r"""Calculates the reverse mode backpropagated gradient of the Cholesky algorithm.

  For an explanation see "Differentiation of the Cholesky algorithm" by
  Iain Murray http://arxiv.org/abs/1602.07527.

  Args:
    l: A `Tensor`. Must be one of the following types: `float32`, `float64`.
      Output of Cholesky algorithm l = chol(A). Shape is `[M, M]`.
      Algorithm depends only on lower triangular part of this matrix.
    grad: A `Tensor`. Must have the same type as `l`.
      df/dl where f is some scalar function. Shape is `[M, M]'.
      Algorithm depends only on lower triangular part of this matrix.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `l`.
    Symmetrized version of df/dA . Shape is `[M, M]'.
  """
  result = _op_def_lib.apply_op("CholeskyGrad", l=l, grad=grad, name=name)
  return result


_matrix_determinant_outputs = ["output"]


def matrix_determinant(input, name=None):
  r"""Calculates the determinant of a square matrix.

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`.
      A tensor of shape `[M, M]`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
    A scalar, equal to the determinant of the input.
  """
  result = _op_def_lib.apply_op("MatrixDeterminant", input=input, name=name)
  return result


_matrix_inverse_outputs = ["output"]


def matrix_inverse(input, adjoint=None, name=None):
  r"""Calculates the inverse of a square invertible matrix or its adjoint (conjugate

  transpose).

  The op uses LU decomposition with partial pivoting to compute the inverse.

  If the matrix is not invertible there is no guarantee what the op does. It
  may detect the condition and raise an exception or it may simply return a
  garbage result.

  Args:
    input: A `Tensor`. Must be one of the following types: `float64`, `float32`.
      Shape is `[M, M]`.
    adjoint: An optional `bool`. Defaults to `False`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
    Shape is `[M, M]`. If `adjoint` is `False` then `output` contains the
    matrix inverse of `input`. If `adjoint` is `True` then `output` contains the
    matrix inverse of the adjoint of `input`.
  """
  result = _op_def_lib.apply_op("MatrixInverse", input=input, adjoint=adjoint,
                                name=name)
  return result


_matrix_solve_outputs = ["output"]


def matrix_solve(matrix, rhs, adjoint=None, name=None):
  r"""Solves a system of linear equations. Checks for invertibility.

  Args:
    matrix: A `Tensor`. Must be one of the following types: `float64`, `float32`.
      Shape is `[M, M]`.
    rhs: A `Tensor`. Must have the same type as `matrix`. Shape is `[M, K]`.
    adjoint: An optional `bool`. Defaults to `False`.
      Boolean indicating whether to solve with `matrix` or its adjoint.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `matrix`.
    Shape is `[M, K]`. If `adjoint` is `False` then `output` that solves
    `matrix` * `output` = `rhs`. If `adjoint` is `True` then `output` that solves
    `adjoint(matrix)` * `output` = `rhs`.
  """
  result = _op_def_lib.apply_op("MatrixSolve", matrix=matrix, rhs=rhs,
                                adjoint=adjoint, name=name)
  return result


_matrix_solve_ls_outputs = ["output"]


def matrix_solve_ls(matrix, rhs, l2_regularizer, fast=None, name=None):
  r"""Solves a linear least-squares problem.

  Below we will use the following notation
  `matrix`=\\(A \in \Re^{m \times n}\\),
  `rhs`=\\(B  \in \Re^{m \times k}\\),
  `output`=\\(X  \in \Re^{n \times k}\\),
  `l2_regularizer`=\\(\lambda\\).

  If `fast` is `True`, then the solution is computed by solving the normal
  equations using Cholesky decomposition. Specifically, if \\(m \ge n\\) then
  \\(X = (A^T A + \lambda I)^{-1} A^T B\\), which solves the least-squares
  problem \\(X = \mathrm{argmin}_{Z \in \Re^{n \times k}} ||A Z - B||_F^2 +
  \lambda ||Z||_F^2\\). If \\(m \lt n\\) then `output` is computed as
  \\(X = A^T (A A^T + \lambda I)^{-1} B\\),
  which (for \\(\lambda = 0\\)) is the minimum-norm solution to the
  under-determined linear system, i.e.
  \\(X = \mathrm{argmin}_{Z \in \Re^{n \times k}} ||Z||_F^2 \\),
  subject to \\(A Z = B\\).
  Notice that the fast path is only numerically stable when \\(A\\) is
  numerically full rank and has a condition number
  \\(\mathrm{cond}(A) \lt \frac{1}{\sqrt{\epsilon_{mach}}}\\)
  or \\(\lambda\\) is sufficiently large.

  If `fast` is `False` an algorithm based on the numerically robust complete
  orthogonal decomposition is used. This computes the minimum-norm
  least-squares solution, even when \\(A\\) is rank deficient. This path is
  typically 6-7 times slower than the fast path. If `fast` is `False` then
  `l2_regularizer` is ignored.

  Args:
    matrix: A `Tensor`. Must be one of the following types: `float64`, `float32`.
      Shape is `[M, N]`.
    rhs: A `Tensor`. Must have the same type as `matrix`. Shape is `[M, K]`.
    l2_regularizer: A `Tensor` of type `float64`.
    fast: An optional `bool`. Defaults to `True`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `matrix`.
    Shape is `[N, K]` containing the tensor that solves
    `matrix * output = rhs` in the least-squares sense.
  """
  result = _op_def_lib.apply_op("MatrixSolveLs", matrix=matrix, rhs=rhs,
                                l2_regularizer=l2_regularizer, fast=fast,
                                name=name)
  return result


_matrix_triangular_solve_outputs = ["output"]


def matrix_triangular_solve(matrix, rhs, lower=None, adjoint=None, name=None):
  r"""Solves a system of linear equations with an upper or lower triangular matrix by

  backsubstitution.

  `matrix` is a matrix of shape `[M, M]`. If `lower` is `True` then the strictly
  upper triangular part of `matrix` is assumed to be zero and not accessed.
  If `lower` is False then the strictly lower triangular part of `matrix` is
  assumed to be zero and not accessed.
  `rhs` is a matrix of shape [M, K]`.

  The output is a matrix of shape `[M, K]`. If `adjoint` is `False` the output
  satisfies the matrix equation `matrix` * `output` = `rhs`.
  If `adjoint` is `False` then `output` satisfies the matrix equation
  `matrix` * `output` = `rhs`.
  If `adjoint` is `True` then `output` satisfies the matrix equation
  `adjoint(matrix)` * `output` = `rhs`.

  Args:
    matrix: A `Tensor`. Must be one of the following types: `float64`, `float32`.
      Shape is `[M, M]`.
    rhs: A `Tensor`. Must have the same type as `matrix`. Shape is `[M, K]`.
    lower: An optional `bool`. Defaults to `True`.
      Boolean indicating whether `matrix` is lower or upper triangular
    adjoint: An optional `bool`. Defaults to `False`.
      Boolean indicating whether to solve with `matrix` or its adjoint.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `matrix`. Shape is `[M, K]`.
  """
  result = _op_def_lib.apply_op("MatrixTriangularSolve", matrix=matrix,
                                rhs=rhs, lower=lower, adjoint=adjoint,
                                name=name)
  return result


_self_adjoint_eig_outputs = ["output"]


def self_adjoint_eig(input, name=None):
  r"""Calculates the Eigen Decomposition of a square Self-Adjoint matrix.

  Only the lower-triangular part of the input will be used in this case. The
  upper-triangular part will not be read.

  The result is a M+1 x M matrix whose first row is the eigenvalues, and
  subsequent rows are eigenvectors.

  Args:
    input: A `Tensor`. Must be one of the following types: `float64`, `float32`.
      Shape is `[M, M]`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`. Shape is `[M+1, M]`.
  """
  result = _op_def_lib.apply_op("SelfAdjointEig", input=input, name=name)
  return result


def _InitOpDefLibrary():
  op_list = op_def_pb2.OpList()
  text_format.Merge(_InitOpDefLibrary.op_list_ascii, op_list)
  op_def_registry.register_op_list(op_list)
  op_def_lib = op_def_library.OpDefLibrary()
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
}
"""


_op_def_lib = _InitOpDefLibrary()
