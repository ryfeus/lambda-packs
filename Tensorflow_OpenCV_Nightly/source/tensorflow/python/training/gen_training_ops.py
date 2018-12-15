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

def apply_adadelta(var, accum, accum_update, lr, rho, epsilon, grad,
                   use_locking=None, name=None):
  r"""Update '*var' according to the adadelta scheme.

  accum = rho() * accum + (1 - rho()) * grad.square();
  update = (update_accum + epsilon).sqrt() * (accum + epsilon()).rsqrt() * grad;
  update_accum = rho() * update_accum + (1 - rho()) * update.square();
  var -= update;

  Args:
    var: A mutable `Tensor`. Must be one of the following types: `float32`, `float64`, `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`, `complex128`, `qint8`, `quint8`, `qint32`, `half`.
      Should be from a Variable().
    accum: A mutable `Tensor`. Must have the same type as `var`.
      Should be from a Variable().
    accum_update: A mutable `Tensor`. Must have the same type as `var`.
      Should be from a Variable().
    lr: A `Tensor`. Must have the same type as `var`.
      Scaling factor. Must be a scalar.
    rho: A `Tensor`. Must have the same type as `var`.
      Decay factor. Must be a scalar.
    epsilon: A `Tensor`. Must have the same type as `var`.
      Constant factor. Must be a scalar.
    grad: A `Tensor`. Must have the same type as `var`. The gradient.
    use_locking: An optional `bool`. Defaults to `False`.
      If True, updating of the var, accum and update_accum tensors will be protected by
      a lock; otherwise the behavior is undefined, but may exhibit less contention.
    name: A name for the operation (optional).

  Returns:
    A mutable `Tensor`. Has the same type as `var`. Same as "var".
  """
  result = _op_def_lib.apply_op("ApplyAdadelta", var=var, accum=accum,
                                accum_update=accum_update, lr=lr, rho=rho,
                                epsilon=epsilon, grad=grad,
                                use_locking=use_locking, name=name)
  return result



def apply_adagrad(var, accum, lr, grad, use_locking=None, name=None):
  r"""Update '*var' according to the adagrad scheme.

  accum += grad * grad
  var -= lr * grad * (1 / sqrt(accum))

  Args:
    var: A mutable `Tensor`. Must be one of the following types: `float32`, `float64`, `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`, `complex128`, `qint8`, `quint8`, `qint32`, `half`.
      Should be from a Variable().
    accum: A mutable `Tensor`. Must have the same type as `var`.
      Should be from a Variable().
    lr: A `Tensor`. Must have the same type as `var`.
      Scaling factor. Must be a scalar.
    grad: A `Tensor`. Must have the same type as `var`. The gradient.
    use_locking: An optional `bool`. Defaults to `False`.
      If `True`, updating of the var and accum tensors will be protected
      by a lock; otherwise the behavior is undefined, but may exhibit less
      contention.
    name: A name for the operation (optional).

  Returns:
    A mutable `Tensor`. Has the same type as `var`. Same as "var".
  """
  result = _op_def_lib.apply_op("ApplyAdagrad", var=var, accum=accum, lr=lr,
                                grad=grad, use_locking=use_locking, name=name)
  return result



def apply_adagrad_da(var, gradient_accumulator, gradient_squared_accumulator,
                     grad, lr, l1, l2, global_step, use_locking=None,
                     name=None):
  r"""Update '*var' according to the proximal adagrad scheme.

  Args:
    var: A mutable `Tensor`. Must be one of the following types: `float32`, `float64`, `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`, `complex128`, `qint8`, `quint8`, `qint32`, `half`.
      Should be from a Variable().
    gradient_accumulator: A mutable `Tensor`. Must have the same type as `var`.
      Should be from a Variable().
    gradient_squared_accumulator: A mutable `Tensor`. Must have the same type as `var`.
      Should be from a Variable().
    grad: A `Tensor`. Must have the same type as `var`. The gradient.
    lr: A `Tensor`. Must have the same type as `var`.
      Scaling factor. Must be a scalar.
    l1: A `Tensor`. Must have the same type as `var`.
      L1 regularization. Must be a scalar.
    l2: A `Tensor`. Must have the same type as `var`.
      L2 regularization. Must be a scalar.
    global_step: A `Tensor` of type `int64`.
      Training step number. Must be a scalar.
    use_locking: An optional `bool`. Defaults to `False`.
      If True, updating of the var and accum tensors will be protected by
      a lock; otherwise the behavior is undefined, but may exhibit less contention.
    name: A name for the operation (optional).

  Returns:
    A mutable `Tensor`. Has the same type as `var`. Same as "var".
  """
  result = _op_def_lib.apply_op("ApplyAdagradDA", var=var,
                                gradient_accumulator=gradient_accumulator,
                                gradient_squared_accumulator=gradient_squared_accumulator,
                                grad=grad, lr=lr, l1=l1, l2=l2,
                                global_step=global_step,
                                use_locking=use_locking, name=name)
  return result



def apply_adam(var, m, v, beta1_power, beta2_power, lr, beta1, beta2, epsilon,
               grad, use_locking=None, use_nesterov=None, name=None):
  r"""Update '*var' according to the Adam algorithm.

  lr_t <- learning_rate * sqrt(1 - beta2^t) / (1 - beta1^t)
  m_t <- beta1 * m_{t-1} + (1 - beta1) * g_t
  v_t <- beta2 * v_{t-1} + (1 - beta2) * g_t * g_t
  variable <- variable - lr_t * m_t / (sqrt(v_t) + epsilon)

  Args:
    var: A mutable `Tensor`. Must be one of the following types: `float32`, `float64`, `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`, `complex128`, `qint8`, `quint8`, `qint32`, `half`.
      Should be from a Variable().
    m: A mutable `Tensor`. Must have the same type as `var`.
      Should be from a Variable().
    v: A mutable `Tensor`. Must have the same type as `var`.
      Should be from a Variable().
    beta1_power: A `Tensor`. Must have the same type as `var`.
      Must be a scalar.
    beta2_power: A `Tensor`. Must have the same type as `var`.
      Must be a scalar.
    lr: A `Tensor`. Must have the same type as `var`.
      Scaling factor. Must be a scalar.
    beta1: A `Tensor`. Must have the same type as `var`.
      Momentum factor. Must be a scalar.
    beta2: A `Tensor`. Must have the same type as `var`.
      Momentum factor. Must be a scalar.
    epsilon: A `Tensor`. Must have the same type as `var`.
      Ridge term. Must be a scalar.
    grad: A `Tensor`. Must have the same type as `var`. The gradient.
    use_locking: An optional `bool`. Defaults to `False`.
      If `True`, updating of the var, m, and v tensors will be protected
      by a lock; otherwise the behavior is undefined, but may exhibit less
      contention.
    use_nesterov: An optional `bool`. Defaults to `False`.
      If `True`, uses the nesterov update.
    name: A name for the operation (optional).

  Returns:
    A mutable `Tensor`. Has the same type as `var`. Same as "var".
  """
  result = _op_def_lib.apply_op("ApplyAdam", var=var, m=m, v=v,
                                beta1_power=beta1_power,
                                beta2_power=beta2_power, lr=lr, beta1=beta1,
                                beta2=beta2, epsilon=epsilon, grad=grad,
                                use_locking=use_locking,
                                use_nesterov=use_nesterov, name=name)
  return result



def apply_centered_rms_prop(var, mg, ms, mom, lr, rho, momentum, epsilon,
                            grad, use_locking=None, name=None):
  r"""Update '*var' according to the centered RMSProp algorithm.

  The centered RMSProp algorithm uses an estimate of the centered second moment
  (i.e., the variance) for normalization, as opposed to regular RMSProp, which
  uses the (uncentered) second moment. This often helps with training, but is
  slightly more expensive in terms of computation and memory.

  Note that in dense implementation of this algorithm, mg, ms, and mom will
  update even if the grad is zero, but in this sparse implementation, mg, ms,
  and mom will not update in iterations during which the grad is zero.

  mean_square = decay * mean_square + (1-decay) * gradient ** 2
  mean_grad = decay * mean_grad + (1-decay) * gradient

  Delta = learning_rate * gradient / sqrt(mean_square + epsilon - mean_grad ** 2)

  mg <- rho * mg_{t-1} + (1-rho) * grad
  ms <- rho * ms_{t-1} + (1-rho) * grad * grad
  mom <- momentum * mom_{t-1} + lr * grad / sqrt(ms - mg * mg + epsilon)
  var <- var - mom

  Args:
    var: A mutable `Tensor`. Must be one of the following types: `float32`, `float64`, `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`, `complex128`, `qint8`, `quint8`, `qint32`, `half`.
      Should be from a Variable().
    mg: A mutable `Tensor`. Must have the same type as `var`.
      Should be from a Variable().
    ms: A mutable `Tensor`. Must have the same type as `var`.
      Should be from a Variable().
    mom: A mutable `Tensor`. Must have the same type as `var`.
      Should be from a Variable().
    lr: A `Tensor`. Must have the same type as `var`.
      Scaling factor. Must be a scalar.
    rho: A `Tensor`. Must have the same type as `var`.
      Decay rate. Must be a scalar.
    momentum: A `Tensor`. Must have the same type as `var`.
    epsilon: A `Tensor`. Must have the same type as `var`.
      Ridge term. Must be a scalar.
    grad: A `Tensor`. Must have the same type as `var`. The gradient.
    use_locking: An optional `bool`. Defaults to `False`.
      If `True`, updating of the var, mg, ms, and mom tensors is
      protected by a lock; otherwise the behavior is undefined, but may exhibit less
      contention.
    name: A name for the operation (optional).

  Returns:
    A mutable `Tensor`. Has the same type as `var`. Same as "var".
  """
  result = _op_def_lib.apply_op("ApplyCenteredRMSProp", var=var, mg=mg, ms=ms,
                                mom=mom, lr=lr, rho=rho, momentum=momentum,
                                epsilon=epsilon, grad=grad,
                                use_locking=use_locking, name=name)
  return result



def apply_ftrl(var, accum, linear, grad, lr, l1, l2, lr_power,
               use_locking=None, name=None):
  r"""Update '*var' according to the Ftrl-proximal scheme.

  accum_new = accum + grad * grad
  linear += grad + (accum_new^(-lr_power) - accum^(-lr_power)) / lr * var
  quadratic = 1.0 / (accum_new^(lr_power) * lr) + 2 * l2
  var = (sign(linear) * l1 - linear) / quadratic if |linear| > l1 else 0.0
  accum = accum_new

  Args:
    var: A mutable `Tensor`. Must be one of the following types: `float32`, `float64`, `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`, `complex128`, `qint8`, `quint8`, `qint32`, `half`.
      Should be from a Variable().
    accum: A mutable `Tensor`. Must have the same type as `var`.
      Should be from a Variable().
    linear: A mutable `Tensor`. Must have the same type as `var`.
      Should be from a Variable().
    grad: A `Tensor`. Must have the same type as `var`. The gradient.
    lr: A `Tensor`. Must have the same type as `var`.
      Scaling factor. Must be a scalar.
    l1: A `Tensor`. Must have the same type as `var`.
      L1 regulariation. Must be a scalar.
    l2: A `Tensor`. Must have the same type as `var`.
      L2 regulariation. Must be a scalar.
    lr_power: A `Tensor`. Must have the same type as `var`.
      Scaling factor. Must be a scalar.
    use_locking: An optional `bool`. Defaults to `False`.
      If `True`, updating of the var and accum tensors will be protected
      by a lock; otherwise the behavior is undefined, but may exhibit less
      contention.
    name: A name for the operation (optional).

  Returns:
    A mutable `Tensor`. Has the same type as `var`. Same as "var".
  """
  result = _op_def_lib.apply_op("ApplyFtrl", var=var, accum=accum,
                                linear=linear, grad=grad, lr=lr, l1=l1, l2=l2,
                                lr_power=lr_power, use_locking=use_locking,
                                name=name)
  return result



def apply_gradient_descent(var, alpha, delta, use_locking=None, name=None):
  r"""Update '*var' by subtracting 'alpha' * 'delta' from it.

  Args:
    var: A mutable `Tensor`. Must be one of the following types: `float32`, `float64`, `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`, `complex128`, `qint8`, `quint8`, `qint32`, `half`.
      Should be from a Variable().
    alpha: A `Tensor`. Must have the same type as `var`.
      Scaling factor. Must be a scalar.
    delta: A `Tensor`. Must have the same type as `var`. The change.
    use_locking: An optional `bool`. Defaults to `False`.
      If `True`, the subtraction will be protected by a lock;
      otherwise the behavior is undefined, but may exhibit less contention.
    name: A name for the operation (optional).

  Returns:
    A mutable `Tensor`. Has the same type as `var`. Same as "var".
  """
  result = _op_def_lib.apply_op("ApplyGradientDescent", var=var, alpha=alpha,
                                delta=delta, use_locking=use_locking,
                                name=name)
  return result



def apply_momentum(var, accum, lr, grad, momentum, use_locking=None,
                   use_nesterov=None, name=None):
  r"""Update '*var' according to the momentum scheme. Set use_nesterov = True if you

  want to use Nesterov momentum.

  accum = accum * momentum + grad
  var -= lr * accum

  Args:
    var: A mutable `Tensor`. Must be one of the following types: `float32`, `float64`, `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`, `complex128`, `qint8`, `quint8`, `qint32`, `half`.
      Should be from a Variable().
    accum: A mutable `Tensor`. Must have the same type as `var`.
      Should be from a Variable().
    lr: A `Tensor`. Must have the same type as `var`.
      Scaling factor. Must be a scalar.
    grad: A `Tensor`. Must have the same type as `var`. The gradient.
    momentum: A `Tensor`. Must have the same type as `var`.
      Momentum. Must be a scalar.
    use_locking: An optional `bool`. Defaults to `False`.
      If `True`, updating of the var and accum tensors will be protected
      by a lock; otherwise the behavior is undefined, but may exhibit less
      contention.
    use_nesterov: An optional `bool`. Defaults to `False`.
      If `True`, the tensor passed to compute grad will be
      var - lr * momentum * accum, so in the end, the var you get is actually
      var - lr * momentum * accum.
    name: A name for the operation (optional).

  Returns:
    A mutable `Tensor`. Has the same type as `var`. Same as "var".
  """
  result = _op_def_lib.apply_op("ApplyMomentum", var=var, accum=accum, lr=lr,
                                grad=grad, momentum=momentum,
                                use_locking=use_locking,
                                use_nesterov=use_nesterov, name=name)
  return result



def apply_proximal_adagrad(var, accum, lr, l1, l2, grad, use_locking=None,
                           name=None):
  r"""Update '*var' and '*accum' according to FOBOS with Adagrad learning rate.

  accum += grad * grad
  prox_v = var - lr * grad * (1 / sqrt(accum))
  var = sign(prox_v)/(1+lr*l2) * max{|prox_v|-lr*l1,0}

  Args:
    var: A mutable `Tensor`. Must be one of the following types: `float32`, `float64`, `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`, `complex128`, `qint8`, `quint8`, `qint32`, `half`.
      Should be from a Variable().
    accum: A mutable `Tensor`. Must have the same type as `var`.
      Should be from a Variable().
    lr: A `Tensor`. Must have the same type as `var`.
      Scaling factor. Must be a scalar.
    l1: A `Tensor`. Must have the same type as `var`.
      L1 regularization. Must be a scalar.
    l2: A `Tensor`. Must have the same type as `var`.
      L2 regularization. Must be a scalar.
    grad: A `Tensor`. Must have the same type as `var`. The gradient.
    use_locking: An optional `bool`. Defaults to `False`.
      If True, updating of the var and accum tensors will be protected by
      a lock; otherwise the behavior is undefined, but may exhibit less contention.
    name: A name for the operation (optional).

  Returns:
    A mutable `Tensor`. Has the same type as `var`. Same as "var".
  """
  result = _op_def_lib.apply_op("ApplyProximalAdagrad", var=var, accum=accum,
                                lr=lr, l1=l1, l2=l2, grad=grad,
                                use_locking=use_locking, name=name)
  return result



def apply_proximal_gradient_descent(var, alpha, l1, l2, delta,
                                    use_locking=None, name=None):
  r"""Update '*var' as FOBOS algorithm with fixed learning rate.

  prox_v = var - alpha * delta
  var = sign(prox_v)/(1+alpha*l2) * max{|prox_v|-alpha*l1,0}

  Args:
    var: A mutable `Tensor`. Must be one of the following types: `float32`, `float64`, `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`, `complex128`, `qint8`, `quint8`, `qint32`, `half`.
      Should be from a Variable().
    alpha: A `Tensor`. Must have the same type as `var`.
      Scaling factor. Must be a scalar.
    l1: A `Tensor`. Must have the same type as `var`.
      L1 regularization. Must be a scalar.
    l2: A `Tensor`. Must have the same type as `var`.
      L2 regularization. Must be a scalar.
    delta: A `Tensor`. Must have the same type as `var`. The change.
    use_locking: An optional `bool`. Defaults to `False`.
      If True, the subtraction will be protected by a lock;
      otherwise the behavior is undefined, but may exhibit less contention.
    name: A name for the operation (optional).

  Returns:
    A mutable `Tensor`. Has the same type as `var`. Same as "var".
  """
  result = _op_def_lib.apply_op("ApplyProximalGradientDescent", var=var,
                                alpha=alpha, l1=l1, l2=l2, delta=delta,
                                use_locking=use_locking, name=name)
  return result



def apply_rms_prop(var, ms, mom, lr, rho, momentum, epsilon, grad,
                   use_locking=None, name=None):
  r"""Update '*var' according to the RMSProp algorithm.

  Note that in dense implementation of this algorithm, ms and mom will
  update even if the grad is zero, but in this sparse implementation, ms
  and mom will not update in iterations during which the grad is zero.

  mean_square = decay * mean_square + (1-decay) * gradient ** 2
  Delta = learning_rate * gradient / sqrt(mean_square + epsilon)

  ms <- rho * ms_{t-1} + (1-rho) * grad * grad
  mom <- momentum * mom_{t-1} + lr * grad / sqrt(ms + epsilon)
  var <- var - mom

  Args:
    var: A mutable `Tensor`. Must be one of the following types: `float32`, `float64`, `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`, `complex128`, `qint8`, `quint8`, `qint32`, `half`.
      Should be from a Variable().
    ms: A mutable `Tensor`. Must have the same type as `var`.
      Should be from a Variable().
    mom: A mutable `Tensor`. Must have the same type as `var`.
      Should be from a Variable().
    lr: A `Tensor`. Must have the same type as `var`.
      Scaling factor. Must be a scalar.
    rho: A `Tensor`. Must have the same type as `var`.
      Decay rate. Must be a scalar.
    momentum: A `Tensor`. Must have the same type as `var`.
    epsilon: A `Tensor`. Must have the same type as `var`.
      Ridge term. Must be a scalar.
    grad: A `Tensor`. Must have the same type as `var`. The gradient.
    use_locking: An optional `bool`. Defaults to `False`.
      If `True`, updating of the var, ms, and mom tensors is protected
      by a lock; otherwise the behavior is undefined, but may exhibit less
      contention.
    name: A name for the operation (optional).

  Returns:
    A mutable `Tensor`. Has the same type as `var`. Same as "var".
  """
  result = _op_def_lib.apply_op("ApplyRMSProp", var=var, ms=ms, mom=mom,
                                lr=lr, rho=rho, momentum=momentum,
                                epsilon=epsilon, grad=grad,
                                use_locking=use_locking, name=name)
  return result



def resource_apply_adadelta(var, accum, accum_update, lr, rho, epsilon, grad,
                            use_locking=None, name=None):
  r"""Update '*var' according to the adadelta scheme.

  accum = rho() * accum + (1 - rho()) * grad.square();
  update = (update_accum + epsilon).sqrt() * (accum + epsilon()).rsqrt() * grad;
  update_accum = rho() * update_accum + (1 - rho()) * update.square();
  var -= update;

  Args:
    var: A `Tensor` of type `resource`. Should be from a Variable().
    accum: A `Tensor` of type `resource`. Should be from a Variable().
    accum_update: A `Tensor` of type `resource`. Should be from a Variable().
    lr: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`, `complex128`, `qint8`, `quint8`, `qint32`, `half`.
      Scaling factor. Must be a scalar.
    rho: A `Tensor`. Must have the same type as `lr`.
      Decay factor. Must be a scalar.
    epsilon: A `Tensor`. Must have the same type as `lr`.
      Constant factor. Must be a scalar.
    grad: A `Tensor`. Must have the same type as `lr`. The gradient.
    use_locking: An optional `bool`. Defaults to `False`.
      If True, updating of the var, accum and update_accum tensors will be protected by
      a lock; otherwise the behavior is undefined, but may exhibit less contention.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  result = _op_def_lib.apply_op("ResourceApplyAdadelta", var=var, accum=accum,
                                accum_update=accum_update, lr=lr, rho=rho,
                                epsilon=epsilon, grad=grad,
                                use_locking=use_locking, name=name)
  return result



def resource_apply_adagrad(var, accum, lr, grad, use_locking=None, name=None):
  r"""Update '*var' according to the adagrad scheme.

  accum += grad * grad
  var -= lr * grad * (1 / sqrt(accum))

  Args:
    var: A `Tensor` of type `resource`. Should be from a Variable().
    accum: A `Tensor` of type `resource`. Should be from a Variable().
    lr: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`, `complex128`, `qint8`, `quint8`, `qint32`, `half`.
      Scaling factor. Must be a scalar.
    grad: A `Tensor`. Must have the same type as `lr`. The gradient.
    use_locking: An optional `bool`. Defaults to `False`.
      If `True`, updating of the var and accum tensors will be protected
      by a lock; otherwise the behavior is undefined, but may exhibit less
      contention.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  result = _op_def_lib.apply_op("ResourceApplyAdagrad", var=var, accum=accum,
                                lr=lr, grad=grad, use_locking=use_locking,
                                name=name)
  return result



def resource_apply_adagrad_da(var, gradient_accumulator,
                              gradient_squared_accumulator, grad, lr, l1, l2,
                              global_step, use_locking=None, name=None):
  r"""Update '*var' according to the proximal adagrad scheme.

  Args:
    var: A `Tensor` of type `resource`. Should be from a Variable().
    gradient_accumulator: A `Tensor` of type `resource`.
      Should be from a Variable().
    gradient_squared_accumulator: A `Tensor` of type `resource`.
      Should be from a Variable().
    grad: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`, `complex128`, `qint8`, `quint8`, `qint32`, `half`.
      The gradient.
    lr: A `Tensor`. Must have the same type as `grad`.
      Scaling factor. Must be a scalar.
    l1: A `Tensor`. Must have the same type as `grad`.
      L1 regularization. Must be a scalar.
    l2: A `Tensor`. Must have the same type as `grad`.
      L2 regularization. Must be a scalar.
    global_step: A `Tensor` of type `int64`.
      Training step number. Must be a scalar.
    use_locking: An optional `bool`. Defaults to `False`.
      If True, updating of the var and accum tensors will be protected by
      a lock; otherwise the behavior is undefined, but may exhibit less contention.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  result = _op_def_lib.apply_op("ResourceApplyAdagradDA", var=var,
                                gradient_accumulator=gradient_accumulator,
                                gradient_squared_accumulator=gradient_squared_accumulator,
                                grad=grad, lr=lr, l1=l1, l2=l2,
                                global_step=global_step,
                                use_locking=use_locking, name=name)
  return result



def resource_apply_adam(var, m, v, beta1_power, beta2_power, lr, beta1, beta2,
                        epsilon, grad, use_locking=None, use_nesterov=None,
                        name=None):
  r"""Update '*var' according to the Adam algorithm.

  lr_t <- learning_rate * sqrt(1 - beta2^t) / (1 - beta1^t)
  m_t <- beta1 * m_{t-1} + (1 - beta1) * g_t
  v_t <- beta2 * v_{t-1} + (1 - beta2) * g_t * g_t
  variable <- variable - lr_t * m_t / (sqrt(v_t) + epsilon)

  Args:
    var: A `Tensor` of type `resource`. Should be from a Variable().
    m: A `Tensor` of type `resource`. Should be from a Variable().
    v: A `Tensor` of type `resource`. Should be from a Variable().
    beta1_power: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`, `complex128`, `qint8`, `quint8`, `qint32`, `half`.
      Must be a scalar.
    beta2_power: A `Tensor`. Must have the same type as `beta1_power`.
      Must be a scalar.
    lr: A `Tensor`. Must have the same type as `beta1_power`.
      Scaling factor. Must be a scalar.
    beta1: A `Tensor`. Must have the same type as `beta1_power`.
      Momentum factor. Must be a scalar.
    beta2: A `Tensor`. Must have the same type as `beta1_power`.
      Momentum factor. Must be a scalar.
    epsilon: A `Tensor`. Must have the same type as `beta1_power`.
      Ridge term. Must be a scalar.
    grad: A `Tensor`. Must have the same type as `beta1_power`. The gradient.
    use_locking: An optional `bool`. Defaults to `False`.
      If `True`, updating of the var, m, and v tensors will be protected
      by a lock; otherwise the behavior is undefined, but may exhibit less
      contention.
    use_nesterov: An optional `bool`. Defaults to `False`.
      If `True`, uses the nesterov update.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  result = _op_def_lib.apply_op("ResourceApplyAdam", var=var, m=m, v=v,
                                beta1_power=beta1_power,
                                beta2_power=beta2_power, lr=lr, beta1=beta1,
                                beta2=beta2, epsilon=epsilon, grad=grad,
                                use_locking=use_locking,
                                use_nesterov=use_nesterov, name=name)
  return result



def resource_apply_centered_rms_prop(var, mg, ms, mom, lr, rho, momentum,
                                     epsilon, grad, use_locking=None,
                                     name=None):
  r"""Update '*var' according to the centered RMSProp algorithm.

  The centered RMSProp algorithm uses an estimate of the centered second moment
  (i.e., the variance) for normalization, as opposed to regular RMSProp, which
  uses the (uncentered) second moment. This often helps with training, but is
  slightly more expensive in terms of computation and memory.

  Note that in dense implementation of this algorithm, mg, ms, and mom will
  update even if the grad is zero, but in this sparse implementation, mg, ms,
  and mom will not update in iterations during which the grad is zero.

  mean_square = decay * mean_square + (1-decay) * gradient ** 2
  mean_grad = decay * mean_grad + (1-decay) * gradient

  Delta = learning_rate * gradient / sqrt(mean_square + epsilon - mean_grad ** 2)

  mg <- rho * mg_{t-1} + (1-rho) * grad
  ms <- rho * ms_{t-1} + (1-rho) * grad * grad
  mom <- momentum * mom_{t-1} + lr * grad / sqrt(ms - mg * mg + epsilon)
  var <- var - mom

  Args:
    var: A `Tensor` of type `resource`. Should be from a Variable().
    mg: A `Tensor` of type `resource`. Should be from a Variable().
    ms: A `Tensor` of type `resource`. Should be from a Variable().
    mom: A `Tensor` of type `resource`. Should be from a Variable().
    lr: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`, `complex128`, `qint8`, `quint8`, `qint32`, `half`.
      Scaling factor. Must be a scalar.
    rho: A `Tensor`. Must have the same type as `lr`.
      Decay rate. Must be a scalar.
    momentum: A `Tensor`. Must have the same type as `lr`.
    epsilon: A `Tensor`. Must have the same type as `lr`.
      Ridge term. Must be a scalar.
    grad: A `Tensor`. Must have the same type as `lr`. The gradient.
    use_locking: An optional `bool`. Defaults to `False`.
      If `True`, updating of the var, mg, ms, and mom tensors is
      protected by a lock; otherwise the behavior is undefined, but may exhibit less
      contention.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  result = _op_def_lib.apply_op("ResourceApplyCenteredRMSProp", var=var,
                                mg=mg, ms=ms, mom=mom, lr=lr, rho=rho,
                                momentum=momentum, epsilon=epsilon, grad=grad,
                                use_locking=use_locking, name=name)
  return result



def resource_apply_ftrl(var, accum, linear, grad, lr, l1, l2, lr_power,
                        use_locking=None, name=None):
  r"""Update '*var' according to the Ftrl-proximal scheme.

  accum_new = accum + grad * grad
  linear += grad + (accum_new^(-lr_power) - accum^(-lr_power)) / lr * var
  quadratic = 1.0 / (accum_new^(lr_power) * lr) + 2 * l2
  var = (sign(linear) * l1 - linear) / quadratic if |linear| > l1 else 0.0
  accum = accum_new

  Args:
    var: A `Tensor` of type `resource`. Should be from a Variable().
    accum: A `Tensor` of type `resource`. Should be from a Variable().
    linear: A `Tensor` of type `resource`. Should be from a Variable().
    grad: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`, `complex128`, `qint8`, `quint8`, `qint32`, `half`.
      The gradient.
    lr: A `Tensor`. Must have the same type as `grad`.
      Scaling factor. Must be a scalar.
    l1: A `Tensor`. Must have the same type as `grad`.
      L1 regulariation. Must be a scalar.
    l2: A `Tensor`. Must have the same type as `grad`.
      L2 regulariation. Must be a scalar.
    lr_power: A `Tensor`. Must have the same type as `grad`.
      Scaling factor. Must be a scalar.
    use_locking: An optional `bool`. Defaults to `False`.
      If `True`, updating of the var and accum tensors will be protected
      by a lock; otherwise the behavior is undefined, but may exhibit less
      contention.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  result = _op_def_lib.apply_op("ResourceApplyFtrl", var=var, accum=accum,
                                linear=linear, grad=grad, lr=lr, l1=l1, l2=l2,
                                lr_power=lr_power, use_locking=use_locking,
                                name=name)
  return result



def resource_apply_gradient_descent(var, alpha, delta, use_locking=None,
                                    name=None):
  r"""Update '*var' by subtracting 'alpha' * 'delta' from it.

  Args:
    var: A `Tensor` of type `resource`. Should be from a Variable().
    alpha: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`, `complex128`, `qint8`, `quint8`, `qint32`, `half`.
      Scaling factor. Must be a scalar.
    delta: A `Tensor`. Must have the same type as `alpha`. The change.
    use_locking: An optional `bool`. Defaults to `False`.
      If `True`, the subtraction will be protected by a lock;
      otherwise the behavior is undefined, but may exhibit less contention.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  result = _op_def_lib.apply_op("ResourceApplyGradientDescent", var=var,
                                alpha=alpha, delta=delta,
                                use_locking=use_locking, name=name)
  return result



def resource_apply_momentum(var, accum, lr, grad, momentum, use_locking=None,
                            use_nesterov=None, name=None):
  r"""Update '*var' according to the momentum scheme. Set use_nesterov = True if you

  want to use Nesterov momentum.

  accum = accum * momentum + grad
  var -= lr * accum

  Args:
    var: A `Tensor` of type `resource`. Should be from a Variable().
    accum: A `Tensor` of type `resource`. Should be from a Variable().
    lr: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`, `complex128`, `qint8`, `quint8`, `qint32`, `half`.
      Scaling factor. Must be a scalar.
    grad: A `Tensor`. Must have the same type as `lr`. The gradient.
    momentum: A `Tensor`. Must have the same type as `lr`.
      Momentum. Must be a scalar.
    use_locking: An optional `bool`. Defaults to `False`.
      If `True`, updating of the var and accum tensors will be protected
      by a lock; otherwise the behavior is undefined, but may exhibit less
      contention.
    use_nesterov: An optional `bool`. Defaults to `False`.
      If `True`, the tensor passed to compute grad will be
      var - lr * momentum * accum, so in the end, the var you get is actually
      var - lr * momentum * accum.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  result = _op_def_lib.apply_op("ResourceApplyMomentum", var=var, accum=accum,
                                lr=lr, grad=grad, momentum=momentum,
                                use_locking=use_locking,
                                use_nesterov=use_nesterov, name=name)
  return result



def resource_apply_proximal_adagrad(var, accum, lr, l1, l2, grad,
                                    use_locking=None, name=None):
  r"""Update '*var' and '*accum' according to FOBOS with Adagrad learning rate.

  accum += grad * grad
  prox_v = var - lr * grad * (1 / sqrt(accum))
  var = sign(prox_v)/(1+lr*l2) * max{|prox_v|-lr*l1,0}

  Args:
    var: A `Tensor` of type `resource`. Should be from a Variable().
    accum: A `Tensor` of type `resource`. Should be from a Variable().
    lr: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`, `complex128`, `qint8`, `quint8`, `qint32`, `half`.
      Scaling factor. Must be a scalar.
    l1: A `Tensor`. Must have the same type as `lr`.
      L1 regularization. Must be a scalar.
    l2: A `Tensor`. Must have the same type as `lr`.
      L2 regularization. Must be a scalar.
    grad: A `Tensor`. Must have the same type as `lr`. The gradient.
    use_locking: An optional `bool`. Defaults to `False`.
      If True, updating of the var and accum tensors will be protected by
      a lock; otherwise the behavior is undefined, but may exhibit less contention.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  result = _op_def_lib.apply_op("ResourceApplyProximalAdagrad", var=var,
                                accum=accum, lr=lr, l1=l1, l2=l2, grad=grad,
                                use_locking=use_locking, name=name)
  return result



def resource_apply_proximal_gradient_descent(var, alpha, l1, l2, delta,
                                             use_locking=None, name=None):
  r"""Update '*var' as FOBOS algorithm with fixed learning rate.

  prox_v = var - alpha * delta
  var = sign(prox_v)/(1+alpha*l2) * max{|prox_v|-alpha*l1,0}

  Args:
    var: A `Tensor` of type `resource`. Should be from a Variable().
    alpha: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`, `complex128`, `qint8`, `quint8`, `qint32`, `half`.
      Scaling factor. Must be a scalar.
    l1: A `Tensor`. Must have the same type as `alpha`.
      L1 regularization. Must be a scalar.
    l2: A `Tensor`. Must have the same type as `alpha`.
      L2 regularization. Must be a scalar.
    delta: A `Tensor`. Must have the same type as `alpha`. The change.
    use_locking: An optional `bool`. Defaults to `False`.
      If True, the subtraction will be protected by a lock;
      otherwise the behavior is undefined, but may exhibit less contention.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  result = _op_def_lib.apply_op("ResourceApplyProximalGradientDescent",
                                var=var, alpha=alpha, l1=l1, l2=l2,
                                delta=delta, use_locking=use_locking,
                                name=name)
  return result



def resource_apply_rms_prop(var, ms, mom, lr, rho, momentum, epsilon, grad,
                            use_locking=None, name=None):
  r"""Update '*var' according to the RMSProp algorithm.

  Note that in dense implementation of this algorithm, ms and mom will
  update even if the grad is zero, but in this sparse implementation, ms
  and mom will not update in iterations during which the grad is zero.

  mean_square = decay * mean_square + (1-decay) * gradient ** 2
  Delta = learning_rate * gradient / sqrt(mean_square + epsilon)

  ms <- rho * ms_{t-1} + (1-rho) * grad * grad
  mom <- momentum * mom_{t-1} + lr * grad / sqrt(ms + epsilon)
  var <- var - mom

  Args:
    var: A `Tensor` of type `resource`. Should be from a Variable().
    ms: A `Tensor` of type `resource`. Should be from a Variable().
    mom: A `Tensor` of type `resource`. Should be from a Variable().
    lr: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`, `complex128`, `qint8`, `quint8`, `qint32`, `half`.
      Scaling factor. Must be a scalar.
    rho: A `Tensor`. Must have the same type as `lr`.
      Decay rate. Must be a scalar.
    momentum: A `Tensor`. Must have the same type as `lr`.
    epsilon: A `Tensor`. Must have the same type as `lr`.
      Ridge term. Must be a scalar.
    grad: A `Tensor`. Must have the same type as `lr`. The gradient.
    use_locking: An optional `bool`. Defaults to `False`.
      If `True`, updating of the var, ms, and mom tensors is protected
      by a lock; otherwise the behavior is undefined, but may exhibit less
      contention.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  result = _op_def_lib.apply_op("ResourceApplyRMSProp", var=var, ms=ms,
                                mom=mom, lr=lr, rho=rho, momentum=momentum,
                                epsilon=epsilon, grad=grad,
                                use_locking=use_locking, name=name)
  return result



def resource_sparse_apply_adadelta(var, accum, accum_update, lr, rho, epsilon,
                                   grad, indices, use_locking=None,
                                   name=None):
  r"""var: Should be from a Variable().

  Args:
    var: A `Tensor` of type `resource`.
    accum: A `Tensor` of type `resource`. Should be from a Variable().
    accum_update: A `Tensor` of type `resource`.
      : Should be from a Variable().
    lr: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`, `complex128`, `qint8`, `quint8`, `qint32`, `half`.
      Learning rate. Must be a scalar.
    rho: A `Tensor`. Must have the same type as `lr`.
      Decay factor. Must be a scalar.
    epsilon: A `Tensor`. Must have the same type as `lr`.
      Constant factor. Must be a scalar.
    grad: A `Tensor`. Must have the same type as `lr`. The gradient.
    indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A vector of indices into the first dimension of var and accum.
    use_locking: An optional `bool`. Defaults to `False`.
      If True, updating of the var and accum tensors will be protected by
      a lock; otherwise the behavior is undefined, but may exhibit less contention.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  result = _op_def_lib.apply_op("ResourceSparseApplyAdadelta", var=var,
                                accum=accum, accum_update=accum_update, lr=lr,
                                rho=rho, epsilon=epsilon, grad=grad,
                                indices=indices, use_locking=use_locking,
                                name=name)
  return result



def resource_sparse_apply_adagrad(var, accum, lr, grad, indices,
                                  use_locking=None, name=None):
  r"""Update relevant entries in '*var' and '*accum' according to the adagrad scheme.

  That is for rows we have grad for, we update var and accum as follows:
  accum += grad * grad
  var -= lr * grad * (1 / sqrt(accum))

  Args:
    var: A `Tensor` of type `resource`. Should be from a Variable().
    accum: A `Tensor` of type `resource`. Should be from a Variable().
    lr: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`, `complex128`, `qint8`, `quint8`, `qint32`, `half`.
      Learning rate. Must be a scalar.
    grad: A `Tensor`. Must have the same type as `lr`. The gradient.
    indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A vector of indices into the first dimension of var and accum.
    use_locking: An optional `bool`. Defaults to `False`.
      If `True`, updating of the var and accum tensors will be protected
      by a lock; otherwise the behavior is undefined, but may exhibit less
      contention.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  result = _op_def_lib.apply_op("ResourceSparseApplyAdagrad", var=var,
                                accum=accum, lr=lr, grad=grad,
                                indices=indices, use_locking=use_locking,
                                name=name)
  return result



def resource_sparse_apply_adagrad_da(var, gradient_accumulator,
                                     gradient_squared_accumulator, grad,
                                     indices, lr, l1, l2, global_step,
                                     use_locking=None, name=None):
  r"""Update entries in '*var' and '*accum' according to the proximal adagrad scheme.

  Args:
    var: A `Tensor` of type `resource`. Should be from a Variable().
    gradient_accumulator: A `Tensor` of type `resource`.
      Should be from a Variable().
    gradient_squared_accumulator: A `Tensor` of type `resource`.
      Should be from a Variable().
    grad: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`, `complex128`, `qint8`, `quint8`, `qint32`, `half`.
      The gradient.
    indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A vector of indices into the first dimension of var and accum.
    lr: A `Tensor`. Must have the same type as `grad`.
      Learning rate. Must be a scalar.
    l1: A `Tensor`. Must have the same type as `grad`.
      L1 regularization. Must be a scalar.
    l2: A `Tensor`. Must have the same type as `grad`.
      L2 regularization. Must be a scalar.
    global_step: A `Tensor` of type `int64`.
      Training step number. Must be a scalar.
    use_locking: An optional `bool`. Defaults to `False`.
      If True, updating of the var and accum tensors will be protected by
      a lock; otherwise the behavior is undefined, but may exhibit less contention.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  result = _op_def_lib.apply_op("ResourceSparseApplyAdagradDA", var=var,
                                gradient_accumulator=gradient_accumulator,
                                gradient_squared_accumulator=gradient_squared_accumulator,
                                grad=grad, indices=indices, lr=lr, l1=l1,
                                l2=l2, global_step=global_step,
                                use_locking=use_locking, name=name)
  return result



def resource_sparse_apply_centered_rms_prop(var, mg, ms, mom, lr, rho,
                                            momentum, epsilon, grad, indices,
                                            use_locking=None, name=None):
  r"""Update '*var' according to the centered RMSProp algorithm.

  The centered RMSProp algorithm uses an estimate of the centered second moment
  (i.e., the variance) for normalization, as opposed to regular RMSProp, which
  uses the (uncentered) second moment. This often helps with training, but is
  slightly more expensive in terms of computation and memory.

  Note that in dense implementation of this algorithm, mg, ms, and mom will
  update even if the grad is zero, but in this sparse implementation, mg, ms,
  and mom will not update in iterations during which the grad is zero.

  mean_square = decay * mean_square + (1-decay) * gradient ** 2
  mean_grad = decay * mean_grad + (1-decay) * gradient
  Delta = learning_rate * gradient / sqrt(mean_square + epsilon - mean_grad ** 2)

  ms <- rho * ms_{t-1} + (1-rho) * grad * grad
  mom <- momentum * mom_{t-1} + lr * grad / sqrt(ms + epsilon)
  var <- var - mom

  Args:
    var: A `Tensor` of type `resource`. Should be from a Variable().
    mg: A `Tensor` of type `resource`. Should be from a Variable().
    ms: A `Tensor` of type `resource`. Should be from a Variable().
    mom: A `Tensor` of type `resource`. Should be from a Variable().
    lr: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`, `complex128`, `qint8`, `quint8`, `qint32`, `half`.
      Scaling factor. Must be a scalar.
    rho: A `Tensor`. Must have the same type as `lr`.
      Decay rate. Must be a scalar.
    momentum: A `Tensor`. Must have the same type as `lr`.
    epsilon: A `Tensor`. Must have the same type as `lr`.
      Ridge term. Must be a scalar.
    grad: A `Tensor`. Must have the same type as `lr`. The gradient.
    indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A vector of indices into the first dimension of var, ms and mom.
    use_locking: An optional `bool`. Defaults to `False`.
      If `True`, updating of the var, mg, ms, and mom tensors is
      protected by a lock; otherwise the behavior is undefined, but may exhibit less
      contention.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  result = _op_def_lib.apply_op("ResourceSparseApplyCenteredRMSProp", var=var,
                                mg=mg, ms=ms, mom=mom, lr=lr, rho=rho,
                                momentum=momentum, epsilon=epsilon, grad=grad,
                                indices=indices, use_locking=use_locking,
                                name=name)
  return result



def resource_sparse_apply_ftrl(var, accum, linear, grad, indices, lr, l1, l2,
                               lr_power, use_locking=None, name=None):
  r"""Update relevant entries in '*var' according to the Ftrl-proximal scheme.

  That is for rows we have grad for, we update var, accum and linear as follows:
  accum_new = accum + grad * grad
  linear += grad + (accum_new^(-lr_power) - accum^(-lr_power)) / lr * var
  quadratic = 1.0 / (accum_new^(lr_power) * lr) + 2 * l2
  var = (sign(linear) * l1 - linear) / quadratic if |linear| > l1 else 0.0
  accum = accum_new

  Args:
    var: A `Tensor` of type `resource`. Should be from a Variable().
    accum: A `Tensor` of type `resource`. Should be from a Variable().
    linear: A `Tensor` of type `resource`. Should be from a Variable().
    grad: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`, `complex128`, `qint8`, `quint8`, `qint32`, `half`.
      The gradient.
    indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A vector of indices into the first dimension of var and accum.
    lr: A `Tensor`. Must have the same type as `grad`.
      Scaling factor. Must be a scalar.
    l1: A `Tensor`. Must have the same type as `grad`.
      L1 regularization. Must be a scalar.
    l2: A `Tensor`. Must have the same type as `grad`.
      L2 regularization. Must be a scalar.
    lr_power: A `Tensor`. Must have the same type as `grad`.
      Scaling factor. Must be a scalar.
    use_locking: An optional `bool`. Defaults to `False`.
      If `True`, updating of the var and accum tensors will be protected
      by a lock; otherwise the behavior is undefined, but may exhibit less
      contention.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  result = _op_def_lib.apply_op("ResourceSparseApplyFtrl", var=var,
                                accum=accum, linear=linear, grad=grad,
                                indices=indices, lr=lr, l1=l1, l2=l2,
                                lr_power=lr_power, use_locking=use_locking,
                                name=name)
  return result



def resource_sparse_apply_momentum(var, accum, lr, grad, indices, momentum,
                                   use_locking=None, use_nesterov=None,
                                   name=None):
  r"""Update relevant entries in '*var' and '*accum' according to the momentum scheme.

  Set use_nesterov = True if you want to use Nesterov momentum.

  That is for rows we have grad for, we update var and accum as follows:

  accum = accum * momentum + grad
  var -= lr * accum

  Args:
    var: A `Tensor` of type `resource`. Should be from a Variable().
    accum: A `Tensor` of type `resource`. Should be from a Variable().
    lr: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`, `complex128`, `qint8`, `quint8`, `qint32`, `half`.
      Learning rate. Must be a scalar.
    grad: A `Tensor`. Must have the same type as `lr`. The gradient.
    indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A vector of indices into the first dimension of var and accum.
    momentum: A `Tensor`. Must have the same type as `lr`.
      Momentum. Must be a scalar.
    use_locking: An optional `bool`. Defaults to `False`.
      If `True`, updating of the var and accum tensors will be protected
      by a lock; otherwise the behavior is undefined, but may exhibit less
      contention.
    use_nesterov: An optional `bool`. Defaults to `False`.
      If `True`, the tensor passed to compute grad will be
      var - lr * momentum * accum, so in the end, the var you get is actually
      var - lr * momentum * accum.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  result = _op_def_lib.apply_op("ResourceSparseApplyMomentum", var=var,
                                accum=accum, lr=lr, grad=grad,
                                indices=indices, momentum=momentum,
                                use_locking=use_locking,
                                use_nesterov=use_nesterov, name=name)
  return result



def resource_sparse_apply_proximal_adagrad(var, accum, lr, l1, l2, grad,
                                           indices, use_locking=None,
                                           name=None):
  r"""Sparse update entries in '*var' and '*accum' according to FOBOS algorithm.

  That is for rows we have grad for, we update var and accum as follows:
  accum += grad * grad
  prox_v = var
  prox_v -= lr * grad * (1 / sqrt(accum))
  var = sign(prox_v)/(1+lr*l2) * max{|prox_v|-lr*l1,0}

  Args:
    var: A `Tensor` of type `resource`. Should be from a Variable().
    accum: A `Tensor` of type `resource`. Should be from a Variable().
    lr: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`, `complex128`, `qint8`, `quint8`, `qint32`, `half`.
      Learning rate. Must be a scalar.
    l1: A `Tensor`. Must have the same type as `lr`.
      L1 regularization. Must be a scalar.
    l2: A `Tensor`. Must have the same type as `lr`.
      L2 regularization. Must be a scalar.
    grad: A `Tensor`. Must have the same type as `lr`. The gradient.
    indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A vector of indices into the first dimension of var and accum.
    use_locking: An optional `bool`. Defaults to `False`.
      If True, updating of the var and accum tensors will be protected by
      a lock; otherwise the behavior is undefined, but may exhibit less contention.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  result = _op_def_lib.apply_op("ResourceSparseApplyProximalAdagrad", var=var,
                                accum=accum, lr=lr, l1=l1, l2=l2, grad=grad,
                                indices=indices, use_locking=use_locking,
                                name=name)
  return result



def resource_sparse_apply_proximal_gradient_descent(var, alpha, l1, l2, grad,
                                                    indices, use_locking=None,
                                                    name=None):
  r"""Sparse update '*var' as FOBOS algorithm with fixed learning rate.

  That is for rows we have grad for, we update var as follows:
  prox_v = var - alpha * grad
  var = sign(prox_v)/(1+alpha*l2) * max{|prox_v|-alpha*l1,0}

  Args:
    var: A `Tensor` of type `resource`. Should be from a Variable().
    alpha: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`, `complex128`, `qint8`, `quint8`, `qint32`, `half`.
      Scaling factor. Must be a scalar.
    l1: A `Tensor`. Must have the same type as `alpha`.
      L1 regularization. Must be a scalar.
    l2: A `Tensor`. Must have the same type as `alpha`.
      L2 regularization. Must be a scalar.
    grad: A `Tensor`. Must have the same type as `alpha`. The gradient.
    indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A vector of indices into the first dimension of var and accum.
    use_locking: An optional `bool`. Defaults to `False`.
      If True, the subtraction will be protected by a lock;
      otherwise the behavior is undefined, but may exhibit less contention.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  result = _op_def_lib.apply_op("ResourceSparseApplyProximalGradientDescent",
                                var=var, alpha=alpha, l1=l1, l2=l2, grad=grad,
                                indices=indices, use_locking=use_locking,
                                name=name)
  return result



def resource_sparse_apply_rms_prop(var, ms, mom, lr, rho, momentum, epsilon,
                                   grad, indices, use_locking=None,
                                   name=None):
  r"""Update '*var' according to the RMSProp algorithm.

  Note that in dense implementation of this algorithm, ms and mom will
  update even if the grad is zero, but in this sparse implementation, ms
  and mom will not update in iterations during which the grad is zero.

  mean_square = decay * mean_square + (1-decay) * gradient ** 2
  Delta = learning_rate * gradient / sqrt(mean_square + epsilon)

  ms <- rho * ms_{t-1} + (1-rho) * grad * grad
  mom <- momentum * mom_{t-1} + lr * grad / sqrt(ms + epsilon)
  var <- var - mom

  Args:
    var: A `Tensor` of type `resource`. Should be from a Variable().
    ms: A `Tensor` of type `resource`. Should be from a Variable().
    mom: A `Tensor` of type `resource`. Should be from a Variable().
    lr: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`, `complex128`, `qint8`, `quint8`, `qint32`, `half`.
      Scaling factor. Must be a scalar.
    rho: A `Tensor`. Must have the same type as `lr`.
      Decay rate. Must be a scalar.
    momentum: A `Tensor`. Must have the same type as `lr`.
    epsilon: A `Tensor`. Must have the same type as `lr`.
      Ridge term. Must be a scalar.
    grad: A `Tensor`. Must have the same type as `lr`. The gradient.
    indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A vector of indices into the first dimension of var, ms and mom.
    use_locking: An optional `bool`. Defaults to `False`.
      If `True`, updating of the var, ms, and mom tensors is protected
      by a lock; otherwise the behavior is undefined, but may exhibit less
      contention.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  result = _op_def_lib.apply_op("ResourceSparseApplyRMSProp", var=var, ms=ms,
                                mom=mom, lr=lr, rho=rho, momentum=momentum,
                                epsilon=epsilon, grad=grad, indices=indices,
                                use_locking=use_locking, name=name)
  return result



def sparse_apply_adadelta(var, accum, accum_update, lr, rho, epsilon, grad,
                          indices, use_locking=None, name=None):
  r"""var: Should be from a Variable().

  Args:
    var: A mutable `Tensor`. Must be one of the following types: `float32`, `float64`, `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`, `complex128`, `qint8`, `quint8`, `qint32`, `half`.
    accum: A mutable `Tensor`. Must have the same type as `var`.
      Should be from a Variable().
    accum_update: A mutable `Tensor`. Must have the same type as `var`.
      : Should be from a Variable().
    lr: A `Tensor`. Must have the same type as `var`.
      Learning rate. Must be a scalar.
    rho: A `Tensor`. Must have the same type as `var`.
      Decay factor. Must be a scalar.
    epsilon: A `Tensor`. Must have the same type as `var`.
      Constant factor. Must be a scalar.
    grad: A `Tensor`. Must have the same type as `var`. The gradient.
    indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A vector of indices into the first dimension of var and accum.
    use_locking: An optional `bool`. Defaults to `False`.
      If True, updating of the var and accum tensors will be protected by
      a lock; otherwise the behavior is undefined, but may exhibit less contention.
    name: A name for the operation (optional).

  Returns:
    A mutable `Tensor`. Has the same type as `var`. Same as "var".
  """
  result = _op_def_lib.apply_op("SparseApplyAdadelta", var=var, accum=accum,
                                accum_update=accum_update, lr=lr, rho=rho,
                                epsilon=epsilon, grad=grad, indices=indices,
                                use_locking=use_locking, name=name)
  return result



def sparse_apply_adagrad(var, accum, lr, grad, indices, use_locking=None,
                         name=None):
  r"""Update relevant entries in '*var' and '*accum' according to the adagrad scheme.

  That is for rows we have grad for, we update var and accum as follows:
  accum += grad * grad
  var -= lr * grad * (1 / sqrt(accum))

  Args:
    var: A mutable `Tensor`. Must be one of the following types: `float32`, `float64`, `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`, `complex128`, `qint8`, `quint8`, `qint32`, `half`.
      Should be from a Variable().
    accum: A mutable `Tensor`. Must have the same type as `var`.
      Should be from a Variable().
    lr: A `Tensor`. Must have the same type as `var`.
      Learning rate. Must be a scalar.
    grad: A `Tensor`. Must have the same type as `var`. The gradient.
    indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A vector of indices into the first dimension of var and accum.
    use_locking: An optional `bool`. Defaults to `False`.
      If `True`, updating of the var and accum tensors will be protected
      by a lock; otherwise the behavior is undefined, but may exhibit less
      contention.
    name: A name for the operation (optional).

  Returns:
    A mutable `Tensor`. Has the same type as `var`. Same as "var".
  """
  result = _op_def_lib.apply_op("SparseApplyAdagrad", var=var, accum=accum,
                                lr=lr, grad=grad, indices=indices,
                                use_locking=use_locking, name=name)
  return result



def sparse_apply_adagrad_da(var, gradient_accumulator,
                            gradient_squared_accumulator, grad, indices, lr,
                            l1, l2, global_step, use_locking=None, name=None):
  r"""Update entries in '*var' and '*accum' according to the proximal adagrad scheme.

  Args:
    var: A mutable `Tensor`. Must be one of the following types: `float32`, `float64`, `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`, `complex128`, `qint8`, `quint8`, `qint32`, `half`.
      Should be from a Variable().
    gradient_accumulator: A mutable `Tensor`. Must have the same type as `var`.
      Should be from a Variable().
    gradient_squared_accumulator: A mutable `Tensor`. Must have the same type as `var`.
      Should be from a Variable().
    grad: A `Tensor`. Must have the same type as `var`. The gradient.
    indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A vector of indices into the first dimension of var and accum.
    lr: A `Tensor`. Must have the same type as `var`.
      Learning rate. Must be a scalar.
    l1: A `Tensor`. Must have the same type as `var`.
      L1 regularization. Must be a scalar.
    l2: A `Tensor`. Must have the same type as `var`.
      L2 regularization. Must be a scalar.
    global_step: A `Tensor` of type `int64`.
      Training step number. Must be a scalar.
    use_locking: An optional `bool`. Defaults to `False`.
      If True, updating of the var and accum tensors will be protected by
      a lock; otherwise the behavior is undefined, but may exhibit less contention.
    name: A name for the operation (optional).

  Returns:
    A mutable `Tensor`. Has the same type as `var`. Same as "var".
  """
  result = _op_def_lib.apply_op("SparseApplyAdagradDA", var=var,
                                gradient_accumulator=gradient_accumulator,
                                gradient_squared_accumulator=gradient_squared_accumulator,
                                grad=grad, indices=indices, lr=lr, l1=l1,
                                l2=l2, global_step=global_step,
                                use_locking=use_locking, name=name)
  return result



def sparse_apply_centered_rms_prop(var, mg, ms, mom, lr, rho, momentum,
                                   epsilon, grad, indices, use_locking=None,
                                   name=None):
  r"""Update '*var' according to the centered RMSProp algorithm.

  The centered RMSProp algorithm uses an estimate of the centered second moment
  (i.e., the variance) for normalization, as opposed to regular RMSProp, which
  uses the (uncentered) second moment. This often helps with training, but is
  slightly more expensive in terms of computation and memory.

  Note that in dense implementation of this algorithm, mg, ms, and mom will
  update even if the grad is zero, but in this sparse implementation, mg, ms,
  and mom will not update in iterations during which the grad is zero.

  mean_square = decay * mean_square + (1-decay) * gradient ** 2
  mean_grad = decay * mean_grad + (1-decay) * gradient
  Delta = learning_rate * gradient / sqrt(mean_square + epsilon - mean_grad ** 2)

  ms <- rho * ms_{t-1} + (1-rho) * grad * grad
  mom <- momentum * mom_{t-1} + lr * grad / sqrt(ms + epsilon)
  var <- var - mom

  Args:
    var: A mutable `Tensor`. Must be one of the following types: `float32`, `float64`, `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`, `complex128`, `qint8`, `quint8`, `qint32`, `half`.
      Should be from a Variable().
    mg: A mutable `Tensor`. Must have the same type as `var`.
      Should be from a Variable().
    ms: A mutable `Tensor`. Must have the same type as `var`.
      Should be from a Variable().
    mom: A mutable `Tensor`. Must have the same type as `var`.
      Should be from a Variable().
    lr: A `Tensor`. Must have the same type as `var`.
      Scaling factor. Must be a scalar.
    rho: A `Tensor`. Must have the same type as `var`.
      Decay rate. Must be a scalar.
    momentum: A `Tensor`. Must have the same type as `var`.
    epsilon: A `Tensor`. Must have the same type as `var`.
      Ridge term. Must be a scalar.
    grad: A `Tensor`. Must have the same type as `var`. The gradient.
    indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A vector of indices into the first dimension of var, ms and mom.
    use_locking: An optional `bool`. Defaults to `False`.
      If `True`, updating of the var, mg, ms, and mom tensors is
      protected by a lock; otherwise the behavior is undefined, but may exhibit less
      contention.
    name: A name for the operation (optional).

  Returns:
    A mutable `Tensor`. Has the same type as `var`. Same as "var".
  """
  result = _op_def_lib.apply_op("SparseApplyCenteredRMSProp", var=var, mg=mg,
                                ms=ms, mom=mom, lr=lr, rho=rho,
                                momentum=momentum, epsilon=epsilon, grad=grad,
                                indices=indices, use_locking=use_locking,
                                name=name)
  return result



def sparse_apply_ftrl(var, accum, linear, grad, indices, lr, l1, l2, lr_power,
                      use_locking=None, name=None):
  r"""Update relevant entries in '*var' according to the Ftrl-proximal scheme.

  That is for rows we have grad for, we update var, accum and linear as follows:
  accum_new = accum + grad * grad
  linear += grad + (accum_new^(-lr_power) - accum^(-lr_power)) / lr * var
  quadratic = 1.0 / (accum_new^(lr_power) * lr) + 2 * l2
  var = (sign(linear) * l1 - linear) / quadratic if |linear| > l1 else 0.0
  accum = accum_new

  Args:
    var: A mutable `Tensor`. Must be one of the following types: `float32`, `float64`, `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`, `complex128`, `qint8`, `quint8`, `qint32`, `half`.
      Should be from a Variable().
    accum: A mutable `Tensor`. Must have the same type as `var`.
      Should be from a Variable().
    linear: A mutable `Tensor`. Must have the same type as `var`.
      Should be from a Variable().
    grad: A `Tensor`. Must have the same type as `var`. The gradient.
    indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A vector of indices into the first dimension of var and accum.
    lr: A `Tensor`. Must have the same type as `var`.
      Scaling factor. Must be a scalar.
    l1: A `Tensor`. Must have the same type as `var`.
      L1 regularization. Must be a scalar.
    l2: A `Tensor`. Must have the same type as `var`.
      L2 regularization. Must be a scalar.
    lr_power: A `Tensor`. Must have the same type as `var`.
      Scaling factor. Must be a scalar.
    use_locking: An optional `bool`. Defaults to `False`.
      If `True`, updating of the var and accum tensors will be protected
      by a lock; otherwise the behavior is undefined, but may exhibit less
      contention.
    name: A name for the operation (optional).

  Returns:
    A mutable `Tensor`. Has the same type as `var`. Same as "var".
  """
  result = _op_def_lib.apply_op("SparseApplyFtrl", var=var, accum=accum,
                                linear=linear, grad=grad, indices=indices,
                                lr=lr, l1=l1, l2=l2, lr_power=lr_power,
                                use_locking=use_locking, name=name)
  return result



def sparse_apply_momentum(var, accum, lr, grad, indices, momentum,
                          use_locking=None, use_nesterov=None, name=None):
  r"""Update relevant entries in '*var' and '*accum' according to the momentum scheme.

  Set use_nesterov = True if you want to use Nesterov momentum.

  That is for rows we have grad for, we update var and accum as follows:

  accum = accum * momentum + grad
  var -= lr * accum

  Args:
    var: A mutable `Tensor`. Must be one of the following types: `float32`, `float64`, `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`, `complex128`, `qint8`, `quint8`, `qint32`, `half`.
      Should be from a Variable().
    accum: A mutable `Tensor`. Must have the same type as `var`.
      Should be from a Variable().
    lr: A `Tensor`. Must have the same type as `var`.
      Learning rate. Must be a scalar.
    grad: A `Tensor`. Must have the same type as `var`. The gradient.
    indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A vector of indices into the first dimension of var and accum.
    momentum: A `Tensor`. Must have the same type as `var`.
      Momentum. Must be a scalar.
    use_locking: An optional `bool`. Defaults to `False`.
      If `True`, updating of the var and accum tensors will be protected
      by a lock; otherwise the behavior is undefined, but may exhibit less
      contention.
    use_nesterov: An optional `bool`. Defaults to `False`.
      If `True`, the tensor passed to compute grad will be
      var - lr * momentum * accum, so in the end, the var you get is actually
      var - lr * momentum * accum.
    name: A name for the operation (optional).

  Returns:
    A mutable `Tensor`. Has the same type as `var`. Same as "var".
  """
  result = _op_def_lib.apply_op("SparseApplyMomentum", var=var, accum=accum,
                                lr=lr, grad=grad, indices=indices,
                                momentum=momentum, use_locking=use_locking,
                                use_nesterov=use_nesterov, name=name)
  return result



def sparse_apply_proximal_adagrad(var, accum, lr, l1, l2, grad, indices,
                                  use_locking=None, name=None):
  r"""Sparse update entries in '*var' and '*accum' according to FOBOS algorithm.

  That is for rows we have grad for, we update var and accum as follows:
  accum += grad * grad
  prox_v = var
  prox_v -= lr * grad * (1 / sqrt(accum))
  var = sign(prox_v)/(1+lr*l2) * max{|prox_v|-lr*l1,0}

  Args:
    var: A mutable `Tensor`. Must be one of the following types: `float32`, `float64`, `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`, `complex128`, `qint8`, `quint8`, `qint32`, `half`.
      Should be from a Variable().
    accum: A mutable `Tensor`. Must have the same type as `var`.
      Should be from a Variable().
    lr: A `Tensor`. Must have the same type as `var`.
      Learning rate. Must be a scalar.
    l1: A `Tensor`. Must have the same type as `var`.
      L1 regularization. Must be a scalar.
    l2: A `Tensor`. Must have the same type as `var`.
      L2 regularization. Must be a scalar.
    grad: A `Tensor`. Must have the same type as `var`. The gradient.
    indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A vector of indices into the first dimension of var and accum.
    use_locking: An optional `bool`. Defaults to `False`.
      If True, updating of the var and accum tensors will be protected by
      a lock; otherwise the behavior is undefined, but may exhibit less contention.
    name: A name for the operation (optional).

  Returns:
    A mutable `Tensor`. Has the same type as `var`. Same as "var".
  """
  result = _op_def_lib.apply_op("SparseApplyProximalAdagrad", var=var,
                                accum=accum, lr=lr, l1=l1, l2=l2, grad=grad,
                                indices=indices, use_locking=use_locking,
                                name=name)
  return result



def sparse_apply_proximal_gradient_descent(var, alpha, l1, l2, grad, indices,
                                           use_locking=None, name=None):
  r"""Sparse update '*var' as FOBOS algorithm with fixed learning rate.

  That is for rows we have grad for, we update var as follows:
  prox_v = var - alpha * grad
  var = sign(prox_v)/(1+alpha*l2) * max{|prox_v|-alpha*l1,0}

  Args:
    var: A mutable `Tensor`. Must be one of the following types: `float32`, `float64`, `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`, `complex128`, `qint8`, `quint8`, `qint32`, `half`.
      Should be from a Variable().
    alpha: A `Tensor`. Must have the same type as `var`.
      Scaling factor. Must be a scalar.
    l1: A `Tensor`. Must have the same type as `var`.
      L1 regularization. Must be a scalar.
    l2: A `Tensor`. Must have the same type as `var`.
      L2 regularization. Must be a scalar.
    grad: A `Tensor`. Must have the same type as `var`. The gradient.
    indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A vector of indices into the first dimension of var and accum.
    use_locking: An optional `bool`. Defaults to `False`.
      If True, the subtraction will be protected by a lock;
      otherwise the behavior is undefined, but may exhibit less contention.
    name: A name for the operation (optional).

  Returns:
    A mutable `Tensor`. Has the same type as `var`. Same as "var".
  """
  result = _op_def_lib.apply_op("SparseApplyProximalGradientDescent", var=var,
                                alpha=alpha, l1=l1, l2=l2, grad=grad,
                                indices=indices, use_locking=use_locking,
                                name=name)
  return result



def sparse_apply_rms_prop(var, ms, mom, lr, rho, momentum, epsilon, grad,
                          indices, use_locking=None, name=None):
  r"""Update '*var' according to the RMSProp algorithm.

  Note that in dense implementation of this algorithm, ms and mom will
  update even if the grad is zero, but in this sparse implementation, ms
  and mom will not update in iterations during which the grad is zero.

  mean_square = decay * mean_square + (1-decay) * gradient ** 2
  Delta = learning_rate * gradient / sqrt(mean_square + epsilon)

  ms <- rho * ms_{t-1} + (1-rho) * grad * grad
  mom <- momentum * mom_{t-1} + lr * grad / sqrt(ms + epsilon)
  var <- var - mom

  Args:
    var: A mutable `Tensor`. Must be one of the following types: `float32`, `float64`, `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`, `complex128`, `qint8`, `quint8`, `qint32`, `half`.
      Should be from a Variable().
    ms: A mutable `Tensor`. Must have the same type as `var`.
      Should be from a Variable().
    mom: A mutable `Tensor`. Must have the same type as `var`.
      Should be from a Variable().
    lr: A `Tensor`. Must have the same type as `var`.
      Scaling factor. Must be a scalar.
    rho: A `Tensor`. Must have the same type as `var`.
      Decay rate. Must be a scalar.
    momentum: A `Tensor`. Must have the same type as `var`.
    epsilon: A `Tensor`. Must have the same type as `var`.
      Ridge term. Must be a scalar.
    grad: A `Tensor`. Must have the same type as `var`. The gradient.
    indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A vector of indices into the first dimension of var, ms and mom.
    use_locking: An optional `bool`. Defaults to `False`.
      If `True`, updating of the var, ms, and mom tensors is protected
      by a lock; otherwise the behavior is undefined, but may exhibit less
      contention.
    name: A name for the operation (optional).

  Returns:
    A mutable `Tensor`. Has the same type as `var`. Same as "var".
  """
  result = _op_def_lib.apply_op("SparseApplyRMSProp", var=var, ms=ms, mom=mom,
                                lr=lr, rho=rho, momentum=momentum,
                                epsilon=epsilon, grad=grad, indices=indices,
                                use_locking=use_locking, name=name)
  return result


def _InitOpDefLibrary():
  op_list = _op_def_pb2.OpList()
  _text_format.Merge(_InitOpDefLibrary.op_list_ascii, op_list)
  _op_def_registry.register_op_list(op_list)
  op_def_lib = _op_def_library.OpDefLibrary()
  op_def_lib.add_op_list(op_list)
  return op_def_lib


_InitOpDefLibrary.op_list_ascii = """op {
  name: "ApplyAdadelta"
  input_arg {
    name: "var"
    type_attr: "T"
    is_ref: true
  }
  input_arg {
    name: "accum"
    type_attr: "T"
    is_ref: true
  }
  input_arg {
    name: "accum_update"
    type_attr: "T"
    is_ref: true
  }
  input_arg {
    name: "lr"
    type_attr: "T"
  }
  input_arg {
    name: "rho"
    type_attr: "T"
  }
  input_arg {
    name: "epsilon"
    type_attr: "T"
  }
  input_arg {
    name: "grad"
    type_attr: "T"
  }
  output_arg {
    name: "out"
    type_attr: "T"
    is_ref: true
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT64
        type: DT_INT32
        type: DT_UINT8
        type: DT_UINT16
        type: DT_INT16
        type: DT_INT8
        type: DT_COMPLEX64
        type: DT_COMPLEX128
        type: DT_QINT8
        type: DT_QUINT8
        type: DT_QINT32
        type: DT_HALF
      }
    }
  }
  attr {
    name: "use_locking"
    type: "bool"
    default_value {
      b: false
    }
  }
}
op {
  name: "ApplyAdagrad"
  input_arg {
    name: "var"
    type_attr: "T"
    is_ref: true
  }
  input_arg {
    name: "accum"
    type_attr: "T"
    is_ref: true
  }
  input_arg {
    name: "lr"
    type_attr: "T"
  }
  input_arg {
    name: "grad"
    type_attr: "T"
  }
  output_arg {
    name: "out"
    type_attr: "T"
    is_ref: true
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT64
        type: DT_INT32
        type: DT_UINT8
        type: DT_UINT16
        type: DT_INT16
        type: DT_INT8
        type: DT_COMPLEX64
        type: DT_COMPLEX128
        type: DT_QINT8
        type: DT_QUINT8
        type: DT_QINT32
        type: DT_HALF
      }
    }
  }
  attr {
    name: "use_locking"
    type: "bool"
    default_value {
      b: false
    }
  }
}
op {
  name: "ApplyAdagradDA"
  input_arg {
    name: "var"
    type_attr: "T"
    is_ref: true
  }
  input_arg {
    name: "gradient_accumulator"
    type_attr: "T"
    is_ref: true
  }
  input_arg {
    name: "gradient_squared_accumulator"
    type_attr: "T"
    is_ref: true
  }
  input_arg {
    name: "grad"
    type_attr: "T"
  }
  input_arg {
    name: "lr"
    type_attr: "T"
  }
  input_arg {
    name: "l1"
    type_attr: "T"
  }
  input_arg {
    name: "l2"
    type_attr: "T"
  }
  input_arg {
    name: "global_step"
    type: DT_INT64
  }
  output_arg {
    name: "out"
    type_attr: "T"
    is_ref: true
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT64
        type: DT_INT32
        type: DT_UINT8
        type: DT_UINT16
        type: DT_INT16
        type: DT_INT8
        type: DT_COMPLEX64
        type: DT_COMPLEX128
        type: DT_QINT8
        type: DT_QUINT8
        type: DT_QINT32
        type: DT_HALF
      }
    }
  }
  attr {
    name: "use_locking"
    type: "bool"
    default_value {
      b: false
    }
  }
}
op {
  name: "ApplyAdam"
  input_arg {
    name: "var"
    type_attr: "T"
    is_ref: true
  }
  input_arg {
    name: "m"
    type_attr: "T"
    is_ref: true
  }
  input_arg {
    name: "v"
    type_attr: "T"
    is_ref: true
  }
  input_arg {
    name: "beta1_power"
    type_attr: "T"
  }
  input_arg {
    name: "beta2_power"
    type_attr: "T"
  }
  input_arg {
    name: "lr"
    type_attr: "T"
  }
  input_arg {
    name: "beta1"
    type_attr: "T"
  }
  input_arg {
    name: "beta2"
    type_attr: "T"
  }
  input_arg {
    name: "epsilon"
    type_attr: "T"
  }
  input_arg {
    name: "grad"
    type_attr: "T"
  }
  output_arg {
    name: "out"
    type_attr: "T"
    is_ref: true
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT64
        type: DT_INT32
        type: DT_UINT8
        type: DT_UINT16
        type: DT_INT16
        type: DT_INT8
        type: DT_COMPLEX64
        type: DT_COMPLEX128
        type: DT_QINT8
        type: DT_QUINT8
        type: DT_QINT32
        type: DT_HALF
      }
    }
  }
  attr {
    name: "use_locking"
    type: "bool"
    default_value {
      b: false
    }
  }
  attr {
    name: "use_nesterov"
    type: "bool"
    default_value {
      b: false
    }
  }
}
op {
  name: "ApplyCenteredRMSProp"
  input_arg {
    name: "var"
    type_attr: "T"
    is_ref: true
  }
  input_arg {
    name: "mg"
    type_attr: "T"
    is_ref: true
  }
  input_arg {
    name: "ms"
    type_attr: "T"
    is_ref: true
  }
  input_arg {
    name: "mom"
    type_attr: "T"
    is_ref: true
  }
  input_arg {
    name: "lr"
    type_attr: "T"
  }
  input_arg {
    name: "rho"
    type_attr: "T"
  }
  input_arg {
    name: "momentum"
    type_attr: "T"
  }
  input_arg {
    name: "epsilon"
    type_attr: "T"
  }
  input_arg {
    name: "grad"
    type_attr: "T"
  }
  output_arg {
    name: "out"
    type_attr: "T"
    is_ref: true
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT64
        type: DT_INT32
        type: DT_UINT8
        type: DT_UINT16
        type: DT_INT16
        type: DT_INT8
        type: DT_COMPLEX64
        type: DT_COMPLEX128
        type: DT_QINT8
        type: DT_QUINT8
        type: DT_QINT32
        type: DT_HALF
      }
    }
  }
  attr {
    name: "use_locking"
    type: "bool"
    default_value {
      b: false
    }
  }
}
op {
  name: "ApplyFtrl"
  input_arg {
    name: "var"
    type_attr: "T"
    is_ref: true
  }
  input_arg {
    name: "accum"
    type_attr: "T"
    is_ref: true
  }
  input_arg {
    name: "linear"
    type_attr: "T"
    is_ref: true
  }
  input_arg {
    name: "grad"
    type_attr: "T"
  }
  input_arg {
    name: "lr"
    type_attr: "T"
  }
  input_arg {
    name: "l1"
    type_attr: "T"
  }
  input_arg {
    name: "l2"
    type_attr: "T"
  }
  input_arg {
    name: "lr_power"
    type_attr: "T"
  }
  output_arg {
    name: "out"
    type_attr: "T"
    is_ref: true
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT64
        type: DT_INT32
        type: DT_UINT8
        type: DT_UINT16
        type: DT_INT16
        type: DT_INT8
        type: DT_COMPLEX64
        type: DT_COMPLEX128
        type: DT_QINT8
        type: DT_QUINT8
        type: DT_QINT32
        type: DT_HALF
      }
    }
  }
  attr {
    name: "use_locking"
    type: "bool"
    default_value {
      b: false
    }
  }
}
op {
  name: "ApplyGradientDescent"
  input_arg {
    name: "var"
    type_attr: "T"
    is_ref: true
  }
  input_arg {
    name: "alpha"
    type_attr: "T"
  }
  input_arg {
    name: "delta"
    type_attr: "T"
  }
  output_arg {
    name: "out"
    type_attr: "T"
    is_ref: true
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT64
        type: DT_INT32
        type: DT_UINT8
        type: DT_UINT16
        type: DT_INT16
        type: DT_INT8
        type: DT_COMPLEX64
        type: DT_COMPLEX128
        type: DT_QINT8
        type: DT_QUINT8
        type: DT_QINT32
        type: DT_HALF
      }
    }
  }
  attr {
    name: "use_locking"
    type: "bool"
    default_value {
      b: false
    }
  }
}
op {
  name: "ApplyMomentum"
  input_arg {
    name: "var"
    type_attr: "T"
    is_ref: true
  }
  input_arg {
    name: "accum"
    type_attr: "T"
    is_ref: true
  }
  input_arg {
    name: "lr"
    type_attr: "T"
  }
  input_arg {
    name: "grad"
    type_attr: "T"
  }
  input_arg {
    name: "momentum"
    type_attr: "T"
  }
  output_arg {
    name: "out"
    type_attr: "T"
    is_ref: true
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT64
        type: DT_INT32
        type: DT_UINT8
        type: DT_UINT16
        type: DT_INT16
        type: DT_INT8
        type: DT_COMPLEX64
        type: DT_COMPLEX128
        type: DT_QINT8
        type: DT_QUINT8
        type: DT_QINT32
        type: DT_HALF
      }
    }
  }
  attr {
    name: "use_locking"
    type: "bool"
    default_value {
      b: false
    }
  }
  attr {
    name: "use_nesterov"
    type: "bool"
    default_value {
      b: false
    }
  }
}
op {
  name: "ApplyProximalAdagrad"
  input_arg {
    name: "var"
    type_attr: "T"
    is_ref: true
  }
  input_arg {
    name: "accum"
    type_attr: "T"
    is_ref: true
  }
  input_arg {
    name: "lr"
    type_attr: "T"
  }
  input_arg {
    name: "l1"
    type_attr: "T"
  }
  input_arg {
    name: "l2"
    type_attr: "T"
  }
  input_arg {
    name: "grad"
    type_attr: "T"
  }
  output_arg {
    name: "out"
    type_attr: "T"
    is_ref: true
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT64
        type: DT_INT32
        type: DT_UINT8
        type: DT_UINT16
        type: DT_INT16
        type: DT_INT8
        type: DT_COMPLEX64
        type: DT_COMPLEX128
        type: DT_QINT8
        type: DT_QUINT8
        type: DT_QINT32
        type: DT_HALF
      }
    }
  }
  attr {
    name: "use_locking"
    type: "bool"
    default_value {
      b: false
    }
  }
}
op {
  name: "ApplyProximalGradientDescent"
  input_arg {
    name: "var"
    type_attr: "T"
    is_ref: true
  }
  input_arg {
    name: "alpha"
    type_attr: "T"
  }
  input_arg {
    name: "l1"
    type_attr: "T"
  }
  input_arg {
    name: "l2"
    type_attr: "T"
  }
  input_arg {
    name: "delta"
    type_attr: "T"
  }
  output_arg {
    name: "out"
    type_attr: "T"
    is_ref: true
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT64
        type: DT_INT32
        type: DT_UINT8
        type: DT_UINT16
        type: DT_INT16
        type: DT_INT8
        type: DT_COMPLEX64
        type: DT_COMPLEX128
        type: DT_QINT8
        type: DT_QUINT8
        type: DT_QINT32
        type: DT_HALF
      }
    }
  }
  attr {
    name: "use_locking"
    type: "bool"
    default_value {
      b: false
    }
  }
}
op {
  name: "ApplyRMSProp"
  input_arg {
    name: "var"
    type_attr: "T"
    is_ref: true
  }
  input_arg {
    name: "ms"
    type_attr: "T"
    is_ref: true
  }
  input_arg {
    name: "mom"
    type_attr: "T"
    is_ref: true
  }
  input_arg {
    name: "lr"
    type_attr: "T"
  }
  input_arg {
    name: "rho"
    type_attr: "T"
  }
  input_arg {
    name: "momentum"
    type_attr: "T"
  }
  input_arg {
    name: "epsilon"
    type_attr: "T"
  }
  input_arg {
    name: "grad"
    type_attr: "T"
  }
  output_arg {
    name: "out"
    type_attr: "T"
    is_ref: true
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT64
        type: DT_INT32
        type: DT_UINT8
        type: DT_UINT16
        type: DT_INT16
        type: DT_INT8
        type: DT_COMPLEX64
        type: DT_COMPLEX128
        type: DT_QINT8
        type: DT_QUINT8
        type: DT_QINT32
        type: DT_HALF
      }
    }
  }
  attr {
    name: "use_locking"
    type: "bool"
    default_value {
      b: false
    }
  }
}
op {
  name: "ResourceApplyAdadelta"
  input_arg {
    name: "var"
    type: DT_RESOURCE
  }
  input_arg {
    name: "accum"
    type: DT_RESOURCE
  }
  input_arg {
    name: "accum_update"
    type: DT_RESOURCE
  }
  input_arg {
    name: "lr"
    type_attr: "T"
  }
  input_arg {
    name: "rho"
    type_attr: "T"
  }
  input_arg {
    name: "epsilon"
    type_attr: "T"
  }
  input_arg {
    name: "grad"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT64
        type: DT_INT32
        type: DT_UINT8
        type: DT_UINT16
        type: DT_INT16
        type: DT_INT8
        type: DT_COMPLEX64
        type: DT_COMPLEX128
        type: DT_QINT8
        type: DT_QUINT8
        type: DT_QINT32
        type: DT_HALF
      }
    }
  }
  attr {
    name: "use_locking"
    type: "bool"
    default_value {
      b: false
    }
  }
  is_stateful: true
}
op {
  name: "ResourceApplyAdagrad"
  input_arg {
    name: "var"
    type: DT_RESOURCE
  }
  input_arg {
    name: "accum"
    type: DT_RESOURCE
  }
  input_arg {
    name: "lr"
    type_attr: "T"
  }
  input_arg {
    name: "grad"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT64
        type: DT_INT32
        type: DT_UINT8
        type: DT_UINT16
        type: DT_INT16
        type: DT_INT8
        type: DT_COMPLEX64
        type: DT_COMPLEX128
        type: DT_QINT8
        type: DT_QUINT8
        type: DT_QINT32
        type: DT_HALF
      }
    }
  }
  attr {
    name: "use_locking"
    type: "bool"
    default_value {
      b: false
    }
  }
  is_stateful: true
}
op {
  name: "ResourceApplyAdagradDA"
  input_arg {
    name: "var"
    type: DT_RESOURCE
  }
  input_arg {
    name: "gradient_accumulator"
    type: DT_RESOURCE
  }
  input_arg {
    name: "gradient_squared_accumulator"
    type: DT_RESOURCE
  }
  input_arg {
    name: "grad"
    type_attr: "T"
  }
  input_arg {
    name: "lr"
    type_attr: "T"
  }
  input_arg {
    name: "l1"
    type_attr: "T"
  }
  input_arg {
    name: "l2"
    type_attr: "T"
  }
  input_arg {
    name: "global_step"
    type: DT_INT64
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT64
        type: DT_INT32
        type: DT_UINT8
        type: DT_UINT16
        type: DT_INT16
        type: DT_INT8
        type: DT_COMPLEX64
        type: DT_COMPLEX128
        type: DT_QINT8
        type: DT_QUINT8
        type: DT_QINT32
        type: DT_HALF
      }
    }
  }
  attr {
    name: "use_locking"
    type: "bool"
    default_value {
      b: false
    }
  }
  is_stateful: true
}
op {
  name: "ResourceApplyAdam"
  input_arg {
    name: "var"
    type: DT_RESOURCE
  }
  input_arg {
    name: "m"
    type: DT_RESOURCE
  }
  input_arg {
    name: "v"
    type: DT_RESOURCE
  }
  input_arg {
    name: "beta1_power"
    type_attr: "T"
  }
  input_arg {
    name: "beta2_power"
    type_attr: "T"
  }
  input_arg {
    name: "lr"
    type_attr: "T"
  }
  input_arg {
    name: "beta1"
    type_attr: "T"
  }
  input_arg {
    name: "beta2"
    type_attr: "T"
  }
  input_arg {
    name: "epsilon"
    type_attr: "T"
  }
  input_arg {
    name: "grad"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT64
        type: DT_INT32
        type: DT_UINT8
        type: DT_UINT16
        type: DT_INT16
        type: DT_INT8
        type: DT_COMPLEX64
        type: DT_COMPLEX128
        type: DT_QINT8
        type: DT_QUINT8
        type: DT_QINT32
        type: DT_HALF
      }
    }
  }
  attr {
    name: "use_locking"
    type: "bool"
    default_value {
      b: false
    }
  }
  attr {
    name: "use_nesterov"
    type: "bool"
    default_value {
      b: false
    }
  }
  is_stateful: true
}
op {
  name: "ResourceApplyCenteredRMSProp"
  input_arg {
    name: "var"
    type: DT_RESOURCE
  }
  input_arg {
    name: "mg"
    type: DT_RESOURCE
  }
  input_arg {
    name: "ms"
    type: DT_RESOURCE
  }
  input_arg {
    name: "mom"
    type: DT_RESOURCE
  }
  input_arg {
    name: "lr"
    type_attr: "T"
  }
  input_arg {
    name: "rho"
    type_attr: "T"
  }
  input_arg {
    name: "momentum"
    type_attr: "T"
  }
  input_arg {
    name: "epsilon"
    type_attr: "T"
  }
  input_arg {
    name: "grad"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT64
        type: DT_INT32
        type: DT_UINT8
        type: DT_UINT16
        type: DT_INT16
        type: DT_INT8
        type: DT_COMPLEX64
        type: DT_COMPLEX128
        type: DT_QINT8
        type: DT_QUINT8
        type: DT_QINT32
        type: DT_HALF
      }
    }
  }
  attr {
    name: "use_locking"
    type: "bool"
    default_value {
      b: false
    }
  }
  is_stateful: true
}
op {
  name: "ResourceApplyFtrl"
  input_arg {
    name: "var"
    type: DT_RESOURCE
  }
  input_arg {
    name: "accum"
    type: DT_RESOURCE
  }
  input_arg {
    name: "linear"
    type: DT_RESOURCE
  }
  input_arg {
    name: "grad"
    type_attr: "T"
  }
  input_arg {
    name: "lr"
    type_attr: "T"
  }
  input_arg {
    name: "l1"
    type_attr: "T"
  }
  input_arg {
    name: "l2"
    type_attr: "T"
  }
  input_arg {
    name: "lr_power"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT64
        type: DT_INT32
        type: DT_UINT8
        type: DT_UINT16
        type: DT_INT16
        type: DT_INT8
        type: DT_COMPLEX64
        type: DT_COMPLEX128
        type: DT_QINT8
        type: DT_QUINT8
        type: DT_QINT32
        type: DT_HALF
      }
    }
  }
  attr {
    name: "use_locking"
    type: "bool"
    default_value {
      b: false
    }
  }
  is_stateful: true
}
op {
  name: "ResourceApplyGradientDescent"
  input_arg {
    name: "var"
    type: DT_RESOURCE
  }
  input_arg {
    name: "alpha"
    type_attr: "T"
  }
  input_arg {
    name: "delta"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT64
        type: DT_INT32
        type: DT_UINT8
        type: DT_UINT16
        type: DT_INT16
        type: DT_INT8
        type: DT_COMPLEX64
        type: DT_COMPLEX128
        type: DT_QINT8
        type: DT_QUINT8
        type: DT_QINT32
        type: DT_HALF
      }
    }
  }
  attr {
    name: "use_locking"
    type: "bool"
    default_value {
      b: false
    }
  }
  is_stateful: true
}
op {
  name: "ResourceApplyMomentum"
  input_arg {
    name: "var"
    type: DT_RESOURCE
  }
  input_arg {
    name: "accum"
    type: DT_RESOURCE
  }
  input_arg {
    name: "lr"
    type_attr: "T"
  }
  input_arg {
    name: "grad"
    type_attr: "T"
  }
  input_arg {
    name: "momentum"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT64
        type: DT_INT32
        type: DT_UINT8
        type: DT_UINT16
        type: DT_INT16
        type: DT_INT8
        type: DT_COMPLEX64
        type: DT_COMPLEX128
        type: DT_QINT8
        type: DT_QUINT8
        type: DT_QINT32
        type: DT_HALF
      }
    }
  }
  attr {
    name: "use_locking"
    type: "bool"
    default_value {
      b: false
    }
  }
  attr {
    name: "use_nesterov"
    type: "bool"
    default_value {
      b: false
    }
  }
  is_stateful: true
}
op {
  name: "ResourceApplyProximalAdagrad"
  input_arg {
    name: "var"
    type: DT_RESOURCE
  }
  input_arg {
    name: "accum"
    type: DT_RESOURCE
  }
  input_arg {
    name: "lr"
    type_attr: "T"
  }
  input_arg {
    name: "l1"
    type_attr: "T"
  }
  input_arg {
    name: "l2"
    type_attr: "T"
  }
  input_arg {
    name: "grad"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT64
        type: DT_INT32
        type: DT_UINT8
        type: DT_UINT16
        type: DT_INT16
        type: DT_INT8
        type: DT_COMPLEX64
        type: DT_COMPLEX128
        type: DT_QINT8
        type: DT_QUINT8
        type: DT_QINT32
        type: DT_HALF
      }
    }
  }
  attr {
    name: "use_locking"
    type: "bool"
    default_value {
      b: false
    }
  }
  is_stateful: true
}
op {
  name: "ResourceApplyProximalGradientDescent"
  input_arg {
    name: "var"
    type: DT_RESOURCE
  }
  input_arg {
    name: "alpha"
    type_attr: "T"
  }
  input_arg {
    name: "l1"
    type_attr: "T"
  }
  input_arg {
    name: "l2"
    type_attr: "T"
  }
  input_arg {
    name: "delta"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT64
        type: DT_INT32
        type: DT_UINT8
        type: DT_UINT16
        type: DT_INT16
        type: DT_INT8
        type: DT_COMPLEX64
        type: DT_COMPLEX128
        type: DT_QINT8
        type: DT_QUINT8
        type: DT_QINT32
        type: DT_HALF
      }
    }
  }
  attr {
    name: "use_locking"
    type: "bool"
    default_value {
      b: false
    }
  }
  is_stateful: true
}
op {
  name: "ResourceApplyRMSProp"
  input_arg {
    name: "var"
    type: DT_RESOURCE
  }
  input_arg {
    name: "ms"
    type: DT_RESOURCE
  }
  input_arg {
    name: "mom"
    type: DT_RESOURCE
  }
  input_arg {
    name: "lr"
    type_attr: "T"
  }
  input_arg {
    name: "rho"
    type_attr: "T"
  }
  input_arg {
    name: "momentum"
    type_attr: "T"
  }
  input_arg {
    name: "epsilon"
    type_attr: "T"
  }
  input_arg {
    name: "grad"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT64
        type: DT_INT32
        type: DT_UINT8
        type: DT_UINT16
        type: DT_INT16
        type: DT_INT8
        type: DT_COMPLEX64
        type: DT_COMPLEX128
        type: DT_QINT8
        type: DT_QUINT8
        type: DT_QINT32
        type: DT_HALF
      }
    }
  }
  attr {
    name: "use_locking"
    type: "bool"
    default_value {
      b: false
    }
  }
  is_stateful: true
}
op {
  name: "ResourceSparseApplyAdadelta"
  input_arg {
    name: "var"
    type: DT_RESOURCE
  }
  input_arg {
    name: "accum"
    type: DT_RESOURCE
  }
  input_arg {
    name: "accum_update"
    type: DT_RESOURCE
  }
  input_arg {
    name: "lr"
    type_attr: "T"
  }
  input_arg {
    name: "rho"
    type_attr: "T"
  }
  input_arg {
    name: "epsilon"
    type_attr: "T"
  }
  input_arg {
    name: "grad"
    type_attr: "T"
  }
  input_arg {
    name: "indices"
    type_attr: "Tindices"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT64
        type: DT_INT32
        type: DT_UINT8
        type: DT_UINT16
        type: DT_INT16
        type: DT_INT8
        type: DT_COMPLEX64
        type: DT_COMPLEX128
        type: DT_QINT8
        type: DT_QUINT8
        type: DT_QINT32
        type: DT_HALF
      }
    }
  }
  attr {
    name: "Tindices"
    type: "type"
    allowed_values {
      list {
        type: DT_INT32
        type: DT_INT64
      }
    }
  }
  attr {
    name: "use_locking"
    type: "bool"
    default_value {
      b: false
    }
  }
  is_stateful: true
}
op {
  name: "ResourceSparseApplyAdagrad"
  input_arg {
    name: "var"
    type: DT_RESOURCE
  }
  input_arg {
    name: "accum"
    type: DT_RESOURCE
  }
  input_arg {
    name: "lr"
    type_attr: "T"
  }
  input_arg {
    name: "grad"
    type_attr: "T"
  }
  input_arg {
    name: "indices"
    type_attr: "Tindices"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT64
        type: DT_INT32
        type: DT_UINT8
        type: DT_UINT16
        type: DT_INT16
        type: DT_INT8
        type: DT_COMPLEX64
        type: DT_COMPLEX128
        type: DT_QINT8
        type: DT_QUINT8
        type: DT_QINT32
        type: DT_HALF
      }
    }
  }
  attr {
    name: "Tindices"
    type: "type"
    allowed_values {
      list {
        type: DT_INT32
        type: DT_INT64
      }
    }
  }
  attr {
    name: "use_locking"
    type: "bool"
    default_value {
      b: false
    }
  }
  is_stateful: true
}
op {
  name: "ResourceSparseApplyAdagradDA"
  input_arg {
    name: "var"
    type: DT_RESOURCE
  }
  input_arg {
    name: "gradient_accumulator"
    type: DT_RESOURCE
  }
  input_arg {
    name: "gradient_squared_accumulator"
    type: DT_RESOURCE
  }
  input_arg {
    name: "grad"
    type_attr: "T"
  }
  input_arg {
    name: "indices"
    type_attr: "Tindices"
  }
  input_arg {
    name: "lr"
    type_attr: "T"
  }
  input_arg {
    name: "l1"
    type_attr: "T"
  }
  input_arg {
    name: "l2"
    type_attr: "T"
  }
  input_arg {
    name: "global_step"
    type: DT_INT64
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT64
        type: DT_INT32
        type: DT_UINT8
        type: DT_UINT16
        type: DT_INT16
        type: DT_INT8
        type: DT_COMPLEX64
        type: DT_COMPLEX128
        type: DT_QINT8
        type: DT_QUINT8
        type: DT_QINT32
        type: DT_HALF
      }
    }
  }
  attr {
    name: "Tindices"
    type: "type"
    allowed_values {
      list {
        type: DT_INT32
        type: DT_INT64
      }
    }
  }
  attr {
    name: "use_locking"
    type: "bool"
    default_value {
      b: false
    }
  }
  is_stateful: true
}
op {
  name: "ResourceSparseApplyCenteredRMSProp"
  input_arg {
    name: "var"
    type: DT_RESOURCE
  }
  input_arg {
    name: "mg"
    type: DT_RESOURCE
  }
  input_arg {
    name: "ms"
    type: DT_RESOURCE
  }
  input_arg {
    name: "mom"
    type: DT_RESOURCE
  }
  input_arg {
    name: "lr"
    type_attr: "T"
  }
  input_arg {
    name: "rho"
    type_attr: "T"
  }
  input_arg {
    name: "momentum"
    type_attr: "T"
  }
  input_arg {
    name: "epsilon"
    type_attr: "T"
  }
  input_arg {
    name: "grad"
    type_attr: "T"
  }
  input_arg {
    name: "indices"
    type_attr: "Tindices"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT64
        type: DT_INT32
        type: DT_UINT8
        type: DT_UINT16
        type: DT_INT16
        type: DT_INT8
        type: DT_COMPLEX64
        type: DT_COMPLEX128
        type: DT_QINT8
        type: DT_QUINT8
        type: DT_QINT32
        type: DT_HALF
      }
    }
  }
  attr {
    name: "Tindices"
    type: "type"
    allowed_values {
      list {
        type: DT_INT32
        type: DT_INT64
      }
    }
  }
  attr {
    name: "use_locking"
    type: "bool"
    default_value {
      b: false
    }
  }
  is_stateful: true
}
op {
  name: "ResourceSparseApplyFtrl"
  input_arg {
    name: "var"
    type: DT_RESOURCE
  }
  input_arg {
    name: "accum"
    type: DT_RESOURCE
  }
  input_arg {
    name: "linear"
    type: DT_RESOURCE
  }
  input_arg {
    name: "grad"
    type_attr: "T"
  }
  input_arg {
    name: "indices"
    type_attr: "Tindices"
  }
  input_arg {
    name: "lr"
    type_attr: "T"
  }
  input_arg {
    name: "l1"
    type_attr: "T"
  }
  input_arg {
    name: "l2"
    type_attr: "T"
  }
  input_arg {
    name: "lr_power"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT64
        type: DT_INT32
        type: DT_UINT8
        type: DT_UINT16
        type: DT_INT16
        type: DT_INT8
        type: DT_COMPLEX64
        type: DT_COMPLEX128
        type: DT_QINT8
        type: DT_QUINT8
        type: DT_QINT32
        type: DT_HALF
      }
    }
  }
  attr {
    name: "Tindices"
    type: "type"
    allowed_values {
      list {
        type: DT_INT32
        type: DT_INT64
      }
    }
  }
  attr {
    name: "use_locking"
    type: "bool"
    default_value {
      b: false
    }
  }
  is_stateful: true
}
op {
  name: "ResourceSparseApplyMomentum"
  input_arg {
    name: "var"
    type: DT_RESOURCE
  }
  input_arg {
    name: "accum"
    type: DT_RESOURCE
  }
  input_arg {
    name: "lr"
    type_attr: "T"
  }
  input_arg {
    name: "grad"
    type_attr: "T"
  }
  input_arg {
    name: "indices"
    type_attr: "Tindices"
  }
  input_arg {
    name: "momentum"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT64
        type: DT_INT32
        type: DT_UINT8
        type: DT_UINT16
        type: DT_INT16
        type: DT_INT8
        type: DT_COMPLEX64
        type: DT_COMPLEX128
        type: DT_QINT8
        type: DT_QUINT8
        type: DT_QINT32
        type: DT_HALF
      }
    }
  }
  attr {
    name: "Tindices"
    type: "type"
    allowed_values {
      list {
        type: DT_INT32
        type: DT_INT64
      }
    }
  }
  attr {
    name: "use_locking"
    type: "bool"
    default_value {
      b: false
    }
  }
  attr {
    name: "use_nesterov"
    type: "bool"
    default_value {
      b: false
    }
  }
  is_stateful: true
}
op {
  name: "ResourceSparseApplyProximalAdagrad"
  input_arg {
    name: "var"
    type: DT_RESOURCE
  }
  input_arg {
    name: "accum"
    type: DT_RESOURCE
  }
  input_arg {
    name: "lr"
    type_attr: "T"
  }
  input_arg {
    name: "l1"
    type_attr: "T"
  }
  input_arg {
    name: "l2"
    type_attr: "T"
  }
  input_arg {
    name: "grad"
    type_attr: "T"
  }
  input_arg {
    name: "indices"
    type_attr: "Tindices"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT64
        type: DT_INT32
        type: DT_UINT8
        type: DT_UINT16
        type: DT_INT16
        type: DT_INT8
        type: DT_COMPLEX64
        type: DT_COMPLEX128
        type: DT_QINT8
        type: DT_QUINT8
        type: DT_QINT32
        type: DT_HALF
      }
    }
  }
  attr {
    name: "Tindices"
    type: "type"
    allowed_values {
      list {
        type: DT_INT32
        type: DT_INT64
      }
    }
  }
  attr {
    name: "use_locking"
    type: "bool"
    default_value {
      b: false
    }
  }
  is_stateful: true
}
op {
  name: "ResourceSparseApplyProximalGradientDescent"
  input_arg {
    name: "var"
    type: DT_RESOURCE
  }
  input_arg {
    name: "alpha"
    type_attr: "T"
  }
  input_arg {
    name: "l1"
    type_attr: "T"
  }
  input_arg {
    name: "l2"
    type_attr: "T"
  }
  input_arg {
    name: "grad"
    type_attr: "T"
  }
  input_arg {
    name: "indices"
    type_attr: "Tindices"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT64
        type: DT_INT32
        type: DT_UINT8
        type: DT_UINT16
        type: DT_INT16
        type: DT_INT8
        type: DT_COMPLEX64
        type: DT_COMPLEX128
        type: DT_QINT8
        type: DT_QUINT8
        type: DT_QINT32
        type: DT_HALF
      }
    }
  }
  attr {
    name: "Tindices"
    type: "type"
    allowed_values {
      list {
        type: DT_INT32
        type: DT_INT64
      }
    }
  }
  attr {
    name: "use_locking"
    type: "bool"
    default_value {
      b: false
    }
  }
  is_stateful: true
}
op {
  name: "ResourceSparseApplyRMSProp"
  input_arg {
    name: "var"
    type: DT_RESOURCE
  }
  input_arg {
    name: "ms"
    type: DT_RESOURCE
  }
  input_arg {
    name: "mom"
    type: DT_RESOURCE
  }
  input_arg {
    name: "lr"
    type_attr: "T"
  }
  input_arg {
    name: "rho"
    type_attr: "T"
  }
  input_arg {
    name: "momentum"
    type_attr: "T"
  }
  input_arg {
    name: "epsilon"
    type_attr: "T"
  }
  input_arg {
    name: "grad"
    type_attr: "T"
  }
  input_arg {
    name: "indices"
    type_attr: "Tindices"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT64
        type: DT_INT32
        type: DT_UINT8
        type: DT_UINT16
        type: DT_INT16
        type: DT_INT8
        type: DT_COMPLEX64
        type: DT_COMPLEX128
        type: DT_QINT8
        type: DT_QUINT8
        type: DT_QINT32
        type: DT_HALF
      }
    }
  }
  attr {
    name: "Tindices"
    type: "type"
    allowed_values {
      list {
        type: DT_INT32
        type: DT_INT64
      }
    }
  }
  attr {
    name: "use_locking"
    type: "bool"
    default_value {
      b: false
    }
  }
  is_stateful: true
}
op {
  name: "SparseApplyAdadelta"
  input_arg {
    name: "var"
    type_attr: "T"
    is_ref: true
  }
  input_arg {
    name: "accum"
    type_attr: "T"
    is_ref: true
  }
  input_arg {
    name: "accum_update"
    type_attr: "T"
    is_ref: true
  }
  input_arg {
    name: "lr"
    type_attr: "T"
  }
  input_arg {
    name: "rho"
    type_attr: "T"
  }
  input_arg {
    name: "epsilon"
    type_attr: "T"
  }
  input_arg {
    name: "grad"
    type_attr: "T"
  }
  input_arg {
    name: "indices"
    type_attr: "Tindices"
  }
  output_arg {
    name: "out"
    type_attr: "T"
    is_ref: true
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT64
        type: DT_INT32
        type: DT_UINT8
        type: DT_UINT16
        type: DT_INT16
        type: DT_INT8
        type: DT_COMPLEX64
        type: DT_COMPLEX128
        type: DT_QINT8
        type: DT_QUINT8
        type: DT_QINT32
        type: DT_HALF
      }
    }
  }
  attr {
    name: "Tindices"
    type: "type"
    allowed_values {
      list {
        type: DT_INT32
        type: DT_INT64
      }
    }
  }
  attr {
    name: "use_locking"
    type: "bool"
    default_value {
      b: false
    }
  }
}
op {
  name: "SparseApplyAdagrad"
  input_arg {
    name: "var"
    type_attr: "T"
    is_ref: true
  }
  input_arg {
    name: "accum"
    type_attr: "T"
    is_ref: true
  }
  input_arg {
    name: "lr"
    type_attr: "T"
  }
  input_arg {
    name: "grad"
    type_attr: "T"
  }
  input_arg {
    name: "indices"
    type_attr: "Tindices"
  }
  output_arg {
    name: "out"
    type_attr: "T"
    is_ref: true
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT64
        type: DT_INT32
        type: DT_UINT8
        type: DT_UINT16
        type: DT_INT16
        type: DT_INT8
        type: DT_COMPLEX64
        type: DT_COMPLEX128
        type: DT_QINT8
        type: DT_QUINT8
        type: DT_QINT32
        type: DT_HALF
      }
    }
  }
  attr {
    name: "Tindices"
    type: "type"
    allowed_values {
      list {
        type: DT_INT32
        type: DT_INT64
      }
    }
  }
  attr {
    name: "use_locking"
    type: "bool"
    default_value {
      b: false
    }
  }
}
op {
  name: "SparseApplyAdagradDA"
  input_arg {
    name: "var"
    type_attr: "T"
    is_ref: true
  }
  input_arg {
    name: "gradient_accumulator"
    type_attr: "T"
    is_ref: true
  }
  input_arg {
    name: "gradient_squared_accumulator"
    type_attr: "T"
    is_ref: true
  }
  input_arg {
    name: "grad"
    type_attr: "T"
  }
  input_arg {
    name: "indices"
    type_attr: "Tindices"
  }
  input_arg {
    name: "lr"
    type_attr: "T"
  }
  input_arg {
    name: "l1"
    type_attr: "T"
  }
  input_arg {
    name: "l2"
    type_attr: "T"
  }
  input_arg {
    name: "global_step"
    type: DT_INT64
  }
  output_arg {
    name: "out"
    type_attr: "T"
    is_ref: true
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT64
        type: DT_INT32
        type: DT_UINT8
        type: DT_UINT16
        type: DT_INT16
        type: DT_INT8
        type: DT_COMPLEX64
        type: DT_COMPLEX128
        type: DT_QINT8
        type: DT_QUINT8
        type: DT_QINT32
        type: DT_HALF
      }
    }
  }
  attr {
    name: "Tindices"
    type: "type"
    allowed_values {
      list {
        type: DT_INT32
        type: DT_INT64
      }
    }
  }
  attr {
    name: "use_locking"
    type: "bool"
    default_value {
      b: false
    }
  }
}
op {
  name: "SparseApplyCenteredRMSProp"
  input_arg {
    name: "var"
    type_attr: "T"
    is_ref: true
  }
  input_arg {
    name: "mg"
    type_attr: "T"
    is_ref: true
  }
  input_arg {
    name: "ms"
    type_attr: "T"
    is_ref: true
  }
  input_arg {
    name: "mom"
    type_attr: "T"
    is_ref: true
  }
  input_arg {
    name: "lr"
    type_attr: "T"
  }
  input_arg {
    name: "rho"
    type_attr: "T"
  }
  input_arg {
    name: "momentum"
    type_attr: "T"
  }
  input_arg {
    name: "epsilon"
    type_attr: "T"
  }
  input_arg {
    name: "grad"
    type_attr: "T"
  }
  input_arg {
    name: "indices"
    type_attr: "Tindices"
  }
  output_arg {
    name: "out"
    type_attr: "T"
    is_ref: true
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT64
        type: DT_INT32
        type: DT_UINT8
        type: DT_UINT16
        type: DT_INT16
        type: DT_INT8
        type: DT_COMPLEX64
        type: DT_COMPLEX128
        type: DT_QINT8
        type: DT_QUINT8
        type: DT_QINT32
        type: DT_HALF
      }
    }
  }
  attr {
    name: "Tindices"
    type: "type"
    allowed_values {
      list {
        type: DT_INT32
        type: DT_INT64
      }
    }
  }
  attr {
    name: "use_locking"
    type: "bool"
    default_value {
      b: false
    }
  }
}
op {
  name: "SparseApplyFtrl"
  input_arg {
    name: "var"
    type_attr: "T"
    is_ref: true
  }
  input_arg {
    name: "accum"
    type_attr: "T"
    is_ref: true
  }
  input_arg {
    name: "linear"
    type_attr: "T"
    is_ref: true
  }
  input_arg {
    name: "grad"
    type_attr: "T"
  }
  input_arg {
    name: "indices"
    type_attr: "Tindices"
  }
  input_arg {
    name: "lr"
    type_attr: "T"
  }
  input_arg {
    name: "l1"
    type_attr: "T"
  }
  input_arg {
    name: "l2"
    type_attr: "T"
  }
  input_arg {
    name: "lr_power"
    type_attr: "T"
  }
  output_arg {
    name: "out"
    type_attr: "T"
    is_ref: true
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT64
        type: DT_INT32
        type: DT_UINT8
        type: DT_UINT16
        type: DT_INT16
        type: DT_INT8
        type: DT_COMPLEX64
        type: DT_COMPLEX128
        type: DT_QINT8
        type: DT_QUINT8
        type: DT_QINT32
        type: DT_HALF
      }
    }
  }
  attr {
    name: "Tindices"
    type: "type"
    allowed_values {
      list {
        type: DT_INT32
        type: DT_INT64
      }
    }
  }
  attr {
    name: "use_locking"
    type: "bool"
    default_value {
      b: false
    }
  }
}
op {
  name: "SparseApplyMomentum"
  input_arg {
    name: "var"
    type_attr: "T"
    is_ref: true
  }
  input_arg {
    name: "accum"
    type_attr: "T"
    is_ref: true
  }
  input_arg {
    name: "lr"
    type_attr: "T"
  }
  input_arg {
    name: "grad"
    type_attr: "T"
  }
  input_arg {
    name: "indices"
    type_attr: "Tindices"
  }
  input_arg {
    name: "momentum"
    type_attr: "T"
  }
  output_arg {
    name: "out"
    type_attr: "T"
    is_ref: true
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT64
        type: DT_INT32
        type: DT_UINT8
        type: DT_UINT16
        type: DT_INT16
        type: DT_INT8
        type: DT_COMPLEX64
        type: DT_COMPLEX128
        type: DT_QINT8
        type: DT_QUINT8
        type: DT_QINT32
        type: DT_HALF
      }
    }
  }
  attr {
    name: "Tindices"
    type: "type"
    allowed_values {
      list {
        type: DT_INT32
        type: DT_INT64
      }
    }
  }
  attr {
    name: "use_locking"
    type: "bool"
    default_value {
      b: false
    }
  }
  attr {
    name: "use_nesterov"
    type: "bool"
    default_value {
      b: false
    }
  }
}
op {
  name: "SparseApplyProximalAdagrad"
  input_arg {
    name: "var"
    type_attr: "T"
    is_ref: true
  }
  input_arg {
    name: "accum"
    type_attr: "T"
    is_ref: true
  }
  input_arg {
    name: "lr"
    type_attr: "T"
  }
  input_arg {
    name: "l1"
    type_attr: "T"
  }
  input_arg {
    name: "l2"
    type_attr: "T"
  }
  input_arg {
    name: "grad"
    type_attr: "T"
  }
  input_arg {
    name: "indices"
    type_attr: "Tindices"
  }
  output_arg {
    name: "out"
    type_attr: "T"
    is_ref: true
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT64
        type: DT_INT32
        type: DT_UINT8
        type: DT_UINT16
        type: DT_INT16
        type: DT_INT8
        type: DT_COMPLEX64
        type: DT_COMPLEX128
        type: DT_QINT8
        type: DT_QUINT8
        type: DT_QINT32
        type: DT_HALF
      }
    }
  }
  attr {
    name: "Tindices"
    type: "type"
    allowed_values {
      list {
        type: DT_INT32
        type: DT_INT64
      }
    }
  }
  attr {
    name: "use_locking"
    type: "bool"
    default_value {
      b: false
    }
  }
}
op {
  name: "SparseApplyProximalGradientDescent"
  input_arg {
    name: "var"
    type_attr: "T"
    is_ref: true
  }
  input_arg {
    name: "alpha"
    type_attr: "T"
  }
  input_arg {
    name: "l1"
    type_attr: "T"
  }
  input_arg {
    name: "l2"
    type_attr: "T"
  }
  input_arg {
    name: "grad"
    type_attr: "T"
  }
  input_arg {
    name: "indices"
    type_attr: "Tindices"
  }
  output_arg {
    name: "out"
    type_attr: "T"
    is_ref: true
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT64
        type: DT_INT32
        type: DT_UINT8
        type: DT_UINT16
        type: DT_INT16
        type: DT_INT8
        type: DT_COMPLEX64
        type: DT_COMPLEX128
        type: DT_QINT8
        type: DT_QUINT8
        type: DT_QINT32
        type: DT_HALF
      }
    }
  }
  attr {
    name: "Tindices"
    type: "type"
    allowed_values {
      list {
        type: DT_INT32
        type: DT_INT64
      }
    }
  }
  attr {
    name: "use_locking"
    type: "bool"
    default_value {
      b: false
    }
  }
}
op {
  name: "SparseApplyRMSProp"
  input_arg {
    name: "var"
    type_attr: "T"
    is_ref: true
  }
  input_arg {
    name: "ms"
    type_attr: "T"
    is_ref: true
  }
  input_arg {
    name: "mom"
    type_attr: "T"
    is_ref: true
  }
  input_arg {
    name: "lr"
    type_attr: "T"
  }
  input_arg {
    name: "rho"
    type_attr: "T"
  }
  input_arg {
    name: "momentum"
    type_attr: "T"
  }
  input_arg {
    name: "epsilon"
    type_attr: "T"
  }
  input_arg {
    name: "grad"
    type_attr: "T"
  }
  input_arg {
    name: "indices"
    type_attr: "Tindices"
  }
  output_arg {
    name: "out"
    type_attr: "T"
    is_ref: true
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT64
        type: DT_INT32
        type: DT_UINT8
        type: DT_UINT16
        type: DT_INT16
        type: DT_INT8
        type: DT_COMPLEX64
        type: DT_COMPLEX128
        type: DT_QINT8
        type: DT_QUINT8
        type: DT_QINT32
        type: DT_HALF
      }
    }
  }
  attr {
    name: "Tindices"
    type: "type"
    allowed_values {
      list {
        type: DT_INT32
        type: DT_INT64
      }
    }
  }
  attr {
    name: "use_locking"
    type: "bool"
    default_value {
      b: false
    }
  }
}
"""


_op_def_lib = _InitOpDefLibrary()
