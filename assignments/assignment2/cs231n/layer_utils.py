from .layers import *
from .fast_layers import *


def affine_relu_forward(x, w, b):
    """Convenience layer that performs an affine transform followed by a ReLU.

    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a, fc_cache = affine_forward(x, w, b)
    out, relu_cache = relu_forward(a)
    cache = (fc_cache, relu_cache)
    return out, cache

def affine_relu_backward(dout, cache):
    """Backward pass for the affine-relu convenience layer.
    """
    fc_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db

# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

pass

# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

def conv_relu_forward(x, w, b, conv_param):
    """A convenience layer that performs a convolution followed by a ReLU.

    Inputs:
    - x: Input to the convolutional layer
    - w, b, conv_param: Weights and parameters for the convolutional layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    out, relu_cache = relu_forward(a)
    cache = (conv_cache, relu_cache)
    return out, cache


def conv_relu_backward(dout, cache):
    """Backward pass for the conv-relu convenience layer.
    """
    conv_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = conv_backward_fast(da, conv_cache)
    return dx, dw, db


def conv_bn_relu_forward(x, w, b, gamma, beta, conv_param, bn_param):
    """Convenience layer that performs a convolution, a batch normalization, and a ReLU.

    Inputs:
    - x: Input to the convolutional layer
    - w, b, conv_param: Weights and parameters for the convolutional layer
    - pool_param: Parameters for the pooling layer
    - gamma, beta: Arrays of shape (D2,) and (D2,) giving scale and shift
      parameters for batch normalization.
    - bn_param: Dictionary of parameters for batch normalization.

    Returns a tuple of:
    - out: Output from the pooling layer
    - cache: Object to give to the backward pass
    """
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    an, bn_cache = spatial_batchnorm_forward(a, gamma, beta, bn_param)
    out, relu_cache = relu_forward(an)
    cache = (conv_cache, bn_cache, relu_cache)
    return out, cache


def conv_bn_relu_backward(dout, cache):
    """Backward pass for the conv-bn-relu convenience layer.
    """
    conv_cache, bn_cache, relu_cache = cache
    dan = relu_backward(dout, relu_cache)
    da, dgamma, dbeta = spatial_batchnorm_backward(dan, bn_cache)
    dx, dw, db = conv_backward_fast(da, conv_cache)
    return dx, dw, db, dgamma, dbeta


def conv_relu_pool_forward(x, w, b, conv_param, pool_param):
    """Convenience layer that performs a convolution, a ReLU, and a pool.

    Inputs:
    - x: Input to the convolutional layer
    - w, b, conv_param: Weights and parameters for the convolutional layer
    - pool_param: Parameters for the pooling layer

    Returns a tuple of:
    - out: Output from the pooling layer
    - cache: Object to give to the backward pass
    """
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    s, relu_cache = relu_forward(a)
    out, pool_cache = max_pool_forward_fast(s, pool_param)
    cache = (conv_cache, relu_cache, pool_cache)
    return out, cache


def conv_relu_pool_backward(dout, cache):
    """Backward pass for the conv-relu-pool convenience layer.
    """
    conv_cache, relu_cache, pool_cache = cache
    ds = max_pool_backward_fast(dout, pool_cache)
    da = relu_backward(ds, relu_cache)
    dx, dw, db = conv_backward_fast(da, conv_cache)
    return dx, dw, db

def affine_batchnorm_relu_forward(x,w,b,gamma,beta,bn_param):
    """
        Convenience layer that performs an affine , a batchnorm and a ReLu

        Inputs:
            - x: Input to the affine layer
            - w, b: Weights for the affine layer
            - gamma: Scale parameter of shape (D,)
            - beta: Shift paremeter of shape (D,)
            - bn_param: Dictionary with the following keys:
              - mode: 'train' or 'test'; required
              - eps: Constant for numeric stability
              - momentum: Constant for running mean / variance.
              - running_mean: Array of shape (D,) giving running mean of features
              - running_var Array of shape (D,) giving running variance of features
        Returns:
            - out: Output from the layer
            - cache: Object to give to the backward pass

    """

    fc_out , fc_cache = affine_forward(x,w,b)
    bn_out , bn_cache = batchnorm_forward(fc_out,gamma,beta,bn_param)
    relu_out , relu_cache = relu_forward(bn_out)

    cache = (fc_cache,bn_cache,relu_cache)

    return relu_out,cache


def affine_batchnorm_relu_backward(dout,cache):
    """
    Backward pass for the fc-bn-relu convenience layer.

    Inputs:
        - dout: Upstream derivative, of shape (N, M)
        - cache : cache

    Returns a tuple of:
        - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
        - dw: Gradient with respect to w, of shape (D, M)
        - db: Gradient with respect to b, of shape (M,)
        - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
        - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """

    fc_cache, bn_cache, relu_cache = cache
    drelu = relu_backward(dout,relu_cache)
    dbn, dgamma, dbeta = batchnorm_backward_alt(drelu,bn_cache)
    dx, dw, db = affine_backward(dbn,fc_cache)

    return dx, dw, db, dgamma, dbeta


def affine_layernorm_relu_forward(x,w,b,gamma,beta,bn_param):
    """
        Convenience layer that performs an affine , a layernorm and a ReLu

        Inputs:
            - x: Input to the affine layer
            - w, b: Weights for the affine layer
            - gamma: Scale parameter of shape (D,)
            - beta: Shift paremeter of shape (D,)
            - bn_param: Dictionary with the following keys:
              - mode: 'train' or 'test'; required
              - eps: Constant for numeric stability
              - momentum: Constant for running mean / variance.
              - running_mean: Array of shape (D,) giving running mean of features
              - running_var Array of shape (D,) giving running variance of features
        Returns:
            - out: Output from the layer
            - cache: Object to give to the backward pass

    """

    fc_out , fc_cache = affine_forward(x,w,b)
    fn_out , fn_cache = layernorm_forward(fc_out,gamma,beta,bn_param)
    relu_out , relu_cache = relu_forward(fn_out)

    cache = (fc_cache,fn_cache,relu_cache)

    return relu_out,cache


def affine_layernorm_relu_backward(dout,cache):
    """
    Backward pass for the fc-fn-relu convenience layer.

    Inputs:
        - dout: Upstream derivative, of shape (N, M)
        - cache : cache

    Returns a tuple of:
        - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
        - dw: Gradient with respect to w, of shape (D, M)
        - db: Gradient with respect to b, of shape (M,)
        - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
        - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """

    fc_cache, fn_cache, relu_cache = cache
    drelu = relu_backward(dout,relu_cache)
    dfn, dgamma, dbeta = layernorm_backward(drelu,fn_cache)
    dx, dw, db = affine_backward(dfn,fc_cache)

    return dx, dw, db, dgamma, dbeta