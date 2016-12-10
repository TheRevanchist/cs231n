import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *

class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=64, filter_size=3,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32, use_batchnorm=True):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    self.use_batchnorm = use_batchnorm
    self.bn_params = {}
    
    # initialize the weights for all layers
    C, H, W = input_dim
    
    # initialization for the convolutional layer
    F = num_filters
    filter_height = filter_size
    filter_width = filter_size
    stride_conv = 1
    pad = (filter_size - 1)/2
    Hc = 1 + (H + 2 * pad - filter_height)/stride_conv
    Wc = 1 + (W + 2 * pad - filter_width)/stride_conv
    
    #W1 = weight_scale * np.random.randn(F, C, filter_height, filter_width)
    W1 = np.random.randn(F, C, filter_height, filter_width) \
      * np.sqrt(2.0/(C * filter_height * filter_width))   
    b1 = np.zeros(F)
    
    # initialization for the pooling layer
    width_pool = 2
    height_pool = 2
    stride_pool = 2
    Hp = 1 + (Hc - height_pool)/stride_pool
    Wp = 1 + (Wc - width_pool)/stride_pool
    
    # initialization for the fully connected (ReLU) layer
    Hh = hidden_dim
    #W2 = weight_scale * np.random.randn(F * Hp * Wp, Hh)
    W2 = np.random.randn(F * Hp * Wp, Hh) * np.sqrt(2.0/(F * Hp * Wp)) 
    b2 = np.zeros(Hh)
    
    # initialization for the softmax layer
    Hc = num_classes
    #W3 = weight_scale * np.random.randn(Hh, Hc)
    W3 = np.random.randn(Hh, Hc) * np.sqrt(2.0/(Hh))
    b3 = np.zeros(Hc)
    
    # store the values into the dictionary
    self.params['W1'] = W1
    self.params['W2'] = W2
    self.params['W3'] = W3
    self.params['b1'] = b1
    self.params['b2'] = b2
    self.params['b3'] = b3
    
    if self.use_batchnorm:
      print 'We use batchnorm here'
      bn_param1 = {'mode': 'train',
                   'running_mean': np.zeros(F),
                   'running_var': np.zeros(F)}
      gamma1 = np.ones(F)
      beta1 = np.zeros(F)

      bn_param2 = {'mode': 'train',
                   'running_mean': np.zeros(Hh),
                   'running_var': np.zeros(Hh)}
      gamma2 = np.ones(Hh)
      beta2 = np.zeros(Hh)

      self.bn_params.update({'bn_param1': bn_param1,
                             'bn_param2': bn_param2})

      self.params.update({'beta1': beta1,
                          'beta2': beta2,
                          'gamma1': gamma1,
                          'gamma2': gamma2})

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)

 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    
    X = X.astype(self.dtype)
    mode = 'test' if y is None else 'train'

    if self.use_batchnorm:
      for key, bn_param in self.bn_params.iteritems():
        bn_param[mode] = mode

    N = X.shape[0] 
    
    # unpack the weight and the biases
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']

    if self.use_batchnorm:
      bn_param1, gamma1, beta1 = self.bn_params[
        'bn_param1'], self.params['gamma1'], self.params['beta1']
      bn_param2, gamma2, beta2 = self.bn_params[
        'bn_param2'], self.params['gamma2'], self.params['beta2']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    
    # forward stage into the conv layer
    if self.use_batchnorm:
      conv_layer, cache_conv_layer = conv_norm_relu_pool_forward(
        X, W1, b1, conv_param, pool_param, gamma1, beta1, bn_param1)
    else:
      conv_layer, cache_conv_layer = conv_relu_pool_forward(X, W1, b1, \
        conv_param, pool_param)
    
    # forward stage for the fully connected layer
    N, F, Hp, Wp = conv_layer.shape
    conv_layer_reshaped = conv_layer.reshape((N, F * Hp * Wp)) 
    
    if self.use_batchnorm:
      hidden_layer, cache_hidden_layer = affine_norm_relu_forward(conv_layer_reshaped, \
        W2, b2, gamma2, beta2, bn_param2)
    else:
      hidden_layer, cache_hidden_layer = affine_relu_forward(conv_layer_reshaped, \
        W2, b2)
        
    # forward stage for the output (softmax) layer
    scores, cache_scores = affine_forward(hidden_layer, W3, b3)    
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    
    # compute the cost (loss function)
    data_loss, dscores = softmax_loss(scores, y)
    reg_loss = 0.5 * self.reg * np.sum(W1 ** 2)
    reg_loss += 0.5 * self.reg * np.sum(W2 ** 2)
    reg_loss += 0.5 * self.reg * np.sum(W3 ** 2)
    loss = data_loss + reg_loss
    
    # back-prop into the output layer
    dX3, dW3, db3 = affine_backward(dscores, cache_scores)
    dW3 += self.reg * W3
    
    # back-prop into the fully connected layer
    if self.use_batchnorm:
      dX2, dW2, db2, dgamma2, dbeta2 = affine_norm_relu_backward(dX3, cache_hidden_layer)
    else:
      dX2, dW2, db2 = affine_relu_backward(dX3, cache_hidden_layer)
    dW2 += self.reg * W2  
      
    # back-prop into the convolutional layer
    dX2 = dX2.reshape(N, F, Hp, Wp)  
    if self.use_batchnorm:
      dX1, dW1, db1, dgamma1, dbeta1 = conv_norm_relu_pool_backward(
                dX2, cache_conv_layer)
    else:
      dX1, dW1, db1 = conv_relu_pool_backward(dX2, cache_conv_layer)  
    dW1 += self.reg * W1          

    grads.update({'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2, 'W3': dW3, 'b3': db3})

    if self.use_batchnorm:
      grads.update({'beta1': dbeta1,
                    'beta2': dbeta2,
                    'gamma1': dgamma1,
                    'gamma2': dgamma2})  
                    
    return loss, grads
