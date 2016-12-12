import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *

class DeeperCNN2(object):
  """ A more deeper CNN, with the following architecture
  conv - relu - max pool - conv - relu - max-pool - affine - relu - 
  affine - softmax/SVM
  
  The network operates on minibatches of data that have shape (N, C, H, W),
  consisting of N images, each with height H and width W and with C input 
  channels """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=[32, 64, 64, 128],
               filter_size=3, hidden_dim=100, num_classes=10, 
               weight_scale=1e-3, reg=0.0, dtype=np.float32, use_batchnorm=True):
                 
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: tuple (C, H, W) giving size of input data
    - num_filters: list containing the number of filters used for each
                   convolutional layer
    - filter_size: the size of the filters
    - hidden_dim: number of neurons used in the fully-connected layer
    - num_classes: number of scores to produce from the final affine layer.
    - weight_scale: scalar giving standard deviation for random initialization
                    of weights 
    - reg: scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation 
    """

    self.params = {}
    self.reg = reg
    self.dtype = dtype
    self.use_batchnorm = use_batchnorm
    self.bn_params = {} 

    # initialize the weights for all layers

    # compute all the information needed in order to initialize the weights 
    # of the first layer
    C, H, W = input_dim
    filter_height1, filter_width1 = filter_size, filter_size
    stride_conv1 = 1
    pad1 = (filter_height1 - 1)/2
    F1 = num_filters[0] 
    Hc1 = 1 + (H - filter_height1 + 2 * pad1)/stride_conv1
    Wc1 = 1 + (W - filter_width1 + 2 * pad1)/stride_conv1 

    # initialize the weights and biases in the first layer
    W1 = np.random.randn(F1, C, filter_height1, filter_width1) \
      * np.sqrt(2.0/(C * filter_height1 * filter_width1 * F1)) 
    b1 = np.zeros(F1)    

    # compute the values for the first pooling layer
    width_pool1, height_pool1, stride_pool1 = 2, 2, 2
    Hp1 = 1 + (Hc1 - height_pool1) / stride_pool1
    Wp1 = 1 + (Wc1 - width_pool1) / stride_pool1

    # compute the information needed to initialize the weights of the second layer
    filter_height2, filter_width2 = filter_size, filter_size
    stride_conv2 = 1
    pad2 = (filter_height2 - 1) / 2
    F2 = num_filters[1]
    Hc2 = 1 + (Hp1 - filter_height2 + 2 * pad2) / stride_conv2       
    Wc2 = 1 + (Wp1 - filter_width2 + 2 * pad2) / stride_conv2 

    # initialize the weights and biases in the second layer
    W2 = np.random.randn(F2, F1, filter_height2, filter_width2) \
      * np.sqrt(2.0/(F1 * filter_height2 * filter_width2 * F2))     
    b2 = np.zeros(F2)

    # compute the values for the second pooling layer
    width_pool2, height_pool2, stride_pool2 = 2, 2, 2
    Hp2 = 1 + (Hc2 - height_pool2) / stride_pool2 
    Wp2 = 1 + (Wc2 - width_pool2) / stride_pool2

    # compute the information needed to initialize the weights of the third layer
    filter_height3, filter_width3 = filter_size, filter_size
    stride_conv3 = 1
    pad3 = (filter_height3 - 1) / 2
    F3 = num_filters[2]
    Hc3 = 1 + (Hp2 - filter_height3 + 2 * pad3) / stride_conv3      
    Wc3 = 1 + (Wp2 - filter_width3 + 2 * pad3) / stride_conv3 

    # initialize the weights and biases in the third layer
    W3 = np.random.randn(F3, F2, filter_height3, filter_width3) \
      * np.sqrt(2.0/(F2 * filter_height3 * filter_width3 * F3))     
    b3 = np.zeros(F3)    

    # compute the values for the third pooling layer
    width_pool3, height_pool3, stride_pool3 = 2, 2, 2
    Hp3 = 1 + (Hc3 - height_pool3) / stride_pool3 
    Wp3 = 1 + (Wc3 - width_pool3) / stride_pool3
    
    # compute the information needed to initialize the weights of the fourth layer
    filter_height4, filter_width4 = filter_size, filter_size
    stride_conv4 = 1
    pad4 = (filter_height4 - 1) / 2
    F4 = num_filters[3]
    Hc4 = 1 + (Hp3 - filter_height4 + 2 * pad4) / stride_conv4      
    Wc4 = 1 + (Wp3 - filter_width4 + 2 * pad4) / stride_conv4
    
    # initialize the weights and biases in the fourth layer
    W4 = np.random.randn(F4, F3, filter_height4, filter_width4) \
      * np.sqrt(2.0/(F3 * filter_height4 * filter_width4 * F4))     
    b4 = np.zeros(F4)    

    # initialization for the fully connected (ReLU) layer
    Hh = hidden_dim
    W5 = np.random.randn(F4 * Hc4 * Wc4, Hh) * np.sqrt(2.0/(F4 * Hc4 * Wc4 * Hh))
    b5 = np.zeros(Hh) 

    # initialization for the output layer
    Hc = num_classes
    W6 = np.random.randn(Hh, Hc) * np.sqrt(2.0/(Hh * Hc))
    b6 = np.zeros(Hc) 
    
    # store some values as fields (in order to synchronize better with the
    # loss function)
    self.filter_size_conv1 = filter_size
    self.filter_size_pool1 = height_pool1
    self.stride_conv1 = stride_conv1
    self.stride_pool1 = stride_pool1
    self.pad1 = pad1
    
    self.filter_size_conv2 = filter_size
    self.filter_size_pool2 = height_pool2
    self.stride_conv2 = stride_conv2
    self.stride_pool2 = stride_pool2
    self.pad2 = pad2
    
    self.filter_size_conv3 = filter_size
    self.filter_size_pool3 = height_pool3
    self.stride_conv3 = stride_conv3
    self.stride_pool3 = stride_pool3
    self.pad3 = pad3 
    
    self.filter_size_conv4 = filter_size
    self.stride_conv4 = stride_conv4
    self.pad4 = pad4

    # store the weights/biases into the dictionary
    self.params.update({'W1': W1, 'b1': b1,
                        'W2': W2, 'b2': b2,
                        'W3': W3, 'b3': b3,
                        'W4': W4, 'b4': b4,
                        'W5': W5, 'b5': b5,
                        'W6': W6, 'b6': b6})  
                        
    # store the parameters for the batchnorm into an another dictionary
    if self.use_batchnorm:
      print 'We use batchnorm here'
      
      bn_param1 = {'mode': 'train',
                   'running_mean': np.zeros(F1),
                   'running_var': np.zeros(F1)}
      gamma1 = np.ones(F1)
      beta1 = np.zeros(F1)
      
      bn_param2 = {'mode': 'train',
                   'running_mean': np.zeros(F2),
                   'running_var': np.zeros(F2)}
      gamma2 = np.ones(F2)
      beta2 = np.zeros(F2)      

      bn_param3 = {'mode': 'train',
                   'running_mean': np.zeros(F3),
                   'running_var': np.zeros(F3)}
      gamma3 = np.ones(F3)
      beta3 = np.zeros(F3)

      bn_param4 = {'mode': 'train',
                   'running_mean': np.zeros(F4),
                   'running_var': np.zeros(F4)}
      gamma4 = np.ones(F4)
      beta4 = np.zeros(F4)

      bn_param5 = {'mode': 'train',
                   'running_mean': np.zeros(Hh),
                   'running_var': np.zeros(Hh)}
      gamma5 = np.ones(Hh)
      beta5 = np.zeros(Hh)

      self.bn_params.update({'bn_param1': bn_param1,
                             'bn_param2': bn_param2,
                             'bn_param3': bn_param3,
                             'bn_param4': bn_param4,
                             'bn_param5': bn_param5})
      
      self.params.update({'beta1': beta1, 'gamma1': gamma1,
                          'beta2': beta2, 'gamma2': gamma2,
                          'beta3': beta3, 'gamma3': gamma3,
                          'beta4': beta4, 'gamma4': gamma4,
                          'beta5': beta5, 'gamma5': gamma5})

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
    
    # unpack the weight and the biases
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    W4, b4 = self.params['W4'], self.params['b4']
    W5, b5 = self.params['W5'], self.params['b5']
    W6, b6 = self.params['W6'], self.params['b6']

    if self.use_batchnorm:
      bn_param1, gamma1, beta1 = self.bn_params[
        'bn_param1'], self.params['gamma1'], self.params['beta1']
      bn_param2, gamma2, beta2 = self.bn_params[
        'bn_param2'], self.params['gamma2'], self.params['beta2']
      bn_param3, gamma3, beta3 = self.bn_params[
        'bn_param3'], self.params['gamma3'], self.params['beta3']  
      bn_param4, gamma4, beta4 = self.bn_params[
        'bn_param4'], self.params['gamma4'], self.params['beta4'] 
      bn_param5, gamma5, beta5 = self.bn_params[
        'bn_param5'], self.params['gamma5'], self.params['beta5']       

    # pass conv_param to the forward pass for the convolutional layers
    conv_param1 = {'stride': self.stride_conv1, 'pad': self.pad1}
    conv_param2 = {'stride': self.stride_conv2, 'pad': self.pad2}
    conv_param3 = {'stride': self.stride_conv3, 'pad': self.pad3}
    conv_param4 = {'stride': self.stride_conv4, 'pad': self.pad4}
    
    # pass pool_param to the forward pass for the max-pooling layers
    pool_param1 = {'pool_height': self.filter_size_pool1, \
      'pool_width': self.filter_size_pool1, 'stride': self.stride_pool1}  
    pool_param2 = {'pool_height': self.filter_size_pool2, \
      'pool_width': self.filter_size_pool2, 'stride': self.stride_pool2}
    pool_param3 = {'pool_height': self.filter_size_pool3, \
      'pool_width': self.filter_size_pool3, 'stride': self.stride_pool3}  
      
    scores = None # the variable which will contain scores for each class
    
    # forward stage into the first conv layer
    if self.use_batchnorm:
      conv_layer1, cache_conv_layer1 = conv_norm_relu_pool_forward(
        X, W1, b1, conv_param1, pool_param1, gamma1, beta1, bn_param1)
    else:
      conv_layer1, cache_conv_layer1 = conv_relu_pool_forward(X, W1, b1, \
        conv_param1, pool_param1)
        
    # forward stage into the second conv layer
    if self.use_batchnorm:
      conv_layer2, cache_conv_layer2 = conv_norm_relu_pool_forward(
        conv_layer1, W2, b2, conv_param2, pool_param2, gamma2, beta2, bn_param2)
    else:
      conv_layer2, cache_conv_layer2 = conv_relu_pool_forward(
        conv_layer1, W2, b2, conv_param2, pool_param2)

    # forward stage into the third conv layer
    if self.use_batchnorm:
      conv_layer3, cache_conv_layer3 = conv_norm_relu_pool_forward(
        conv_layer2, W3, b3, conv_param3, pool_param3, gamma3, beta3, bn_param3)
    else:
      conv_layer3, cache_conv_layer3 = conv_relu_pool_forward(
        conv_layer2, W3, b3, conv_param3, pool_param3)
        
    # forward stage into the fourth conv layer
    if self.use_batchnorm:
      conv_layer4, cache_conv_layer4 = conv_norm_relu_forward(
        conv_layer3, W4, b4, conv_param4, gamma4, beta4, bn_param4)
    else:
      conv_layer4, cache_conv_layer4 = conv_relu_forward(
        conv_layer3, W4, b4, conv_param4)        

    # forward stage for the fully connected layer
    N, F, Hp, Wp = conv_layer4.shape
    conv_layer_reshaped = conv_layer4.reshape((N, F * Hp * Wp))     

    if self.use_batchnorm:
      hidden_layer, cache_hidden_layer = affine_norm_relu_forward(
        conv_layer_reshaped, W5, b5, gamma5, beta5, bn_param5)
    else:
      hidden_layer, cache_hidden_layer = affine_relu_forward(conv_layer_reshaped, \
        W5, b5)    

    # forward stage for the output (softmax) layer
    scores, cache_scores = affine_forward(hidden_layer, W6, b6)    
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    
    # compute the cost (loss function)
    data_loss, dscores = softmax_loss(scores, y)
    reg_loss = 0.5 * self.reg * np.sum(W1 ** 2)
    reg_loss += 0.5 * self.reg * np.sum(W2 ** 2)
    reg_loss += 0.5 * self.reg * np.sum(W3 ** 2)
    reg_loss += 0.5 * self.reg * np.sum(W4 ** 2)
    reg_loss += 0.5 * self.reg * np.sum(W5 ** 2)
    reg_loss += 0.5 * self.reg * np.sum(W6 ** 2)
    loss = data_loss + reg_loss    

    # back-prop into the output layer
    dX6, dW6, db6 = affine_backward(dscores, cache_scores)
    dW6 += self.reg * W6

    # back-prop into the fully connected layer
    if self.use_batchnorm:
      dX5, dW5, db5, dgamma5, dbeta5 = affine_norm_relu_backward(
        dX6, cache_hidden_layer)
    else:
      dX5, dW5, db5 = affine_relu_backward(dX6, cache_hidden_layer)
    dW5 += self.reg * W5 
    
    # back-prop into the fourth convolutional layer
    dX5 = dX5.reshape(N, F, Hp, Wp)  
    if self.use_batchnorm:
      dX4, dW4, db4, dgamma4, dbeta4 = conv_norm_relu_backward(
                dX5, cache_conv_layer4)
    else:
      dX4, dW4, db4 = conv_relu_backward(dX5, cache_conv_layer4)  
    dW4 += self.reg * W4    

    # back-prop into the third convolutional layer 
    if self.use_batchnorm:
      dX3, dW3, db3, dgamma3, dbeta3 = conv_norm_relu_pool_backward(
                dX4, cache_conv_layer3)
    else:
      dX3, dW3, db3 = conv_relu_pool_backward(dX4, cache_conv_layer3)  
    dW3 += self.reg * W3

    # back-prop into the second convolutional layer 
    if self.use_batchnorm:
      dX2, dW2, db2, dgamma2, dbeta2 = conv_norm_relu_pool_backward(
                dX3, cache_conv_layer2)
    else:
      dX2, dW2, db2 = conv_relu_pool_backward(dX3, cache_conv_layer2)  
    dW2 += self.reg * W2 
    
    # back-prop into the first convolutional layer 
    if self.use_batchnorm:
      dX1, dW1, db1, dgamma1, dbeta1 = conv_norm_relu_pool_backward(
                dX2, cache_conv_layer1)
    else:
      dX1, dW1, db1 = conv_relu_pool_backward(dX2, cache_conv_layer1)  
    dW1 += self.reg * W1 

    grads.update({'W1': dW1, 'b1': db1,
                  'W2': dW2, 'b2': db2,
                  'W3': dW3, 'b3': db3,
                  'W4': dW4, 'b4': db4,
                  'W5': dW5, 'b5': db5,
                  'W6': dW6, 'b6': db6})  
                  
    if self.use_batchnorm:
      grads.update({'beta1': dbeta1, 'gamma1': dgamma1,
                    'beta2': dbeta2, 'gamma2': dgamma2,
                    'beta3': dbeta3, 'gamma3': dgamma3,
                    'beta4': dbeta4, 'gamma4': dgamma4,
                    'beta5': dbeta5, 'gamma5': dgamma5})                  
    
    return loss, grads