import numpy as np


"""
This file defines layer types that are commonly used for recurrent neural
networks.
"""


def rnn_step_forward(x, prev_h, Wx, Wh, b):
  """
  Run the forward pass for a single timestep of a vanilla RNN that uses a tanh
  activation function.

  The input data has dimension D, the hidden state has dimension H, and we use
  a minibatch size of N.

  Inputs:
  - x: Input data for this timestep, of shape (N, D).
  - prev_h: Hidden state from previous timestep, of shape (N, H)
  - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
  - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
  - b: Biases of shape (H,)

  Returns a tuple of:
  - next_h: Next hidden state, of shape (N, H)
  - cache: Tuple of values needed for the backward pass.
  """
  next_h, cache = None, None
  
  h_t = np.dot(prev_h, Wh) + np.dot(x, Wx) + b
  next_h = np.tanh(h_t)
  cache = x, prev_h, Wx, Wh, h_t
  
  return next_h, cache


def rnn_step_backward(dnext_h, cache):
  """
  Backward pass for a single timestep of a vanilla RNN.
  
  Inputs:
  - dnext_h: Gradient of loss with respect to next hidden state
  - cache: Cache object from the forward pass
  
  Returns a tuple of:
  - dx: Gradients of input data, of shape (N, D)
  - dprev_h: Gradients of previous hidden state, of shape (N, H)
  - dWx: Gradients of input-to-hidden weights, of shape (N, H)
  - dWh: Gradients of hidden-to-hidden weights, of shape (H, H)
  - db: Gradients of bias vector, of shape (H,)
  """
  dx, dprev_h, dWx, dWh, db = None, None, None, None, None
  x, prev_h, Wx, Wh, h_t = cache
  
  # dtanh(h_t)/dh_t = dsinh(h_t)/dcosh(h_t) = (cosh^2(h_t) - sinh^2(h_t))/cosh^2(h_t)
  # = 1 - sinh^2(h_t)/cosh^2(h_t) = 1 - tanh^2(h_t)
  dh_t = (1 - np.tanh(h_t) ** 2) * dnext_h
  db = np.sum(dh_t, axis = 0)
  dprev_h = np.dot(Wh, dh_t.T).T 
  dWh = np.dot(prev_h.T, dh_t)
  dx = np.dot(dh_t, Wx.T)
  dWx = np.dot(x.T, dh_t)
  
  return dx, dprev_h, dWx, dWh, db


def rnn_forward(x, h0, Wx, Wh, b):
  """
  Run a vanilla RNN forward on an entire sequence of data. We assume an input
  sequence composed of T vectors, each of dimension D. The RNN uses a hidden
  size of H, and we work over a minibatch containing N sequences. After running
  the RNN forward, we return the hidden states for all timesteps.
  
  Inputs:
  - x: Input data for the entire timeseries, of shape (N, T, D).
  - h0: Initial hidden state, of shape (N, H)
  - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
  - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
  - b: Biases of shape (H,)
  
  Returns a tuple of:
  - h: Hidden states for the entire timeseries, of shape (N, T, H).
  - cache: Values needed in the backward pass
  """
  h, cache = None, None
  N, T, D = x.shape
  H = h0.shape[1]
  h = np.zeros((T, N, H))
  x = x.transpose(1, 0, 2)
  cache = []
  for i in xrange(T):
    if i == 0:
      prev_h = h0
    else:
      prev_h = h[i - 1]
    h[i], cache_i = rnn_step_forward(x[i], prev_h, Wx, Wh, b) 
    cache.append(cache_i)
  h = h.transpose(1, 0, 2)  
  return h, cache


def rnn_backward(dh, cache):
  """
  Compute the backward pass for a vanilla RNN over an entire sequence of data.
  
  Inputs:
  - dh: Upstream gradients of all hidden states, of shape (N, T, H)
  
  Returns a tuple of:
  - dx: Gradient of inputs, of shape (N, T, D)
  - dh0: Gradient of initial hidden state, of shape (N, H)
  - dWx: Gradient of input-to-hidden weights, of shape (D, H)
  - dWh: Gradient of hidden-to-hidden weights, of shape (H, H)
  - db: Gradient of biases, of shape (H,)
  """
  dx, dh0, dWx, dWh, db = None, None, None, None, None
  
  # initialize the derivatives to 0
  D = cache[0][0].shape[1]
  N, T, H = dh.shape
  dx = np.zeros((T, N, D))
  dh0 = np.zeros((N, H))
  dWx = np.zeros((D, H))
  dWh = np.zeros((H, H))
  db = np.zeros(H)
  dh = dh.transpose(1, 0, 2)
  dprev_h = np.zeros((N, H))
  
  for i in reversed(xrange(T)):
    dcurrent_h = dh[i] + dprev_h
    dx_i, dprev_h, dWx_i, dWh_i, db_i = rnn_step_backward(dcurrent_h, cache[i])
    dx[i] += dx_i
    dWh += dWh_i
    dWx += dWx_i
    db += db_i    
  dh0 = dprev_h
  
  dx = dx.transpose(1, 0, 2)  
  
  return dx, dh0, dWx, dWh, db


def word_embedding_forward(x, W):
  """
  Forward pass for word embeddings. We operate on minibatches of size N where
  each sequence has length T. We assume a vocabulary of V words, assigning each
  to a vector of dimension D.
  
  Inputs:
  - x: Integer array of shape (N, T) giving indices of words. Each element idx
    of x muxt be in the range 0 <= idx < V.
  - W: Weight matrix of shape (V, D) giving word vectors for all words.
  
  Returns a tuple of:
  - out: Array of shape (N, T, D) giving word vectors for all input words.
  - cache: Values needed for the backward pass
  """
  out, cache = None, None
  out = W[x, :]
  cache = x, W
  return out, cache


def word_embedding_backward(dout, cache):
  """
  Backward pass for word embeddings. We cannot back-propagate into the words
  since they are integers, so we only return gradient for the word embedding
  matrix.
  
  HINT: Look up the function np.add.at
  
  Inputs:
  - dout: Upstream gradients of shape (N, T, D)
  - cache: Values from the forward pass
  
  Returns:
  - dW: Gradient of word embedding matrix, of shape (V, D).
  """
  dW = None

  x, W = cache
  dW = np.zeros_like(W)
  np.add.at(dW, x, dout)

  return dW


def sigmoid(x):
  """
  A numerically stable version of the logistic sigmoid function.
  """
  pos_mask = (x >= 0)
  neg_mask = (x < 0)
  z = np.zeros_like(x)
  z[pos_mask] = np.exp(-x[pos_mask])
  z[neg_mask] = np.exp(x[neg_mask])
  top = np.ones_like(x)
  top[neg_mask] = z[neg_mask]
  return top / (1 + z)


def lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b):
  """
  Forward pass for a single timestep of an LSTM.
  
  The input data has dimension D, the hidden state has dimension H, and we use
  a minibatch size of N.
  
  Inputs:
  - x: Input data, of shape (N, D)
  - prev_h: Previous hidden state, of shape (N, H)
  - prev_c: previous cell state, of shape (N, H)
  - Wx: Input-to-hidden weights, of shape (D, 4H)
  - Wh: Hidden-to-hidden weights, of shape (H, 4H)
  - b: Biases, of shape (4H,)
  
  Returns a tuple of:
  - next_h: Next hidden state, of shape (N, H)
  - next_c: Next cell state, of shape (N, H)
  - cache: Tuple of values needed for backward pass.
  """
  next_h, next_c, cache = None, None, None

  N, H = prev_h.shape
  # 1
  a = np.dot(x, Wx) + np.dot(prev_h, Wh) + b
  
  # 2
  ai = a[:, :H]
  af = a[:, H:2*H]
  ao = a[:, 2*H:3*H]
  ag = a[:, 3*H:]
  
  # 3
  i = sigmoid(ai)
  f = sigmoid(af)
  o = sigmoid(ao)
  g = np.tanh(ag)
  
  # 4
  next_c = f * prev_c + i * g
  
  # 5
  next_h = o * np.tanh(next_c)
  
  cache = i, f, o, g, a, ai, af, ao, ag, Wx, Wh, b, prev_h, prev_c, x, next_c, next_h
  
  return next_h, next_c, cache


def lstm_step_backward(dnext_h, dnext_c, cache):
  """
  Backward pass for a single timestep of an LSTM.
  
  Inputs:
  - dnext_h: Gradients of next hidden state, of shape (N, H)
  - dnext_c: Gradients of next cell state, of shape (N, H)
  - cache: Values from the forward pass
  
  Returns a tuple of:
  - dx: Gradient of input data, of shape (N, D)
  - dprev_h: Gradient of previous hidden state, of shape (N, H)
  - dprev_c: Gradient of previous cell state, of shape (N, H)
  - dWx: Gradient of input-to-hidden weights, of shape (D, 4H)
  - dWh: Gradient of hidden-to-hidden weights, of shape (H, 4H)
  - db: Gradient of biases, of shape (4H,)
  """
  dx, dh, dc, dWx, dWh, db = None, None, None, None, None, None

  i, f, o, g, a, ai, af, ao, ag, Wx, Wh, b, prev_h, prev_c, x, next_c, next_h = cache
  
  # backprop into step 5
  do = np.tanh(next_c) * dnext_h
  dnext_c += o * (1 - np.tanh(next_c) ** 2) * dnext_h 
  
  # backprop into 4
  df = prev_c * dnext_c
  dprev_c = f * dnext_c
  di = g * dnext_c
  dg = i * dnext_c
  
  # backprop into 3
  dai = sigmoid(ai) * (1 - sigmoid(ai)) * di
  daf = sigmoid(af) * (1 - sigmoid(af)) * df
  dao = sigmoid(ao) * (1 - sigmoid(ao)) * do
  dag = (1 - np.tanh(ag) ** 2) * dg
  
  # backprop into 2
  da = np.hstack((dai, daf, dao, dag))
  
  # backprop into 1
  db = np.sum(da, axis = 0)
  dprev_h = np.dot(Wh, da.T).T 
  dWh = np.dot(prev_h.T, da)
  dx = np.dot(da, Wx.T)
  dWx = np.dot(x.T, da)

  return dx, dprev_h, dprev_c, dWx, dWh, db


def lstm_forward(x, h0, Wx, Wh, b):
  """
  Forward pass for an LSTM over an entire sequence of data. We assume an input
  sequence composed of T vectors, each of dimension D. The LSTM uses a hidden
  size of H, and we work over a minibatch containing N sequences. After running
  the LSTM forward, we return the hidden states for all timesteps.
  
  Note that the initial cell state is passed as input, but the initial cell
  state is set to zero. Also note that the cell state is not returned; it is
  an internal variable to the LSTM and is not accessed from outside.
  
  Inputs:
  - x: Input data of shape (N, T, D)
  - h0: Initial hidden state of shape (N, H)
  - Wx: Weights for input-to-hidden connections, of shape (D, 4H)
  - Wh: Weights for hidden-to-hidden connections, of shape (H, 4H)
  - b: Biases of shape (4H,)
  
  Returns a tuple of:
  - h: Hidden states for all timesteps of all sequences, of shape (N, T, H)
  - cache: Values needed for the backward pass.
  """
  h, cache = None, None
  N, T, D = x.shape
  N, H = h0.shape
  prev_h = h0
  prev_c = np.zeros_like(prev_h)
  h = np.zeros((T, N, H))
  x = x.transpose(1, 0, 2)  
  cache = []
  
  h[0], next_c, cache_next = lstm_step_forward(x[0], prev_h, prev_c, Wx, Wh, b)
  cache.append(cache_next)
  for i in xrange(1, T):
    prev_h = h[i - 1]
    prev_c = next_c
    h[i], next_c, cache_next = lstm_step_forward(x[i], prev_h, prev_c, Wx, Wh, b)
    cache.append(cache_next)
  h = h.transpose(1, 0, 2) 
  
  return h, cache


def lstm_backward(dh, cache):
  """
  Backward pass for an LSTM over an entire sequence of data.]
  
  Inputs:
  - dh: Upstream gradients of hidden states, of shape (N, T, H)
  - cache: Values from the forward pass
  
  Returns a tuple of:
  - dx: Gradient of input data of shape (N, T, D)
  - dh0: Gradient of initial hidden state of shape (N, H)
  - dWx: Gradient of input-to-hidden weight matrix of shape (D, 4H)
  - dWh: Gradient of hidden-to-hidden weight matrix of shape (H, 4H)
  - db: Gradient of biases, of shape (4H,)
  """
  N, T, H = dh.shape
  i, f, o, g, a, ai, af, ao, ag, Wx, Wh, b, prev_h, prev_c, x, next_c, next_h = cache[
        0]
  D = x.shape[-1]

    # Initialize dx,dh0,dWx,dWh,db
  dx = np.zeros((T, N, D))
  dh0 = np.zeros((N, H))
  db = np.zeros((4 * H,))
  dWh = np.zeros((H, 4 * H))
  dWx = np.zeros((D, 4 * H))

    # On transpose dh
  dh = dh.transpose(1, 0, 2)
  dh_prev = np.zeros((N, H))
  dc_prev = np.zeros_like(dh_prev)

  for t in reversed(xrange(T)):
    dh_current = dh[t] + dh_prev
    dc_current = dc_prev
    dx_t, dh_prev, dc_prev, dWx_t, dWh_t, db_t = lstm_step_backward(
      dh_current, dc_current, cache[t])
    dx[t] += dx_t
    dh0 = dh_prev
    dWx += dWx_t
    dWh += dWh_t
    db += db_t

  dx = dx.transpose(1, 0, 2)
  return dx, dh0, dWx, dWh, db
  


def temporal_affine_forward(x, w, b):
  """
  Forward pass for a temporal affine layer. The input is a set of D-dimensional
  vectors arranged into a minibatch of N timeseries, each of length T. We use
  an affine function to transform each of those vectors into a new vector of
  dimension M.

  Inputs:
  - x: Input data of shape (N, T, D)
  - w: Weights of shape (D, M)
  - b: Biases of shape (M,)
  
  Returns a tuple of:
  - out: Output data of shape (N, T, M)
  - cache: Values needed for the backward pass
  """
  N, T, D = x.shape
  M = b.shape[0]
  out = x.reshape(N * T, D).dot(w).reshape(N, T, M) + b
  cache = x, w, b, out
  return out, cache


def temporal_affine_backward(dout, cache):
  """
  Backward pass for temporal affine layer.

  Input:
  - dout: Upstream gradients of shape (N, T, M)
  - cache: Values from forward pass

  Returns a tuple of:
  - dx: Gradient of input, of shape (N, T, D)
  - dw: Gradient of weights, of shape (D, M)
  - db: Gradient of biases, of shape (M,)
  """
  x, w, b, out = cache
  N, T, D = x.shape
  M = b.shape[0]

  dx = dout.reshape(N * T, M).dot(w.T).reshape(N, T, D)
  dw = dout.reshape(N * T, M).T.dot(x.reshape(N * T, D)).T
  db = dout.sum(axis=(0, 1))

  return dx, dw, db


def temporal_softmax_loss(x, y, mask, verbose=False):
  """
  A temporal version of softmax loss for use in RNNs. We assume that we are
  making predictions over a vocabulary of size V for each timestep of a
  timeseries of length T, over a minibatch of size N. The input x gives scores
  for all vocabulary elements at all timesteps, and y gives the indices of the
  ground-truth element at each timestep. We use a cross-entropy loss at each
  timestep, summing the loss over all timesteps and averaging across the
  minibatch.

  As an additional complication, we may want to ignore the model output at some
  timesteps, since sequences of different length may have been combined into a
  minibatch and padded with NULL tokens. The optional mask argument tells us
  which elements should contribute to the loss.

  Inputs:
  - x: Input scores, of shape (N, T, V)
  - y: Ground-truth indices, of shape (N, T) where each element is in the range
       0 <= y[i, t] < V
  - mask: Boolean array of shape (N, T) where mask[i, t] tells whether or not
    the scores at x[i, t] should contribute to the loss.

  Returns a tuple of:
  - loss: Scalar giving loss
  - dx: Gradient of loss with respect to scores x.
  """

  N, T, V = x.shape
  
  x_flat = x.reshape(N * T, V)
  y_flat = y.reshape(N * T)
  mask_flat = mask.reshape(N * T)
  
  probs = np.exp(x_flat - np.max(x_flat, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  loss = -np.sum(mask_flat * np.log(probs[np.arange(N * T), y_flat])) / N
  dx_flat = probs.copy()
  dx_flat[np.arange(N * T), y_flat] -= 1
  dx_flat /= N
  dx_flat *= mask_flat[:, None]
  
  if verbose: print 'dx_flat: ', dx_flat.shape
  
  dx = dx_flat.reshape(N, T, V)
  
  return loss, dx

