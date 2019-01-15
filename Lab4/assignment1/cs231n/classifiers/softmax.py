import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass
  
  N = X.shape[0]
  C = W.shape[1]
  
  for i in range(N):
      # Loss:
      f = np.dot(X[i], W)
      f -= np.max(f) # [1, C]
      exps = np.exp(f) # [1, C]
      exps_sum = np.sum(exps)
      p = exps / exps_sum # [1, C]
      loss += -np.log(p[y[i]])

      # Gradient:
      for j in range(C):
          dW[:, j] += (p[j] - (j == y[i])) * X[i]
         
  # Regularization
  loss = loss / N + 0.5 * reg * np.sum(W * W)
  dW = dW / N + reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass

  N = X.shape[0]

  # Loss
  f = X @ W # [N, C] -> (np.dot(X, W))
  f -= np.max(f, axis=1, keepdims=True) # [N, C] -> normalized for a numerically stable softmax function -> see 'https://deepnotes.io/softmax-crossentropy'
  exps = np.exp(f) # [N, C]
  exps_sum = np.sum(exps, axis=1, keepdims=True) # [N, 1]
  p = exps / exps_sum # [N, C]
  loss = -np.sum(np.log(p[np.arange(N), y]))
  
  # Gradient
  tmp = np.zeros(p.shape) # [N, C] of 0s
  tmp[np.arange(N), y] = 1 # only 1s at class index (col)
  dW = X.T @ (p - tmp)
  
  # Regularization
  loss = loss / N + 0.5 * reg * np.sum(W * W)
  dW = dW / N + reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

