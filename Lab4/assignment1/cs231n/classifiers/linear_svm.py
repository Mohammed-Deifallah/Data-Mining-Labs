import numpy as np
from random import shuffle


def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

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
    
    #Ref: https://cs231n.github.io/linear-classify/#svm
    #Ref: https://math.stackexchange.com/questions/2572318/derivation-of-gradient-of-svm-loss
    
    #L = sum(max(0, score[j] - correct_score + Delta))
    #derivative=I(something−wy∗x>0)∗(−x)
    dW = np.zeros(W.shape) # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        #W_T.X
        scores = X[i].dot(W)
        #The score of the correct class
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            #Compute for only misclassifications
            if j == y[i]:
                continue
            #L = sum(max(0, score[j] - correct_score + Delta))
            margin = scores[j] - correct_class_score + 1 # note delta = 1
            if margin > 0:  #Indicator
                loss += margin
                #∇wLi=[dL_i.dw_1, dL_i.dw_2 ⋯ dL_i.dw_C]
                dW[:, j] += X[i, :]
                dW[:, y[i]] -= X[i, :]

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train

    # Add regularization to the loss.
    # To remove ambiguity over uniqueness of W parameters
    loss += reg * np.sum(W * W)
    dW += reg * W

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    #Ref: https://mlxai.github.io/2017/01/06/vectorized-implementation-of-svm-loss-and-gradient-update.html
    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero
    num_train = X.shape[0]

    scores = X.dot(W)
    yi_scores = scores[range(num_train),y]
    margins = np.maximum(0, scores - np.matrix(yi_scores).T + 1) #Formula
    margins[np.arange(num_train),y] = 0 #Correctly classified points have 0 margin
    loss = np.mean(np.sum(margins, axis=1))
    loss += reg * np.sum(W * W)

    margins[margins > 0] = 1 #Misclassified
    row_sum = np.sum(margins, axis=1)
    margins[np.arange(num_train), y] = -row_sum.T
    dW = np.dot(X.T, margins)

    dW /= num_train

    dW += reg * W
    return loss, dW
