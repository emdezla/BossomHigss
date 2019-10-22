# -*- coding: utf-8 -*-
"""Function used in the project 1."""
import numpy as np

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """
    Gradient descent algorithm that returns the weights that minimilize 
    the mean square error (mse) over a given number of iterations max_iters, 
    from a starting point initial_w.
    Also returns the corresponding mse.
    """
    w = initial_w
    
    for n_iter in range(max_iters):
        grad=compute_gradient(y,tx,w)
        w = w-gamma*grad
    loss=compute_mse(y, tx, w)
    
    return w, loss

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Stochastic version of the gradient descent algorithm for mse."""
    w = initial_w
    
    for n_iter in range(max_iters):
        grad_iter=compute_stoch_gradient(y,tx,w)
        w = w-gamma*grad_iter
    loss=compute_mse(y, tx, w)
    
    return w, loss

def least_squares(y, tx):
    """calculate the least squares solution."""
    
    N=y.shape[0]
    gram=tx.T.dot(tx)
    w = np.linalg.solve(gram, tx.T.dot(y))
    loss=1/(2*N)*(y-tx.dot(w)).dot(y-tx.dot(w))
    
    return w, loss

def ridge_regression(y, tx, lambda_):
    """
    Returns the weights and the mse obtained by the ridge regression method. 
    Mse does not include the lambda term.
    """
    D=tx.shape[1]
    N=tx.shape[0]
    A=tx.T.dot(tx)+lambda_*(2*N)*np.eye(D)
    w=np.linalg.solve(A,tx.T.dot(y))
    loss=compute_mse(y, tx, w) #Loss without the regularizer.
    
    return w, loss

def logistic_regression(y, tx, initial_w, max_iter, gamma):
    """
        Returns the weights that minimalize the negative log-likelihood function, 
        and the corresponding negative log-likelihood loss. This is done using a
        gradient descent algorithm.
    """
    w = initial_w
    D=tx.shape[1]
    
    # start the logistic regression
    for iter in range(max_iter):
        # get loss and update w.
 
        loss=calculate_log_loss(y, tx, w)
        grad=calculate_log_loss_gradient(y, tx, w)
        w=w-gamma*grad
        
        # log info
        if ((iter % 100 == 0) and display):
            print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
            
    return w, loss

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iter, gamma):
    """ 
        Performs a logistic regression with a regularization term. The loss is calculated without
        this term.
    """
    w = initial_w
    D=tx.shape[1]
    
    # start the logistic regression
    for iter in range(max_iter):
        # get loss and update w.
 
        loss=calculate_log_loss(y, tx, w) #Loss without the regularizer.
        grad=calculate_log_loss_gradient(y, tx, w)+lambda_*w
        w=w-gamma*grad
        
        # log info
        if ((iter % 1000 == 0) and display):
            print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
            
    return w, loss

##########################################################################
# Sub-functions used in the implementations of the methods shown above.
##########################################################################

def compute_mse(y, tx, w):
    """Compute MSE."""
    N=y.shape[0]
    e=y-tx.dot(w)
    loss=1/(2*N)*e.T.dot(e)
    return loss

##########################################################################
# Implemented in least-squares GD and SGD.

def compute_gradient(y, tx, w):
    """Compute gradient for MSE."""
    N=y.shape[0]
    e=y-tx.dot(w)
    grad=-1/N*tx.T.dot(e)
    return grad

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]
            
def compute_stoch_gradient(y, tx, w, batch_size=1):
    """Compute a stochastic gradient for mse from just few examples n and their corresponding y_n labels."""
    d=tx.shape[1]
    for element_y, element_x in batch_iter(y, tx, batch_size):
        grad_element=np.zeros((d,batch_size))
        for n in range(batch_size):
            y_n=np.array([element_y[n]])
            x_n=np.array([element_x[n,:]])
            grad_n=compute_gradient(y_n, x_n, w)
            grad_element[n]=grad_n
        grad_batch=np.mean(grad_element,axis=1)
    return grad_batch

##########################################################################
# Implemented in logistic regression.

def sigmoid(t):
    """apply sigmoid function on t."""
    sig=np.exp(t)/(1+np.exp(t))
    return sig

def calculate_log_loss(y, tx, w):
    """compute the cost by negative log likelihood."""
    N=y.shape[0]
    loss= np.sum(np.log(1+np.exp(tx.dot(w))),axis=0)-y.T.dot(tx.dot(w))
    return loss

def calculate_log_loss_gradient(y, tx, w):
    """compute the gradient of loss."""
    grad=tx.T.dot(sigmoid(tx.dot(w))-y)
    return grad