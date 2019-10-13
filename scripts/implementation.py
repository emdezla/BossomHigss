# -*- coding: utf-8 -*-

import numpy as np
from numpy.linalg import inv

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
       """Gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        # ***************************************************
        # INSERT YOUR CODE HERE
        grad = compute_gradient(y,tx,w)
        loss = compute_loss(y, tx, w)
      
        # TODO: compute gradient and loss
        # ***************************************************
        # ***************************************************
        # INSERT YOUR CODE HERE
        w = w-gamma*grad
        # TODO: update w by gradient
        # ***************************************************
        # store w and loss
        ws.append(w)
        losses.append(loss)
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return losses, ws


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    
    ws = [initial_w]
    losses = []
    w = initial_w
   
    for n_iter in range(max_iters):
        for yn, xn in batch_iter(y, tx, batch_size=1, num_batches=1):

            grad = compute_stoch_gradient(yn,xn,w)
            w = w-gamma*grad
            loss = compute_loss(y, tx, w)
            ws.append(w)
            losses.append(loss)

    return losses, ws
    
    
def least_squares(y, tx):
    """calculate the least squares solution."""

    GM = tx.T.dot(tx);
    opt = inv(GM).dot(tx.T.dot(y))
    rmse = math.sqrt(2*compute_mse(y,tx,opt))
    
    return rmse, opt



def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""

    u = 2*len(y)*lambda_*np.identity(tx.shape[1])
    R = tx.T.dot(tx)+ u;
    opt = inv(R).dot(tx.T.dot(y))
    
    return opt

def logistic_regression(y, tx, initial_w, max_iters, gamma):

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    
