# -*- coding: utf-8 -*-

import numpy as np
from implementation import *

def calculate_mse(e):
    """Calculate the mse for vector e."""
    return 1/2*np.mean(e**2)


def compute_mae(e):
    """Calculate the mae for vector e."""
    return np.mean(np.abs(e))


def compute_loss(y, tx, w):
    """Calculate the loss."""
    e = y - tx.dot(w)
    return calculate_mse(e)
    # return calculate_mae(e)

    
def compute_gradient(y, tx, w):
    """Compute the gradient."""
    e = y - tx.dot(w)
    gradient = - tx.T.dot(e) / len(e)
    return gradient

def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""
    e = y - (tx@w)
    gradient = -tx.T@e /len(e)
    return gradient


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    poly = np.ones((len(x), 1))
    for deg in range(1, degree+1):
        poly = np.c_[poly, np.power(x, deg)]
    return poly

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)


def cross_validation(y, x, k_indices, k, lambda_, degree):
    """return the loss of ridge regression."""
    # ***************************************************
    # INSERT YOUR CODE HERE

    index_te = k_indices[k]
    index_tr = k_indices[~(np.arange(k_indices.shape[0]) == k)]
    index_tr = index_tr.reshape(-1);
    x_tr = x[index_tr]
    x_te = x[index_te]
    y_tr = y[index_tr]
    y_te = y[index_te]

    phi_tr = build_poly(x_tr,degree)
    phi_te = build_poly(x_te,degree)
    weight = ridge_regression(y_tr, phi_tr, lambda_)
    loss_tr = np.sqrt(2 * compute_loss(y_tr, phi_tr, weight))
    loss_te = np.sqrt(2 * compute_loss(y_te, phi_te, weight))

    return loss_tr, loss_te

