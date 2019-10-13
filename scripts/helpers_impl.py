# -*- coding: utf-8 -*-

import numpy as np


def calculate_mse(e):
    """Calculate the mse for vector e."""
    return 1/2*np.mean(e**2)


def calculate_mae(e):
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