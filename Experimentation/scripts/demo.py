# -*- coding: utf-8 -*-

import numpy as np
from data_helpers import*
from implementation import*
from plots import *

def gradient_decent_demo():

    # Define the parameters of the algorithm.
    max_iters = 50
    gamma = 0.7

    # Initialization
    w_initial = np.array([0, 0])

    # Start gradient descent.
    start_time = datetime.datetime.now()
    gradient_losses, gradient_ws = least_squares_GD(y, tx, w_initial, max_iters, gamma)
    end_time = datetime.datetime.now()

    # Print result
    exection_time = (end_time - start_time).total_seconds()
    print("Gradient Descent: execution time={t:.3f} seconds".format(t=exection_time))



def stochastic_gradient_decent_demo():

    # Define the parameters of the algorithm.
    max_iters = 50
    gamma = 0.7

    # Initialization
    w_initial = np.array([0, 0])

    # Start SGD.
    start_time = datetime.datetime.now()
    sgd_losses, sgd_ws = least_squares_SGD( y, tx, w_initial, max_iters, gamma)
    end_time = datetime.datetime.now()

    # Print result
    exection_time = (end_time - start_time).total_seconds()
    print("SGD: execution time={t:.3f} seconds".format(t=exection_time))
    
    
def cross_validation_demo(degree,k_fold,y,x):
    seed = 20
    lambdas = np.logspace(-9, -5, 20 )
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    # define lists to store the loss of training data and test data
    rmse_tr = []
    rmse_te = []
    # ***************************************************
    # INSERT YOUR CODE HERE
    for lambda_ in lambdas:
        tmp_tr = []
        tmp_te = []
        for k in range(k_fold):
            a,b = cross_validation(y, x, k_indices, k, lambda_, degree)
            tmp_tr.append(a)
            tmp_te.append(b)
        rmse_tr.append(np.mean(tmp_tr))
        rmse_te.append(np.mean(tmp_te))
    # cross validation: TODO
    # ***************************************************    
    cross_validation_visualization(lambdas, rmse_tr, rmse_te)
    