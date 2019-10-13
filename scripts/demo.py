# -*- coding: utf-8 -*-

import numpy as np

def gradient_decent_demo()

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



def stochastic_gradient_decent_demo()    

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
    