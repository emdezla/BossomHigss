import numpy as np
import matplotlib.pyplot as plt
from helpers import *
from implementations import *

def reg_logistic_implementation(y, x, degrees, ratio, seed, max_iters, gamma):
    
    from helpers import build_poly, split_data
    
    # Split the data based on the input ratio into training and testing data
    x_tr, y_tr, x_te, y_te = split_data(x,y,ratio,seed)
    
    losses_tr = []
    losses_te = []
    
    
    for degree in degrees:
        print('degree = ',degree)
        
        # Build a training polynomial basis based on the choice of degree
        tx_tr = build_poly(x_tr, degree)
        
        # Initialize starting point of the gradient descent
        initial_w = np.zeros(tx_tr.shape[1])
        
        # Perform iteration - calculate w(t+1) and calculate the new loss
        w, loss_tr = reg_logistic_regression(y_tr, tx_tr, initial_w, max_iters, gamma)
        
        np.append(losses_tr,loss_tr)
        
        # Build a testing polynomial basis based on the choice of degree
        tx_te = build_poly(x_te, degree)
        
        # Test the validity of the predictions with the help of the test data
        correct_percentage, loss_te = reg_logistic_test(y_te,tx_te,w,degree)
        
        np.append(losses_te,loss_te)
    
    
    return

def reg_logistic_regression (y, tx, lambda_, initial_w, max_iters, gamma, lambda_):
    
    threshold = 1e-8
    w = initial_w
    loss = []
    
    for iter in max_iters:
        
        w, loss = reg_logistic_descent(y,tx,w,gamma,lambda_)
        np.append(losses,loss)
        
        if len(loss) > 1 and np.abs(losses[-1]-losses[-2]) < threshold:
            break
        
    return w, loss
      
def sigmoid(t):
    return np.exp(t)/(1+np.exp(t))

def reg_logistic_loss(y,tx,w,lambda_):
    loss = np.sum( np.log(1+np.exp(tx@w)) ,axis=0 ) - y.T@tx@w + lambda_*np.linalg.norm(w)
    return loss

def reg_logistic_gradient(y,tx,w,lambda_):
    gradient = -tx.T@y + tx.T@(sigmoid(tx@w))*2*lambda_*w
    return gradient

def reg_logistic_descent(y,tx,w,gamma,lambda_):
    gradient = reg_logistic_gradient(y,tx,w,lambda_)
    loss = reg_logistic_loss(y,tx,w,lambda_)
    
    w = w - gamma*gradient
    
    return w, loss

def classification(tx_te,w):
    """computes probability of a data point x of being a boson y=1 or not y=-1, classifies each data point"""
    
    n = tx_te.shape[0]
    y_pred = np.zeros(n)
    
    a = tx_te@w
    ind_1 = np.where(a.ravel()>=0.5)
    ind_0 = np.where(a.ravel()<0.5)
    
    y_pred[ind_1] = 1
    y_pred[ind_0] = -1

    return y_pred

def reg_logistic_test(y_te,tx_te,w,degree):
    
    loss_te = logistic_loss(y_te,tx_te,w)
    y_pred = classification(tx_te,w)
    
    n_total = y_te.shape[0]
    correct_indices = np.where(y_pred == y_te)
    
    n_correct = len(correct_indices)
    correct_percentage = n_correct/n_total*100
    
    print("For a degree of ", degree)
    print("We guessed ",correct_percentage," percent right")
    print("Logistic loss = ",loss_te)
    
    return correct_percentage, loss_te
    