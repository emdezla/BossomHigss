import numpy as np
import matplotlib.pyplot as plt
from helpers import *
from implementations import *

def logistic_implementation(y, x, degrees, ratio, seed, max_iters, gamma,Newton=False):
    from helpers import build_poly, split_data
    
    x_tr, y_tr, x_te, y_te = split_data(x,y,ratio,seed)
    
    losses_tr = []
    losses_te = []
    
    
    for degree in degrees:
        print('degree = ',degree)
        tx_tr = build_poly(x_tr, degree)
        initial_w = np.zeros(tx_tr.shape[1])
        
        
        if Newton == False:
            w, loss_tr = logistic_regression(y_tr, tx_tr, initial_w, max_iters, gamma)
        else:
            w, loss_tr = logistic_newton(y_tr, tx_tr ,initial_w , max_iters)
        
        np.append(losses_tr,loss_tr)
        
        tx_te = build_poly(x_te, degree)
        correct_percentage, loss_te = logistic_test(y_te,tx_te,w,degree)
        
        np.append(losses_te,loss_te)
    
    #plt.plot(degrees,losses_tr,'r',degrees,losses_te,'b')
    
    return




def logistic_regression(y, tx, initial_w, max_iters, gamma):
    
    #Changing y_n =-1 to y_n = 0 in order to implement classification algorithms studied in class
    ind_0 = np.where(y == -1)
    y[ind_0] = 0
    
    w = initial_w
    
    for iter in range(max_iters):
        if iter % 100 == 0:
            print('iteration ', iter)
        # Gradient iteration
        w, loss = logistic_descent(y,tx,w,gamma)
           
    #Changing y_n = 0 back to -1
    y[ind_0] = -1
    
    return w, loss

def logistic_newton(y,tx,initial_w,max_iters):
    ind_0 = np.where(y==-1)
    y[ind_0] = 0
    
    w = initial_w
    
    for iter in range(max_iters):
        if iter % 100 == 0:
            print('iteration ', iter)
        gradient = logistic_gradient(y,tx,w)
        hessian = logistic_hessian(y,tx,w)
        loss = logistic_loss(y,tx,w)
        w = w - np.linalg.inv(hessian)@gradient
    return w, loss
        
def sigmoid(t):
    return np.exp(t)/(1+np.exp(t))

def logistic_loss(y,tx,w):
    loss = np.sum( np.log(1+np.exp(tx@w)) ,axis=0 ) - y.T@tx@w
    return loss

def logistic_gradient(y,tx,w):
    gradient = -tx.T@y + tx.T@(sigmoid(tx@w))
    return gradient

def logistic_descent(y,tx,w,gamma):
    gradient = logistic_gradient(y,tx,w)
    loss = logistic_loss(y,tx,w)
    
    w = w - gamma*gradient
    
    return w, loss

def logistic_hessian(y, tx, w):
    S = (sigmoid(tx@w)*(1-sigmoid(tx@w)))
    S = S.ravel()
    
    hessian = tx.T@np.diag( S )@tx
    return hessian

def classification(tx_te,w):
    """computes probability of a data point x of being a boson y=1 or not y=-1, classifies each data point"""
    
    n = tx_te.shape[0]
    y_pred = np.zeros(n)
    
    a = tx_te@w
    ind_1 = np.where(a.ravel()>=0.5)
    ind_0 = np.where(a.ravel()<0.5)
    
    y_pred[ind_1] = 1
    y_pred[ind_0] = -1
    
    #for i in range(n):
    #    if sigmoid(tx_te[i,:]@w)>= 0.5:
    #        y_pred[i] = 1
    #    else:
    #        y_pred[i] = -1
    
    return y_pred

def logistic_test(y_te,tx_te,w,degree):
    
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
    