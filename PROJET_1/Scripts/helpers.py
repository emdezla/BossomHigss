# -*- coding: utf-8 -*-
"""some personal helper functions for project 1."""
import numpy as np
<<<<<<< HEAD
from implementations import*
from proj1_helpers import*
=======
from implementations import *
>>>>>>> 6d821103c13681dbfbc66399a15c41fb5b57e173


def data_preprocessing(y, tx, ids, dataTreatment="none"):
    """ Treatment of the meaningless data (-999)
            "none": nothing is done.
            "zero": meaningless data are replaced with zeros.
            "discard": meaningless data points are discarded.
            "mean": Replaces meaningless data with the mean of the meaningful data points.
    """
    if (dataTreatment=="discard"):
        meaningfull_ind=(tx!=-999).all(1)
        y = y[meaningfull_ind]
        tx = tx[meaningfull_ind]
      #  ids = ids[meaningfull_ind]
    elif (dataTreatment=="zero"):
        tx[tx==-999]=0
    elif (dataTreatment=="mean"):
        TX = np.copy(tx)
        TX[TX==-999]=0
        S = np.sum(TX,axis=0)
        TX[TX!=0]=1
        N = np.sum(TX,axis=0)
        means = S/N
        
        for index, mean in enumerate(means):
            indy = np.where(tx[:,index]==-999)
            tx[indy,index] = mean
        
    else:
        print("Careful, some data points have meaningless features!")
    return y, tx, ids
        
##################################################################################

def build_poly(x, degrees, constant_feature=True):
    """Polynomial basis functions for input data x. Each feature x_k is extended with x_k^j, j=1 up to j=degrees[k]. 
       Adds a column of ones in the front of x if constant_feature = True"""
    N=x.shape[0]
    D=x.shape[1]
    phi=np.ones((N,1))
    
    if (isinstance(degrees,int)):
        degrees=degrees*np.ones(D).astype(int)
    elif (isinstance(degrees,list)):
        degrees=np.array(degrees)
    assert(degrees.shape[0]==x.shape[1])
           
    for ind_feat, degree in enumerate(degrees):
        for i in range(degree):
            phi=np.c_[phi,x[:,ind_feat]**(i+1)]
    if (constant_feature==False):
        phi=np.delete(phi, 0, 1)
    return phi

##################################################################################

def check_accuracy(y_predicted,y_true):
    """ Return the accuracy and the F1 score of the predicted y w.r.t. the true y"""
    N=y_true.shape[0]
    assert(N==y_predicted.shape[0])
    accuracy=np.sum(y_predicted==y_true)/N
    
    ones=np.ones_like(y_true)   
    precision=np.sum((y_predicted+y_true==2*ones))/np.sum(y_predicted==ones)
    recall=np.sum((y_predicted+y_true==2*ones))/np.sum(y_true==ones)
    F1=2*precision*recall/(precision+recall)
    
    return accuracy, F1

##################################################################################

def prediction(x_train, y_train, x_test, degrees=1, method="RegLogReg", initial_w=None,
               max_iters=10000, gamma=1e-1, lambda_=0):
    """
        Function that alows to train the model with a given method and corresponding parameters. 
        It also predicts the label from a given test data. The arguments are:
        
        - x_train, y_train: Train data
        - x_test: Test data
        - degrees: 1. If an int: Max degree for the augmentation of each feature.
                   2. If an array of size (# of feature): Each element of degrees represents the max degree
                      for the augmentation of the corresponding feature.
        - method: The method used to predict the label:
                   1. LS: Least-squares
                   2. LS_GD: Least-squares gradient descent
                   3. LS_SGD: Least-squares stochastic gradient descent
                   4. RR: Ridge regression
                   5. LR: Logistic regression
                   6. RLR: Regularized logistic regression
        - initial_w: Starting points for the iterative methods (LS_GD, LS_SGD, LR, RLR)
        - max_iters: Number of iterations for the iterative methods (LS_GD, LS_SGD, LR, RLR)
        - gamma: Step size for the iterative methods (LS_GD, LS_SGD, LR, RLR)
        - lambda_: Regularization term for RR and RLR.
                   
    """
    
    N_te=x_test.shape[0]
    y_test_predicted=np.zeros(N_te)
    phi_train = build_poly(x_train, degrees)
    phi_test = build_poly(x_test, degrees)
    D=phi_train.shape[1]
    
    if ((initial_w==None) and (method!="LS") and (method!="RR")):
        initial_w = np.zeros(D)
    
    if (method=="LS"):
        w, loss_tr = least_squares(y_train, phi_train)
        
    elif (method=="LS_GD"):
        w, loss_tr = least_squares_GD(y_train, phi_train, initial_w, max_iters, gamma)
        
    elif (method=="LS_SGD"):
        w, loss_tr = least_squares_SGD(y_train, phi_train, initial_w, max_iters, gamma)
        
    elif (method=="RR"):
        w, loss_tr = ridge_regression(y_train, phi_train, lambda_)
        
    elif (method=="LR"):
        w, loss_tr = logistic_regression(y_train, phi_train, initial_w, max_iters, gamma)
        
    elif (method=="RLR"):
        w, loss_tr = reg_logistic_regression(y_train, phi_train, lambda_, initial_w, max_iters, gamma)
        
    else:
        raise Exception('The method is not valid!')
    
    y_test_predicted = predict_labels(w, phi_test)
    y_train_predicted = predict_labels(w, phi_train)
    accuracy_train, f1_train=check_accuracy(y_train_predicted,y_train)
    
    #print("The train data accuracy of the model is ",accuracy_train, "\nThe train data f1 score of the model is ", f1_train)
    
    return y_train_predicted, y_test_predicted

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold cross-validation."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)

def cross_validation(y, x, k_indices, k, lambda_, degrees):
    """return the accuracy of ridge regression."""
    y_test=y[k_indices[k,:]]
    x_test=x[k_indices[k,:]]   
    y_train=np.delete(y,k)
    x_train=np.delete(x,k,0)
    
    y_pred_train, y_pred_test = prediction(x_train, y_train, x_test, degrees, lambda_)
    accuracy_train, F1_train = check_accuracy(y_pred_train, y_train)
    accuracy_test, F1_test = check_accuracy(y_pred_test, y_test)
    return accuracy_train, accuracy_test, F1_train, F1_test

def cross_validation_visualization(lambdas, acc_tr, acc_te, f1_tr, f1_te):
    """visualization of the accuracy and the f1 score for the train data and the test data."""
    fig = plt.figure()
    fig.set_size_inches(12,4)
    ax_acc = fig.add_subplot(1, 2, 1)
    ax_f1 = fig.add_subplot(1, 2, 2)
    
    ax_acc.set_xlabel('lambda')
    ax_acc.set_ylabel('accuracy')
    ax_acc.semilogx(lambdas, acc_tr, marker=".", color='b', label='train accuracy')
    ax_acc.semilogx(lambdas, acc_te, marker=".", color='r', label='test accuracy')
    ax_acc.set_title('Accuracy')           
    ax_acc.grid(True)
    ax_acc.legend(loc=2)
    
    ax_f1.set_xlabel('lambda')
    ax_f1.set_ylabel('f1 score')
    ax_f1.semilogx(lambdas, f1_tr, marker=".", color='b', label='train f1 score')
    ax_f1.semilogx(lambdas, f1_te, marker=".", color='r', label='test f1 score')
    ax_f1.set_title('F1 score')           
    ax_f1.grid(True)
    ax_f1.legend(loc=2)
    
    fig.savefig('cross_validation')


def cross_validation_demo(y, x, k_fold, lambdas, degrees,seed=1):
    """to do"""
    k_indices = build_k_indices(y, k_fold,seed)
    acc_tr = []
    acc_te = []
    f1_tr = []
    f1_te = []
    for lambda_ in lambdas:
        acc_tr_lambda=0;
        acc_te_lambda=0;
        f1_tr_lambda=0;
        f1_te_lambda=0;
        for k in range(k_fold):
            accuracy_train, accuracy_test, f1_train, f1_test = cross_validation(y, x, k_indices, k, lambda_, degrees)
            
            acc_tr_lambda += accuracy_train/k_fold
            acc_te_lambda += accuracy_test/k_fold
            f1_tr_lambda += f1_train/k_fold
            f1_te_lambda += f1_test/k_fold
            
        print("The train data accuracy of the model is ",acc_tr_lambda, " with degree=", degrees," and lambda=", lambda_)
        
        acc_tr.append(acc_tr_lambda)
        acc_te.append(acc_te_lambda)
        f1_tr.append(f1_tr_lambda)
        f1_te.append(f1_te_lambda)
       
    cross_validation_visualization(lambdas, acc_tr, acc_te, f1_tr, f1_te)
