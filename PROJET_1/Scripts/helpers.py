# -*- coding: utf-8 -*-
"""some personal helper functions for project 1."""
import numpy as np


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
        ids = ids[meaningfull_ind]
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

def prediction(x_train, y_train, x_test, degrees, lambda_):
    N_te=x_test.shape[0]
    y_test_predicted=np.zeros(N_te)
    phi_train = build_poly(x_train, degrees)
    phi_test = build_poly(x_test, degrees)
    
    w, mse_tr = ridge_regression(y_train, phi_train, lambda_)
    y_test_predicted = predict_labels(w, phi_test)
    y_train_predicted = predict_labels(w, phi_train)
    accuracy_train, f1_train=check_accuracy(y_train_predicted,y_train)
    
    print("The train data accuracy of the model is ",accuracy_train, "\nThe train data f1 score of the model is ", f1_train)
    
    return y_train_predicted, y_test_predicted