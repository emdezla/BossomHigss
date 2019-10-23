# -*- coding: utf-8 -*-
"""some personal helper functions for project 1."""
import numpy as np
from implementations import *
from proj1_helpers import *

def data_processing(DATA):
                       
    y, tx, ids = load_csv_data(DATA,sub_sample=False)
    
    for jet in range(4):
        
        # select data in function of its jet number and delete the corresponding feature
        x_jet = tx[tx[:, 22] == jet]
        y_jet = y[tx[:, 22] == jet]
        ids_jet = ids[tx[:, 22] == jet]

        x_jet = np.delete(x_jet, 22, 1)
        
        # replace the Nan values by the mean of the mass with same jet number
        TX = np.copy(x_jet[:,0])
        TX[TX==-999]=0
        mean = np.sum(TX)/len(TX)
        TX[TX==0]=mean
        x_jet[:,0] = TX
        
    
        # extract the meaningless features depending on the jet number
        if jet==0: 
            jet0_delete = [4,5,6,12,22,23,24,25,26,27,28] 
            jet0 = np.delete(x_jet,jet0_delete,1)
            print("tx_0_train:",jet0.shape)
        elif jet==1:
            jet1_delete = [4,5,6,12,25,26,27]
            jet1 = np.delete(x_jet,jet1_delete,1)
            print("tx_1_train:",jet1.shape)
        elif jet==2:
            jet2 = x_jet
            print("tx_2_train:",jet2.shape)
        elif jet==3:
            jet3 = x_jet
            print("tx_3_train:",jet3.shape)
        else: print("Fatal error - unexpected jet number")
            
    return jet0,jet1,jet2,jet3

        
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

def prediction(x_train, y_train, x_test, degrees=1,lambda_=0, method="RR", initial_w=None,
               max_iters=10000, gamma=1e-1):
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
        y_test_predicted = predict_labels(w, phi_test)
        y_train_predicted = predict_labels(w, phi_train)
        
    elif (method=="LS_GD"):
        w, loss_tr = least_squares_GD(y_train, phi_train, initial_w, max_iters, gamma)
        y_test_predicted = predict_labels(w, phi_test)
        y_train_predicted = predict_labels(w, phi_train)
        
    elif (method=="LS_SGD"):
        w, loss_tr = least_squares_SGD(y_train, phi_train, initial_w, max_iters, gamma)
        y_test_predicted = predict_labels(w, phi_test)
        y_train_predicted = predict_labels(w, phi_train)
        
    elif (method=="RR"):
        w, loss_tr = ridge_regression(y_train, phi_train, lambda_)
        y_test_predicted = predict_labels(w, phi_test)
        y_train_predicted = predict_labels(w, phi_train)
        
    elif (method=="LR"):
        y_train[y_train==-1]=0
        w, loss_tr = logistic_regression(y_train, phi_train, initial_w, max_iters, gamma)
        y_test_predicted = predict_labels(w, phi_test)
        y_train_predicted = predict_labels(w, phi_train, reg_log=True)
        
    elif (method=="RLR"):
        y_train[y_train==-1]=0
        w, loss_tr = reg_logistic_regression(y_train, phi_train, lambda_, initial_w, max_iters, gamma)
        y_test_predicted = predict_labels(w, phi_test)
        y_train_predicted = predict_labels(w, phi_train,reg_log=True)
        
    else:
        raise Exception('The method is not valid!')
    
    accuracy_train, f1_train=check_accuracy(y_train_predicted,y_train)
    
    print("Lambda=1e{0:1.0f}:\n The train data accuracy of the model is {1} \nThe train data f1 score of the model is {2} ".format(np.log10(lambda_),accuracy_train,f1_train))
    
    return y_train_predicted, y_test_predicted



"""def data_preprocessing(y, tx, ids, dataTreatment="none"):
     Treatment of the meaningless data (-999)
            "none": nothing is done.
            "zero": meaningless data are replaced with zeros.
            "discard": meaningless data points are discarded.
            "mean": Replaces meaningless data with the mean of the meaningful data points.
    
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
    return y, tx, ids"""