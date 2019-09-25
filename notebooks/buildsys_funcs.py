import matplotlib.pyplot as plt
import os
import numpy as np
import datetime
from statsmodels.tsa import stattools
from sklearn import preprocessing
import random
import copy
import scipy
import sklearn.metrics
from sklearn.linear_model import LogisticRegression

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

#state transition matrix functions
class linear_nnet(nn.Module):
    #linear model for kernelized inputs
    #to do logistic regression use criterion = nn.CrossEntropyLoss() & num class output
    def __init__(self, params):
        super(linear_nnet, self).__init__()
        self.D_in = params['FEATURE_DIM']
        self.D_out = params['OUTPUT_DIM']
        self.l1 = nn.Linear(self.D_in, self.D_out)
    
    def forward(self, x):
        x = self.l1(x) #linear weights for model interpretability
        return(x)
        
def minibatch_X_Y_arrays(X_arr, Y_arr, batchsize):    
    if batchsize == 1:
        return([X_arr], [Y_arr])
    #list of training, target pair tuples
    remainder = X_arr.shape[1] % batchsize
    diff = batchsize - remainder
    tail_X = X_arr[:,-diff:] 
    tail_Y = Y_arr[:,-diff:]
    out_X = [ X_arr[:,i*batchsize:(i+1)*batchsize] for i in range(int(float(X_arr.shape[1])/float(batchsize))) ]
    out_Y = [ Y_arr[:,i*batchsize:(i+1)*batchsize] for i in range(int(float(Y_arr.shape[1])/float(batchsize)))]
    out_X = out_X + [tail_X]
    out_Y = out_Y + [tail_Y]
    return(out_X, out_Y)
        
def train_linear_state_estimation(net, params, X_train, X_val, Y_train, Y_val, lrate=0.01, epochs=1000, batch_size=100, l="mse", verbose=True, validate=True):
    if l == "mse":
        loss_func = nn.MSELoss()#SmoothL1Loss()
    if l == "bce":
        loss_func = nn.CrossEntropyLoss()
        print("Using Binary Cross Entropy Loss")
    optimizer = optim.SGD(net.parameters(),lr=lrate, momentum=0.9)
    for e in range(epochs):
        training_losses = []
        X_train_list, Y_train_list = minibatch_X_Y_arrays(X_train, Y_train, batch_size)
        for i in enumerate(X_train_list):
            if l == "bce":
                inp = Variable(torch.Tensor(X_train_list[i[0]].T))
                label = Variable(torch.Tensor(Y_train_list[i[0]].T)).type(torch.long)[:,0]#.unsqueeze(-1)
            else:
                inp = Variable(torch.Tensor(X_train_list[i[0]].T))
                label = Variable(torch.Tensor(Y_train_list[i[0]].T))

            out = net(inp)
            loss = loss_func(out, label)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    if validate==True:
        if l == "bce":
            inp_val = Variable(torch.Tensor(X_val.T))
            label_val = Variable(torch.Tensor(Y_val.T)).type(torch.long)[:,0]
        else:
            inp_val = Variable(torch.Tensor(X_val.T))
            label_val = Variable(torch.Tensor(Y_val.T))
        out_val = net(inp_val)
        loss_val = loss_func(out_val, label_val)
    if verbose==True:
        print("Validation MSE: ", loss_val)
        
#Bayesian classifier projection features
def mat_C(x):
    #x shape = px1
    C = np.dot(x, x.T) + np.eye(x.shape[0])
    return(C)

def mat_D(x, y, A):
    #x shape px1, y shape nx1, A nxp
    D = np.dot(y, x.T) + A
    return(D)

def likelihood_point(x, y, A):
    C = mat_C(x)
    D = mat_D(x, y, A)
    const = x.shape[0]*np.log(np.linalg.det(np.linalg.inv(C)))
    var = np.trace(y.dot(y.T) - 2*A.dot(x).dot(y.T) + x.T.dot(A.T).dot(A).dot(x))
    var2 = np.trace(y.dot(y.T) + A.dot(A.T) - D.dot(np.linalg.inv(C)).dot(D.T))   #mixed term doesn't look right
    out = var - var2 + const
    return(out)
    
def sample_classification_transform(A, Xallpoly, Xallpolyf, Yall, Yallf):
    #use inputs to classification rule as features; try logistic regression classifier

    X = []
    Y = []

    for i in range(Xallpoly.shape[1]):
        if i % 1000 == 0:
            print("No fault data: ", np.around(100*(i/Xallpoly.shape[1])), "%")
        x = np.expand_dims(Xallpoly[:,i], axis=1)
        y = np.expand_dims(Yall[:,i], axis=1)
        C = mat_C(x)
        D = mat_D(x, y, A)
        C_inv = np.linalg.inv(C)
        term1 = x.shape[0]*np.log(np.linalg.det(C_inv))
        term2 = np.trace(C_inv.dot(D.T).dot(D))
        term3 = -1.0*np.trace(A.dot(A.T))
        term4 = -2.0*np.trace(A.dot(x).dot(y.T))
        term5 = np.trace(x.T.dot(A.T).dot(A).dot(x))

        feat = [term1, term2, term3, term4, term5]
        features_app = np.asarray(feat)#(feat + list(x[:,0]) + list(y[:,0])) #np.expand_dims(A.flatten(), axis=1), x and y themselves seem to be a waste
        X.append(features_app)
        Y.append(0)

    X_f = []
    Y_f = []

    for i in range(Xallpolyf.shape[1]):
        if i % 1000 == 0:
            print("Fault data: ", np.around(100*(i/Xallpoly.shape[1])), "%")
        x = np.expand_dims(Xallpolyf[:,i], axis=1) #tried with fault data, using normal operational data
        y = np.expand_dims(Yallf[:,i], axis=1)
        C = mat_C(x)
        D = mat_D(x, y, A)
        C_inv = np.linalg.inv(C)
        term1 = x.shape[0]*np.log(np.linalg.det(C_inv))
        term2 = np.trace(C_inv.dot(D.T).dot(D))
        term3 = -1.0*np.trace(A.dot(A.T))
        term4 = -2.0*np.trace(A.dot(x).dot(y.T))
        term5 = np.trace(x.T.dot(A.T).dot(A).dot(x))

        feat = [term1, term2, term3, term4, term5]
        features_app = np.asarray(feat)#(feat + list(x[:,0]) + list(y[:,0])) #np.expand_dims(A.flatten(), axis=1)
        X_f.append(features_app)
        Y_f.append(1)

    X = np.asarray(X)
    Y = np.asarray(Y)
    X_f = np.asarray(X_f)
    Y_f = np.asarray(Y_f)
    
    Y = np.expand_dims(Y, axis=1)
    Y_f = np.expand_dims(Y_f, axis=1)
    
    X = X.T
    Y = Y.T
    X_f = X_f.T
    Y_f = Y_f.T
    
    return(X, Y, X_f, Y_f)
    
def lag_samples_array(X, lag):
    X_lag = np.zeros((lag*X.shape[0], X.shape[1] - lag))
    for i in range(X.shape[1] - lag):
        X_lag[:,i] = np.asarray(X[:,i:i+lag]).flatten()
    return(X_lag)
    
def plot_polling_val_data(X_val_log_reg, Y_val_log_reg, sklearn_predictor):
    fault_votes = [0.0]
    nofault_votes = [0.0]
    base = [0.0]

    for i in range(X_val_log_reg.shape[1]):
        #if no fault
        if Y_val_log_reg[0,i] == 0:
            Y_hat = sklearn_predictor.predict(X_val_log_reg[:,i].T.reshape(1, -1))
            if Y_hat == 0:
                nofault_votes.append(nofault_votes[-1] - 1)
            if Y_hat == 1:
                nofault_votes.append(nofault_votes[-1] + 1)
        #if fault
        if Y_val_log_reg[0,i] == 1:
            Y_hat = sklearn_predictor.predict(X_val_log_reg[:,i].T.reshape(1, -1))
            if Y_hat == 0:
                fault_votes.append(fault_votes[-1] - 1)
            if Y_hat == 1:
                fault_votes.append(fault_votes[-1] + 1)
            base.append(base[-1] + 0.5)
    
    plt.plot(fault_votes, label="fault data")
    plt.plot(nofault_votes, label="no fault data")
    plt.plot(base, label="1/2")
    plt.plot(np.zeros((len(fault_votes,))))
    plt.ylabel("net positive fault classifications")
    plt.xlabel("number of samples")
    plt.title("logistic regression classifier performance")
    plt.legend()
    plt.show()
    
