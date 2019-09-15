import math
import copy
import numpy as np
import h5py
import pandas as pd
import torch
from torch import nn, optim
from torch.autograd import Variable
import multiprocessing as mp
from torch.nn.functional import mse_loss
from matplotlib import pyplot as plt
#import tensorflow as tf
from scipy.io import loadmat
import scipy.stats
import sys, os
from skimage import img_as_float
from skimage.transform import downscale_local_mean
from scipy.stats import pearsonr

"""
This file implemented the Projection Pursuit Regression in Pytorch, please refer
to the original paper http://inspirehep.net/record/152302/files/slac-pub-2466.pdf
for more detail.
This algorithm works on one neuron at a time and it can be run very fast so this
file is in CPU version can call cuda() to transfer it into a GPU version.
"""

def check_converge(r1, r2, r3, a, b, c, epsilon=1e-5):
    s = abs(r1-a) + abs(r2-b) + abs(r3-c)
    return s < epsilon

class PPR(nn.Module):
    #a class that performs projection pursuit optimization after selection.
    def __init__(self, lr = 1e-3, wd = 1e-5, input_size=20, nlayers=5, seed=None):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        self.lr = lr #learning rate
        self.wd = wd #weight decay
        self.nlayers = nlayers #number of layers
        self.input_size = input_size #my input image is 20x20
        self.filter = [] 
        self.theta = [] #parameters for quadratic function
        for i in range(nlayers):
            F = torch.nn.Parameter(0.01*torch.ones(input_size**2))
            self.filter.append(F)
            l = torch.nn.Parameter(torch.tensor([0.01,0.01]))
            self.theta.append(l)
    
    def predict(self, X, ind_set = None):
        sum = None
        if ind_set == None:
            ind_set = range(len(self.filter))
        for i in ind_set:
            if sum is None:
                sum = self.forward(X,i)
            else:
                sum = sum + self.forward(X,i)
        return sum

    def forward(self, X, layer):
        X = torch.matmul(X, self.filter[layer])
        X = self.theta[layer][0]*X + self.theta[layer][1]*(X**2)
        return X
    
    def evaluate(self, PP, X, y):
        """This function sort the sub units by their contribution to the model, and
        eliminate the sub units one-by-one until only one subunit left. Then choose
        the F with smallest number of sub units but still without significant loss in
        CC.
        """
        k = len(self.filter)
        arrinds = PP.argsort()
        #print(arrinds)
        self.reorder(arrinds)
        sorted_PP = PP[arrinds]
        prev_cc = self.eval_func_2(X, y)
        for i in range(k):
            if(i <= k-2):
                #print(prev_cc)
                temp_F = self.filter[i]
                pred = self.predict(X, range(i+1, k))
                y_pred = pred.data.numpy().reshape(-1)
                cc = pearsonr(y.numpy(), y_pred)[0]
                if(cc <= 0.9*prev_cc):
                    self.reorder(range(i,k))
                    self.out = k-i
                    break
                else:
                    prev_cc = cc
    

    def reorder(self, order):
        new_F = []
        new_theta = []
        for i in order:
            new_F.append(self.filter[i])
            new_theta.append(self.theta[i])
        self.filter = new_F
        self.theta = new_theta
        
    def eval_func_2(self, X, y):
        y_pred = self.predict(X)
        y_pred = y_pred.data.numpy().reshape(-1)
        return pearsonr(y, y_pred)[0]
        

    def eval_func(self, loader, layer):
        Y = None
        for X, y in loader:
            X = X.reshape(X.shape[0], self.input_size*self.input_size)
            y_pred = self.predict(X, layer)
            y_pred = y_pred.data.numpy().reshape(-1)
            if Y is None:
                Y = y.view(-1).numpy()
                Y_pred = y_pred
            else:
                Y = np.concatenate((Y, y.view(-1).numpy()))
                Y_pred = np.concatenate((Y_pred, y_pred))
                
        return pearsonr(Y, Y_pred)[0]


def convert(F, theta, layers = 5, inputD = 20):
    #convert the tensor into a numpy to facilitate save and visualization
    res_F = np.zeros((layers,inputD,inputD))
    res_T = np.zeros((layers,2))
    for i in range(len(F)):
        res_F[i] = F[i].data.numpy().reshape(inputD,inputD)
        for j in range(2):
            res_T[i,j] = theta[i][j].data.numpy()
    return (res_F, res_T)

