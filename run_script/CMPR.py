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



class CMPR(nn.Module):
    #a class that performs projection pursuit optimization after selection.
    def __init__(self, weights, lr = 1e-3, wd = 1e-5, input_size=20, nlayers = 5, seed=None):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)

        self.lr = lr
        self.wd = wd
        self.nlayers = nlayers
        self.input_size = input_size
        self.weights = weights
        self.dic_size = weights.shape[0]
        self.filter_size = weights.shape[-1]
        #faster computation for selecting out the filter during training. 
        self.temp_conv, self.temp_theta = self.init_temp_conv()  
        self.conv = [] # actual filter layer in referencing
        self.theta = []
        for i in range(nlayers):
            self.conv.append(nn.Conv2d(1, 1, kernel_size = self.filter_size))
            l = torch.nn.Parameter(torch.tensor([0.01,0.01]))
            self.theta.append(l)
    
    def init_temp_conv(self):
        #init the temporary filter layer with dictionary to be selected from.
        layer = nn.Conv2d(1, self.dic_size, kernel_size = self.filter_size)
        t = torch.from_numpy(self.weights)
        t = t.type(torch.FloatTensor)
        t = nn.Parameter(t)
        layer.weight = t
        
        theta = []
        for i in range(self.nlayers):
            l = torch.nn.Parameter(torch.ones(self.dic_size, 2)*0.01)
            theta.append(l)
        
        return layer, theta 
    
        
    def predict_ref(self, X, ind_set = None):
        #prediction used in referencing
        sum = None
        if ind_set == None:
            ind_set = range(self.nlayers)
        for i in ind_set:
            if sum is None:
                sum = self.forward_ref(X,i)
            else:
                sum = sum + self.forward_ref(X,i)
        return sum
    

    def forward_ref(self, X, ind):
        #forward used in referencing 
        if ind >= len(self.conv):
            return torch.zeros(X.shape[0])
        else:
            x = self.conv[ind](X)
        x = x.abs()
        x = x.reshape(x.shape[0], x.shape[-1]**2)
        x = torch.max(x, 1)[0]
        x = self.theta[ind][0]*x + self.theta[ind][1]*(x**2)
        return x
    
    def eval_func_ref(self, X, y):
        #evaluation function used in reference
        y_pred = self.predict_ref(X)
        y_pred = y_pred.data.numpy()
        return pearsonr(y, y_pred)[0]
    
    
    def predict(self, X, layer):
        #predict function used in training
        res = torch.zeros(X.shape[0], self.dic_size)
        for i in range(layer+1):
            if i == layer:
                res += self.forward(X, layer)
            else:
                temp = self.forward_ref(X,i)
                res += (torch.ones(self.dic_size, len(temp))*temp).t()
        return res
            
    
    def forward(self, X, ind):
        #forward used in training
        x = self.temp_conv(X)  #B X K X O X O
        x = x.abs()
        x = x.reshape(x.shape[0], x.shape[1], x.shape[-1]**2)
        x = torch.max(x, 2)[0] #B X K
        x = x*self.temp_theta[ind][:,0] + (x*x)*self.temp_theta[ind][:,1] #B X K
        return x
    
    
    def eval_func(self, val_loader, layer):
        #evaluation function used in training
        Y = None
        for X, y in val_loader:
            X = Variable(X, requires_grad = False)
            y_pred = self.predict(X, layer)
            y_pred = y_pred.data.numpy()
            if Y is None:
                Y = y.view(-1).numpy()
                Y_pred = y_pred
            else:
                Y = np.concatenate((Y, y.view(-1).numpy()))
                Y_pred = np.concatenate((Y_pred, y_pred))
        val_best = -100
        idx_best = None
        for i in range(self.dic_size):
            val_corr = pearsonr(Y, Y_pred[:,i])[0]
            if val_corr > val_best:
                val_best = val_corr
                idx_best = i
            
        return val_best, idx_best, copy.deepcopy(self.temp_theta[layer][idx_best,:].detach())

    
    def evaluate(self, PP, X, y):
        """This function sort the sub units by their contribution to the model, and
        eliminate the sub units one-by-one until only one subunit left. Then choose
        the F with smallest number of sub units but still without significant loss in
        CC.
        """
        k = len(self.conv)
        arrinds = PP.argsort()
        #print(arrinds)
        self.reorder(arrinds)
        sorted_PP = PP[arrinds]
        prev_cc = self.eval_func_ref(X, y)
        for i in range(k):
            if(i <= k-2):
                #print(prev_cc)
                temp_F = self.conv[i]
                pred = self.predict_ref(X, range(i+1, k))
                y_pred = pred.data.numpy()
                cc = pearsonr(y, y_pred)[0]
                if(cc <= 0.9*prev_cc):
                    self.reorder(range(i,k))
                    self.nlayers = k-i
                    break
                else:
                    prev_cc = cc
        #print(sorted_PP)



    def reorder(self, order):
        new_F = []
        new_theta = []
        for i in order:
            new_F.append(self.conv[i])
            new_theta.append(self.theta[i])
        self.conv = new_F
        self.theta = new_theta
            