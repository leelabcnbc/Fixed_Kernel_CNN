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



class CPPR(nn.Module):
    #a class that performs projection pursuit optimization after selection.
    def __init__(self, lr = 1e-3, wd = 1e-5, input_size=20, nlayers = 3, seed=None):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        self.lr = lr
        self.wd = wd
        self.nlayers = nlayers
        self.input_size = input_size
        self.conv = []
        self.theta = []
        for i in range(nlayers):
            self.conv.append(nn.Conv2d(1, 1, kernel_size = 13))
            l = torch.nn.Parameter(torch.tensor([0.01,0.01]))
            self.theta.append(l)

    
    def predict(self, X, ind_set = None):
        sum = None
        if ind_set == None:
            ind_set = range(self.nlayers)
        for i in ind_set:
            if sum is None:
                sum = self.forward(X,i)
            else:
                sum = sum + self.forward(X,i)
        return sum

    def forward(self, X, ind):
        if self.conv[ind] is not None:
            x = self.conv[ind](X)
        else:
            x = X
        x = x.abs()
        x = x.reshape((int(x.size()[0]), int(x.size()[3])**2))
        x = torch.max(x, 1)[0]
        x = self.theta[ind][0]*x + self.theta[ind][1]*(x**2)
        return x


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
        prev_cc = self.eval_func(X, y)
        for i in range(k):
            if(i <= k-2):
                #print(prev_cc)
                temp_F = self.conv[i]
                pred = self.predict(X, range(i+1, k))
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
            
            
        

    def eval_func(self, X, y):
        y_pred = self.predict(X)
        y_pred = y_pred.data.numpy()
        return pearsonr(y, y_pred)[0]
