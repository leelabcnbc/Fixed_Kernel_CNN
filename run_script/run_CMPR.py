import numpy as np
import h5py
import os
from collections import OrderedDict
import matplotlib.pyplot as plt
import time
from skimage import img_as_float
from torch import nn, FloatTensor
import torch
from torch import nn, optim
from torch.autograd import Variable
from torchvision.utils import make_grid
import sys
import copy
import importlib
from scipy.stats import pearsonr
sys.path.append('/home/ziniuw/Fixed_Kernel_CNN')
from util.get_data_pattern import prepare_dataset_pattern
from scipy.io import loadmat
from torch.nn.functional import mse_loss
import sys
from models_8K.utils import make_dataloader
from CMPR import CMPR


def get_residual(model,X,y,layer):
    if layer == 0:
        return y
    else:
        pred = 0
        for i in range(layer):
            pred += model.forward_ref(X, i)
        return y - pred
    
def to_tensor(X, y, size):
    X = torch.from_numpy(X)
    X = Variable(X, requires_grad = False)
    X = X.type(torch.FloatTensor)
    y = torch.from_numpy(y.reshape(-1))
    y = Variable(y, requires_grad = False)
    y = y.type(torch.FloatTensor)
    return X,y

def eval_func(model, loader, layer):
    Y = None
    for X, y in loader:
        X = Variable(X, requires_grad = False)
        y_pred = model.predict_ref(X, layer)
        y_pred = y_pred.data.numpy().reshape(-1)
        if Y is None:
            Y = y.view(-1).numpy()
            Y_pred = y_pred
        else:
            Y = np.concatenate((Y, y.view(-1).numpy()))
            Y_pred = np.concatenate((Y_pred, y_pred))

    return pearsonr(Y, Y_pred)[0]


def fix_learnt_layer(model, best_idx, best_theta, layer):
    t = torch.from_numpy(model.weights[best_idx].reshape(1,1,model.filter_size, model.filter_size))
    t = t.type(torch.FloatTensor)
    t = nn.Parameter(t)
    model.conv[layer].weight = t
    model.theta[layer] = nn.Parameter(best_theta)
    return None

def train_one(model, data, max_iter = [100,70,50,25,10], epsilon = 1e-6, lr = 1e-3, wd=2e-5, show_every = 10):
    """Note that in my PPR for Tang's data I only used the top stimulus so
       the batch is the entire data but if you want to use other dataset, 8k
       for example, you will need to write your own dataloader in pytorch.
       Note this function is not efficient enough.
    """
    #To understand the training algoritm please refer to the original paper of PPR
    train_loader = make_dataloader(data[0], data[1], batch_size=64, is_train=True)
    valid_loader = make_dataloader(data[2], data[3], batch_size=64, is_train=False)
    test_loader = make_dataloader(data[4], data[5], batch_size=64, is_train=False)
    train_X, train_y, val_X, val_y, test_X, test_y = data
    project_pursuit = np.zeros(shape=model.nlayers)
    old_residual = None
    for layer in range(model.nlayers): #optimizing layer by layer
        best_valCC = 0
        best_idx = None #control overfitting
        best_theta = None
        optimizer = optim.Adam([{'params': model.temp_theta[layer]}], lr=lr, weight_decay=wd)
        if layer >= len(max_iter):
            max_iteration = 10
        else:
            max_iteration = max_iter[layer]
        for j in range(max_iteration):
            epoch_loss = -100
            for X, y in train_loader:
                X = Variable(X, requires_grad = False)
                y = y.reshape(-1)
                R = get_residual(model, X, y, layer)
                R = (torch.ones(model.dic_size, len(R))*R).t()
                optimizer.zero_grad()
                out = model.forward(X, layer)
                loss = mse_loss(out, R)
                loss.backward(retain_graph=True)
                optimizer.step()  
                epoch_loss += loss.item()
            if (j+1) % 5 == 0:
                valid_CC, idx, curr_theta = model.eval_func(valid_loader, layer)
                if valid_CC > best_valCC:
                    best_valCC = valid_CC
                    best_idx = idx
                    del best_theta
                    best_theta = curr_theta
            if (j+1) % show_every ==0:
                print(f"Layer {layer}, Epoch {j+1}")
                print(valid_CC)
        fix_learnt_layer(model, best_idx, best_theta, layer)
        
        if old_residual is None:
            old_residual = np.sum(train_y*train_y)
        
        X, y = to_tensor(train_X, train_y, model.input_size)
        new_res = get_residual(model, X, y, layer).detach().numpy().reshape(-1)
        new_residual = np.sum(new_res*new_res)
        
        if old_residual - new_residual < -0.05: break
        project_pursuit[layer] = 1 - (new_residual/old_residual)
        old_residual = new_residual
        
    val_X, val_y = to_tensor(val_X, val_y, model.input_size)
    model.evaluate(project_pursuit, val_X, val_y)  #rank and eliminate layers
    test_corr = eval_func(model, test_loader, None)
    return model, test_corr


def run(monkey = 'A'):
    data = prepare_dataset_pattern(monkey)
    weight = np.load("/home/ziniuw/Tangdata/filter_512.npy")
    weight = weight.reshape(79,1,9,9)
    corr = []
    num = 0
    for i in data:
        print(i)
        model=CMPR(weight)
        res = train_one(model, data[i], show_every = 100)
        corr.append(res[1])
        print(res[1])
        num += 1
        if (num+1) % 20 == 0:
            np.save(f"CMPR_corr_{monkey}",np.asarray(corr))
    np.save(f"CMPR_corr_{monkey}",np.asarray(corr))
        
#run('A')
#run('E')
