#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import copy
import importlib
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import ParameterGrid
from torch.nn.functional import mse_loss
import sys
sys.path.append("/home/ziniuw/Fixed_Kernel_CNN/models_8K")
from adam import Adam
import model
from sync_batchnorm import DataParallelWithCallback
from utils import make_dataloader, load_var_noise, make_dataloader, print_param, rmse, fev, pcc, plot_responses_fit, \
    plot_spatial_mask, plot_stimuli
from torch.autograd import Variable

def train(model, data_loader, optimizer):
    model.train()
    mse = 0
    batch = 0
    for data, target in data_loader:
        batch+=1
        optimizer.zero_grad()
        data, target = Variable(data.cuda()), Variable(target.cuda())
        loss = mse_loss(model(data), target)
        loss.backward()
        optimizer.step()
        mse+=float(loss.detach().cpu().numpy())
    return mse/batch
        


def test(net, data_loader, mode, output_best=False, show_best=False, save_best=False, show_all=False, save_all=False,
         show_vis=False):
    net.eval()
    xs, ys, y_preds = [], [], []
    for data, target in data_loader:
        xs.append(data.numpy())
        ys.append(target.numpy())
        data = Variable(data).cuda()
        y_preds.append(net(data).cpu().detach().data.numpy())
    x, y, y_pred = np.vstack(xs).transpose(0, 2, 3, 1).astype(np.int64), np.vstack(ys), np.vstack(y_preds)
    test_RMSE, test_PCC = rmse(y, y_pred), pcc(y, y_pred)
    #print("{0:s} RMSE = {1:.4f}\xB1{2:.4f}, {0:s} FEV = {3:.4f}\xB1{4:.4f}, {0:s} PCC = {5:.4f}\xB1{6:.4f}".format(
        #mode, np.mean(test_RMSE, axis=0), np.std(test_RMSE, axis=0), np.mean(test_FEV, axis=0),
        #np.std(test_FEV, axis=0), np.mean(test_PCC, axis=0), np.std(test_PCC, axis=0)))
    if output_best or show_best or save_best:
        I_max = np.argmax(test_PCC)
        best_fit = "{:d} (RMSE = {:.4f}, FEV = {:.4f}, PCC = {:.4f})".format(I_max + 1, test_RMSE[I_max],
                                                                             test_FEV[I_max], test_PCC[I_max])
    if output_best:
        print(f"Best {mode} Fitting Neuron = {best_fit}")
    if show_best or save_best:
        plots = list(zip([y[:, I_max], y_pred[:, I_max]], ["b-", "r-"], ["Real Response", "Predict Response"]))
        title = f"Response vs Stimulus for Neuron {best_fit}"
        plot_responses_fit(plots, title=title, show=show_best, save=save_best, path=f'./{model_name}_responses_fit.pdf')
        if show_vis:
            plot_spatial_mask(
                best_net.module.spatial_mask.weight.cpu().detach().numpy().squeeze()[I_max, :, :][np.newaxis],
                show=True, save=False, path="./visualization.pdf")
            stimuli_like = y[::-1, I_max].argsort(0)[:16]
            stimuli_like_pred = y_pred[::-1, I_max].argsort(0)[:16]
            plot_stimuli(x[stimuli_like, :, :, :], show=True, save=False, path="./visualization.pdf")
            plot_stimuli(x[stimuli_like_pred, :, :, :], show=True, save=False, path="./visualization.pdf")
    if show_all or save_all:
        for i in range(y.shape[1]):
            i_fit = "{:d} (RMSE = {:.4f}, FEV = {:.4f}, PCC = {:.4f})".format(i + 1, test_RMSE[i], test_FEV[i],
                                                                              test_PCC[i])
            plots = list(zip([y[:, i], y_pred[:, i]], ["b-", "r-"], ["Real Response", "Predict Response"]))
            title = f"Response vs Stimulus for Neuron {i_fit}"
            plot_responses_fit(plots, title=title, show=show_all, save=save_all,
                               path=f'./{model_name}_responses_fit_{i + 1}.pdf')
            if show_vis:
                plot_spatial_mask(
                    best_net.module.spatial_mask.weight.cpu().detach().numpy().squeeze()[i, :, :][np.newaxis],
                    show=True, save=False, path="./visualization.pdf")
                stimuli_like = y[::-1, i].argsort(0)[:16]
                stimuli_like_pred = y_pred[::-1, i].argsort(0)[:16]
                plot_stimuli(x[stimuli_like, :, :, :], show=True, save=False, path="./visualization.pdf")
                plot_stimuli(x[stimuli_like_pred, :, :, :], show=True, save=False, path="./visualization.pdf")
    return test_RMSE, test_PCC, y, y_pred

def init_CNN(model, weight):
    t = torch.from_numpy(weight)
    t = t.type(torch.FloatTensor)
    t = nn.Parameter(t.cuda())
    model.first_layer.weight = t
        
    
    #print(t.type())
    #print(model.conv_module_list[0].weight.size())
    return model
def train_one(model, data, param, weight, first_layer_no_learn = False, return_model = False):
    
    tic = time.time()
    batch_size = param['batch_size']
    lr = param['lr']
    l1 = param['l1']
    l2 = param['l2']
    max_epoch = param['max_epoch']
    seed = param['seed']
    
    if seed != -1:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    input_channel, input_size = data[0].shape[1], data[0].shape[2]
    output_size = data[1].shape[0]
    train_loader = make_dataloader(data[0], data[1], batch_size=batch_size)
    valid_loader = make_dataloader(data[2], data[3], batch_size=batch_size)
    test_loader = make_dataloader(data[4], data[5], batch_size=batch_size)
    best_valCC = 0
    best_model = None
    
    if first_layer_no_learn:
        model = init_CNN(model, weight)
        optimizer = Adam([{'params': model.conv.parameters()},
                {'params': model.fc.parameters()}], 
                          lr=lr, l1=l1, weight_decay=l2, amsgrad=True)
    else:
        optimizer = Adam(model.parameters(), lr=lr, l1=l1, weight_decay=l2, amsgrad=True)
    for epoch in range(max_epoch):
        print(f"===> Training Epoch {epoch + 1}:")
        train(model, train_loader, optimizer)
        valid_CC = test(model, valid_loader, 'Validation')[-1]
        print(valid_CC)
        if valid_CC > best_valCC:
            best_valCC = valid_CC
            del best_model
            best_model = copy.deepcopy(model)

    print("===========>")
    test_corr = test(best_model, test_loader, 'Test')[-1][0]
    print(test_corr)
    torch.cuda.empty_cache()
    print("Finished.")
    toc = time.time()
    print("Elapsed time is {:.6f} seconds.".format(toc - tic))
    if return_model:
        return best_model, test_corr, toc-tic
    else:
        return test_corr, toc-tic
