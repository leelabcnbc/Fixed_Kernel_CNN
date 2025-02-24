import numpy as np
import h5py
import os
from collections import OrderedDict
import matplotlib.pyplot as plt
import time
from skimage import img_as_float
from torch import nn, FloatTensor
import torch
from torchvision.utils import make_grid
import sys
import copy
import importlib
from torch.nn.functional import mse_loss
sys.path.append("/home/ziniuw/thesis-yimeng-v2")
from thesis_v2.data.prepared.yuanyuan_8k import get_data

sys.path.append('/home/ziniuw/Fixed_Kernel_CNN')
from util.get_data_NS import reorginize_data
from util.noisify import noise_pred
from models_8K.model import FKCNN_2l, FKCNN_3l

sys.path.append("/home/ziniuw/Fixed_Kernel_CNN/models_8K")
from adam import Adam
from utils import make_dataloader, load_var_noise, make_dataloader, print_param, rmse, fev, pcc, plot_responses_fit, \
    plot_spatial_mask, plot_stimuli
from training_8K import train, test

def init_CNN(model, weight):
    t = torch.from_numpy(weight)
    t = t.type(torch.FloatTensor)
    t = nn.Parameter(t.cuda())
    model.first_layer.weight = t
    return model

def train_one(model, data, param, weight, first_layer_no_learn = False, show_every=1, return_model = False):
    
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
    train_loader = make_dataloader(data[0], data[1], batch_size=batch_size, is_train=True)
    valid_loader = make_dataloader(data[2], data[3], batch_size=batch_size, is_train=False)
    test_loader = make_dataloader(data[4], data[5], batch_size=batch_size, is_train=False)
    best_valCC = 0
    best_model = None
    
    if first_layer_no_learn:
        model = init_CNN(model, weight)
        optimizer = Adam([{'params': model.conv.parameters()},
                {'params': model.fc.parameters()}], 
                          lr=lr, l1=l1, weight_decay=l2, amsgrad=True)
    else:
        optimizer = Adam(model.parameters(), lr=lr, l1=l1, weight_decay=l2, amsgrad=True)
    loss = []
    val_corr = []
    for epoch in range(max_epoch):
        if (epoch + 1) % show_every == 0:
            print(f"Epoch {epoch + 1}:")
        loss.append(train(model, train_loader, optimizer))
        valid_CC = test(model, valid_loader, 'Validation')[1]
        valid_CC = sum(valid_CC)/len(valid_CC)
        val_corr.append(valid_CC)
        if (epoch + 1) % show_every == 0:
            print(valid_CC)
        if valid_CC > best_valCC:
            #recover the best model by validation set
            best_valCC = valid_CC
            del best_model
            best_model = copy.deepcopy(model)

    print("Done Training")
    res = test(best_model, test_loader, 'Test')
    test_corr = res[1]
    pred = res[-1]
    test_corr = sum(test_corr)/len(test_corr)
    print(test_corr)
    torch.cuda.empty_cache()
    print("Finished.")
    if return_model:
        return best_model, test_corr, loss, val_corr,pred
    else:
        return test_coic, loss, val_corr, pred
    
    
def main():
    data = get_data('a', 256, 128, ['042318'], read_only=False, scale=0.5)
    data_Sep = reorginize_data(data) #train model one at a time
    weight = np.load("/home/ziniuw/Tangdata/filter_79.npy")
    gabor = np.load("/home/ziniuw/Tangdata/gabor.npy")
    gabor = gabor.reshape(24,1,10,10)
    weight = weight.reshape(79,1,9,9)
    weight = weight[[0,3,5,8,9,11,14,17,18,20,23,25,27,32,36,37,40,44,53,57,58,64,65,74],:,:,:]
    #fit one neuron at a time and document the performance, convergence and noise test
    model_arch = {'l1_c': 24, 'l1_k': 9, 'p1_k':3, 'p1_s':2, 'l2_c':16, 'l2_k':5, 'l2_s':1, 'p2_k':3, 'p2_s':2,
             'l3_c':16, 'l3_k':5, 'l3_s':1, 'p3_k':3, 'p3_s':2}
    optm_param = {'batch_size': 64,'lr': 1e-4, 'l1': 3e-5, 'l2': 1e-5, 'max_epoch': 150, 'seed': 1}
    n = len(data_Sep)
    FK_corr = np.zeros(n)
    CNN_corr = np.zeros(n)
    G_corr = np.zeros(n)
    FK_val_corr = np.zeros((n,150))
    CNN_val_corr = np.zeros((n,150))
    G_val_corr = np.zeros((n,150))
    FK_noise = np.zeros((n,3))
    CNN_noise = np.zeros((n,3))
    G_noise = np.zeros((n,3))
    
    for i in range(len(data_Sep)):
        print(i)
        print('FKCNN:')
        model = FKCNN_3l(model_arch, input_size=128)
        res = train_one(model.cuda(), data_Sep[i], optm_param, weight, show_every=1000, first_layer_no_learn=True, return_model=True)
        best_model = res[0]
        FK_corr[i] = res[1]
        FK_val_corr[i,:] = np.asarray(res[3])
        FK_noise[i,:] = np.asarray(noise_pred(best_model, data_Sep[i]))
        print('CNN:')
        model = FKCNN_3l(model_arch, input_size=128)
        res = train_one(model.cuda(), data_Sep[i], optm_param, None, show_every=1000, first_layer_no_learn=False, return_model=True)
        best_model = res[0]
        CNN_corr[i] = res[1]
        CNN_val_corr[i,:] = np.asarray(res[3])
        CNN_noise[i,:] = np.asarray(noise_pred(best_model, data_Sep[i]))
        print('GaborCNN:')
        model = FKCNN_3l(model_arch, input_size=128)
        res = train_one(model.cuda(), data_Sep[i], optm_param, gabor, show_every=1000, first_layer_no_learn=True, return_model=True)
        best_model = res[0]
        G_corr[i] = res[1]
        G_val_corr[i,:] = np.asarray(res[3])
        G_noise[i,:] = np.asarray(noise_pred(best_model, data_Sep[i]))
    
    
    np.save('FK_corr',FK_corr)
    np.save('CNN_corr',CNN_corr)
    np.save('G_corr',G_corr)
    np.save('FK_val_corr',FK_val_corr)
    np.save('CNN_val_corr',CNN_val_corr)
    np.save('G_val_corr',G_val_corr)
    np.save('FK_noise', FK_noise)
    np.save('CNN_noise', CNN_noise)
    np.save('G_noise', G_noise)
main()
