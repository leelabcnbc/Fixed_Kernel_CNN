import numpy as np
from torch import nn
import torch
from tang_jcompneuro.cnn import CNN
from tang_jcompneuro.model_fitting_cnn import (opt_configs_to_explore, models_to_train, opt_configs_to_explore_2l,
                                               init_config_to_use_fn, train_one_case, save_one_model)

from util.fine_tune import fine_tune_all, test_FT

def init_CNN(model:CNN, weight):
    t = torch.from_numpy(weight)
    t = t.type(torch.FloatTensor)
    t = nn.Parameter(t)
    model.conv_module_list[0].weight = t
        
    
    #print(t.type())
    #print(model.conv_module_list[0].weight.size())
    return model
def train_one_model(neuron_idx, data, weight, opt_config_to_use = '1e-2L2_1e-2L2_adam002_mse',
                   model_config='b.24', first_layer_nolearning=False, batch_size = 128):
    # change num_channel to 9 for 9 channel CNN, etc.
    # print `models_to_train` for more options.
    arch_config = models_to_train[model_config]
    
    # load datasets splitted in train/val/test.
    
    (X_train_, y_train_,X_val_, y_val_, X_test_, y_test_) = data
    
    datasets_local = (X_train_, y_train_,X_val_, y_val_, X_test_, y_test_)
    # init CNN.
    model = CNN(arch_config, init_config_to_use_fn(), mean_response=datasets_local[1].mean(axis=0),
                    seed=0,
                    input_size=20
                    )
    if first_layer_nolearning:
        #This is an indication of whether we use the fixed first layer.
        model = init_CNN(model, weight)
        
    val_cc, y_test_hat, new_cc = train_one_case(model.cuda(), datasets_local,
                                                      opt_configs_to_explore[opt_config_to_use],
                                                      seed=2, show_every=1000,
                                                      return_val_perf=True,
                                                      max_epoch=20000,first_layer_nolearning = first_layer_nolearning, batch_size = batch_size)
    
    print(val_cc)
    
    return model,val_cc

def train_one_NS(key, data_train, data_test, weight, num_channel = 24, lr = 1e-4, wd = 0, opt="Adam", max_epoch = 1000):
    #training NS data with top stimulus, then we must enforce batch_size equals to the whole training size. 
    arch_config = models_to_train[f'b.{num_channel}']
    model = CNN(arch_config, init_config_to_use_fn(), mean_response=data_train[1].mean(axis=0),
                    seed=0,
                    input_size=20
                    )
    model.cuda()
    model = fine_tune_all(model, data_train, lr = lr, wd = wd, opt=opt, max_epoch = max_epoch)
    cc = test_FT(model, data_test)
    
    return cc
    
    
def train_one_model_8K(data, weight, arch_config, opt_config_to_use = '1e-2L2_1e-2L2_adam002_mse',
                   first_layer_nolearning=False, batch_size = 128):
    # change num_channel to 9 for 9 channel CNN, etc.
    # print `models_to_train` for more options.
    
    # init CNN.
    model = CNN(arch_config, init_config_to_use_fn(), mean_response=data[1].mean(axis=0),
                    seed=0,
                    input_size=64
                    )
    if first_layer_nolearning:
        #This is an indication of whether we use the fixed first layer.
        model = init_CNN(model, weight)
        
    val_cc, y_test_hat, new_cc = train_one_case(model.cuda(), data,
                                                      opt_configs_to_explore_2l[opt_config_to_use],
                                                      seed=2, show_every=1000,
                                                      return_val_perf=True,
                                                      max_epoch=8000,first_layer_nolearning = first_layer_nolearning, batch_size = batch_size)
    
    print(new_cc)
    
    return model,new_cc

    
    