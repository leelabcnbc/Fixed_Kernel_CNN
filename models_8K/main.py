#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import ast
import copy
import importlib
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import ParameterGrid

import model
from adam import Adam
from sync_batchnorm import DataParallelWithCallback
from utils import load_dataset, load_var_noise, make_dataloader, print_param, rmse, fev, pcc, plot_responses_fit, \
    plot_spatial_mask, plot_stimuli


def train(net, data_loader):
    net.train()
    n_train = len(data_loader.dataset)
    n_trained = 0
    running_loss = []
    percent = 20
    for data, target in data_loader:
        optimizer.zero_grad()
        loss = criterion(net(data.to(device)), target.to(device))
        loss.backward()
        optimizer.step()
        n_trained += len(data)
        running_loss.append(loss.item())
        while n_trained / n_train >= percent / 100:
            print(f"---{percent}%", end='', flush=True)
            percent += 20
    print("    Running RMSE = {:.4f}".format(np.sqrt(np.mean(running_loss, axis=0))))


def test(net, data_loader, mode, output_best=False, show_best=False, save_best=False, show_all=False, save_all=False,
         show_vis=False):
    net.eval()
    xs, ys, y_preds = [], [], []
    for data, target in data_loader:
        xs.append(data.numpy())
        ys.append(target.numpy())
        with torch.no_grad():
            y_preds.append(net(data.to(device)).cpu().detach().numpy())
    x, y, y_pred = np.vstack(xs).transpose(0, 2, 3, 1).astype(np.int64), np.vstack(ys), np.vstack(y_preds)
    test_RMSE, test_FEV, test_PCC = rmse(y, y_pred), fev(y, y_pred, var_noise), pcc(y, y_pred)
    print("{0:s} RMSE = {1:.4f}\xB1{2:.4f}, {0:s} FEV = {3:.4f}\xB1{4:.4f}, {0:s} PCC = {5:.4f}\xB1{6:.4f}".format(
        mode, np.mean(test_RMSE, axis=0), np.std(test_RMSE, axis=0), np.mean(test_FEV, axis=0),
        np.std(test_FEV, axis=0), np.mean(test_PCC, axis=0), np.std(test_PCC, axis=0)))
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
    return test_RMSE, test_FEV, test_PCC


def parse_arguments():
    # Command-line flags are defined here.
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model-name', type=str, required=True, help="name of the model")
    parser.add_argument('-d', '--data-file', type=str, required=True, help="path to the data file")
    parser.add_argument('-n', '--max-n-epoch', type=int, required=True, help="maximum number of epochs to train on")
    parser.add_argument('-p', '--param-grid', type=str, default=None, help="grid of hyper-parameters")
    parser.add_argument('-s', '--seed', type=int, default=0, help="random seed")
    parser.add_argument('--use-cuda', type=int, choices={0, 1}, default=1, help="whether to use CUDA if available")
    parser.add_argument('--save-model', type=int, choices={0, 1}, default=0, help="whether to save the best model")
    return parser.parse_args()


if __name__ == '__main__':
    tic = time.time()
    # Parse command-line arguments.
    args = parse_arguments()
    model_name, data_file, max_n_epoch = args.model_name, args.data_file, args.max_n_epoch
    param_grid, seed, use_cuda, save_model = args.param_grid, args.seed, args.use_cuda, args.save_model
    if param_grid is not None:
        param_grid = ast.literal_eval(param_grid)
        for k, v in param_grid.items():
            if not isinstance(v, list):
                param_grid[k] = [v]
    elif model_name == 'cnn0' or model_name == 'cnn1':
        param_grid = {'batch_size': [64, 128, 256], 'lr': [1e-3], 'l1': [0, 1e-4, 1e-3], 'l2': [0, 1e-4, 1e-3]}
    elif model_name == 'vgg19' or model_name == 'cnn2' or model_name == 'cnn3':
        param_grid = {'batch_size': [64, 128, 256], 'lr': [1e-3], 'l1_channel': [0, 1e-4, 1e-3, 1e-2],
                      'l1_spatial': [0, 1e-4, 1e-3, 1e-2], 'l2': [0]}
    device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.deterministic = True
    train_set = load_dataset(data_file, 'data_train', 'labels_train')
    valid_set = load_dataset(data_file, 'data_valid', 'labels_valid')
    test_set = load_dataset(data_file, 'data_test', 'labels_test')
    var_noise = load_var_noise(data_file, 'var_noise')
    input_channel, input_size = train_set.tensors[0].size(1), train_set.tensors[0].size(2)
    output_size = train_set.tensors[1].size(1)
    valid_loader = make_dataloader(valid_set, 'valid_set', batch_size=max(param_grid['batch_size']))
    test_loader = make_dataloader(test_set, 'test_set', batch_size=max(param_grid['batch_size']))
    criterion = nn.MSELoss()
    best_valid_RMSE = np.full(1, np.inf)
    for grid in ParameterGrid(param_grid):
        print(f"===> Hyper-parameters = {grid}:")
        # random.seed(seed)
        # numpy.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        importlib.reload(model)
        net = getattr(model, model_name.upper())(input_channel=input_channel, input_size=input_size,
                                                 output_size=output_size).to(device)
        if use_cuda:
            net = DataParallelWithCallback(net)
        print(f"===> Model:\n{list(net.modules())[0]}")
        print_param(net)
        if model_name == 'cnn0' or model_name == 'cnn1':
            optimizer = Adam(net.parameters(), lr=grid['lr'], l1=grid['l1'], weight_decay=grid['l2'], amsgrad=True)
        elif model_name == 'vgg19' or model_name == 'cnn2' or model_name == 'cnn3':
            optimizer = Adam([
                {'params': iter(param for name, param in net.named_parameters() if 'channel_mask' in name),
                 'l1': grid['l1_channel']},
                {'params': iter(param for name, param in net.named_parameters() if 'spatial_mask' in name),
                 'l1': grid['l1_spatial']},
                {'params': iter(param for name, param in net.named_parameters() if
                                'mask' not in name and 'bn' not in name and 'weight' in name),
                 'weight_decay': grid['l2']},
                {'params': iter(param for name, param in net.named_parameters() if 'bn' in name or 'bias' in name)}
            ], lr=grid['lr'], amsgrad=True)
        train_loader = make_dataloader(train_set, 'train_set', batch_size=grid['batch_size'])
        for epoch in range(max_n_epoch):
            print(f"===> Training Epoch {epoch + 1}:")
            train(net, train_loader)
            valid_RMSE = test(net, valid_loader, 'Validation')[0]
            if np.mean(valid_RMSE, axis=0) < np.mean(best_valid_RMSE, axis=0):
                best_net = copy.deepcopy(net)
                best_grid = grid
                best_n_epoch = epoch + 1
                best_valid_RMSE = valid_RMSE
    print("===========>")
    print(f"Best Model:\n{list(best_net.modules())[0]}")
    print_param(best_net)
    print(f"Best Hyper-parameters = {best_grid}")
    print(f"Best Number of Epoch = {best_n_epoch}")
    test(best_net, valid_loader, 'Best Validation')
    test(best_net, test_loader, 'Test', output_best=True)
    if save_model:
        torch.save(best_net, f'./{model_name}_best_model.pkl')
    torch.cuda.empty_cache()
    print("Finished.")
    toc = time.time()
    print("Elapsed time is {:.6f} seconds.".format(toc - tic))
