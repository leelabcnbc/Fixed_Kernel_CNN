#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
from scipy import io
from scipy.stats.stats import pearsonr
from torch.utils.data import TensorDataset, DataLoader



def load_var_noise(data_file, var_noise):
    return io.loadmat(data_file, variable_names=[var_noise])[var_noise]


def make_dataloader(X, Y, is_train=True, batch_size=64):
    dataset = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(Y).float())
    if is_train:
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)
    else:
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def print_param(net):
    max_len_name = 0
    n_param = 0
    for name, param in net.named_parameters():
        if param.requires_grad:
            if len(name) > max_len_name:
                max_len_name = len(name)
            n_param += np.prod(param.size())
    for name, param in net.named_parameters():
        if param.requires_grad:
            print("{:{:d}s}    {:s}".format(name, max_len_name, str(param.size())))
    print(f"Number of Parameters = {n_param}")


def rmse(y, y_pred):
    return np.sqrt(np.mean((y - y_pred) ** 2, axis=0))


def fev(y, y_pred, var_noise):
    RSS = np.sum((y - y_pred) ** 2, axis=0)
    TSS = np.sum((y - np.mean(y, axis=0)) ** 2, axis=0)
    NSS = (y.shape[0] - 1) * np.squeeze(var_noise)
    return 1 - (RSS - NSS) / (TSS - NSS)


def pcc(y, y_pred):
    return np.array([pearsonr(y[:, i], y_pred[:, i])[0] for i in range(np.shape(y)[1])])


def plot_responses_fit(plots, title="Response vs Stimulus", show=True, save=False, path='./responses_fit.pdf'):
    responses, styles, labels = zip(*plots)
    n = len(responses[0])
    x = range(1, n + 1)
    if not show:
        import matplotlib
        matplotlib.use('agg', warn=False)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    I_sort = np.flip(np.argsort(responses[0]), axis=0)
    for i in range(len(responses)):
        plt.plot(x, responses[i][I_sort], styles[i], label=labels[i])
    plt.legend()
    plt.xlim(0, n)
    if n <= 10:
        x_ticks = range(0, n + 1)
    else:
        x_ticks = range(0, n + 1, int(np.ceil(n / 10)))
    plt.xticks(x_ticks)
    plt.xlabel("Stimulus")
    plt.ylabel("Response")
    plt.title(title)
    if save:
        plt.savefig(path, pad_inches=0)
    if show:
        plt.show()
    else:
        plt.close()


def plot_spatial_mask(data, show=True, save=False, path="./visualization.pdf"):
    n = data.shape[0]
    width = data.shape[2]
    n_cols = max(min(640 // width, 8), 8)
    n_rows = int(np.ceil(n / n_cols))
    if not show:
        import matplotlib
        matplotlib.use('agg', warn=False)
    import matplotlib.pyplot as plt
    f, ax = plt.subplots(n_rows, n_cols, figsize=(16, 16 * n_rows / n_cols))
    for i, x in zip(range(n_rows * n_cols), ax.ravel()):
        x.axis('off')
        if i < n:
            x.imshow(data[i, :, :], cmap=plt.get_cmap('gray'))
            if n > 1:
                x.set_title(f"Neuron {i + 1}")
    if save:
        plt.savefig(path, pad_inches=0)
    if show:
        plt.show()
    else:
        plt.close()


def plot_stimuli(data, show=True, save=False, path="./visualization.pdf"):
    n = data.shape[0]
    width = data.shape[2]
    n_cols = max(min(640 // width, 8), 8)
    n_rows = int(np.ceil(n / n_cols))
    if not show:
        import matplotlib
        matplotlib.use('agg', warn=False)
    import matplotlib.pyplot as plt
    f, ax = plt.subplots(n_rows, n_cols, figsize=(16, 16 * n_rows / n_cols))
    for i, x in zip(range(n_rows * n_cols), ax.ravel()):
        x.axis('off')
        if i < n:
            x.imshow(data[i, :, :, :].squeeze(), cmap=plt.get_cmap('gray'))
            if n > 1:
                x.set_title(f"Stimuli {i + 1}")
    if save:
        plt.savefig(path, pad_inches=0)
    if show:
        plt.show()
    else:
        plt.close()
