import numpy as np
from torch import nn
import torch
import copy
from torch.autograd import Variable
from scipy.stats import pearsonr

def sparse_trim(model, FC_weight, X, y, sparse_l, gpu = True):
    X = torch.from_numpy(X)
    X = X.type(torch.FloatTensor)
    X = Variable(X, requires_grad = False)
    if gpu:
        X = X.cuda()
    new_FC_w = copy.deepcopy(FC_weight)
    temp = copy.deepcopy(FC_weight)
    temp = temp.reshape(-1)
    temp = np.sort(abs(temp))[::-1]
    sparse_val = temp[int(sparse_l*len(temp))]
    new_FC_w[abs(new_FC_w) < sparse_val] = 0
    #print(np.sum(abs(new_FC_w) >= sparse_val))
    
    
    t = torch.from_numpy(new_FC_w)
    t = t.type(torch.FloatTensor)
    t = nn.Parameter(t)
    model.fc.fc.weight = t
    if gpu:
        model = model.cuda()
    y_pred = model(X).data.cpu().numpy()
    corr = pearsonr(y_pred, y)[0]
    return corr[0], abs(new_FC_w) > sparse_val

def test_sparsity(model, X, y, corr):
    res = []
    w = model.fc.fc.weight.data.cpu().numpy()
    for s_l in [0.5, 0.2, 0.1, 0.05, 0.02]:
        cur_corr,idex = sparse_trim(model,w, X, y, s_l)
        if cur_corr > 0.85*corr:
            res_corr = cur_corr
            res = idex
        else:
            break
    t = torch.from_numpy(w)
    t = t.type(torch.FloatTensor)
    t = nn.Parameter(t)
    model.fc.fc.weight = t
    if len(res) == 0:
        res = idex
        res_corr = cur_corr
    return res_corr, res