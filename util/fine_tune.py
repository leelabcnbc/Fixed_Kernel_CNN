import numpy as np
import h5py
from torch import nn
import torch
from torch.autograd import Variable
from torch.nn.functional import mse_loss
from scipy.stats import pearsonr

class trim_FT(nn.Module):
    #this class has all dependency for fine-tuning a trimed model
    def __init__(self, model, param_idx):
        super(trim_FT, self).__init__()
        self.model = model
        param_idx = torch.from_numpy(param_idx.astype(int)).type(torch.FloatTensor)
        self.param_idx = param_idx
    
    def forward(self,input):
        if self.model.conv is not None:
            x = self.model.conv(input)
        else:
            x = input
        if self.model.reshape_conv:
            x = x.view(x.size(0), -1)
        mask = torch.autograd.Variable(self.param_idx, requires_grad = False)
        mask = self.model.fc.fc.weight.cuda() * mask.cuda()
        return torch.mm(x,mask.view(-1,1))
        

    def fine_tune_sparse(self, train_data, lr=1e-4, wd = 0, opt="Adam", max_epoch=100):
        train_X = train_data[0]
        train_y = train_data[1]
        if opt == "sgd":
            optimizer = torch.optim.SGD(self.model.fc.fc.parameters(), lr=lr, weight_decay = wd)
        else:
            optimizer = torch.optim.Adam(self.model.fc.fc.parameters(), lr=lr, weight_decay = wd)
        idx = np.arange(train_X.shape[0])
        for i in range(max_epoch):
            np.random.shuffle(idx)
            X = torch.from_numpy(train_X[idx])
            y = torch.from_numpy(train_y[idx]).type(torch.FloatTensor)
            X, y = Variable(X.cuda()), Variable(y.cuda())
            optimizer.zero_grad()
            outputs = self.forward(X)
            loss = mse_loss(outputs, y)
            loss.backward()
            optimizer.step()
        return self.model
            
    def test(self, test_data):
        X = torch.from_numpy(test_data[0])
        y = test_data[1]
        X = Variable(X.cuda())
        outputs = self.forward(X)
        outputs = outputs.data.cpu().numpy()
        return pearsonr(y, outputs)[0]
        
        
    

def fine_tune_all(model, train_data, lr=1e-4, wd = 0, opt="Adam", max_epoch=100):
    #fine-tune on all connections
    #here we use the whole dataset as a single batch
    train_X = train_data[0]
    train_y = train_data[1]
    if opt == "sgd":
        optimizer = torch.optim.SGD(model.fc.fc.parameters(), lr=lr, weight_decay = wd)
    else:
        optimizer = torch.optim.Adam(model.fc.fc.parameters(), lr=lr, weight_decay = wd)
    idx = np.arange(train_X.shape[0])
    for i in range(max_epoch):
        np.random.shuffle(idx)
        X = torch.from_numpy(train_X[idx])
        y = torch.from_numpy(train_y[idx]).type(torch.FloatTensor)
        X, y = Variable(X.cuda()), Variable(y.cuda())
        optimizer.zero_grad()
        outputs = model(X)
        loss = mse_loss(outputs, y)
        loss.backward()
        optimizer.step()
    return model

def test_FT(model, test_data):
    X = torch.from_numpy(test_data[0])
    y = test_data[1]
    X = Variable(X.cuda())
    y_pred = model(X).data.cpu().numpy()
    corr = pearsonr(y_pred, y)[0]
    return corr
    
    
    
    
