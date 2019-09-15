import numpy as np
import os
import torch
from scipy.stats import pearsonr
import copy
from torch.autograd import Variable


def noisy(noise_typ,image, p = 1, amount = 0.2):
    ch,_,row,col= image.shape
    if noise_typ == "gauss":
        mean = 0
        var = 0.1
        sigma = var**0.5
        gauss = p*np.random.normal(mean,sigma,(ch,row,col))
        gauss = gauss.reshape(ch,1,row,col)
        noisy = image + gauss
        ar = ch
        ac = row*col
        a = np.reshape(noisy, (ar, ac))
        noisy = (a-np.amin(a, axis = 1, keepdims = True))/np.amax(a, axis = 1, keepdims = True)
        noisy = np.reshape(noisy, (ch,1, row,col))
        return noisy
    elif noise_typ == "s&p":
        s_vs_p = 0.2
        image = image.reshape((ch,row,col))
        out = np.copy(image)
        
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                for i in image.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                for i in image.shape]
        out[coords] = 0
        out = np.reshape(out, (ch,1, row,col))
        return out
    
    
def noise_pred(model,dataset,gpu=True):
    img = dataset[-2]
    y = dataset[-1]
    X = copy.deepcopy(img)
    corr = np.zeros(3)
    i=0
    for level in [0.1,0.2,0.3]:
        noise_X = noisy("s&p", X, amount = level)
        noise_X = torch.from_numpy(noise_X)
        noise_X = noise_X.type(torch.FloatTensor)
        if gpu:
            noise_X=noise_X.cuda()
        noise_X = Variable(noise_X, requires_grad = False)
        y_hat = model(noise_X).data.cpu().numpy()
        corr[i] = pearsonr(y_hat, y)[0]
        i+=1
    print(corr)
    return corr
    
    




    
