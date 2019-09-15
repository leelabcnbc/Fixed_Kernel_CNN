import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision.utils
from skimage.io import imsave
def imshow(img, figsize, savefile):
    plt.close('all')
    plt.figure(figsize=figsize)
    npimg = img.numpy()
    img = np.transpose(npimg, (1, 2, 0))
    plt.imshow(img)
    plt.show()
    if savefile is not None:
        imsave(savefile, img)
    
    
def display_one_network(kernels, scale_each=True, normalize=True,figsize=(8,6), nrow=6, savefile=None):
    #kernels = model.first_layer.weight.data.cpu().detach()
    kernels = torch.from_numpy(kernels)
    imshow(torchvision.utils.make_grid(kernels, nrow=nrow, normalize=normalize, scale_each=scale_each),
          figsize=figsize, savefile=savefile)