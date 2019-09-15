import numpy as np
import h5py
import sys, os
sys.path.append('/home/ziniuw/Fixed_Kernel_CNN/util')
from collections import OrderedDict
import matplotlib.pyplot as plt
from skimage import img_as_float
from contrast_norm import local_contrast_normalization
from skimage import img_as_float
from skimage.transform import downscale_local_mean

def get_all_data():
    #make sure to put data in correct path.
    f = h5py.File('/home/ziniuw/Tangdata/tang_stimulus.hdf5', 'r') #load stimulus file
    NS_image = f['NS_2250/gray']
    f_neuron = h5py.File('/home/ziniuw/Tangdata/tang_neural_data.hdf5', 'r') #load firing rate
    y = f_neuron['monkeyA/NS_2250/corrected_20160313/mean']
    X = img_as_float(NS_image[:,80-28:80+28,80-28:80+28]) #load the center 56x56 image
    y = np.asarray(y)
    X = local_contrast_normalization(X, True) #preprocess the data.
    X = X.reshape(2250,1,20,20)
    assert X.shape == (2250,1,20,20)
    assert y.shape == (2250,1225)
    return X, y

def prepare_dataset_NS(neu_reference):
    (X, y) = get_all_data()
    num = np.sum(y>0.2, axis=0)
    #get rid of neurons that do not have enough response to train and neurons that hasn't been recorded in pattern stimulus
    index_set = np.intersect1d(np.where(num>15)[0], np.where(num<100)[0])
    index_set = np.intersect1d(index_set, np.where(neu_reference!=0)[0])
    train_list = dict() #a small portion of data that used to train
    test_list = dict() #I use all data to test
    neu_ref = dict()
    for i in range(len(index_set)):
        indx = index_set[i]
        top_stimulus_indx = np.where(y[:,indx]>0.2)[0]
        low_stimulus_indx = np.where(y[:,indx]<0.2)[0]
        np.random.shuffle(low_stimulus_indx)
        stimulus_indx = np.union1d(top_stimulus_indx, low_stimulus_indx[0:len(top_stimulus_indx)])
        res_y = y[stimulus_indx, indx]
        res_X = X[stimulus_indx]
        res_y = res_y.reshape(len(res_y),1)
        train_list[indx] = generate_CV(res_X, res_y, fold = 0)
        test_list[indx] = X, y[:,indx].reshape(len(y),1)
        neu_ref[indx] = neu_reference[indx] -1 #the reference is loaded from matlab which starts at 1.
    return train_list, test_list, neu_ref

def generate_CV(X, y, fold = 5):
    #create a five-fold cross-validation for this dataset.
    assert X.shape[0] == len(y)
    if fold ==0:
        return X, y
    num_im = X.shape[0]
    k = int(num_im/fold)
    neu_list = []
    rand_vec = list(range(num_im))
    for i in range(fold):
        if i == fold -1:
            test_ind = rand_vec
        else:
            test_ind = np.random.choice(rand_vec, k, replace = False)
            rand_vec = set(rand_vec)
            rand_vec = rand_vec - set(test_ind)
            rand_vec = list(rand_vec)
        neu_list.append(X[test_ind])
        neu_list.append(y[test_ind])
        
    return neu_list

def show_top_stimuli(neuron_idx_this, X, y):
    # load actual neuron response
    actual_response_this_ALL = y[:, neuron_idx_this]
    assert actual_response_this_ALL.shape == (2250,)
    actual_response_sort_idx_ALL = np.argsort(actual_response_this_ALL)[::-1]
    # show top 20 stimuli, 10 x 2.
    X_top_this_ALL = X[actual_response_sort_idx_ALL[:20]]

    # use torchvision to get a 10 x 2 grid and show it.
    X_top_this_ALL = make_grid(FloatTensor(X_top_this_ALL)[:20], nrow=10, normalize=False, scale_each=False)
    X_top_this_ALL = np.transpose(X_top_this_ALL.numpy(), (1, 2, 0))
    plt.close('all')
    plt.imshow(X_top_this_ALL)
    plt.show()
    
def reorginize_data(data):
    res = dict()
    train_X = data[0]
    train_Y = data[1]
    val_X = data[2]
    val_Y = data[3]
    test_X = data[4]
    test_Y = data[5]
    num_neuron = train_Y.shape[-1]
    for i in range(num_neuron):
        res[i] = (train_X, train_Y[:,i].reshape(len(train_Y),1), val_X, val_Y[:,i].reshape(len(val_Y),1), test_X, test_Y[:,i].reshape(len(test_Y),1))
    return res
    
