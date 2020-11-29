import numpy as np
import h5py
import os
from matplotlib import image
import matplotlib.pyplot as plt
from skimage import img_as_float
from skimage.transform import downscale_local_mean

def load_dataset(monkey='A', seed=0):
    # access neural responses
    #f_neuron = h5py.File('/Users/eric/Desktop/Research/tang-image-data/tang_neural_data.hdf5', 'r')
    f_neuron = h5py.File('/home/ziniuw/Tangdata/tang_neural_data.hdf5', 'r') # load file
    # pattern stimuli
    if monkey == 'A':
        y = f_neuron['monkeyA/Shape_9500/corrected_20160313/mean']
        X = np.load("/home/ziniuw/Tangdata/normalized_image.npy")
    elif monkey == 'E':
        f = h5py.File('/home/ziniuw/Tangdata/tang_stimulus.hdf5', 'r')
        y = f_neuron['monkeyE2/Shape_4605/2017/mean']
        Pattern_Stimuli_E = f['Shape_4605']['original']
        X = img_as_float(Pattern_Stimuli_E[:,80-20:80+20,80-20:80+20])
    X = 1-downscale_local_mean(X, (1, 2, 2))[:,np.newaxis]
    y = np.asarray(y)
    return (X, y)

def generate_CV(X, y, fold = 5):
    assert X.shape[0] == len(y)
    num_im = X.shape[0]
    k = int(num_im/fold)
    neu_list = []
    rand_vec = list(range(num_im))
    test_ind = np.random.choice(rand_vec, k, replace = False)
    rand_vec = set(rand_vec)
    train_ind = rand_vec - set(test_ind)
    train_ind = list(train_ind)
    neu_list.append(X[train_ind])
    neu_list.append(y[train_ind])
    neu_list.append(X[test_ind])
    neu_list.append(y[test_ind])
    neu_list.append(X[test_ind])
    neu_list.append(y[test_ind])
    return neu_list

def prepare_dataset_pattern(monkey='A'):
    #prepare the dataset
    #I eliminate the bad neurons where significant response (more than 0.2)
    #is less than 50. Resulting dataset is roughly 75% of the original dataset.
    (X, y) = load_dataset(monkey)
    num = np.sum(y>0.2, axis=0) #eliminating bad neuron
    index_set = np.where(num>50)[0]
    data_list = dict()
    for i in range(len(index_set)):
        indx = index_set[i]
        res_y = y[:, indx]
        res_y = res_y.reshape(len(res_y),1)
        data_list[indx] = generate_CV(X, res_y)
    
    return data_list

def prepare_dataset_pattern_demo():
    X = np.zeros((9500, 40, 40))
    for i in range(9500):
        img = image.imread(f"./data/stimuli/{i+1}.png")
        #extract the center 40x40
        X[i,:,:] = np.asarray(img[80-20:80+20, 80-20:80+20])
    #downscale the image to 20x20
    X = 1-downscale_local_mean(X, (1, 2, 2))[:,np.newaxis]
    demo_neurons = ["neu75", "neu255", "neu381", "neu504"]
    data = dict()
    for neu in demo_neurons:
        y = np.load(f"./data/neurons/{neu}.npy")
        data[neu] = generate_CV(X, y)

    return data



