"""load neural data and stimuli

main improvements over previous code are as follows.

1. unify subset selection.
"""

import os.path

import h5py
import skimage
import numpy as np
from skimage.transform import downscale_local_mean, rescale

from . import dir_dictionary
from .stimulus_classification import get_subset_slice
from .data_preprocessing import split_file_name_gen

_global_io_dict = {
    'image': os.path.join(dir_dictionary['datasets'], 'tang_stimulus.hdf5'),
    'MkA_Shape': os.path.join(dir_dictionary['datasets'], 'monkeyA.hdf5'),
    'MkE2_Shape': os.path.join(dir_dictionary['datasets'], 'monkeyE2.hdf5'),
}


def _image_dataset_dict_constructor(*, image_dataset_path, color, trans, gray_version=None):
    assert len(trans) == 2
    assert not color and gray_version is None, 'invalid for pattern stimulus'
    return {
        'image_dataset_path': image_dataset_path,
        'color': color,  # colorful or not.
        'trans': trans,  # ax + b
        'gray_version': gray_version,  # this is only meaningful for colorful data sets, such as NS_2250.
        'neural_datasets': []  # for reverse check.
    }


image_dataset_dict = {
    # another place about info on these data sets are in `stimulus_classification.py`.
    # remember to change them both for consistency.
    'Shape_9500': _image_dataset_dict_constructor(image_dataset_path='Shape_9500/original', color=False, trans=(-1, 1)),
    'Shape_4605': _image_dataset_dict_constructor(image_dataset_path='Shape_4605/original', color=False, trans=(-1, 1)),
}


def _neural_dataset_dict_constructor(*, neural_dataset_key, monkey, image_dataset_key,
                                     neural_dataset_path, trial_count):
    image_dataset_dict[image_dataset_key]['neural_datasets'].append(neural_dataset_key)
    return (neural_dataset_key, {
        'monkey': monkey,
        'image_dataset_key': image_dataset_key,
        'neural_dataset_path': neural_dataset_path,
        # first `trial_count` trials in `all` (after some filling in) will be used for estimating ccmax.
        'trial_count': trial_count,
    })


# for trial count number, check
# <https://github.com/leelabcnbc/tang-paper-2017/blob/master/population_analysis/noise_correlation_analysis.ipynb>
# main criterion for choosing trial_count is not discarding existing data.

neural_dataset_dict = dict([
    _neural_dataset_dict_constructor(
        neural_dataset_key='MkA_Shape',
        monkey='A',
        image_dataset_key='Shape_9500',
        neural_dataset_path='monkeyA/Shape_9500/corrected_20160313',
        trial_count=5,
    ),
    _neural_dataset_dict_constructor(
        neural_dataset_key='MkE2_Shape',
        monkey='E2',
        image_dataset_key='Shape_4605',
        neural_dataset_path='monkeyE2/Shape_4605/2017',
        trial_count=6,
    )
])


def _print_neural_dataset_info(neural_dataset_key):
    print(neural_dataset_dict[neural_dataset_key])
    # then print number of neurons.
    # then nubmer of trials.
    # then nubmer of trials to use.
    num_im, num_trial, num_neuron = get_neural_dataset_shape(neural_dataset_key, use_mean=False)
    print(f'num_im {num_im}, num_trial {num_trial}, num_neuron {num_neuron}')


def _print_image_dataset_info(image_dataset_key):
    print(image_dataset_dict[image_dataset_key])
    # then print number of neurons.
    # then nubmer of trials.
    # then nubmer of trials to use.
    image_shape = get_image_dataset_shape(image_dataset_key)
    print(f'image shape {image_shape}')


def get_neural_dataset_shape(neural_dataset_key, use_mean=True):
    with h5py.File(_global_io_dict['neural'], 'r') as f:
        this_dataset = f[
            neural_dataset_dict[neural_dataset_key]['neural_dataset_path'] + ('/mean' if use_mean else '/all')]
        return this_dataset.shape


def get_image_dataset_shape(image_dataset_key):
    with h5py.File(_global_io_dict['image'], 'r') as f:
        this_dataset = f[image_dataset_dict[image_dataset_key]['image_dataset_path']]
        return this_dataset.shape


def _load_image_dataset_normalize_cnn_format(images, color):
    if not color:
        assert images.ndim == 3
        # for inverting the color, for Shape dataset. This is like what people usually do in MNIST.
        images = images[:, np.newaxis]
    else:
        assert images.ndim == 4
        images = np.transpose(images, (0, 3, 1, 2))
    return images


def _load_image_dataset_normalize_trans(images, trans):
    trans_a, trans_b = trans
    return images * trans_a + trans_b


def load_image_dataset(image_dataset_key, patch_size=40, color=False, trans=False, scale=None,
                       normalize_cnn_format=True, subset=None, raw=False,
                       subtract_mean=False, legacy_rescale=False):
    # combine original stuff with some of load_data_for_cnn_fitting
    # always return in [0,1] format, followed by some trans
    # first, get shape.
    gray_version_key = image_dataset_dict[image_dataset_key]['gray_version']
    is_original_dataset_color = image_dataset_dict[image_dataset_key]['color']
    assert not color, "that is old behavior"
    if color:
        assert is_original_dataset_color
        key_to_use = image_dataset_key
    else:
        if is_original_dataset_color:
            assert gray_version_key is not None
            key_to_use = gray_version_key
        else:
            key_to_use = image_dataset_key

    num_im, height, width = get_image_dataset_shape(key_to_use)
    assert height == width  # this is true for our datasets.
    assert height >= patch_size
    # if color:
    #     assert channel == [3]
    # else:

    crop_slice = slice(height // 2 - patch_size // 2, height // 2 + patch_size // 2)

    # get subset slice
    subset_slice = get_subset_slice(key_to_use, subset)

    with h5py.File(_global_io_dict['image'], 'r') as f:
        this_dataset = f[image_dataset_dict[image_dataset_key]['image_dataset_path']]
        image_data = this_dataset[subset_slice, crop_slice, crop_slice]

    # normalize data
    if not raw:
        image_data = skimage.img_as_float(image_data)
    assert image_data.ndim == 3
    # rescale
    if scale is not None:
        if not legacy_rescale:
            # well. probably whether using pooling, which has less aliasing in theory,
            # doesn't matter, as here we are using pattern images.
            # for simplicity, just stick to what I have before.

            # well, let's go that way.
            new_height, new_width = int(np.round(height * scale)), int(np.round(width * scale))
            scale_factor = height // new_height
            assert scale_factor * new_height == height and scale_factor * new_width == width
            image_data = downscale_local_mean(image_data, (1, scale_factor, scale_factor))
        else:
            image_data = np.asarray([rescale(im, scale=scale, mode='edge', preserve_range=True) for im in image_data])

    # normalize shape
    if normalize_cnn_format:
        image_data = _load_image_dataset_normalize_cnn_format(image_data, color)

    if trans:
        image_data = _load_image_dataset_normalize_trans(image_data, image_dataset_dict[image_dataset_key]['trans'])

    if subtract_mean:
        assert image_data.ndim >= 2
        # this is primarily for Gabor models, to compensate for DC in filters.
        image_data -= image_data.mean(axis=tuple(range(1, image_data.ndim)), keepdims=True)

    return image_data


def load_neural_dataset(neural_dataset_key, use_mean=True, return_positive=False,
                        return_positive_bias=0.000001, subset=None):
    neural_group_to_study = neural_dataset_dict[neural_dataset_key]['neural_dataset_path']
    image_dataset_key = neural_dataset_dict[neural_dataset_key]['image_dataset_key']

    subset_slice = get_subset_slice(image_dataset_key, subset)
    
    with h5py.File(_global_io_dict[neural_dataset_key], 'r') as f_neural_data:
        dataset_to_check = neural_group_to_study + ('/mean' if use_mean else '/all')
        assert dataset_to_check in f_neural_data
        y = f_neural_data[dataset_to_check][subset_slice, ...]

    if return_positive:
        # for fitting where positive number is needed.
        assert use_mean
        y -= np.min(y, axis=0, keepdims=True)
        # tried different bias. 0.01, 0.000001, 0.1, 1. seems smaller gives better performance.
        y += return_positive_bias

    return y


def get_num_neuron_all_datasets():
    return {dataset: get_neural_dataset_shape(dataset)[-1] for dataset in neural_dataset_dict}


def get_num_im_all_datasets():
    return {dataset: get_image_dataset_shape(dataset)[0] for dataset in image_dataset_dict}


def load_split_dataset(dataset_key, subset, with_val, neuron_idx_slice, *,
                       percentage=100, seed=0, last_val=True, suffix=None,
                       top_dim=None, subtract_mean=False):
    assert isinstance(with_val, bool)
    val_part = 'with_val' if with_val else 'without_val'
    if isinstance(neuron_idx_slice, int):
        neuron_idx_slice = slice(neuron_idx_slice, neuron_idx_slice + 1)
    assert isinstance(neuron_idx_slice, slice)

    datafile_x = split_file_name_gen(suffix)
    datafile_y = split_file_name_gen()
    # when datafile_x == datafile_y, it's fine.
    # https://github.com/h5py/h5py/issues/332
    with h5py.File(datafile_x, 'r') as f_x, h5py.File(datafile_y, 'r') as f_y:
        g_this_x = f_x[f'/{dataset_key}/{subset}/{val_part}/{percentage}/{seed}']
        g_this_y = f_y[f'/{dataset_key}/{subset}/{val_part}/{percentage}/{seed}']
        # load X_train/test/val
        # load y_train/test/val
        X_train = g_this_x['train/X'][...]
        y_train = g_this_y['train/y'][:, neuron_idx_slice]
        X_test = g_this_x['test/X'][...]
        y_test = g_this_y['test/y'][:, neuron_idx_slice]

        X_val = g_this_x['val/X'][...] if 'val' in g_this_x else None
        y_val = g_this_y['val/y'][:, neuron_idx_slice] if 'val' in g_this_y else None

        if top_dim is not None:
            # extracting top K dims. for disassociating the effect of conv
            # and thresholding.
            assert X_train.ndim == 2 and X_train.shape[1] >= top_dim
            X_train = X_train[:, :top_dim]

            assert X_test.ndim == 2 and X_test.shape[1] >= top_dim
            X_test = X_test[:, :top_dim]

            # this should be fine, as I will only use this stuff in my CNN experiments,
            # where I always have the validation set.

            assert X_val.ndim == 2 and X_val.shape[1] >= top_dim
            X_val = X_val[:, :top_dim]

        if subtract_mean:
            # this is for gabor
            assert last_val and not with_val
            X_train -= X_train.mean(axis=tuple(range(1, X_train.ndim)), keepdims=True)
            X_test -= X_test.mean(axis=tuple(range(1, X_test.ndim)), keepdims=True)

        if last_val:
            result = (X_train, y_train, X_test, y_test, X_val, y_val)
        else:
            result = (X_train, y_train, X_val, y_val, X_test, y_test)

    return result


def load_split_dataset_idx(dataset_key, subset, with_val, *,
                           percentage=100, seed=0, last_val=True):
    assert isinstance(with_val, bool)
    val_part = 'with_val' if with_val else 'without_val'
    datafile_y = split_file_name_gen()
    # when datafile_x == datafile_y, it's fine.
    # https://github.com/h5py/h5py/issues/332
    with h5py.File(datafile_y, 'r') as f_y:
        g_this_y = f_y[f'/{dataset_key}/{subset}/{val_part}/{percentage}/{seed}']
        # load X_train/test/val
        # load y_train/test/val
        y_train_idx = g_this_y['train'].attrs['index']
        y_test_idx = g_this_y['test'].attrs['index']
        y_val_idx = g_this_y['val'].attrs['index'] if 'val' in g_this_y else None

        if last_val:
            result = (y_train_idx, y_test_idx, y_val_idx)
        else:
            result = (y_train_idx, y_val_idx, y_test_idx)

    return result


def load_split_dataset_pretrained_pca_params(dataset_key, subset, *,
                                             suffix, seed=0):
    assert suffix is not None
    datafile_x = split_file_name_gen(suffix)
    val_part = 'without_val'
    with h5py.File(datafile_x, 'r') as f_y:
        g_this_x = f_y[f'/{dataset_key}/{subset}/{val_part}/100/{seed}/pca_params']
        mean = g_this_x['mean'][...]
        components = g_this_x['components'][...]
    return mean, components
