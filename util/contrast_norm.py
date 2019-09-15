import os

import numpy as np
from skimage.transform import downscale_local_mean
import h5py

import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.ndimage.filters import uniform_filter, correlate
from itertools import product


def sanity_check_valid_idx(idx_list, n):
    idx_all = np.concatenate(idx_list)
    assert np.array_equal(np.sort(idx_all), np.arange(n))


def sanity_check_return_value(dataset_tuple, idx_tuple, n):
    sanity_check_valid_idx(idx_tuple, n)
    assert len(idx_tuple) == 3 and len(dataset_tuple) == 6
    im_shape = None
    for idx, idx_this_set in enumerate(idx_tuple):
        X_this_set = dataset_tuple[2 * idx]
        y_this_set = dataset_tuple[2 * idx + 1]
        assert isinstance(X_this_set, np.ndarray)
        assert isinstance(y_this_set, np.ndarray)
        im_num_this, *im_shape_this = X_this_set.shape
        if im_shape is None:
            im_shape = im_shape_this
        assert im_shape == im_shape_this
        assert im_num_this > 0 and idx_this_set.shape == (im_num_this,)
        assert y_this_set.ndim == 2 and y_this_set.shape[0] == im_num_this


def save_dataset(dataset_tuple, idx_tuple, key, f_out, *addons):
    assert key not in f_out
    key_names = ('train', 'val', 'test')
    assert len(idx_tuple) == len(key_names) == 3 and len(dataset_tuple) == 6
    grp = f_out.create_group(key)
    for idx, (idx_this_set, key_this_set) in enumerate(
            zip(idx_tuple, key_names)):
        X_this_set = dataset_tuple[2 * idx]
        y_this_set = dataset_tuple[2 * idx + 1]
        grp_this = grp.create_group(key_this_set)
        grp_this.create_dataset('X', data=X_this_set)
        grp_this.create_dataset('y', data=y_this_set)
        grp_this.attrs['idx'] = idx_this_set
    if len(addons) != 0:
        # mean and scale
        print('save mean and scale!')
        mean, scale = addons
        grp.attrs['mean'] = mean
        grp.attrs['scale'] = scale
    f_out.flush()


def return_idx_sets(num_im, seed):
    # ok. time to get indices.
    rng_state = np.random.RandomState(seed=seed)
    num_train = int(num_im * 0.8 * 0.8)
    num_val = int(num_im * 0.8) - num_train
    num_test = num_im - num_train - num_val
    print('train', num_train, 'val', num_val, 'test', num_test)
    assert num_train > 0 and num_val > 0 and num_test > 0
    perm = rng_state.permutation(num_im)
    idx_train = perm[:num_train]
    idx_val = perm[num_train:num_train + num_val]
    idx_test = perm[-num_test:]

    idx_preprocessing = perm[:-num_test]

    sanity_check_valid_idx((idx_train, idx_val, idx_test), num_im)
    sanity_check_valid_idx((idx_preprocessing, idx_test), num_im)

    return idx_train, idx_val, idx_test, idx_preprocessing


def split_normalize_stuff(X, y, idx_sets, return_trans=True, normalize=True):
    if return_trans:
        assert normalize

    idx_train, idx_val, idx_test, idx_preprocessing = idx_sets
    X_train_ = X[idx_train]
    Y_train_ = y[idx_train]
    X_val_ = X[idx_val]
    Y_val_ = y[idx_val]
    X_test_ = X[idx_test]
    Y_test_ = y[idx_test]

    num_im = X.shape[0]
    im_shape = X.shape[1:]
    n_channel, im_h, im_w = im_shape
    assert n_channel == 1 and im_h == im_w
    im_element = n_channel * im_h * im_w

    if normalize:
        scaler = StandardScaler()
        scaler.fit(
            X[idx_preprocessing].reshape(idx_preprocessing.size, im_element))
        #     print(scaler.mean_.shape, scaler.var_.shape)
        X_train_ = scaler.transform(
            X_train_.reshape(idx_train.size, im_element)).reshape(
            idx_train.size, *im_shape)
        X_val_ = scaler.transform(
            X_val_.reshape(idx_val.size, im_element)).reshape(idx_val.size,
                                                              *im_shape)
        X_test_ = scaler.transform(
            X_test_.reshape(idx_test.size, im_element)).reshape(idx_test.size,
                                                                *im_shape)
    else:
        scaler = None

    datasets = X_train_, Y_train_, X_val_, Y_val_, X_test_, Y_test_
    tuples = idx_train, idx_val, idx_test

    if not return_trans:
        return datasets, tuples, num_im
    else:
        return datasets, tuples, num_im, scaler.mean_.copy(), scaler.scale_.copy()


# well, I think it's simplest to copy the preprocessing code here...
# given that my earlier implementation is not Py3 friendly.
#

def _normin(im, params, legacy=False):
    assert not legacy
    if not legacy:
        assert im.dtype == np.float32
    if legacy:
        raise RuntimeError
        # assert im.ndim == 2 or im.ndim == 3
        # if im.ndim == 2:
        #     result = v1s_funcs.v1s_norm(im[:, :, np.newaxis], **params)[:, :, 0]
        # else:
        #     result = v1s_funcs.v1s_norm(im, **params)
    else:
        result = local_normalization(im, **params)
    if not legacy:
        assert result.dtype == np.float32
    return result


# add naive gaussian
# https://github.com/torch/image/blob/705393fabf51581b98142c9223c5aee6b62bb131/test/test.lua#L97-L134
def naive_gaussian(size, normalize=True, amplitude=1, sigma=0.25):
    height, width = size
    center_x = sigma * width + 0.5
    center_y = sigma * height + 0.5
    # local gauss = torch.Tensor(height, width)
    #    for i=1,height do
    #       for j=1,width do
    #          gauss[i][j] = amplitude * math.exp(-(math.pow((j-center_x)
    #                                                     /(sigma_horz*width),2)/2
    #                                            + math.pow((i-center_y)
    #                                                    /(sigma_vert*height),2)/2))
    #       end
    #    end
    #    if normalize then
    #       gauss:div(gauss:sum())
    #    end
    #    return gauss
    result = np.empty((size, size), dtype=np.float64)
    for i, j in product(range(size), range(size)):
        result[i, j] = amplitude * np.exp(-(np.power((j + 1 - center_x)
                                                     / (sigma * width), 2) / 2
                                            + np.power((i + 1 - center_y)
                                                       / (sigma * height),
                                                       2) / 2))
    if normalize:
        result /= result.sum()
    return result


def local_normalization_naive(im, kshape=(9, 9),
                              center=True, scale=True):
    kh, kw = kshape
    assert im.ndim == 2
    assert kh % 2 == 1 and kw % 2 == 1, "kernel must have odd shape"

    h_slice = slice(kh // 2, -(kh // 2))
    w_slice = slice(kw // 2, -(kw // 2))

    imh, imw = im.shape
    im_mean = np.empty((imh - kh // 2 * 2, imw - kw // 2 * 2), dtype=np.float64)
    im_std = np.empty((imh - kh // 2 * 2, imw - kw // 2 * 2), dtype=np.float64)
    it_im_mean = np.nditer(im_mean,
                           flags=['multi_index'], op_flags=['writeonly'])
    it_im_std = np.nditer(im_std,
                          flags=['multi_index'], op_flags=['writeonly'])
    while not it_im_mean.finished:
        idx_i, idx_j = it_im_mean.multi_index
        it_im_mean[0] = im[idx_i:idx_i + kh, idx_j:idx_j + kw].mean()
        it_im_mean.iternext()

    while not it_im_std.finished:
        idx_i, idx_j = it_im_std.multi_index
        it_im_std[0] = im[idx_i:idx_i + kh, idx_j:idx_j + kw].std()
        it_im_std.iternext()
    assert np.all(im_std >= 0)
    not_scaling_part = im_std < 1
    print('no scaling', not_scaling_part.mean())
    im_std[not_scaling_part] = 1
    im_new = im[h_slice, w_slice].copy()
    if center:
        im_new -= im_mean
    if scale:
        im_new /= im_std

    return im_new


def local_normalization(im, kshape, threshold, eps=1e-5, gauss=False,
                        norm_std=False, return_div=False):
    assert not gauss
    if norm_std:
        # I think using norm_std is the correct thing.
        # threshold should be always same to the replaced value.
        assert threshold == 1 and eps == 0
    # a more efficient version of input normalization.
    # perhaps this is across channel normalization.
    kh, kw = kshape
    assert im.ndim == 2 or im.ndim == 3
    assert kh % 2 == 1 and kw % 2 == 1, "kernel must have odd shape"

    if im.ndim == 2:
        original_2d = True
        im = im[:, :, np.newaxis]
    else:
        original_2d = False
    assert im.ndim == 3
    depth = im.shape[2]
    ksize = kh * kw * depth if not norm_std else 1
    h_slice = slice(kh // 2, -(kh // 2))
    w_slice = slice(kw // 2, -(kw // 2))
    d_slice = slice(depth // 2,
                    depth // 2 + 1)  # this always works, for both even and odd depth.

    # TODO: add an option to do per channel or across channel normalization.
    # first compute local mean (on 3d stuff).

    # local mean kernel
    # local_mean_kernel = np.ones((kh, kw, depth), dtype=im.dtype) / ksize
    # local_mean = conv(im, local_mean_kernel, mode='valid')  # it's 3D.
    if gauss:
        filter_custom = naive_gaussian(kshape, normalize=True)
    else:
        filter_custom = None

    if not gauss:
        local_mean = uniform_filter(im, size=(kh, kw, depth), mode='constant')[
            h_slice, w_slice, d_slice]
    else:
        local_mean = \
            correlate(im, filter_custom[..., np.newaxis], mode='constant')[
                h_slice, w_slice, d_slice]

    assert local_mean.ndim == 3
    im_without_mean = im[
                          h_slice, w_slice] - local_mean  # this is ``hnum`` in the original implementation.

    # then compute local divisor
    # actually, this divisor is not the sum of square of ``im_without_mean``,
    # that is, it's not $\sum_{i \in N(c)} (x_i-\avg{x_i})^2), where N(c) is some neighborhood of location c, and
    # $\avg{x_i}$ is the local mean with center at i.
    # instead, it's $\sum_{i \in N(c)} (x_i-\avg{x_c})^2). so there's a different offset mean subtracted,
    # when contributing to the normalization of different locations.
    # the above is explanation of line ``val = (hssq - (hsum**2.)/size)`` in the original code.
    # I don't know whether this is intended or not. But if we want to stick to the first definition,
    # then we need more padding, in order to compute $\avg{x_i}$ when i is at the border (which requires some padding)

    # local_sum_kernel = np.ones((kh, kw, depth), dtype=im.dtype)

    # hssq = conv(np.square(im), local_sum_kernel, mode='valid')
    # uniform filter is faster.
    if not gauss:
        hssq = ksize * uniform_filter(np.square(im), size=(kh, kw, depth),
                                      mode='constant')[
            h_slice, w_slice, d_slice]
    else:
        hssq = ksize * correlate(np.square(im), filter_custom[..., np.newaxis],
                                 mode='constant')[
            h_slice, w_slice, d_slice]

    hssq_normed = hssq - ksize * np.square(local_mean)
    np.putmask(hssq_normed, hssq_normed < 0, 0)
    h_div = np.sqrt(hssq_normed) + eps

    if return_div:
        return h_div

    np.putmask(h_div, h_div < (threshold + eps), 1)
    result = im_without_mean / h_div
    if original_2d:
        result = result[:, :, 0]
    return result.astype(np.float32, copy=False)

def local_contrast_normalization(old_data_array, prenorm):
    # here, I will first downsample image by 2x, as in Olshausen's overcomplete sparse coding paper (2013)
    # (mostly because the image resolution is relatively high here, I guess).
    # and then perform local contrast normalization (zero mean, and divide by norm, roughly speaking)
    # using my implementation in the early vision toolbox.
    # kernel size will be set to 9 to match Koray Kavukcuoglu's ConvPSD paper (NIPS 2010).

    # scale down.
    # check https://github.com/leelabcnbc/thesis-proposal-yimeng-201804/blob/master/results_ipynb/debug/convsc/debug_vanhateren_images.ipynb
    # for a debug on whether downsampling all images (with factor 1 on the image idx dimension)
    # is safe or not.
    im_down = downscale_local_mean(old_data_array, (1, 2, 2)).astype(np.float32)

    processed_images = []
    # 9 accounts for summation.
    # in the original code of lcn of unsup,
    # 1 is used to threshold std, not 9*std.
    threshold = 1.0 if prenorm else 0.25
    for idx, x in enumerate(im_down):
        #print(idx)
        if prenorm:
            x_mean = x.mean()
            x_std = x.std()
            #print('mean', x_mean, 'std', x_std)
            x_new = (x - x_mean) / x_std
        else:
            x_new = x
        img_this = _normin(x_new, {'kshape': (9, 9),
                                   # I guess this simply balances noise,
                                   # should be determined relative to input images's distribution.
                                   # based on
                                   # https://github.com/leelabcnbc/thesis-proposal-yimeng-201804/blob/f82820a5b7bab7ee65c1f80a48499c5d2331d72e/results_ipynb/debug/convsc/debug_vanhateren_images.ipynb
                                   #
                                   # I use 0.25, which roughly consider things below %5 in variance to be
                                   # below.
                                   # notice that is is normalizing norm, not std. they differ by a factor of \sqrt{n}.
                                   # here n = 81, \sqrt{n}=9.
                                   # check
                                   # https://github.com/leelabcnbc/thesis-proposal-yimeng-201804/blob/master/results_ipynb/debug/convsc/debug_vanhateren_images_sc.ipynb
                                   # where I have `x = x_*9`
                                   'threshold': threshold,
                                   'norm_std': prenorm,
                                   'eps': 1e-5 if not prenorm else 0})
        img_this = img_this - np.min(img_this)
        img_this = img_this/np.max(img_this)
        processed_images.append(img_this)

    processed_images = np.asarray(processed_images)
    # figure out
    # https://github.com/leelabcnbc/early-vision-toolbox/blob/master/early_vision_toolbox/v1like/v1like.py#L641-L657
    return processed_images

