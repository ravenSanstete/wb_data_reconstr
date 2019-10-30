## this file explores the statistics of the parameter update

import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import pickle as pickle
from scipy.stats import describe
from sklearn.decomposition import FastICA, PCA
from sklearn import random_projection
import torch
from torchvision import transforms


import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

TASK = 'mnist'
PREFIX = 'data/attack/{}/'.format(TASK)
if(TASK == 'mnist'):
    from mnist import to_img
else:
    from cifar10 import to_img






def imscatter(x, y, images, ax=None, zoom=0.5):
    if ax is None:
        ax = plt.gca()
    x, y = np.atleast_1d(x, y)
    artists = []
    for x0, y0, img0 in zip(x, y, images):
        img0 = to_img(torch.FloatTensor(img0)).numpy()
        print(img0.shape)
        img0 = np.transpose(img0[0,:, :, :], axes = [1,2,0])
        if(TASK == 'mnist'):
            img0 = img0[:, :, 0]
        im = OffsetImage(img0, zoom=zoom)
        ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=False)
        artists.append(ax.add_artist(ab))
    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()
    return artists


def concat_param(params):
    params = [param.flatten() for param in params]
    params = np.concatenate(params)
    return params

def dump(obj, path):
    f = open(PREFIX + path, 'w+b')
    pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load(path):
    f = open(PREFIX + path, 'rb')
    return pickle.load(f)
    

def get_updates(number = 100):
    theta_0 = load('{}_theta_0.pkl'.format(TASK))
    theta_0 = concat_param(theta_0)
    print("Loading atk dataset ...")
    dataset = load('{}_atk_dataset.pkl'.format(TASK))
    updates = []
    pics = []
    count = 0
    for pic, theta_1 in dataset:
        pics.append(pic)
        updates.append(theta_1 - theta_0)
        # print(theta_1 - theta_0)
        count += 1
        if(count > number):
            break
    updates = np.array(updates)
    return pics, updates



def do_ICA():
    pics, updates = get_updates(10000)
    # transformer = random_projection.SparseRandomProjection(n_components = 2)
    transformer = FastICA(n_components = 2, algorithm = 'parallel')
    X_new = transformer.fit_transform(updates)
    print(X_new)
    fig, ax = plt.subplots(figsize = (10, 10), ncols = 1, nrows =  1)
    imscatter(X_new[:, 0], X_new[:,1], pics, ax)
    # ax.set_xlim(-2.5, 2.5)
    # ax.set_ylim(-2.5, 2.5)
    plt.savefig("scatter_image_ica.png", dpi = 108)
    
    # print("doing ICA ...")
    # transformer = FastICA(algorithm = 'parallel')
    # updates_transformed = transformer.fit_transform(updates)
    # print(updates_transformed.shape)
    # print(transformer.components_.shape)

    
    # pos_non_zeros = []
    # print(updates.shape[1])
    # for i in range(updates.shape[1]):
    #     pos_non_zeros.append(np.sum(np.abs(updates[:, i]) > 1e-4))
    # pos_non_zeros = np.array(pos_non_zeros)
    # # print(describe(pos_non_zeros))
    # plt.hist(pos_non_zeros, bins = 100)
    # plt.savefig('param_hist.png')
    # print(describe(updates))
    # plt.hist(updates, bins = 100)
    # plt.xlim(-1.0, 1.0)
    # plt.savefig('param_update_hist.png')


        
if __name__ == '__main__':
    do_ICA()
