## this file explores the statistics of the parameter update

import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import pickle as pickle

PREFIX = 'data/attack/cifar10/'
TASK = 'cifar10'


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
    

def main():
    theta_0 = load('{}_theta_0.pkl'.format(TASK))
    theta_0 = concat_param(theta_0)
    print("Loading atk dataset ...")
    dataset = load('{}_atk_dataset.pkl'.format(TASK))
    updates = []
    count = 0
    for _, theta_1 in dataset:
        updates.append(theta_1 - theta_0)
        print(theta_1 - theta_0)
        count += 1
        if(count > 1000):
            break
    updates = np.array(updates).flatten()
    plt.hist(updates, density = True)
    plt.savefig('param_update_hist.png')


        
if __name__ == '__main__':
    main()
