## multi-sample attack in white-box

import os
import numpy as np
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image
import torch.nn.functional as F
import torch.autograd as autograd
from torch.nn import Linear
from tqdm import tqdm
import torch.utils.data as data_util
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import argparse
parser = argparse.ArgumentParser(description='MultiSample Attack')
parser.add_argument("-r", action='store_true', help = 'whether to be conditioned on the gradient or not')
parser.add_argument("--func", type = str, default = 'atk', help = 'the functional of this code')


ARGS = parser.parse_args()
TASK = 'cifar10'
REGRESSION = ARGS.r
GP = True

MODEL_PATH = 'mulsample_attacker_new.cpt'

if(REGRESSION):
    PLOT_PREFIX = "mulsample_new"
    print("FROM NOISE")
else:
    PLOT_PREFIX = "mulsample_save"
    print("FROM GRADIENT")

if(TASK == 'mnist'):
    from mnist import img_transform, dataset, test_dataset, to_img, postprocess, classifier, small_classifier, Reconstructor
    INPUT_DIM = 28 * 28
    PREFIX = 'data/attack/mnist/'
else:
    from cifar10 import img_transform, dataset, test_dataset, to_img, postprocess, classifier, small_classifier, Reconstructor
    INPUT_DIM = 32 * 32 * 3
    PREFIX = 'data/attack/cifar10/'
    
from nn_extractor import Extractor
import pickle as pickle
import torch.optim as optim
from tqdm import tqdm


def dump(obj, path):
    f = open(PREFIX + path, 'w+b')
    pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load(path):
    f = open(PREFIX + path, 'rb')
    return pickle.load(f)

def concat_param(params):
    params = [param.flatten() for param in params]
    params = np.concatenate(params)
    return params

def reform(theta):
    return [torch.FloatTensor(param).cuda() for param in theta]


def show(img, name):
    fig, ax = plt.subplots(figsize=(20, 10))
    npimg = img.numpy()
    ax.imshow(np.transpose(npimg, (1,2,0)))
    plt.savefig("{}/result_{}.png".format(PLOT_PREFIX, name))
    print("Plot in {}".format("{}/result_{}.png".format(PLOT_PREFIX, name)))
    plt.close(fig)

num_epochs = 1
batch_size = 4



dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
one_dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size = 64, shuffle = True)







## attacker model
DIM = 128 # This overfits substantially; you're probably better off with 64
LAMBDA = 10 # Gradient penalty lambda hyperparameter
CRITIC_ITERS = 5 # How many critic iterations per generator iteration
BATCH_SIZE = 64 # Batch size
ITERS = 200000 # How many generator iterations to train for
OUTPUT_DIM = 3072 # Number of pixels in CIFAR10 (3*32*32)


class Generator(nn.Module):
    def __init__(self, in_dim = 128):
        super(Generator, self).__init__()
        preprocess = nn.Sequential(
            nn.Linear(in_dim, 4 * 4 * 4 * DIM),
            nn.BatchNorm1d(4 * 4 * 4 * DIM),
            nn.ReLU(),
        )

        block1 = nn.Sequential(
            nn.ConvTranspose2d(4 * DIM, 2 * DIM, 2, stride=2),
            nn.BatchNorm2d(2 * DIM),
            nn.ReLU(),
        )
        block2 = nn.Sequential(
            nn.ConvTranspose2d(2 * DIM, DIM, 2, stride=2),
            nn.BatchNorm2d(DIM),
            nn.ReLU(),
        )
        deconv_out = nn.ConvTranspose2d(DIM, 3, 2, stride=2)
        self.preprocess = preprocess
        self.block1 = block1
        self.block2 = block2
        self.deconv_out = deconv_out
        self.tanh = nn.Tanh()

    def forward(self, input):
        output = self.preprocess(input)
        output = output.view(-1, 4 * DIM, 4, 4)
        output = self.block1(output)
        output = self.block2(output)
        output = self.deconv_out(output)
        output = self.tanh(output)
        return output.view(-1, 3, 32, 32)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        main = nn.Sequential(
            nn.Conv2d(3, DIM, 3, 2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(DIM, 2 * DIM, 3, 2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(2 * DIM, 4 * DIM, 3, 2, padding=1),
            nn.LeakyReLU()
        )

        self.main = main
        self.linear = nn.Linear(4*4*4*DIM, 1)

    def forward(self, input):
        output = self.main(input)
        output = output.view(-1, 4*4*4*DIM)
        output = self.linear(output)
        return output
    




class Attacker(nn.Module):
    def __init__(self, param, rep_dims, feature_dims, batch_size, noise_dim = 4):
        super(Attacker, self).__init__()
        self.extractor = Extractor(param, rep_dims, feature_dims)
        self.linear_mapping = nn.Sequential(Linear(2 * feature_dims[-1], feature_dims[-1]),
                                            nn.ReLU(True))
        # the feature dim. 
        self.batch_size = batch_size
        self.noise_dim = noise_dim
        self.in_dim = feature_dims[-1] + self.noise_dim
        self.netG = Generator(self.in_dim)

        print("Noise Dim: {}".format(noise_dim))
        # print(self.batch_size)

        
        
        

    def forward(self, param_old, param_new):
        with torch.no_grad():
            param_old = self.extractor(param_old)
        param_new = self.extractor(param_new)
        param_new = torch.cat([param_old, param_new], dim = 1)
        param_new = self.linear_mapping(param_new)
        # obtain the gradient information
        if(REGRESSION):
            noise = torch.randn(self.batch_size, self.in_dim)
        else:
            noise = torch.randn(self.batch_size, self.noise_dim)
        noise = noise.cuda()

        if(not REGRESSION):
            param_new = param_new.repeat_interleave(self.batch_size, dim = 0)
            param_new = torch.cat([param_new, noise], dim = 1)
        else:
            param_new = noise
 
        # concatenate with the parameter feature
        fake = self.netG(param_new)
        return fake
        
        

def calc_gradient_penalty(netD, real_data, fake_data, batch_size):
    # print "real_data: ", real_data.size(), fake_data.size()
    batch_size = len(real_data)
    alpha = torch.rand(batch_size, 1)
    alpha = alpha.expand(batch_size, real_data.nelement()//batch_size).contiguous().view(batch_size, 3, 32, 32)
    alpha = alpha.cuda()
   #  print(alpha.size())
   #  print(real_data.size())
   #  print(fake_data.size())

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    interpolates = interpolates.cuda()
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty 


def mean_square(img0, img1):
    img0 = img0.flatten()
    img1 = img1.flatten()
    return np.mean((img0 - img1) ** 2)

def match_img(x0, x1):
    x1_matched = np.zeros_like(x1)
    avg_dist = 0.0
    n = len(x0)
    cost_matrix = np.zeros((n, n))
    
    for i in range(len(x0)):
        cost_matrix[i, :] = [mean_square(x0[i], xp) for xp in x1]
    print(cost_matrix)
    
    from munkres import Munkres
    m = Munkres()
    indexes = m.compute(cost_matrix.copy())

    for i, idx in indexes:
        avg_dist += cost_matrix[i, idx]
        x1_matched[i, :, :, :] = x1[idx, :, :, :]
        avg_dist += cost_matrix[i, idx]
    avg_dist /= len(x0)
    print("MSE: {:.4f}".format(avg_dist))
    return x1_matched
        
        
        

def evaluate_attacker(path):
    # first load the data
    PRINT_FREQ = 100
    PLOT_FREQ = 100
    CACHED = False
    theta_0 = load('{}_theta_0_new.pkl'.format(TASK))
    theta_0_arr = torch.FloatTensor(concat_param(theta_0))
    theta_0 = reform(theta_0)
    rep_dims = [500, 200, 100, 50]
    feature_dims = [128] # 64
    use_l1_loss = True
    batch_size = 64
    NOISE_DIM = 8 # 8
    CRITIC_ITERS = 5 # How many critic iterations per generator iteration
    best_w_dist = 1000.0
    
    attacker = Attacker(theta_0, rep_dims, feature_dims, batch_size, NOISE_DIM)
    attacker.cuda()
    print("Loading Model and Resume ...")
    attacker.load_state_dict(torch.load(path))
    
    print("Loading atk test dataset ...")
    dataset = load('{}_atk_dataset_new.test.pkl'.format(TASK))
    print("convert to torch dataset ...")
    X = []
    Y = []
    labels = []
    for x, y, label in dataset:
        X.append(torch.FloatTensor(x))
        Y.append(torch.FloatTensor(y).unsqueeze(0))
        labels.append(label)
    X = torch.cat(X, dim = 0)[:batch_size,:, :, :]
    Y = torch.cat(Y, dim = 0)[:batch_size,:]
    labels = torch.LongTensor(labels)
    
    print(X.size())
    print(Y.size())
    print(labels.size())
    theta_1 = Y.cuda()
    x = X.cuda()
    labels = labels.cuda()
    theta_1 = torch.mean(theta_1, dim=0, keepdim = True)
    _theta_0 = theta_0_arr.unsqueeze(0)
    _theta_0 = _theta_0.cuda()
    faked = attacker(_theta_0, theta_1)
    print(faked.size())

    faked = to_img(faked.detach().cpu().data)
    x = to_img(x.detach().cpu().data)
        
    faked = torch.FloatTensor(match_img(x.numpy(), faked.numpy()))
   
    # print(faked.size())

    # to match
    figname = "{}_eval".format(path.split('.')[0])
    imgs = []
    for i in range(x.size(0)):
        imgs.append(faked[i, :, :, :].unsqueeze(0))
        imgs.append(x[i, :, :, :].unsqueeze(0))
    
    imgs = torch.cat(imgs, dim = 0)
    grid = torchvision.utils.make_grid(imgs, nrow = 16)
    show(grid, figname)
    



def train_attacker():
    # first load the data
    PRINT_FREQ = 100
    PLOT_FREQ = 5000
    CACHED = False
    theta_0 = load('{}_theta_0_new.pkl'.format(TASK))
    theta_0_arr = torch.FloatTensor(concat_param(theta_0))
    theta_0 = reform(theta_0)
    rep_dims = [500, 200, 100, 50]
    feature_dims = [128] # 64
    use_l1_loss = True
    batch_size = 64
    NOISE_DIM = 128
    CRITIC_ITERS = 5 # How many critic iterations per generator iteration
    best_w_dist = 1000.0
    
    attacker = Attacker(theta_0, rep_dims, feature_dims, batch_size, NOISE_DIM)
    attacker.cuda()
    netD = Discriminator()
    netD.cuda()


    
    # PATH = PREFIX + 'attacker_delta_with_label.cpt'
    # if(CACHED):
    #     print("Loading Model and Resume ...")
    #     attacker.load_state_dict(torch.load(PATH))

    # print(attacker)
    # if(not CACHED): pretrain(attacker)

    

    # if(not CACHED): autoencoder = pretrain(attacker)
    # print(get_parameter(attacker.reconstructor))
    # print(get_parameter(autoencoder.reconstructor))
    ## load training set
    print("Loading atk dataset ...")
    dataset = load('{}_atk_dataset_new.pkl'.format(TASK))
    criterion = nn.MSELoss()  # nn.L1Loss() #
    max_epoch = 500

    running_loss = 0.0
    count = 0
    # batch_size = 16
    batch_count = 0
    loss = 0.0
    TRUNCATE = True

    # param = concat_param(dataset[0][1])
    # print(param.shape)
    # X = torch.FloatTensor([p for _, p in dataset])
    # Y = torch.FloatTensor([y for y, _ in dataset])

    # Z = torch.FloatTensor([theta_0 for i in range(len(dataset))])
    
    # print(X.shape)
    # print(Y.shape)
    # print(Z.shape)
    print("convert to torch dataset ...")
    X = []
    Y = []
    labels = []
    for x, y, label in dataset:
        X.append(torch.FloatTensor(x))
        Y.append(torch.FloatTensor(y).unsqueeze(0))
        labels.append(label)

    X = torch.cat(X, dim = 0)
    Y = torch.cat(Y, dim = 0)
    labels = torch.LongTensor(labels)
    
    print(X.size())
    print(Y.size())
    print(labels.size())
    dataset = data_util.TensorDataset(X, Y, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last = True)

    
    if(GP):
        optimizerD = optim.Adam(netD.parameters(), lr=1e-4, betas=(0.5, 0.9))
        optimizerG = optim.Adam(attacker.parameters(), lr=1e-4, betas=(0.5, 0.9))
    else:
        optimizerD = optim.Adam(netD.parameters(), lr=5e-5, betas=(0.5, 0.99))
        optimizerG = optim.Adam(attacker.parameters(), lr=5e-5, betas=(0.5, 0.99))
    
    print(theta_0_arr.shape)
    # do pretrain the reconstructor part
    best_loss = 100.0
    running_G_cost = 0.0
    running_D_cost = 0.0
    running_reconstruction_cost = 0.0
    running_w_dist = 0.0
    running_gp = 0.0
    CLAMP_LOWER = -0.01
    CLAMP_UPPER = 0.01

    print("Attack Arch:{}".format(attacker))
    print("D Arch: {}".format(netD))

    
    for epoch in range(max_epoch):
        print("Epoch {} ...".format(epoch))
        # do shuffling per epoch, construct the batch and do batch normalization manually
        for x, theta_1, label in dataloader:
            count += 1
            theta_1, x = theta_1.cuda(), x.cuda()
            theta_1 = torch.mean(theta_1, dim=0, keepdim = True)
            _theta_0 = theta_0_arr.unsqueeze(0)
            _theta_0 = _theta_0.cuda()


            # print(theta_0_arr.size())
            # prepare the training set
            # first extract feature 
            # training
            for p in netD.parameters():
                p.requires_grad = True
            # critical phase
            for i in range(CRITIC_ITERS):
                # for p in netD.parameters():
                #     p.data.clamp_(CLAMP_LOWER, CLAMP)
                netD.zero_grad()
                faked = attacker(_theta_0, theta_1)
                D_real = netD(x).mean()
                D_fake = netD(faked).mean()
                gradient_penalty = calc_gradient_penalty(netD, x.data, faked.data, batch_size)
                D_cost = D_fake - D_real + gradient_penalty
                Wasserstein_D = D_real - D_fake
                D_cost.backward()
                optimizerD.step()
                # print(D_cost)
            
            for p in netD.parameters():
                p.requires_grad = False  # to avoid computation

            attacker.zero_grad()
            faked = attacker(_theta_0, theta_1)
            G = netD(faked)
            G = G.mean()
            reconstruction_loss = 0.0
            for i in range(batch_size):
                reconstruction_loss += criterion(faked[i, :, :, :].unsqueeze(0).repeat_interleave(batch_size, dim = 0), x)
            

            if(not REGRESSION):
                G_cost = -G + reconstruction_loss
            else:
                G_cost = -G
            G_cost.backward()
            optimizerG.step()
            running_G_cost += G_cost.data
            running_D_cost += D_cost.data
            running_reconstruction_cost += reconstruction_loss.data
            running_w_dist += Wasserstein_D.data
            running_gp += gradient_penalty.data
            
                        
            if(count % PRINT_FREQ == 0):
                running_D_cost /= PRINT_FREQ
                running_G_cost /= PRINT_FREQ
                running_reconstruction_cost /= PRINT_FREQ
                running_w_dist /= PRINT_FREQ
                running_gp /= PRINT_FREQ
                print("Iter {} G Cost: {:.4f}, D Cost: {:.4f} Rec Cost: {:.4f} W Dist: {:.4f} GP: {:.4f}".format(count, running_G_cost, running_D_cost, running_reconstruction_cost, running_w_dist, running_gp))
                # save model
                if(running_w_dist < best_w_dist):
                    best_w_dist = running_w_dist
                    print("Save Attacker {:.4f}".format(best_w_dist))
                    torch.save(attacker.state_dict(), MODEL_PATH)
                running_D_cost = 0.0
                running_G_cost = 0.0
                running_reconstruction_cost = 0.0
                running_w_dist = 0.0
                running_gp = 0.0


            if(count % PLOT_FREQ == 0):
                # evaluate the generator
                figname = 'iter_{}'.format(count // PRINT_FREQ)
                imgs = []
                faked = to_img(faked.detach().cpu().data)
                x = to_img(x.detach().cpu().data)
                for i in range(x.size(0)):
                    imgs.append(faked[i, :, :, :].unsqueeze(0))
                    imgs.append(x[i, :, :, :].unsqueeze(0))
                imgs = torch.cat(imgs, dim = 0)
                grid = torchvision.utils.make_grid(imgs, nrow = 16)
                show(grid, figname)

                    
    # save data

if __name__ == '__main__':
    if(ARGS.func == 'atk'):
        train_attacker()
    else:
        evaluate_attacker(MODEL_PATH)
    
    
