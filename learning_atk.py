## implement a learning-based attack in the white-box setting

# in this file, implement a demo for white-box data reconstruction
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
import time


TASK = 'cifar10'
PLOT_PREFIX = 'single'
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
    
if not os.path.exists('./mlp_img'):
    os.mkdir('./mlp_img')





num_epochs = 1
batch_size = 128



dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
one_dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size = 64, shuffle = True)


def concat_param(params):
    params = [param.flatten() for param in params]
    params = np.concatenate(params)
    return params

# from modelB to modelA
def copy_from(modelA, modelB):
    for a, b in zip(modelA.parameters(), modelB.parameters()):
        a.data.copy_(b.data)

def copy_from_param(model, parameters):
    for a, b in zip(model.parameters(), parameters):
        a.data.copy_(b.data)

def get_parameter_flatten(model):
    params = []
    _queue = [model]
    while(len(_queue) > 0):
        cur = _queue[0]
        _queue = _queue[1:] # dequeue
        if("weight" in cur._parameters):
            params.append(cur._parameters['weight'].view(-1))
        if("bias" in cur._parameters and not (cur._parameters["bias"] is None)):
            params.append(cur._parameters['bias'].view(-1))
        for module in cur.children():
            _queue.append(module)
    return torch.cat(params)


def get_named_parameter(model):
    return [(name, param.clone().detach()) for name, param in model.named_parameters()]

def get_parameter(model):
    return [param.clone().detach() for param in model.parameters()]
def get_dumpable_param(model):
    return list([param.clone().detach().cpu().numpy() for param in model.parameters()])


def eval_classifier(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('Test set:  Accuracy: {}/{} ({:.2f}%)'.format(
        correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def reform(theta):
    return [torch.FloatTensor(param).cuda() for param in theta]



def collect_test_data():
    one_dataloader = DataLoader(test_dataset, batch_size = 1, shuffle = True)
    model = small_classifier().cuda()
    theta_0 = load('{}_theta_0_new.pkl'.format(TASK))
    theta_0 = reform(theta_0)
    copy_from_param(model, theta_0)
    TEST_SIZE = 128
    criterion = nn.CrossEntropyLoss()
    learning_rate = 1.0
    optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)
    theta_t_0 = get_parameter(model)
    data = []
    count = 0
    
    for x, y in one_dataloader:
        tmp = get_dumpable_param(model)
        x, y = x.cuda(), y.cuda()
        preds = model(x)
        # obtain the gradient
        loss = criterion(preds, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print(loss)
        ### =================== CUSTOMIZED =====================
            # theta_t_0 = [param.clone() for param in theta_t_1]
            # after update, obtain the parameters
        # print(model.named_parameters())
        theta_t_1 = get_dumpable_param(model)
        data.append((x.cpu().numpy(), concat_param(theta_t_1), y.cpu().numpy()))
        theta_t_1 = get_parameter(model)
            # copy the theta_t_0 back
        copy_from_param(model, theta_t_0)

        count += 1
        delta_theta = [(paramA - paramB) for paramA, paramB in zip(theta_t_1, theta_t_0)]
        # print(delta_theta[-1])        
        if(count > TEST_SIZE):
            break
            # delta = [(-p_1 +p_0).detach() for p_1, p_0 in zip(theta_t_1, theta_t_0)]
            # print(delta)
    print("Dumping ...")
    dump(data, '{}_atk_dataset_new.test.pkl'.format(TASK))
    
    


def truncate(x, threshold = 0.001):
    x[x < threshold] = 0
    return x
    
    
    

def collect_training_data():
    one_dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    model =  small_classifier().cuda()
    criterion = nn.CrossEntropyLoss()
    learning_rate = 1.0
    optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)
    PRINT_FREQ = 100

    
    # save the theta_t_0 data
    # theta_t_1 = theta_t_0
    # extractor = Extractor(theta_t_0, [50, 30, 10, 10], [50, 32])
    # extractor = extractor.cuda()
    # nn_feature = extractor(theta_t_0)
    # print(nn_feature)
    data = []
    count = 0
    train_optimizer = torch.optim.Adam(model.parameters(), lr = 0.005)
    running_loss = 0.0
    for i in range(3):
        for x, y in dataloader:
            x, y = x.cuda(), y.cuda()
            preds = model(x)
            # obtain the gradient
            loss = criterion(preds, y)
            train_optimizer.zero_grad()
            loss.backward()
            train_optimizer.step()
            running_loss += loss.data
            count += 1
            if(count % PRINT_FREQ == 0):
                print("Pretrain Iteration: {} Loss: {}".format(count, running_loss / PRINT_FREQ))
                eval_classifier(model, test_loader)
                running_loss = 0.0
    count = 0
    theta_t_0 = get_parameter(model)
    dump(get_dumpable_param(model), '{}_theta_0_new.pkl'.format(TASK))

    
    for x, y in one_dataloader:
        tmp = get_dumpable_param(model)
        x, y = x.cuda(), y.cuda()
        preds = model(x)
        # obtain the gradient
        loss = criterion(preds, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print(loss)
        ### =================== CUSTOMIZED =====================
            # theta_t_0 = [param.clone() for param in theta_t_1]
            # after update, obtain the parameters
        # print(model.named_parameters())
        theta_t_1 = get_dumpable_param(model)
        data.append((x.cpu().numpy(), concat_param(theta_t_1), y.cpu().numpy()))
        theta_t_1 = get_parameter(model)
            # copy the theta_t_0 back
        copy_from_param(model, theta_t_0)

        count += 1
        delta_theta = [(paramA - paramB) for paramA, paramB in zip(theta_t_1, theta_t_0)]
        # print(delta_theta[-1])        
        if(count > 10000):
            break
            # delta = [(-p_1 +p_0).detach() for p_1, p_0 in zip(theta_t_1, theta_t_0)]
            # print(delta)
    print("Dumping ...")
    dump(data, '{}_atk_dataset_new.pkl'.format(TASK))
        # optimizer.step()
        


class AutoEncoder(nn.Module):
    def __init__(self, reconstructor, in_dim):
        super(AutoEncoder, self).__init__()
        self.reconstructor = reconstructor
        self.module = nn.Sequential(
            Linear(INPUT_DIM, in_dim),
            nn.BatchNorm1d(in_dim),
            nn.ReLU(True),
            self.reconstructor
        )

    def forward(self, x):
        return self.module(x)
    
        

class Attacker(nn.Module):
    def __init__(self, param, rep_dims, feature_dims):
        super(Attacker, self).__init__()
        self.extractor = Extractor(param, rep_dims, feature_dims)
        self.delta = False
        self.concat = True
        self.in_dim = feature_dims[-1]
        self.linear_mapping = nn.Sequential(Linear( 2 * feature_dims[-1], feature_dims[-1]),
                                                    nn.BatchNorm1d(feature_dims[-1]),
                                                    nn.ReLU(True))
        self.reconstructor = Reconstructor(self.in_dim, INPUT_DIM)
        

    def forward(self, param_old, param_new, y):
        # donnot allow the gradient to pass to the old parameter
        scale = 1.0
        # param_delta = [(paramA - paramB) * scale  for paramA, paramB in zip(param_new, param_old)]
        # print(param_delta[-1])
        # with torch.no_grad():
        if(not self.delta):       
            with torch.no_grad():
                param_old = self.extractor(param_old)
            param_new = self.extractor(param_new)
            
            if(self.concat):
                param_new = torch.cat([param_old, param_new], dim = 1)
                param_new = self.linear_mapping(param_new)
            else:
                param_new = param_new + param_old
            # print("Before Reconstructor:{}".format(param_new.shape))
            param_new = self.reconstructor(param_new, y)

            return param_new
        else:
            param_delta = param_new - param_old
            param_delta = self.extractor(param_delta)
            param_delta = self.reconstructor(param_delta, y)
            
            
            return param_delta
            
        

def reform(theta):
    return [torch.FloatTensor(param).cuda() for param in theta]


def pretrain(attacker):
    ae = AutoEncoder(attacker.reconstructor, attacker.in_dim)
    ae.cuda()
    PRINT_FREQ = 100
    count = 0
    train_optimizer = torch.optim.Adam(ae.parameters(), lr = 0.005)
    running_loss = 0.0
    criterion = nn.MSELoss()# nn.MSELoss()
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

    for i in range(5):
        for x, _ in dataloader:
            x = x.cuda()
            x = x.reshape(-1, INPUT_DIM)
            x_prime = ae(x)
            # obtain the gradient
            loss = criterion(x, x_prime)
            train_optimizer.zero_grad()
            loss.backward()
            train_optimizer.step()
            running_loss += loss.data
            count += 1
            if(count % PRINT_FREQ == 0):
                print("Pretrain Iteration: {} Loss: {}".format(count, running_loss / PRINT_FREQ))
                running_loss = 0.0
            if(count % 1000 == 0):              
                pic = to_img(x_prime[0, :].detach().cpu().unsqueeze(0).data)
                save_image(pic, PREFIX + '{}_image_{}_pretrain.png'.format(TASK, count // 1000))
                pic = to_img(x[0, :].detach().cpu().unsqueeze(0).data)
                save_image(pic, PREFIX + '{}_image_{}_gt_pretrain.png'.format(TASK, count // 1000))
    return ae
    

    


def train_attacker():
    # first load the data
    PRINT_FREQ = 10
    CACHED = False
    theta_0 = load('{}_theta_0_new.pkl'.format(TASK))
    theta_0_arr = torch.FloatTensor(concat_param(theta_0))
    theta_0 = reform(theta_0)
    rep_dims = [500, 200, 100, 50]
    feature_dims = [256] # 64
    use_l1_loss = True

    
    attacker = Attacker(theta_0, rep_dims, feature_dims)
    PATH = PREFIX + 'attacker_delta_with_label.cpt'
    if(CACHED):
        print("Loading Model and Resume ...")
        attacker.load_state_dict(torch.load(PATH))
    attacker.cuda()
    print(attacker)
    # if(not CACHED): pretrain(attacker)
  

    # if(not CACHED): autoencoder = pretrain(attacker)
    # print(get_parameter(attacker.reconstructor))
    # print(get_parameter(autoencoder.reconstructor))
    ## load training set
    print("Loading atk dataset ...")
    dataset = load('{}_atk_dataset_new.pkl'.format(TASK))
    criterion = nn.L1Loss()  # nn.L1Loss() #
    max_epoch = 500
    optimizer = optim.Adam([{'params': attacker.reconstructor.parameters()},
                            {'params': attacker.extractor.parameters()}], lr=0.005)
    running_loss = 0.0
    count = 0
    batch_size = 128
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
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print(theta_0_arr.shape)
    # do pretrain the reconstructor part
    best_loss = 100.0
    for epoch in range(max_epoch):
        print("Epoch {} ...".format(epoch))
        # do shuffling per epoch, construct the batch and do batch normalization manually
        for x, theta_1, label in dataloader:
            count += 1
            x, theta_1, label = x.cuda(), theta_1.cuda(), label.cuda()
            _theta_0 = theta_0_arr.unsqueeze(0).repeat_interleave(theta_1.size(0), dim = 0).cuda()
            # print(theta_1[-1] - theta_0[-1])
            x_prime = attacker(_theta_0, theta_1, label)
            x = x.reshape(-1, INPUT_DIM).cuda()
            loss = criterion(x_prime, x)
            running_loss += loss.data
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if(count % PRINT_FREQ == 0):
                loss = 0.0
                running_loss = running_loss / (PRINT_FREQ)
                print("Iteration {} Loss: {:.4f}".format(count, running_loss))

                # pic = to_img(x_prime[0, :].detach().cpu().unsqueeze(0).data)
                # save_image(pic, PREFIX + '{}_image_{}.png'.format(TASK, count))
                # pic = to_img(x[0, :].detach().cpu().unsqueeze(0).data)
                # save_image(pic, PREFIX + '{}_image_{}_gt.png'.format(TASK, count))
                if(running_loss < best_loss):
                    best_loss = running_loss
                    print("save model best loss {:.4f}".format(best_loss))
                    torch.save(attacker.state_dict(), PATH)
                running_loss = 0.0
    
                    
    # save data

    
def evaluate_attacker(path, figname):
    # first load the data
    PRINT_FREQ = 10
    CACHED = False
    theta_0 = load('{}_theta_0_new.pkl'.format(TASK))
    theta_0_arr = torch.FloatTensor(concat_param(theta_0))
    theta_0 = reform(theta_0)
    rep_dims = [500, 200, 100, 50]
    feature_dims = [256] # 64

    attacker = Attacker(theta_0, rep_dims, feature_dims)
    # attacker.eval()
    attacker.cuda()
    PATH = PREFIX + path
    print("Loading Model and Resume ...")
    attacker.load_state_dict(torch.load(PATH))

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
    X = torch.cat(X, dim = 0)
    Y = torch.cat(Y, dim = 0)
    labels = torch.LongTensor(labels)
    
    print(X.size())
    print(Y.size())
    print(labels.size())
    theta_1 = Y.cuda()
    x = X.cuda()
    labels = labels.cuda()
    _theta_0 = theta_0_arr.unsqueeze(0).repeat_interleave(Y.size(0), dim = 0).cuda()
    x_prime = attacker(_theta_0, theta_1, labels)
    
    mse_loss = F.mse_loss(x.view(x.size(0), 3, 32, 32), x_prime.view(x.size(0), 3, 32, 32))
    x = to_img(x.detach().cpu().data)
    x_prime = to_img(x_prime.detach().cpu().data)
    print(x.shape)
    print(x_prime.shape)
    mse_loss = F.mse_loss(x.view(x.size(0), 3, 32, 32), x_prime.view(x.size(0), 3, 32, 32))
    # mse_loss = F.mse_loss(x, x_prime)
    print("MSE Loss: {}".format(mse_loss.data))
    x = x[1:, :, :, :]
    x_prime = x_prime[1:, :, :, :]
    
    def show(img, name):
        fig, ax = plt.subplots(figsize=(20, 10))
        npimg = img.numpy()
        ax.imshow(np.transpose(npimg, (1,2,0)))
        plt.savefig("result_{}.png".format(name))
    imgs = []
    for i in range(x.size(0)):
        imgs.append(x[i, :, :, :].unsqueeze(0))
        imgs.append(x_prime[i, :, :, :].unsqueeze(0))
    imgs = torch.cat(imgs, dim = 0)
    print(imgs.shape)
    grid = torchvision.utils.make_grid(imgs, nrow = 32)
    show(grid, figname)
    
def show(img, name):
    fig, ax = plt.subplots(figsize=(20, 10))
    npimg = img.numpy()
    ax.imshow(np.transpose(npimg, (1,2,0)))
    plt.savefig("{}/result_{}.png".format(PLOT_PREFIX, name))
    print("Plot in {}".format("{}/result_{}.png".format(PLOT_PREFIX, name)))
    plt.close(fig)



## implement the method in "Deep Leakage from Gradients" (https://arxiv.org/pdf/1906.08935.pdf)
def reconstruction():
    one_dataloader = DataLoader(test_dataset, batch_size = 1, shuffle = True)
    model = small_classifier().cuda()
    theta_0 = load('{}_theta_0_new.pkl'.format(TASK))
    theta_0 = reform(theta_0)
    copy_from_param(model, theta_0)
    TEST_SIZE = 128
    criterion = nn.CrossEntropyLoss()
    learning_rate = 1
    rep_dims = [500, 200, 100, 50]
    feature_dims = [256] # 64

    attacker = Attacker(theta_0, rep_dims, feature_dims)
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    theta_t_0 = get_parameter(model)
    data = []
    count = 0
    IMG_SIZE = (1, 3, 32, 32)
    CLS_NUM = 10
    # then load the test data
    hat_x = torch.tensor(np.random.randn(*IMG_SIZE)*0.1, requires_grad = True, dtype = torch.float32)
    hat_y = torch.tensor(np.random.randn(CLS_NUM), requires_grad = True, dtype = torch.float32)
    # hat_y = F.softmax(hat_y)
    # add some softmax to hat_y
    TEST_SIZE = 2
    # learning_rate = 0.005
    
    ## load the test_data 
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
    X = torch.cat(X, dim = 0)
    Y = torch.cat(Y, dim = 0)
    labels = torch.LongTensor(labels)
    
    print(X.size())
    print(Y.size())
    print(labels.size())
    theta_1 = Y.cuda()
    x = X.cuda()
    labels = labels.cuda()

    hat_x, hat_y = hat_x.cuda(), hat_y.cuda()

    sample_lr = 0.1
    ITER_NUM = 100000
    PRINT_FREQ = 100
    padding = torch.FloatTensor(np.ones((CLS_NUM)) * 0.01).cuda()

    
    imgs = []

    # sample_optimizer = torch.optim.Adam([hat_x, hat_y], lr = sample_lr)
    for i in range(TEST_SIZE):

        theta_1_list = attacker.extractor.split(theta_1[i, :].unsqueeze(0))
        delta_w = [(x - y).squeeze(0) for x, y in zip(theta_1_list, theta_0)]
        imgs.append(to_img(x[i,:,:,:].unsqueeze(0).detach().cpu().data))
        
        start = time.time()
        for j in range(ITER_NUM):
            # hat_x.zero_grad()
            # hat_y.zero_grad()
            model.zero_grad()
            # print(delta_w)
            # now use the hat_x to generate the gradient
            preds = F.softmax(model(hat_x))
            hat_y = F.softmax(hat_y)
            # print(preds)
            # turn off the gradient at hat_x and hat_y
            
            loss = (preds * (preds / hat_y).log()).sum()
            # print(loss)
            f_delta_theta = autograd.grad([loss], model.parameters(), retain_graph = True, create_graph = True)
            # print(f_delta_theta)
            # print(len(f_delta_theta))
            # compute the sample_loss
            sample_loss = torch.stack([F.mse_loss(learning_rate * x, y) for x, y in zip(f_delta_theta, delta_w)], dim = 0).sum()
            # print(sample_loss)
            delta_sample = autograd.grad([sample_loss], [hat_x, hat_y])
            # print(delta_sample)
            
            hat_x = hat_x - sample_lr * delta_sample[0]
            hat_y = hat_y - sample_lr * delta_sample[1]
            mse = F.mse_loss(hat_x, x[i,:,:,:].unsqueeze(0))
    
            if(j % PRINT_FREQ == 0):
                end = time.time()
                print("Iter {} MSE: {:.4f} Grad: {:.4f} Loss: {:.4f} Time: {:.4f}".format(j, mse.data, delta_sample[0].norm(), sample_loss.data, end - start))
                print(hat_y)
                # print(hat_x)
                start = end
                
            # clear the retained graph
            mse = None
            delta_sample = None
            sample_loss = None
            f_delta_theta = None
        imgs.append(to_img(hat_x.detach().cpu().data))
    figname = "test_{}".format(TEST_SIZE)
    imgs = torch.cat(imgs, dim = 0)
    print(imgs.shape)
    grid = torchvision.utils.make_grid(imgs, nrow = 32)
    show(grid, figname)
    

    
                
        
        
        
        

    
    
    

    
    
    
    
    
      
            
            

    
    
    
            
if __name__ == '__main__':
    PATH = 'attacker_delta_with_label.cpt'
    # evaluate_attacker(PATH, 'delta_eval_with_label')
    # train_attacker()
    # collect_training_data()
    # collect_test_data()
    reconstruction()
