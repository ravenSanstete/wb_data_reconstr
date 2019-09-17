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
from mnist import img_transform, dataset, test_dataset, to_img, postprocess, classifier, small_classifier
from nn_extractor import Extractor
import pickle as pickle
import torch.optim as optim
from tqdm import tqdm

PREFIX = 'data/attack/'

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
learning_rate = 1e-3


dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
one_dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size = 64, shuffle = True)


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
  


def collect_training_data():
    one_dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    model =  small_classifier().cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

    theta_t_0 = get_parameter(model)
    dump(get_dumpable_param(model), 'theta_0.pkl')
    
    # save the theta_t_0 data
    theta_t_1 = theta_t_0
    # extractor = Extractor(theta_t_0, [50, 30, 10, 10], [50, 32])
    # extractor = extractor.cuda()
    # nn_feature = extractor(theta_t_0)
    # print(nn_feature)
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
        ### =================== CUSTOMIZED =====================
        with torch.no_grad():
            # theta_t_0 = [param.clone() for param in theta_t_1]
            # after update, obtain the parameters
        # print(model.named_parameters())
            theta_t_1 = get_dumpable_param(model)
            data.append((x.cpu().numpy(), theta_t_1))
            # copy the theta_t_0 back
            copy_from_param(model, theta_t_0)

        count += 1
        # print(theta_t_1)        
        if(count > 10000):
            break
            # delta = [(-p_1 +p_0).detach() for p_1, p_0 in zip(theta_t_1, theta_t_0)]
            # print(delta)
    print("Dumping ...")
    dump(data, 'atk_dataset.pkl')
        # optimizer.step()
        


class Reconstructor(nn.Module):
    def __init__(self, in_dim):
        super(Reconstructor, self).__init__()
        self.module = nn.Sequential(
            Linear(in_dim, 50),
            nn.Sigmoid(),
            Linear(50, 50),
            nn.Sigmoid(),
            Linear(50, 28 * 28))

    def forward(self, x):
        return self.module(x)


class Attacker(nn.Module):
    def __init__(self, param, rep_dims, feature_dims):
        super(Attacker, self).__init__()
        self.extractor = Extractor(param, rep_dims, feature_dims)
        self.reconstructor = Reconstructor(feature_dims[-1])

    def forward(self, param_old, param_new):
        param_old = self.extractor(param_old)
        param_new = self.extractor(param_new)
        # param_new = torch.cat([param_old, param_new])
        param_new = param_new + param_old
        param_new = self.reconstructor(param_new)
        return param_new
        

def reform(theta):
    return [torch.FloatTensor(param).cuda() for param in theta]

def train_attacker():
    # first load the data
    PRINT_FREQ = 10
    theta_0 = load('theta_0.pkl')
    theta_0 = reform(theta_0)
    rep_dims = [50, 30, 10, 10]
    feature_dims = [50, 50]
  
    attacker = Attacker(theta_0, rep_dims, feature_dims)
    attacker.cuda()
    ## load training set
    print("Loading atk dataset ...")
    dataset = load('atk_dataset.pkl')
    criterion = nn.MSELoss()
    max_epoch = 10
    optimizer = optim.Adam(attacker.parameters(), lr=0.001)
    running_loss = 0.0
    count = 0
    batch_size = 64
    batch_count = 0
    loss = 0.0
    for epoch in range(max_epoch):
        print("Epoch {} ...".format(epoch))
        for x, theta_1 in dataset:
            count += 1
            x_prime = attacker(theta_0, reform(theta_1))
            x = torch.FloatTensor(x).reshape(-1).cuda()
            current_loss = criterion(x_prime, x)
            loss += current_loss
            running_loss += current_loss.data
            if(count % batch_size == 0):
                loss /= batch_size
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                batch_count += 1
                loss = 0.0
                if(batch_count % PRINT_FREQ == 0):
                    print("Iteration {} Loss: {:.4f}".format(batch_count, running_loss / (PRINT_FREQ * batch_size)))
                    running_loss = 0.0
                    pic = to_img(x_prime.detach().cpu().unsqueeze(0).data)
                    save_image(pic, PREFIX + 'mnist_image_{}.png'.format(count))
                    pic = to_img(x.detach().cpu().unsqueeze(0).data)
                    save_image(pic, PREFIX + 'mnist_image_{}_gt.png'.format(count))
                
    
            
            

    
    
    
            
if __name__ == '__main__':
    train_attacker()
    # collect_training_data()
