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
from tqdm import tqdm
from cifar10 import img_transform, dataset, test_dataset, to_img, postprocess, classifier, small_classifier




if not os.path.exists('./mlp_img'):
    os.mkdir('./mlp_img')



num_epochs = 100
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
  

def get_parameter(model):
    return [param.clone().detach() for param in model.parameters()]


def train_classifier(num_epochs = 1):
    learning_rate = 1
    model =  small_classifier().cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

    theta_t_0 = get_parameter(model)
    theta_t_1 = theta_t_0
    print(len(theta_t_0))
    VERBOSE = False
    for epoch in range(num_epochs):
        running_loss = 0.0
        counter = 0
        for x, y in one_dataloader:
            # x = x.view(x.size(0), -1)
            counter += 1
            x, y = x.cuda(), y.cuda()
            print(x.mean())
            # print(x)
            preds = model(x)
            loss = criterion(preds, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ### =================== CUSTOMIZED =====================
            with torch.no_grad():
                theta_t_0 = [param.clone() for param in theta_t_1]
                # after update, obtain the parameters
                theta_t_1 = get_parameter(model)
                delta = [(-p_1 +p_0).detach() for p_1, p_0 in zip(theta_t_1, theta_t_0)]
                # print(delta)
 
                # here I have the delta, which is the gradient
            # then create a variable
            # print(x.size())
            print("============= BEGIN RECONSTRUCTION ==================")
            hat_x = torch.tensor(np.random.randn(*[n for n in x.size()])*0.1 + x.mean().item(), requires_grad = True, dtype = torch.float32)
            # hat_x = torch.tensor(np.random.randn(shape = x.size()), requires_grad = True, dtype = torch.float32)
            hat_x  = hat_x.cuda()
            hat_y = y.clone().cuda()
            reconstruction_iter_num = 500000
            eta = 0.1
            running_hvp = 0.0
            print_freq = 5000
            # small_optimizer = torch.optim.Adam([hat_x])
            # model.freeze()
            copy_from_param(model, theta_t_0)

            # model.zero_grad()
            for k in tqdm(range(reconstruction_iter_num)):
                model.zero_grad()
                preds = model(hat_x)
                loss = criterion(preds, hat_y)
                # loss.backward()
                # compute the \delta_\theta(f)
                f_delta_theta = autograd.grad([loss], model.parameters(), retain_graph = True, create_graph = True)
                # do the hessian vector product
                hvp = 0.0
                for i, g in enumerate(f_delta_theta):
                    # hvp += F.l1_loss(g, delta[i], reduction = 'sum')
                    hvp += (g * (g - 2 * delta[i]) + delta[i] * delta[i]).sum()
                    if i == 0 and VERBOSE:
                        print("g : {}".format(g))
                        print("delta[i]: {}".format(delta[i]))
                        print("g * delta[i]: {}".format(g * delta[i]))
                        print("g * g: {}".format(g * g))
                    # print(g)
                delta_x = autograd.grad([hvp], [hat_x])[0]
                # print(delta_x)
                # print(delta_x)

                # delta_x = hat_x.grad
                with torch.no_grad():
                    hat_x -= eta * delta_x
                    running_hvp += hvp.detach().cpu().data
                # import sys; sys.exit()
                # projection
                # hat_x = hat_x.clamp(-1.0, 1.0)
                if(k % print_freq == 0):
                    print("Iteration {} Loss {:.4f} HVP {:.4f} Norm {:.4f} Grad Nrom: {:.4f}".format(k, F.l1_loss(hat_x, x, reduction = 'mean'), running_hvp / print_freq, torch.norm(hat_x, p = 1), torch.norm(delta_x, p=1)))
                    if(k == 0):
                        pic = postprocess(x.detach().cpu().data)
                        save_image(pic, "one_sample/mnist_image_original.png")
                    else:
                        pic = postprocess(hat_x.detach().cpu().data)
                        save_image(pic, 'one_sample/mnist_image_{}.png'.format(k))
                        running_hvp = 0.0
            pic = postprocess(hat_x.detach().cpu().data)
            save_image(pic, 'one_sample/mnist_image_{}.png'.format(k))
            running_hvp = 0.0
            # 
            running_loss += loss.item()
            break
        print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, num_epochs, running_loss / counter))
        eval_classifier(model, test_loader)
    
        
        

def main():
   # train_autoencoder()
    train_classifier()


if __name__ == '__main__':
    main()
