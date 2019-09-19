from torchvision import transforms
from torchvision.datasets import MNIST

import torch
import torchvision
from torch import nn
import torch.nn.functional as F
from torch.nn import Linear

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


dataset = MNIST('./data', transform=img_transform, download = True)
test_dataset = MNIST('./data', train=False, transform=img_transform)



def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)

    return x
def postprocess(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    return x

    
class classifier(nn.Module):
    def __init__(self):
        super(classifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        return

class small_classifier(nn.Module):
    def __init__(self):
        super(small_classifier, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        x = self.fc4(x)
        return x

    

    
class Reconstructor(nn.Module):
    def __init__(self, in_dim, input_dim):
        super(Reconstructor, self).__init__()
        self.module = nn.Sequential(
            Linear(in_dim, input_dim),
            nn.Tanh())

    def forward(self, x):
        return self.module(x)

