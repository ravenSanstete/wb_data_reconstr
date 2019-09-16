from torchvision import transforms
from torchvision.datasets import CIFAR10
import numpy as np
import torch
import torchvision
from torch import nn
import torch.nn.functional as F
from scipy import ndimage


img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])



dataset = CIFAR10('./data', transform=img_transform, download = True)
test_dataset = CIFAR10('./data', train=False, transform=img_transform)


def to_img(x):
    mean = torch.FloatTensor([0.2023, 0.1994, 0.2010])
    std = torch.FloatTensor([0.4914, 0.4822, 0.4465])
    unnormalize = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())
    x = torch.FloatTensor(np.array([unnormalize(x[i, :, :, :]).numpy() for i in range(x.size(0))]))
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 32, 32)
    return x
def postprocess(x):
    mean = torch.FloatTensor([0.2023, 0.1994, 0.2010])
    std = torch.FloatTensor([0.4914, 0.4822, 0.4465])
    unnormalize = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())
    x = torch.FloatTensor(np.array([unnormalize(x[i, :, :, :]).numpy() for i in range(x.size(0))]))
    # x = x.clamp(0, 1)
    # blurred x
    filter_x = torch.FloatTensor(ndimage.gaussian_filter(x, 1))
    alpha = 0.3
    sharpened = x + alpha * (x - filter_x)
    return sharpened


class classifier(nn.Module):
    """CNN."""

    def __init__(self):
        """CNN Builder."""
        super(classifier, self).__init__()

        self.conv_layer = nn.Sequential(

            # Conv Layer block 1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv Layer block 2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.05),

            # Conv Layer block 3
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )


        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512, 10)
        )


    def forward(self, x):
        """Perform forward."""
        
        # conv layers
        x = self.conv_layer(x)
        
        # flatten
        x = x.view(x.size(0), -1)
        
        # fc layer
        x = self.fc_layer(x)

        return x


class small_classifier(nn.Module):
    def __init__(self):
        super(small_classifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
