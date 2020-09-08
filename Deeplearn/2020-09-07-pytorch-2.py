import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
import torchvision
from matplotlib import pyplot as plt
from utils import plot_image, plot_curve,one_hot

# 加载数据包
batch_size=512

train_loader=torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('Data\lesson05',train=True,download=True,
    transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            (0.1307,),(0.3081,))
    ])),
    batch_size=batch_size,shuffle=True)

test_loader=torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('Data\lesson05',train=False,download=True,
    transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            (0.1307,),(0.3081,))
    ])),
    batch_size=batch_size,shuffle=False)

x,y=next(iter(train_loader))
print(x.shape,y.shape,x.min(),x.max())

class Net(nn.Module):

    def __init__(self):
        super().__init__()