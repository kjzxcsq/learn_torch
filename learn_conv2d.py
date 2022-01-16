#!/home/kjzxcsq/dev/anaconda3/envs/pytorch/bin/python3
from asyncore import write
from modulefinder import Module
from turtle import forward
import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10(
    root="./dataset/CIFAR10",
    train=False,
    transform=torchvision.transforms.ToTensor(),
    download=False)

dataloader = DataLoader(dataset, 64)

class Model(torch.nn.Module):
    def __init__(self) -> None:
        super(Model, self).__init__()
        self.conv2d = torch.nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)

    def forward(self, x):
        x = self.conv2d(x)
        return x

model = Model()
writer = SummaryWriter("logs")
step = 0
for data in dataloader:
    imgs, targets = data
    output = model(imgs)
    print(imgs.shape)
    print(output.shape)

    # torch.Size([64, 3, 32, 32])
    writer.add_images("input", imgs, step)

    # torch.Size([64, 6, 30, 30])  ->  [xx, 3, 30, 30]
    output = torch.reshape(output, (-1, 3, 30, 30))
    writer.add_images("output", output, step)

    step = step + 1

writer.close()
