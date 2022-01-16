#!/home/kjzxcsq/dev/anaconda3/envs/pytorch/bin/python3
from turtle import forward
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.relu = nn.ReLU(inplace=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, input):
        output = self.sigmoid(input)
        return output

model = Model()

dataset = torchvision.datasets.CIFAR10(
    root="./dataset/CIFAR10",
    train=False,
    transform=torchvision.transforms.ToTensor(),
    download=False
)
dataloader = DataLoader(dataset, batch_size=64)

writer = SummaryWriter("logs")
step = 0
for data in dataloader:
    imgs ,targets = data
    output = model(imgs)
    writer.add_images("input", imgs, step)
    writer.add_images("output", output, step)
    step = step + 1

writer.close()

