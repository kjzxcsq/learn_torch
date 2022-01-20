#!/home/kjzxcsq/dev/anaconda3/envs/pytorch/bin/python3
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=3, ceil_mode=True)
    def forward(self, input):
        output = self.maxpool(input)
        return output


dataset = torchvision.datasets.CIFAR10(
    root="./dataset/CIFAR10",
    train=False,
    transform=torchvision.transforms.ToTensor(),
    download=False
)

dataloader = DataLoader(dataset, batch_size=64)

model = Model()
writer = SummaryWriter("logs")
step = 0
for data in dataloader:
    imgs, targets = data
    writer.add_images("input", imgs, step)
    output = model(imgs)
    writer.add_images("output", output, step)
    step = step + 1

writer.close()
