#!/home/kjzxcsq/dev/anaconda3/envs/pytorch/bin/python3
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model1 = nn.Sequential(
            nn.Conv2d(3, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1024, 64),
            nn.Linear(64, 10)
        )
    def forward(self, input):
        input = self.model1(input)
        return input

model = Model()
print(model)

input = torch.ones((64, 3, 32, 32))
output = model(input)
print(output.shape)

writer = SummaryWriter("logs")
writer.add_graph(model, input)
writer.close()
