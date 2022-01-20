#!/home/kjzxcsq/dev/anaconda3/envs/pytorch/bin/python3
import torch
import torchvision
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10(
    root="./dataset/CIFAR10",
    train=False,
    transform=torchvision.transforms.ToTensor(),
    download=False
)
dataloader = DataLoader(dataset, batch_size=1)

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

loss = nn.CrossEntropyLoss()
optim1 = torch.optim.SGD(model.parameters(), lr=0.01)
for epoch in range(20):
    running_loss = 0.0
    for data in dataloader:
        imgs, targets = data
        outputs = model(imgs)
        result_loss = loss(outputs, targets)
        optim1.zero_grad()
        result_loss.backward()
        optim1.step()
        running_loss = result_loss + running_loss
    print(running_loss)


# writer = SummaryWriter("logs")
# writer.add_graph(model, input)
# writer.close()

