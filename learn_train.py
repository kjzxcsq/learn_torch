#!/home/kjzxcsq/dev/anaconda3/envs/pytorch/bin/python3
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import time

train_data = torchvision.datasets.CIFAR10(
    root="./dataset/CIFAR10",
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=False
)

test_data = torchvision.datasets.CIFAR10(
    root="./dataset/CIFAR10",
    train=False,
    transform=torchvision.transforms.ToTensor(),
    download=False
)

train_data_len = len(train_data)
test_data_len = len(test_data)
print("train data len:", train_data_len)
print("test data len:", test_data_len)

train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# 搭建网络
class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model1 = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*4*4, 64),
            nn.Linear(64, 10)
        )
    def forward(self, x):
        x = self.model1(x)
        return x

# 创建网络模型
model = Model()
model = model.cuda()

# 损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.cuda()

# 优化器
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 训练网络的一些参数
total_train_step = 0
total_test_step = 0
epochs = 10

# tensorboard
writer = SummaryWriter("logs")

start_time = time.time()

for i in range(epochs):
    print("--------第{}轮--------".format(i+1))
    # 训练
    model.train()
    for data in train_dataloader:
        imgs, targets = data
        imgs = imgs.cuda()
        targets = targets.cuda()
        outputs = model(imgs)
        loss = loss_fn(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            end_time = time.time()
            print("训练次数:{}, Loss:{}, 用时:{}".format(total_train_step, loss, end_time-start_time))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 测试
    model.eval()
    total_test_loss = 0
    total_test_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            imgs = imgs.cuda()
            targets = targets.cuda() 
            outputs = model(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_test_accuracy = total_test_accuracy + accuracy
    print("整体Loss:{}".format(total_test_loss))
    print("整体正确率:{}".format(total_test_accuracy/test_data_len))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_test_accuracy/test_data_len, total_test_step)
    total_test_step = total_test_step + 1

    # torch.save(model.state_dict(), "model_{}.pth".format(i))

writer.close()


