#!/home/kjzxcsq/dev/anaconda3/envs/pytorch/bin/python3
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

test_set = torchvision.datasets.CIFAR10(
    root="./dataset/CIFAR10", 
    train=False, 
    transform=torchvision.transforms.ToTensor(), 
    download=True)

test_loader = DataLoader(
    dataset=test_set,
    batch_size = 64,
    shuffle=True,
    num_workers=0,
    drop_last=False)

img, target = test_set[0]
print(img.shape)
print(target)

writer = SummaryWriter("logs")
step = 0
for data in test_loader:
    imgs, targets = data
    # print(imgs.shape)
    # print(targets)
    writer.add_images("test_set", imgs, step)
    step = step + 1
writer.close()
