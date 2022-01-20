#!/home/kjzxcsq/dev/anaconda3/envs/pytorch/bin/python3
import torchvision
import torch

vgg16_false = torchvision.models.vgg16(pretrained=False)
vgg16_true = torchvision.models.vgg16(pretrained=False)

train_data = torchvision.datasets.CIFAR10(
    root="./dataset/CIFAR10",
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=False
)

vgg16_true.classifier.add_module(
    name="add_linear",
    module=torch.nn.Linear(1000, 10)
)
print(vgg16_true)

vgg16_false.classifier[6] = torch.nn.Linear(4096, 10)
print(vgg16_false)
