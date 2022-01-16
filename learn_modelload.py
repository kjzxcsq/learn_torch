#!/home/kjzxcsq/dev/anaconda3/envs/pytorch/bin/python3
import torch
import torchvision
# from learn_sequential import Model

# method 1
model_1 = torch.load("./models/vgg16_model_1.pth")
print(model_1)

# method 2
model_2 = torchvision.models.vgg16(pretrained=False)
model_2.load_state_dict(torch.load("./models/vgg16_model_2.pth"))
print(model_2)

