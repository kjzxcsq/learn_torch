#!/home/kjzxcsq/dev/anaconda3/envs/pytorch/bin/python3
import torch
import torchvision

vgg16_model = torchvision.models.vgg16(pretrained=False)

# method 1  模型结构+模型参数
torch.save(vgg16_model, "./models/vgg16_model_1.pth")

# method 2  模型参数（官方推荐）
torch.save(vgg16_model.state_dict(), "./models/vgg16_model_2.pth")

