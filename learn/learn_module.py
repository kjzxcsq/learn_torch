#!/home/kjzxcsq/dev/anaconda3/envs/pytorch/bin/python3
import torch.nn as nn
import torch.nn.functional as F
import torch

class Model(nn.Module):
    def __init__(self) -> None:
        super(Model, self).__init__()
    def forward(self, input):
        output = input + 1
        return output

model = Model()
model_in = torch.tensor(0.5)
model_out = model(model_in)
print(model_in)
print(model_out)

