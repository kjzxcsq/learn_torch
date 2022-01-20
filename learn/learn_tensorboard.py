#!/home/kjzxcsq/dev/anaconda3/envs/pytorch/bin/python3
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image

image_path = "dataset/practice/train/ants_image/0013035.jpg"
img_PIL = Image.open(image_path)
img_array = np.array(img_PIL)
# print(img_array.shape)

writer = SummaryWriter("logs")
writer.add_image("test", img_array, 1, dataformats='HWC')
# for i in range(100):
#     writer.add_scalar("y=x^2", pow(i, 0.5), i)

writer.close()