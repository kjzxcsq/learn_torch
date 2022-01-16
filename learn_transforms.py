#!/home/kjzxcsq/dev/anaconda3/envs/pytorch/bin/python3
from torchvision import transforms
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("logs")
img = Image.open("dataset/practice/train/ants_image/0013035.jpg")
# print(img)

# ToTensor
trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)
writer.add_image("ToTensor", img_tensor)

# Normalize
trans_norm = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
img_norm = trans_norm(img_tensor)
writer.add_image("Normalize", img_norm)

# Resize
# print(img.size)
trans_resize = transforms.Resize((640, 640))
img_resize = trans_resize(img)
img_resize = trans_totensor(img_resize)
# print(img_resize)
writer.add_image("Resize", img_resize, 0)

# Compose - resize -2
trans_resize_2 = transforms.Resize(512)
# PIL -> PIL -> Tensor
trans_compose = transforms.Compose([trans_resize_2, trans_totensor])
img_compose = trans_compose(img)
writer.add_image("Resize", img_compose, 1)

# RandomCrop
trans_random = transforms.RandomCrop((50, 100))
trans_compose_2 = transforms.Compose([trans_random, trans_totensor])
for i in range(10):
    img_crop = trans_compose_2(img)
    writer.add_image("RandomCropHW", img_crop, i)


writer.close()