import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets, utils
from torch.utils import data
import time
from tqdm import tqdm
import numpy as np
import torchvision
from torch.autograd import Variable
from PIL import Image

classes = ['c1', 'c10', 'c11', 'c12', 'c13', 'c14', 'c15', 'c16', 'c17', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9']

image_path = r'D:\17flowers'

transforms_test=transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

device = torch.device('cuda:0')
model = torch.load(r'Acc0.9919117647058824.pth')
model.eval()
model.to(device)

dataset_test = datasets.ImageFolder(root=image_path + "/val", transform=transforms_test)
print(f'length:{len(dataset_test)}')

#===================================================================================================
#test checkpoint model by val dataset
test_acc=0
for index in tqdm(range(len(dataset_test))):
    item = dataset_test[index]
    img, label = item
    img.unsqueeze_(0)
    data = Variable(img).to(device)
    output = model(data)
    _, pred = torch.max(output.data, 1)


    print(f'Image Name:{dataset_test.imgs[index][0]}, predict:{classes[pred.data.item()]}')
    acc_count = torch.sum(pred == label).item()
    test_acc+=acc_count
print(f'test acc count:{test_acc}')
print(f'test acc:{100*test_acc/len(dataset_test)}')
#=================================================================================================

#====================================================================================================
# single image infere
# input_img = r'D:\17flowers\train\c17/image_1281.jpg'
# # image = cv2.imread(input_img)
# # image = image.transpose((2,0,1))
# # image = np.array(image, dtype='float')
# image = Image.open(input_img)
# image = transforms_test(image)
#
# image = image.unsqueeze(0)
#
# data = Variable(image).to(device)
# with torch.no_grad():
#     output = model(data)
#     _, pred = torch.max(output.data, 1)
#     print(f'predict: {classes[pred.data.item()]}')
#     topk_prob_values, topk_class_indices = torch.topk(output.data, k=5, dim=-1)
#     print(f'topk: {topk_prob_values} -- topk_indices:{topk_class_indices}')
#============================================================================================

















