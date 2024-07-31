import os
from shutil import copy
import random

def mkfile(file):
    if not os.path.exists(file):
        os.makedirs(file)

file_path = r'D:\17flowers'
flower_class = [cla for cla in os.listdir(file_path) if ".txt" not in cla]
print(flower_class)

if "train" not in flower_class:
    mkfile(file_path+'/train')
    for cla in flower_class:
        mkfile(file_path + '/train/' + cla)

if "val" not in flower_class:
    mkfile(file_path+'/val')
    for cla in flower_class:
        mkfile(file_path + '/val/' + cla)

#all_in
count_class = 0
for cla in flower_class:
    if cla == "train" or cla == "val":
        continue
    cla_path = file_path + '/' + cla + '/'
    images = os.listdir(cla_path)
    print(images)
    num = len(images)
    print(f'num: {num}')


    for index, image in enumerate(images):
        #train

        image_path = cla_path + image
        train_class_path = file_path + '/train/' + cla
        copy(image_path, train_class_path)

        #val
        val_class_path = file_path + '/val/' + cla
        copy(image_path, val_class_path)

        print(f'[{count_class+1}: {cla}] processing [{index+1}/{num}]')
        if int(index)==79:
            count_class+=1

print("process done!!!")







