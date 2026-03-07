# -*- coding: utf-8 -*-

import os
import pandas as pd
from PIL import Image, ImageFile
from torchvision import transforms
from torch.utils.data import Dataset
ImageFile.LOAD_TRUNCATED_IMAGES = True


class MyDataset(Dataset):
    def __init__(self, csv_path, mode, class_name, image_size):
        self.mode = mode
        self.class_name = class_name
        self.image_size = image_size
        data = pd.read_csv(csv_path)
        labels = data[class_name]
        labels = labels.fillna(value=0)
        labels = labels.replace(-1, 1)
        labels = labels.values
        imgs = []
        for index, row in data.iterrows():
            label = labels[index, :]
            imgs.append((row['Path'], label))
            
        self.imgs = imgs
        
    def __getitem__(self, index):
        path, label = self.imgs[index]
        p1 = path.replace("images","Augmentation_mask_files/transformed_images").replace(".png","_t0.png")
        p2 = path.replace("images","Augmentation_mask_files/transformed_images").replace(".png","_t1.png")
        p3 = path.replace("images","Augmentation_mask_files/transformed_images").replace(".png","_t2.png")
        p4 = path.replace("images","Augmentation_mask_files/transformed_images").replace(".png","_t3.png")
        p5 = path.replace("images","Augmentation_mask_files/transformed_images").replace(".png","_t4.png")
        
        im = Image.open(path).convert('L')
        filename = os.path.basename(path)  # "0.png"
        name_without_ext = os.path.splitext(filename)[0]  # "0"
        dir_name = os.path.basename(os.path.dirname(path))  # "CXR3030_IM-1405"
        img_paths = f"{dir_name}_{name_without_ext}"  # "CXR3030_IM-1405_0"
        
        im = im.convert('RGB')

        im1 = Image.open(p1).convert('L').convert('RGB')
        im2 = Image.open(p2).convert('L').convert('RGB')
        im3 = Image.open(p3).convert('L').convert('RGB')
        im4 = Image.open(p4).convert('L').convert('RGB')
        im5 = Image.open(p5).convert('L').convert('RGB')
        
        transform = get_transform(self.mode, self.image_size)
        transform_transformation = get_transform_transformation(self.mode, self.image_size)
        im = transform(im)      
        im_t1 = transform_transformation(im1)
        im_t2 = transform_transformation(im2)
        im_t3 = transform_transformation(im3)
        im_t4 = transform_transformation(im4)
        im_t5 = transform_transformation(im5)
        return {'im': im, 'im1': im_t1, 'im2': im_t2, 'im3': im_t3, 'im4': im_t4, 'im5': im_t5, 'label': label, "img_path" : img_paths} #'

    def __len__(self):
        return len(self.imgs)

def get_transform_transformation(mode='train', image_size=224):
    return transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
def get_transform(mode='train', image_size=224):
    transform_list = []
    
    if image_size == 224:        
        transform_list = [transforms.Resize(256)]
        if mode == 'train':
            transform_list += [transforms.RandomCrop(224)]
            transform_list += [transforms.RandomHorizontalFlip()]
            
        else:
            transform_list += [transforms.CenterCrop(224)]
    else:
        if mode == 'train':
            transform_list += [transforms.RandomHorizontalFlip()]
    
    transform_list += [transforms.ToTensor()]
    transform_list += [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        
    return transforms.Compose(transform_list)
