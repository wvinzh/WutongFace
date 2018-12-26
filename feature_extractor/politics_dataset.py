import os
import torch
from torch.utils import data
import torchvision.transforms
from PIL import Image
import cv2
import numpy as np
class PoliticalDataset(data.Dataset):

    def __init__(self, root_folder, image_list, image_list_file, transform=None):
        self.root = root_folder
        self.img_list = image_list
        self.transform = transform
        self.data_info = []
        with open(image_list,'r') as f:
            for i, line in enumerate(f):
                line = line.strip()
                category = line.split('/')[0]
                line_split = line.split()
                image_file = os.path.join(root_folder,line_split[0])
                if len(line_split) > 1:
                    label = int(line_split[1])
                else:
                    label = -1
                basename = os.path.basename(image_file)
                basename = os.path.splitext(basename)[0]
                self.data_info.append((image_file,category,basename,label))
        # print(len(self.data_info),self.data_info[20])
    
    def __len__(self):
        return len(self.data_info) 

    def __getitem__(self, index):
        img_file,category,basename,label = self.data_info[index]
        img = self.transform(img_file)
        return (img,category,basename,label)


def transform_resnet(img_file):
    img = Image.open(img_file).convert('RGB')
    img = torchvision.transforms.Resize(256)(img)
    img = torchvision.transforms.CenterCrop(224)(img)
    img = np.array(img, dtype=np.uint8)
    mean_bgr = np.array([91.4953, 103.8827, 131.0912])
    img = img[:, :, ::-1]  # RGB -> BGR
    img = img.astype(np.float32)
    img -= mean_bgr
    img = img.transpose(2, 0, 1)  # C x H x W
    img = torch.from_numpy(img).float()
    return img

def transform_insightface(img_file):
    img = Image.open(img_file).convert('RGB')
    img = torchvision.transforms.Resize(112)(img)
    # img = torchvision.transforms.CenterCrop(112)(img)
    img = np.array(img, dtype=np.uint8)
    img = img[:,:,::-1] # transform RGB 2 BGR
    img = Image.fromarray(img)
    # img.show()
    img = torchvision.transforms.ToTensor()(img)
    img = torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(img)
    return img

def transform_sphereface(img_file):
    img = Image.open(img_file).convert('RGB')
    img = torchvision.transforms.Resize((116,100))(img)
    img = torchvision.transforms.CenterCrop((112,96))(img)
    img = np.array(img, dtype=np.uint8)
    img = img[:,:,::-1] # transform RGB 2 BGR
    img = img.transpose(2, 0, 1)
    img = ( img - 127.5 ) / 128.0
    img = torch.from_numpy(img).float()
    return img

def transform_cosface(img_file):
    img = Image.open(img_file).convert('RGB')
    img = torchvision.transforms.Resize((112,96))(img)
    img = torchvision.transforms.ToTensor()(img)
    img = torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(img)
    return img
