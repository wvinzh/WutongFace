import torchvision
import torch
from PIL import Image
import numpy as np

class PoliticTransforms():
    def __init__(self, *args, **kwargs):
        return super().__init__(*args, **kwargs)
    
    def transform_resnet(img):
            # img = Image.open(img_file).convert('RGB')
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

    def transform_insightface(img):
        # img = Image.open(img_file).convert('RGB')
        img = torchvision.transforms.Resize(112)(img)
        # img = torchvision.transforms.CenterCrop(112)(img)
        img = np.array(img, dtype=np.uint8)
        img = img[:,:,::-1] # transform RGB 2 BGR
        img = Image.fromarray(img)
        # img.show()
        img = torchvision.transforms.ToTensor()(img)
        img = torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(img)
        return img

    def transform_sphereface(img):
        # img = Image.open(img_file).convert('RGB')
        img = torchvision.transforms.Resize((116,100))(img)
        img = torchvision.transforms.CenterCrop((112,96))(img)
        img = np.array(img, dtype=np.uint8)
        img = img[:,:,::-1] # transform RGB 2 BGR
        # img = Image.fromarray(img)
        # img.show()
        img = img.transpose(2, 0, 1)
        # print(img.shape)
        img = ( img - 127.5 ) / 128.0
        img = torch.from_numpy(img).float()
        return img

    def transform_cosface(img):
        # img = Image.open(img_file).convert('RGB')
        img = torchvision.transforms.Resize((112,96))(img)
        # img.show()
        # img = torchvision.transforms.CenterCrop((112,96))(img)
        img = torchvision.transforms.ToTensor()(img)
        img = torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(img)
        return img 