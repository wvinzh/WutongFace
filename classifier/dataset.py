import torch
from torch.utils.data import Dataset
import numpy as np
import os

def read_npy(npy_path):
    return np.load(npy_path).reshape(-1)

def load_dataset(file_root, train_txt):
    train_x = []
    train_y = []
    val_x = []
    val_y = []
    with open(train_txt, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines:
        file_path, label = line.strip().split()
        train_x.append(read_npy(os.path.join(file_root, file_path)))
        train_y.append(int(label))

    return (np.array(train_x), np.array(train_y))

class ClassifierDataset(Dataset):
    def __init__(self,file_root, train_txt):
        self.file_root = file_root
        self.data = load_dataset(file_root,train_txt)

    def __len__(self):
        return len(self.data[0])

    
    def __getitem__(self,index):

        x = self.data[0][index]
        x = torch.from_numpy(x)
        y = self.data[1][index]+1
        y2 = 0
        if y > 0:
            y2 = 1
        return (x,y,y2)