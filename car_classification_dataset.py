import os
import io
import numpy as np
from PIL import Image

import paddle
from paddle.io import Dataset

DATA_PATH = "datasets/"
CAR_CLASSIFICATION_PATH = DATA_PATH + "car_classification/"

class CarsForClassification(Dataset):
    """
    Cars Dataset for image classification. 
    """

    def __init__(self,transform=None,
                 mode='train'):
        assert mode.lower() in ['train', 'test'], \
                "mode should be 'train' or 'test', but got {}".format(mode)
        self.mode = mode.lower()
        self.data_file = CAR_CLASSIFICATION_PATH + ('train.list' if self.mode=="train" else 'test.list')
        self.transform = transform
        self.num_classes = 14
        self.data = self.read_data()

    def read_data(self):
        ret = []
        with open(self.data_file, 'r') as f:
            lines = f.readlines()
        for line in lines:
            img, label = line.split('*')
            ret.append((img,int(label)))
        return ret

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        return self.transform(img_path), int(label)

    def __len__(self):
        return len(self.data)
