from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy 
import matplotlib.pyplot as plt
from torch._C import set_flush_denormal
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from load_mnist import load_mnist
import sklearn.preprocessing


class FMN(Dataset):
    def __init__(self, transform=None):
 
        self.root_dir, self.landmarks_frame = load_mnist(dataset="training", path=".")
        self.transform = transform

    
        self.landmarks_frame = self.landmarks_frame.numpy()
        result1 = numpy.where(self.landmarks_frame == 0)
        result2 = numpy.where(self.landmarks_frame == 1)
        result = numpy.concatenate((result1,result2),axis = 1)
        result = result.ravel()
        result = numpy.sort(result)
        self.id = torch.tensor(result)
        self.landmarks_frame = self.landmarks_frame[result]
        self.root_dir = self.root_dir[result]

#Image scaling 
        arr = numpy.empty((28,28))
        for i in range(len(self.root_dir)):
            np_array = self.root_dir[i].numpy()
            arr1 = sklearn.preprocessing.minmax_scale(np_array, feature_range=(-1,1))
            arr = numpy.concatenate((arr,arr1))
        arr = arr[28:,:]   
        arr = numpy.reshape(arr,self.root_dir.shape) 
        self.root_dir = arr 

#Function __len__
    def __len__(self):
        return len(self.landmarks_frame)

#Function __getitem__
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = self.landmarks_frame
        img  = self.root_dir[idx,:,:]
        label = self.landmarks_frame[idx]
        sample = {'image' : img, 'label':label}
        return sample         

#Testing for the Dataset
fmn = FMN(Dataset)

for i in range(12000):
    sample = fmn[i]
    print(sample['image'].shape,sample['label'])

#Dataloader    
dataloader = torch.utils.data.DataLoader(fmn, batch_size=32,shuffle=True, drop_last=False)

for (idx,sample) in enumerate(dataloader):
    print(idx,sample['image'].shape,sample['label'])