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


import warnings
warnings.filterwarnings("ignore")

plt.ion()


class FMN(Dataset):
#class FMN():
    """Face Landmarks dataset."""

    def __init__(self, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        # self.landmarks_frame = pd.DataFrame(label)
        # self.root_dir = 
        self.root_dir, self.landmarks_frame = load_mnist(dataset="training", path=".")
        self.transform = transform

    ######## 0 and 1 index
        self.landmarks_frame = self.landmarks_frame.numpy()
        result1 = numpy.where(self.landmarks_frame == 0)
        result2 = numpy.where(self.landmarks_frame == 1)
        result = numpy.concatenate((result1,result2),axis = 1)
        result = result.ravel()
        result = numpy.sort(result)
        self.id = torch.tensor(result)
        
        #print(len(result))
        
        #print(idx)
        #idx = idx.ravel()
        self.landmarks_frame = self.landmarks_frame[result]
        self.root_dir = self.root_dir[result]
        
        # print(self.root_dir.shape)
        #print(self.landmarks_frame , len(self.root_dir))

    #######mean and variance


        # dataset = TensorDataset(self.root_dir, self.landmarks_frame)
        # loader = DataLoader(dataset, batch_size=8)

        # nimages = 0
        # mean = 0.
        # std = 0.
        # for batch, _ in loader:
        #     # Rearrange batch to be the shape of [B, C, W * H]
        #     batch = batch.view(batch.size(0), batch.size(1), -1)
        #     # Update total number of images
        #     nimages += batch.size(0)
        #     # Compute mean and std here
        #     mean += batch.mean(2).sum(0) 
        #     std += batch.std(2).sum(0)

        #     # Final step
        # mean /= nimages
        # std /= nimages

        # print(mean)
        # print(std)

        # self.mean = 
        # self.variance = 


    ######## image scale
        self.root_dir = sklearn.preprocessing.minmax_scale(self.root_dir, feature_range=(-1,1))
        

    ########processing step iinclude
        


    def __len__(self):
        return len(self.landmarks_frame)


    def __getitem__(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # print("idx",idx)
        #self.landmarks_frame = str(self.landmarks_frame[idx])
        img_name = self.landmarks_frame
        
        
        #image = io.imread(img_name)
        #print(image)
        #landmarks = self.landmarks_frame[idx]
        # landmarks = numpy.array([landmarks])
        # landmarks = landmarks.astype('float').reshape(-1, 2)
        # sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        img  = self.root_dir[idx,:,:]
        label = self.landmarks_frame[idx]
        sample = {'image' : img, 'label':label}
        return sample         

# #dataloader = torch.utils.data.DataLoader(dataset, batch_size=32,
# # shuffle=True, drop_last=False)



fmn = FMN(Dataset)

for i in range(12000):

    sample = fmn[i]
    print(sample['image'].shape,sample['label'])
    # print(sample)
# dataloader = torch.utils.data.DataLoader(fmn, batch_size=32,
# shuffle=True, drop_last=False)
