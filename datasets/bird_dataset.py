#-----------------------------------------------------------------------------
# Torch Dataset Class object of Bird data
# Tutorial: Read https://github.com/utkuozbulak/pytorch-custom-dataset-examples#a-custom-custom-custom-dataset
# Date Created: 1/23/21
# Author: Nick Lin
# ----------------------------------------------------------------------------
import torch
import os
import pandas as pd
import numpy as np
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

class BirdDataset(Dataset):
    """
    Images of birds dataset
    """
    def __init__(self, csv_path:str, root_dir:str):
        """
        @param root_dir: directory with all the images of bird. Directory
        structure should be Directory -> Bird Category -> Images
        @param csv_path: path to csv file
        """
        self.root_dir = root_dir
        self.df = pd.read_csv(csv_path)
        self.labels = self.df.iloc[:,1] # for stratified train-test-val split
        # Static transformation of bird image. 
        # Basic Preprocessing: https://pytorch.org/hub/pytorch_vision_alexnet
        self.transform = transforms.Compose([
            #transforms.Resize(256),
            transforms.Resize(224), # Want to utilize the full img
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], # imgNet mean, std
                                 [0.229, 0.224, 0.225])
        ])

    def __len__(self) -> int:
        """
        @return number of samples in bird dataset
        """
        return len(self.df)
    
    def _onehot(self, label: str) -> torch.Tensor:
        """
        Helper Func, one-hot encode labels of the Bird Dataset
        @param label: what bird (based on spelling on folders)
        @return: tensor of onehot 
        """
        categories = os.listdir(self.root_dir)
        categories_to_int = {category: index for index, category in 
                enumerate(categories)}
        one_hot_arr = np.zeros(len(categories))
        one_hot_arr[categories_to_int[label]] = 1
        one_hot_tensor = torch.Tensor(one_hot_arr)
        return one_hot_tensor
    
    def _intlabel(self, label: str) -> torch.Tensor:
        """
        Helper Func, int encode labels of the Bird Dataset
        @param label: what bird (based on spelling on folders)
        @return: int label 
        """
        categories = os.listdir(self.root_dir)
        categories_to_int = {category: index for index, category in 
                enumerate(categories)}
        one_hot_arr = np.zeros(len(categories))
        int_label = categories_to_int[label]
        return int_label

    def getFilename(self, index:int) -> str:
        """
        returns the filename
        """
        return self.df[index, 0]
    
    def __getitem__(self, index:int) -> (Image, torch.Tensor):
        """
        @param index: sample index
        @return transformed image, and label of index
        """
        row = self.df.iloc[index]
        # img
        img_path = row[0]
        img = Image.open(os.path.join("..", "data", "images", img_path))
        # label
        label = row[1]
        return (self.transform(img), self._intlabel(label))
