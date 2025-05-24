#--------------------             
# Author : Serge Zaugg
# Description : Functions an classes specific to ML/PyTorch backend
#--------------------

import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torchvision.io import decode_image

class ImageDataset(Dataset):
    """
    Description: A simple PyTorch dataset (loader) to batch process images from file
    """
    def __init__(self, imgpath, label_path, preprocess):
        """
        imgpath (str) : path to a dir that contains JPG images. 
        label_path (str) : path to a csv file which matches PNG filenames with labels
        preprocess ('torchvision.transforms._presets.ImageClassification'>) : preprocessing transforms provided with the pretrained models
        """
        self.all_img_files = np.array([a for a in os.listdir(imgpath) if '.jpg' in a])
        self.imgpath = imgpath   
        self.preprocess = preprocess  
        # get of class names from csv (not same order as files!)
        df_labels = pd.read_csv(label_path)
        # converts 2-cols df to a handy dict
        df_labels = df_labels.set_index(keys="ImageId")
        dilab = df_labels.to_dict()
        self.dilab = dilab['ClassName']

    def __getitem__(self, index):     
        img = decode_image(os.path.join(self.imgpath,  self.all_img_files[index]))  
        # Apply inference preprocessing transforms
        img = self.preprocess(img) # .unsqueeze(0)
        # get filename 
        filename = self.all_img_files[index]
        # make array of true classes
        true_class = self.dilab[filename.item()] 
        # 
        return (img, filename, true_class)
    
    def __len__(self):
        return (len(self.all_img_files))
    