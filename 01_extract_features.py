

# data from 
# https://www.kaggle.com/datasets/bjoernjostein/food-classification

# code initial from from
# https://docs.pytorch.org/vision/stable/models.html

from torchvision.io import decode_image
from torchvision.models import resnet50, ResNet50_Weights
import os 
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

image_path = "D:/image_clust/food_images/train_images"
label_path = "D:/image_clust/food_images/train_img.csv"
featu_path = "D:/image_clust/features/features.npz"

# Step 1: Initialize model with the best available weights
weights = ResNet50_Weights.DEFAULT

weights = ResNet50_Weights.IMAGENET1K_V2

model = resnet50(weights=weights)
model.eval()

# Step 2: Initialize the inference transforms
preprocess = weights.transforms()




class SpectroImageDataset(Dataset):
    """
    
    """

    def __init__(self, imgpath):
        self.all_img_files = np.array([a for a in os.listdir(imgpath) if '.jpg' in a])
        self.imgpath = imgpath
    
    def __getitem__(self, index):     
        img = decode_image(os.path.join(self.imgpath,  self.all_img_files[index] )  )  
        # Apply inference preprocessing transforms
        img = preprocess(img) # .unsqueeze(0)
        # get filename 
        filename = self.all_img_files[index]
        return (img, filename)
    
    def __len__(self):
        return (len(self.all_img_files))

dataset = SpectroImageDataset(image_path)
loader = torch.utils.data.DataLoader(dataset, batch_size=32,  shuffle=False, drop_last=False)

X_li = []
Y_li = []
for ii, (batch, finam) in enumerate(loader, 0):
    print(batch.shape)
    print(np.array(finam).shape)
    prediction = model(batch).detach().numpy()  #.squeeze(0)
    file_names = np.array(finam)
    X_li.append(prediction)
    Y_li.append(file_names)

X = np.concatenate(X_li)
Y = np.concatenate(Y_li)

X.shape
Y.shape

# get array of class names from filenames 
# load class labels 
df_labels = pd.read_csv(label_path)
df_labels.shape
df_labels.head()
# converts 2-cols df to a handy lil dict
df_labels = df_labels.set_index(keys="ImageId")
dilab = df_labels.to_dict()
dilab = dilab['ClassName']

Y = np.array([dilab[a.item()] for a in  Y])

X.shape
Y.shape

np.savez(file = featu_path, X = X, Y = Y)



