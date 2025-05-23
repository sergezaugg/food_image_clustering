#--------------------             
# Author : Serge Zaugg
# Description : 
#--------------------

import os 
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.io import decode_image
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import densenet121, DenseNet121_Weights 
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights 
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
from torchvision.models import vit_b_16, ViT_B_16_Weights


torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

image_path = "D:/image_clust/food_images/train_images"
label_path = "D:/image_clust/food_images/train_img.csv"
featu_path = "./extracted_features"

class SpectroImageDataset(Dataset):
    """
    Description: A simple PyTorch dataset (loader) to batch process images from file
    """
    def __init__(self, imgpath, label_path):
        self.all_img_files = np.array([a for a in os.listdir(imgpath) if '.jpg' in a])
        self.imgpath = imgpath     
        # get of class names from csv (not same order as files!)
        df_labels = pd.read_csv(label_path)
        # converts 2-cols df to a handy dict
        df_labels = df_labels.set_index(keys="ImageId")
        dilab = df_labels.to_dict()
        self.dilab = dilab['ClassName']

    def __getitem__(self, index):     
        img = decode_image(os.path.join(self.imgpath,  self.all_img_files[index] )  )  
        # Apply inference preprocessing transforms
        img = preprocess(img) # .unsqueeze(0)
        # get filename 
        filename = self.all_img_files[index]
        # make array of true classes
        true_class = self.dilab[filename.item()] 
        # 
        return (img, filename, true_class)
    
    def __len__(self):
        return (len(self.all_img_files))
    

# Step 1: Initialize model with the best available weights

# model_tag = "ResNet50_IMAGENET1K_V2"
# weights = ResNet50_Weights.IMAGENET1K_V2
# model = resnet50(weights=weights)

# model_tag = "DenseNet121_IMAGENET1K_V1"
# weights = DenseNet121_Weights.IMAGENET1K_V1
# model = densenet121(weights=weights)

# model_tag = "MobileNet_V3_Large_IMAGENET1K_V2"
# weights = MobileNet_V3_Large_Weights.IMAGENET1K_V2
# model = mobilenet_v3_large(weights=weights)

model_tag = "vit_b_16_Weights"
weights = ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1
model = vit_b_16(weights=weights)


# vit_b_16


# Step 2: Initialize the inference transforms
model.eval()
preprocess = weights.transforms()
dataset = SpectroImageDataset(image_path, label_path)
loader = torch.utils.data.DataLoader(dataset, batch_size=16,  shuffle=False, drop_last=False)

X_li = []
N_li = []
Y_li = []
for ii, (batch, finam, true_class) in enumerate(loader, 0):
    print(batch.shape)
    print(np.array(finam).shape)
    prediction = model(batch).detach().numpy()  #.squeeze(0)
    file_names = np.array(finam)
    true_class = np.array(true_class)
    X_li.append(prediction)
    N_li.append(file_names)
    Y_li.append(true_class)

X = np.concatenate(X_li)
N = np.concatenate(N_li)
Y = np.concatenate(Y_li)

# check dims
print(X.shape, Y.shape, N.shape)

# save as npz
out_name = os.path.join(featu_path, 'features_' + model_tag + '.npz')
np.savez(file = out_name, X = X, Y = Y, N = N)



