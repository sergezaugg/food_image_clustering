#--------------------             
# Author : Serge Zaugg
# Description : A flat-script to extract features from images via last layer of pre-trained PyTorch models
#--------------------

import os 
import numpy as np
import datetime
import torch
from pt_extract_features.utils_pt import ImageDataset, load_pretraind_model
torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#-------------------------
# set params 
image_path = "D:/image_clust/food_images/train_images"
label_path = "D:/image_clust/food_images/train_img.csv"
featu_path = "./extracted_features"
batch_size = 16

# extraction_mode = "real"
extraction_mode = "dummy"

model_tag = "ResNet50"
# model_tag = "DenseNet121"
# model_tag = "MobileNet_V3_Large"
# model_tag = "Vit_b_16"
# model_tag = "vgg16"
# model_tag = 'MobileNet_randinit'
# model_tag = "MaxVit_T"
# model_tag = "Swin_S"

#-------------------------
# Step 1: Initialize model with pre-trained weights
model, weights = load_pretraind_model(model_tag)

#-------------------------
# Step 2: Extract features 
model.eval()
preprocess = weights.transforms()
dataset = ImageDataset(image_path, label_path, preprocess)
loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,  shuffle=False, drop_last=False)

X_li = [] # features
N_li = [] # file Nanes
Y_li = [] # true class 
for ii, (batch, finam, true_class) in enumerate(loader, 0):
    print(batch.shape)
    print(np.array(finam).shape)
    if extraction_mode == "real":
        prediction = model(batch).detach().numpy()  #.squeeze(0)
    if extraction_mode == "dummy":
        prediction = np.random.uniform(0.0, 1.0, [batch.shape[0], 1000])
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
tstmp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_")
out_name = os.path.join(featu_path, tstmp + 'Feat_from_' + model_tag + '_' + extraction_mode + '.npz')
np.savez(file = out_name, X = X, Y = Y, N = N)



