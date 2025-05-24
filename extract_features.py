#--------------------             
# Author : Serge Zaugg
# Description : 
#--------------------

import os 
import numpy as np
import datetime
import torch
from utils_ml import ImageDataset
torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#-------------------------
# set params 
image_path = "D:/image_clust/food_images/train_images"
label_path = "D:/image_clust/food_images/train_img.csv"
featu_path = "./extracted_features"
batch_size = 16

# model_tag = "ResNet50"
# model_tag = "DenseNet121"
# model_tag = "MobileNet_V3_Large"
# model_tag = "Vit_b_16"
model_tag = "vgg16"

#-------------------------
# Step 1: Initialize model with pre-trained weights
if model_tag == "ResNet50":
    from torchvision.models import resnet50, ResNet50_Weights
    weights = ResNet50_Weights.IMAGENET1K_V2
    model = resnet50(weights=weights)
elif model_tag == "DenseNet121":
    from torchvision.models import densenet121, DenseNet121_Weights 
    weights = DenseNet121_Weights.IMAGENET1K_V1
    model = densenet121(weights=weights)
elif model_tag == "MobileNet_V3_Large":
    from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
    weights = MobileNet_V3_Large_Weights.IMAGENET1K_V2
    model = mobilenet_v3_large(weights=weights)
elif model_tag == "Vit_b_16":
    from torchvision.models import vit_b_16, ViT_B_16_Weights
    weights = ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1
    model = vit_b_16(weights=weights)
elif model_tag == "vgg16":
    from torchvision.models import vgg16, VGG16_Weights
    weights = VGG16_Weights.IMAGENET1K_V1
    model = vgg16(weights=weights)
else:
    print("not a valid model_tag")


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
tstmp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_")
out_name = os.path.join(featu_path, tstmp + 'Feat_from_' + model_tag + '.npz')
np.savez(file = out_name, X = X, Y = Y, N = N)



