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
    


def load_pretraind_model(model_tag):
    """
    """
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
    elif model_tag == "MobileNet_randinit":
        from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
        weights = MobileNet_V3_Large_Weights.IMAGENET1K_V2
        model = mobilenet_v3_large(weights=None)
    elif model_tag == "vgg16":
        from torchvision.models import vgg16, VGG16_Weights
        weights = VGG16_Weights.IMAGENET1K_V1
        model = vgg16(weights=weights)
    # Transformers 
    elif model_tag == "Vit_b_16":
        from torchvision.models import vit_b_16, ViT_B_16_Weights
        weights = ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1
        model = vit_b_16(weights=weights)
    elif model_tag == "MaxVit_T":
        from torchvision.models import maxvit_t, MaxVit_T_Weights  
        weights = MaxVit_T_Weights.IMAGENET1K_V1     
        model = maxvit_t(weights=weights)  
    elif model_tag == "Swin_S":
        from torchvision.models import swin_s, Swin_S_Weights
        weights = Swin_S_Weights.IMAGENET1K_V1
        model = swin_s(weights=weights)
    else:
        print("not a valid model_tag")
    return(model, weights)    






