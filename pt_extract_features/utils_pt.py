#--------------------             
# Author : Serge Zaugg
# Description : Functions an classes specific to ML/PyTorch backend
#--------------------

import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torchvision.io import decode_image
import datetime
import yaml
import torch
# from pt_extract_features.utils_pt import ImageDataset, load_pretraind_model
torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




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


def extract_features_from_images(config, model_tag, dev = False):
    """
    Description: Extract linear features from multiple images and also labels from a csv file
    Args:
        config (path): A yaml file with source and destination paths
        model_tag (str): a model name defined in load_pretraind_model()
        dev (logical): if True only a few batches are processed
    Return value: None, saves npz file to disk    
    """
    # load params
    with open(config) as f:
        conf = yaml.safe_load(f)
    image_path = conf['image_path']
    label_path = conf['label_path']
    featu_path = conf['featu_path']
    batch_size = conf['batch_size']
    extrc_mode = conf['extrc_mode']
    # Step 1: Initialize model with pre-trained weights
    model, weights = load_pretraind_model(model_tag)
    # Step 2: Extract features 
    model.eval()
    preprocess = weights.transforms()
    dataset = ImageDataset(image_path, label_path, preprocess)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,  shuffle=False, drop_last=False)
    X_li = [] # features
    N_li = [] # file Nanes
    Y_li = [] # true class 
    for ii, (batch, finam, true_class) in enumerate(loader, 0):
        if dev and ii > 2: 
            break
        print(batch.shape)
        print(np.array(finam).shape)
        if extrc_mode == "real":
            prediction = model(batch).detach().numpy()  #.squeeze(0)
        if extrc_mode == "dummy":
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
    out_name = os.path.join(featu_path, tstmp + 'Feat_from_' + model_tag + '_' + extrc_mode + '.npz')
    np.savez(file = out_name, X = X, Y = Y, N = N)





