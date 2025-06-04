#--------------------             
# Author : Serge Zaugg
# Description : A flat-script to extract features from images via last layer of pre-trained PyTorch models
# All PyTorch functionality defined in utils_pt.py
# Recommendation: Run each line in a fresh session to avoid memory issues
#--------------------

from pt_extract_features.utils_pt import extract_features_from_images

# uniform random features 
extract_features_from_images(config = 'pt_extract_features/fex_session_dummy.yaml', model_tag= "ResNet50", dev = True)
# CNN NOT pre-trained
extract_features_from_images(config = 'pt_extract_features/fex_session_prod.yaml', model_tag= "MobileNet_randinit", dev = True)
# CNN pre-trained
extract_features_from_images(config = 'pt_extract_features/fex_session_prod.yaml', model_tag= "vgg16", dev = True)
extract_features_from_images(config = 'pt_extract_features/fex_session_prod.yaml', model_tag= "ResNet50", dev = True)
extract_features_from_images(config = 'pt_extract_features/fex_session_prod.yaml', model_tag= "DenseNet121", dev = True)
extract_features_from_images(config = 'pt_extract_features/fex_session_prod.yaml', model_tag= "MobileNet_V3_Large", dev = True)
# Visual transformers pre-trained
extract_features_from_images(config = 'pt_extract_features/fex_session_prod.yaml', model_tag= "Vit_b_16", dev = True)
extract_features_from_images(config = 'pt_extract_features/fex_session_prod.yaml', model_tag= "MaxVit_T", dev = True)
extract_features_from_images(config = 'pt_extract_features/fex_session_prod.yaml', model_tag= "Swin_S", dev = True)




