# CLUSTER IMAGES WITH DNN FEATURES AND DIM REDUCTION

### Overview
* This is a didactic Streamlit dashboard to cluster-analyse data from images
* Features must first be pre-extracted from images offline with ```pt_extract_features./extract_features.py```
* The resulting npz file(s) must be loaded to a Kaggle dataset 
* The images must also be loaded to the same Kaggle dataset
* See the deployed Kaggle Dataset [here](https://www.kaggle.com/datasets/sezaugg/food-classification-features-v01)
* Then, the Streamlit process is started ```streamlit run stmain.py``` (e.g. locally of on https://share.streamlit.io)
* The path to Kaggle dataset is currently hardcoded here: ```page03.py```.
* See the deployed version [here](https://food-image-clustering.streamlit.app)

### Data
* This project is based on images from **Food Classification Dataset** shared on Kaggle by Bjorn.
* https://www.kaggle.com/datasets/bjoernjostein/food-classification
* Over 9300 hand-annotated images with 61 classes

### Feature extraction (image to vector)
* Features extracted with image classification models pre-trained with the Imagenet datataset.
* Details see on [PyTorch documentation](https://docs.pytorch.org/vision/main/models.html)
* As features we used output from last linear layer of image CNNs: 1000 continuous values. 
* These CNNs were trained specifically for the Imagenet classes, let's hope the feature are informative for our task.
* Pre-extracted features available [here](https://www.kaggle.com/datasets/sezaugg/food-classification-features-v01)

### Clustering, dim-reduction, and visualization
* First features are dim-reduced with UMAP
* Second cluster-IDs are obtained with DBSCAN (unsupervised -> without using the ground truth)
* Third, cluster-IDs and ground truth are compared visually and with metrics.

### Dependencies / Intallation
* Developed under Python 3.12.8
* Make a fresh venv!
* For Streamlit deployment only
```bash 
pip install -r requirements.txt
```
* For feature extraction you also need to install **torch** and **torchvision**
* This code was developed under Windows with CUDA 12.6 and Python 3.12.8 
```bash 
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```
* If other CUDA version needed, check instructions here https://pytorch.org/get-started/locally

### Usage 
*  To extract features, see **pt_extract_features./extract_features.py**
*  Start dashboard
```bash 
streamlit run stmain.py
```


