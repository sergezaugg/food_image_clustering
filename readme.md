# CLUSTER IMAGES WITH DNN FEATURES AND DIM REDUCTION

### Data
* Data from Food Classification dataset published on Kaggle by Bjorn.
* https://www.kaggle.com/datasets/bjoernjostein/food-classification
* Over 9300 hand-annotated images with 61 classes

### Clustering
* Here the focus is on clustering (i.e. without using the labels)
* Labels are only used to assess the quality of clustering

### Feature extraction (image to vector)
* Features extracted with image classification models pre-trained with the Imagenet datataset
* Details see here : https://docs.pytorch.org/vision/main/models.html
* As output we used the last linear layer which outputs 1000 continuous features (ommited Softamx) 

### A bit of context
* These models were trained specifically for the 1000 Imagenet classes, so let's hope the feature are informative for our task

### Dependencies / Intallation
* Developed under Python 3.12.8
* First make a venv, then:
* For Streamlit deployment only
```
pip install -r requirements.txt
```
* For feature extraction (Pytorch / GPU) and Streamlit deployment 
```
pip install -r req_torchcuda.txt
```

### Usage / Sample code
*  Start dashboard
```bash 
streamlit run stmain.py
```
*  Extract features see **01_extract_features.py**, done rarely!

