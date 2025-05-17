
# from
# https://docs.pytorch.org/vision/stable/models.html

from torchvision.io import decode_image
from torchvision.models import resnet50, ResNet50_Weights
import os 
import numpy as np
import pandas as pd
import plotly.express as px
import torch
from torch.utils.data import Dataset

torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



# Step 1: Initialize model with the best available weights
weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights)
model.eval()

# Step 2: Initialize the inference transforms
preprocess = weights.transforms()




class SpectroImageDataset(Dataset):

    def __init__(self, imgpath):
        self.all_img_files = [a for a in os.listdir(imgpath) if '.jpg' in a]
        self.imgpath = imgpath
    
    def __getitem__(self, index):     
        img = decode_image(os.path.join(self.imgpath,  self.all_img_files[index] )  )        
        # Apply inference preprocessing transforms
        img = preprocess(img) # .unsqueeze(0)
        return (img)
    
    def __len__(self):
        return (len(self.all_img_files))



image_path = "C:/Users/sezau/Downloads/food_images/train_images/train_images"

dataset = SpectroImageDataset(image_path)

loader = torch.utils.data.DataLoader(dataset, batch_size=32,  shuffle=False, drop_last=False)

X_li = []
for ii, batch in enumerate(loader, 0):
    print(batch.shape)
    prediction = model(batch).detach().numpy()  #.squeeze(0)
    X_li.append(prediction)

X = np.concatenate(X_li)

X.shape
# type(prediction)


fig = px.line(data_frame=pd.DataFrame(X.T))
fig.show()








import umap.umap_ as umap
from sklearn.preprocessing import StandardScaler


n_neighbors = 5
n_dims_red = 2

# umap 
reducer = umap.UMAP(
    n_neighbors = n_neighbors, 
    n_components = n_dims_red, 
    metric = 'euclidean',
    n_jobs = -1
    )

# reducer.fit(X[0:25000])
# X_trans = reducer.transform(X)
X_trans = reducer.fit_transform(X)

X_trans.shape

# standardize
scaler = StandardScaler()
scaler.fit(X_trans)
X_scaled = scaler.transform(X_trans)


X_scaled.shape



fig = px.scatter(
    x =X_scaled[:,0],
    y =X_scaled[:,1]
    )


fig.show()



# image_path = "C:/Users/sezau/Downloads/food_images/train_images/train_images/fe9bbe121b.jpg"

# img = decode_image(image_path)
# img.shape
# type(img)

# # Step 3: Apply inference preprocessing transforms
# batch = preprocess(img).unsqueeze(0)
# batch.shape

# # Step 4: Use the model and print the predicted category
# prediction = model(batch)  .squeeze(0)


# prediction.shape
# prediction.min()
# prediction.max()

# # class_id = prediction.argmax().item()
# score = prediction[class_id].item()
# category_name = weights.meta["categories"][class_id]
# print(f"{category_name}: {100 * score:.1f}%")



