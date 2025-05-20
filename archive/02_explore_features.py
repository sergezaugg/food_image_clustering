#--------------------             
# Author : Serge Zaugg
# Description : 
#--------------------

import os 
import numpy as np
import pandas as pd
import plotly.express as px
import umap.umap_ as umap
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import v_measure_score, rand_score
from sklearn.metrics.cluster import contingency_matrix



featu_path = "D:/image_clust/features/features.npz"




#-------------------------------------------
# (1) Load data  
npzfile = np.load(featu_path)
X = npzfile['X']
clusters_true = npzfile['Y']
pd.Series(clusters_true).value_counts()
#-------------------------------------------




# UMAP 
n_neighbors = 10
n_dims_red = 32

# DBSCAN
eps = 0.6
min_samples = 10










#-------------------------------------------
# (2) UMAP dim reduction for 2D plot 
reducer = umap.UMAP(
    n_neighbors = n_neighbors, 
    n_components = 2, 
    metric = 'euclidean',
    n_jobs = -1
    )
X2D_trans = reducer.fit_transform(X)
# standardize
scaler = StandardScaler()
X2D_scaled = scaler.fit_transform(X2D_trans)
X2D_scaled.shape
#-------------------------------------------


#-------------------------------------------
# (2) UMAP dim reduction for clustering
reducer = umap.UMAP(
    n_neighbors = n_neighbors, 
    n_components = n_dims_red, 
    metric = 'euclidean',
    n_jobs = -1
    )
X_trans = reducer.fit_transform(X)
# standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_trans)
X_scaled.shape
#-------------------------------------------



#-------------------------------------------
# (3) Clustering  
clu = DBSCAN(eps = eps, min_samples=min_samples, metric='euclidean', n_jobs = 8) 
clusters_pred = clu.fit_predict(X_scaled)
pd.Series(clusters_pred).value_counts()
#-------------------------------------------


#-------------------------------------------
# (4) Compare  

clusters_pred.shape
clusters_true.shape
v_measure_score(labels_true = clusters_true , labels_pred = clusters_pred, beta=1.0)
adjusted_rand_score(labels_true = clusters_true , labels_pred = clusters_pred)
# z = contingency_matrix(labels_true = clusters_true , labels_pred = clusters_pred)
# fig = px.imshow(z, text_auto=True)
# fig.show()




# plot ground truth
fig = px.scatter(
    x =X2D_scaled[:,0],
    y =X2D_scaled[:,1],
    color = clusters_true,
    template='plotly_dark',
    height=800,
    width=1100,
    color_discrete_sequence = px.colors.qualitative.Light24
    )
_ = fig.update_layout(margin=dict(t=10, b=10, l=15, r=300))
fig.show()

# plot predicted clusters 
fig = px.scatter(
    x =X2D_scaled[:,0],
    y =X2D_scaled[:,1],
    color = clusters_pred.astype('str'),
    template='plotly_dark',
    height=800,
    width=1100,
    color_discrete_sequence = px.colors.qualitative.Light24,
    )
_ = fig.update_layout(margin=dict(t=10, b=10, l=15, r=300))
fig.show()

#-------------------------------------------


# color_discrete_sequence = px.colors.sequential.Turbo
# color_discrete_sequence = px.colors.sequential.Rainbow

# @st.cache_data
# def make_fig(df, dot_colors):
#     fig00 = px.scatter(
#         data_frame = df,
#         x = 'proba_score',
#         y = 'jitter',
#         color = 'class',
#         color_discrete_sequence = dot_colors,
#         template='plotly_dark',
#         width = 900,
#         height = 300,
#         labels={"proba_score": "Score", "jitter": ""},
#         title = "",
#         )
#     _ = fig00.update_xaxes(showline = True, linecolor = 'white', linewidth = 2, row = 1, col = 1, mirror = True)
#     _ = fig00.update_yaxes(showline = True, linecolor = 'white', linewidth = 2, row = 1, col = 1, mirror = True)
#     _ = fig00.update_traces(marker=dict(size=4))
#     _ = fig00.update_layout(xaxis=dict(showgrid=False, zeroline=False), yaxis=dict(showgrid=False, zeroline=False))
#     _ = fig00.update_layout(xaxis_range=[-0.00001, +1.00001])
#     _ = fig00.update_layout(paper_bgcolor="#000000") # "#350030"
#     _ = fig00.update_yaxes(showticklabels=False)
#     # text font sizes 
#     # _ = fig00.update_layout(title_font_size=25)
#     _ = fig00.update_layout(xaxis_title_font_size=25)
#     _ = fig00.update_layout(yaxis_title_font_size=25)
#     _ = fig00.update_layout(xaxis_tickfont_size=25)
#     _ = fig00.update_layout(legend_font_size=20)
#     # _ = fig00.update_layout(title_y=0.96)
#     _ = fig00.update_layout(showlegend=False)
#     _ = fig00.update_layout(yaxis_title=None)
#     _ = fig00.update_layout(margin=dict(t=10, b=10, l=15, r=15))
#     # _ = fig00.update_layout(xaxis={'side': 'top'}) # , yaxis={'side': 'right'}  )
#     # 
#     return(fig00)

