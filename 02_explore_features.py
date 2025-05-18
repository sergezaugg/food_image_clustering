


# data from 
# https://www.kaggle.com/datasets/bjoernjostein/food-classification

# code initial from from
# https://docs.pytorch.org/vision/stable/models.html

import os 
import numpy as np
import pandas as pd
import plotly.express as px
import umap.umap_ as umap
from sklearn.preprocessing import StandardScaler

featu_path = "D:/image_clust/features/features.npz"

npzfile = np.load(featu_path)
X = npzfile['X']
C = npzfile['Y']
X.shape
C.shape


pd.Series(C).value_counts()

# # type(prediction)
# fig = px.line(data_frame=pd.DataFrame(X.T))
# fig.show()


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
    y =X_scaled[:,1],
    color = C,
    template='plotly_dark',
    color_discrete_sequence = px.colors.qualitative.Light24
    # color_discrete_sequence = px.colors.sequential.Turbo
    # color_discrete_sequence = px.colors.sequential.Rainbow
    )

fig.show()










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

