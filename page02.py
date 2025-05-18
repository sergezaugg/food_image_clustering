#--------------------             
# Author : Serge Zaugg
# Description : Main interactive streamlit page
#--------------------

import os
import streamlit as st
from streamlit import session_state as ss
import numpy as np
import gc
import pandas as pd 

import os 
import numpy as np
import pandas as pd
import plotly.express as px
import umap.umap_ as umap
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import v_measure_score, rand_score
from sklearn.metrics.cluster import contingency_matrix


gc.collect()

# render page only id data is available 
if ss['dapar']['feat_path'] == 'empty' :
    st.text("First activate data (navigation bar left)")
else :
    with st.form("form01", border=False):
                featu_path = st.selectbox("Select extracted features", options = os.listdir(ss['dapar']['feat_path']))
                submitted_1 = st.form_submit_button("Activate", type = "primary")   
                if submitted_1:    
                    npzfile = np.load(os.path.join(ss['dapar']['feat_path'], featu_path))
                    ss['dapar']['X'] = npzfile['X']
                    ss['dapar']['clusters_true'] = npzfile['Y']


if len(ss['dapar']['X']) > 0 :
    st.write("Shape of feature array", ss['dapar']['X'].shape)
    st.write(pd.Series(ss['dapar']['clusters_true']).value_counts())


        
    #-------------------------------------------
    # (1) Load data  
    X = ss['dapar']['X']
    clusters_true = ss['dapar']['clusters_true']
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
    rand_score(labels_true = clusters_true , labels_pred = clusters_pred)
    # z = contingency_matrix(labels_true = clusters_true , labels_pred = clusters_pred)
    # fig = px.imshow(z, text_auto=True)
    # fig.show()




    # plot ground truth
    fig01 = px.scatter(
        x =X2D_scaled[:,0],
        y =X2D_scaled[:,1],
        color = clusters_true,
        template='plotly_dark',
        height=800,
        width=1100,
        color_discrete_sequence = px.colors.qualitative.Light24
        )
    _ = fig01.update_layout(margin=dict(t=10, b=10, l=15, r=300))
    # fig01.show()

    # plot predicted clusters 
    fig02 = px.scatter(
        x =X2D_scaled[:,0],
        y =X2D_scaled[:,1],
        color = clusters_pred.astype('str'),
        template='plotly_dark',
        height=800,
        width=1100,
        color_discrete_sequence = px.colors.qualitative.Light24,
        )
    _ = fig02.update_layout(margin=dict(t=10, b=10, l=15, r=300))
    # fig02.show()

    #-------------------------------------------


    st.plotly_chart(fig01, use_container_width=True, theme=None)
                    
    st.plotly_chart(fig02, use_container_width=True, theme=None)





    