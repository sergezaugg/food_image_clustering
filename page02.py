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
import plotly.express as px
from sklearn.metrics import v_measure_score, rand_score
from sklearn.metrics.cluster import contingency_matrix
from utils import dim_reduction_for_2D_plot, dim_reduction_for_clustering, perform_dbscan_clusterin, update_ss
from utils import make_sorted_df, make_scatter_plot
import kagglehub
gc.collect()


cols = st.columns(5)


# First, get data or local path to data
if ss['dapar']['feat_path'] == 'empty' :
    st.text("Preparing data ...")
    # download the data from kaggle (https://www.kaggle.com/datasets/sezaugg/food-classification-features-v01)
    kgl_ds = "sezaugg/" + 'food-classification-features-v01' # link on Kaggle , is fixed
    kgl_path = kagglehub.dataset_download(kgl_ds, force_download = False) # get local path where downloaded
    ss['dapar']['feat_path'] = kgl_path
    st.rerun()

# main dashboard
else :
    with cols[0]:
        with st.form("form01", border=False):
            featu_path = st.selectbox("Select extracted features", options = os.listdir(ss['dapar']['feat_path']))
            submitted_1 = st.form_submit_button("Choose", type = "primary")   
            if submitted_1:    
                npzfile = np.load(os.path.join(ss['dapar']['feat_path'], featu_path))
                ss['dapar']['X'] = npzfile['X']
                ss['dapar']['clusters_true'] = npzfile['Y']


if len(ss['dapar']['X']) > 0 :
   
    with cols[1]:
        _ = st.select_slider(label = "UMAP reduce dim", options=[2,4,8,16,32,64,128], 
                            key = "k_UMAP_dim", value = ss['upar']["umap_n_dims_red"], on_change=update_ss, args=["k_UMAP_dim", "umap_n_dims_red"])
        st.text("Will recompute UMAP (slow) ")
    with cols[2]:    
        _ = st.select_slider(label = "UMAP nb neighbors", options=[2,5,10,15,20,30,40,50,75,100], 
                         key = "k_UMAP_n_neigh", value=ss['upar']["umap_n_neighbors"], on_change=update_ss, args=["k_UMAP_n_neigh", "umap_n_neighbors"])
    with cols[3]:
        _ = st.select_slider(label = "DBSCAN eps", options=np.arange(0.05, 2.05, 0.05).round(2), 
                            key = "k_dbscan_eps", value=ss['upar']["dbscan_eps"], on_change=update_ss, args=["k_dbscan_eps", "dbscan_eps"])
        st.text("Will recompute DBSCAN")
    with cols[4]:
        _ = st.select_slider(label = "DBSCAN min samples", options=np.arange(5, 51, 5), 
                            key = "k_dbscan_min", value=ss['upar']["dbscan_min_samples"], on_change=update_ss, args=["k_dbscan_min", "dbscan_min_samples"])

    #-------------------------------------------
    # computational block (st-cached)
    X2D_scaled = dim_reduction_for_2D_plot(X = ss['dapar']['X'], n_neigh = ss['upar']['umap_n_neighbors'])
    X_scaled = dim_reduction_for_clustering(X = ss['dapar']['X'], n_neigh = ss['upar']['umap_n_neighbors'], n_dims_red = ss['upar']['umap_n_dims_red'])
    clusters_pred = perform_dbscan_clusterin(X = X_scaled, eps = ss['upar']['dbscan_eps'], min_samples = ss['upar']['dbscan_min_samples']) 
    clusters_pred_str = np.array([format(a, '03d') for a in clusters_pred])
    df_true = make_sorted_df(cat = ss['dapar']['clusters_true'], cat_name = 'True class', X = X2D_scaled)
    df_pred = make_sorted_df(cat = clusters_pred_str, cat_name = 'Predicted cluster', X = X2D_scaled)
    fig01 = make_scatter_plot(df = df_true, cat_name = 'True class')
    fig02 = make_scatter_plot(df = df_pred, cat_name = 'Predicted cluster')


    #-------------------------------------------
    # show plots 
    c01, c02 = st.columns(2)
    with c01:
        st.text("Ground truth")
        st.plotly_chart(fig01, use_container_width=False, theme=None)
        st.text("Plots are zoomable and categories can be selectively hidden/shown by click in legend.")
    with c02:
        st.text("Predicted clusters")             
        st.plotly_chart(fig02, use_container_width=False, theme=None)


  

    # v_measure_score(labels_true = clusters_true , labels_pred = clusters_pred, beta=1.0)
    # rand_score(labels_true = clusters_true , labels_pred = clusters_pred)


