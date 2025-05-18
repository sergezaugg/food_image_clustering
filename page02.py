#--------------------             
# Author : Serge Zaugg
# Description : Main interactive streamlit page
#--------------------

import os
import streamlit as st
from streamlit import session_state as ss
import numpy as np
import gc
# import pandas as pd 
import plotly.express as px
# import umap.umap_ as umap
# from sklearn.preprocessing import StandardScaler
# from sklearn.cluster import DBSCAN
from sklearn.metrics import v_measure_score, rand_score
from sklearn.metrics.cluster import contingency_matrix
from utils import dim_reduction_for_2D_plot, dim_reduction_for_clustering, perform_dbscan_clusterin, update_ss

gc.collect()

# render page only if data is available 
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
    # st.write(pd.Series(ss['dapar']['clusters_true']).value_counts())

    _ = st.select_slider(label = "UMAP dim", options=[2,4,8,16,32,64,128], 
                         key = "k_UMAP_dim", value = ss['upar']["umap_n_dims_red"], on_change=update_ss, args=["k_UMAP_dim", "umap_n_dims_red"])
    
    _ = st.select_slider(label = "UMAP n_neighbors", options=[2,5,10,15,20,30,40,50,75,100], 
                         key = "k_UMAP_n_neigh", value=ss['upar']["umap_n_neighbors"], on_change=update_ss, args=["k_UMAP_n_neigh", "umap_n_neighbors"])
    
    _ = st.select_slider(label = "DBSCAN eps", options=np.arange(0.05, 3.0, 0.05).round(2), 
                         key = "k_dbscan_eps", value=ss['upar']["dbscan_eps"], on_change=update_ss, args=["k_dbscan_eps", "dbscan_eps"])

    _ = st.select_slider(label = "DBSCAN min samples", options=np.arange(5, 51, 5), 
                         key = "k_dbscan_min", value=ss['upar']["dbscan_min_samples"], on_change=update_ss, args=["k_dbscan_min", "dbscan_min_samples"])




    #-------------------------------------------
    # (1) Load data  
    X = ss['dapar']['X']
    clusters_true = ss['dapar']['clusters_true']
    #-------------------------------------------
    X2D_scaled = dim_reduction_for_2D_plot(X = ss['dapar']['X'], n_neigh = ss['upar']['umap_n_neighbors'])
    X_scaled = dim_reduction_for_clustering(X = ss['dapar']['X'], n_neigh = ss['upar']['umap_n_neighbors'], n_dims_red = ss['upar']['umap_n_dims_red'])
    clusters_pred = perform_dbscan_clusterin(X = X_scaled, eps = ss['upar']['dbscan_eps'], min_samples = ss['upar']['dbscan_min_samples']) 



    # pd.Series(clusters_pred).value_counts()
    








    #-------------------------------------------
    # (4) Compare  

    v_measure_score(labels_true = clusters_true , labels_pred = clusters_pred, beta=1.0)
    rand_score(labels_true = clusters_true , labels_pred = clusters_pred)
  
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


    c01, c02 = st.columns(2)
    with c01:
        st.plotly_chart(fig01, use_container_width=True, theme=None)
    with c02:             
        st.plotly_chart(fig02, use_container_width=True, theme=None)





    