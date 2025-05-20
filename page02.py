#--------------------             
# Author : Serge Zaugg
# Description : Main interactive streamlit page
#--------------------

import os
import streamlit as st
from streamlit import session_state as ss
import numpy as np
import pandas as pd
import gc
import kagglehub
from sklearn.metrics import v_measure_score, adjusted_rand_score, adjusted_mutual_info_score
from utils import dim_reduction_for_2D_plot, dim_reduction_for_clustering, perform_dbscan_clusterin, update_ss
from utils import make_sorted_df, make_scatter_plot, load_data_from_npz
gc.collect()

cols = st.columns([0.1, 0.35, 0.1, 0.3, 0.25])

# Handle state on app startup
if ss['dapar']['feat_path'] == 'empty' :
    st.text("Preparing data ...")
    # download the data from kaggle (https://www.kaggle.com/datasets/sezaugg/food-classification-features-v01)
    kgl_ds = "sezaugg/" + 'food-classification-features-v01' # link on Kaggle , is fixed
    kgl_path = kagglehub.dataset_download(kgl_ds, force_download = False) # get local path where downloaded
    ss['dapar']['feat_path'] = kgl_path
    ss['upar']['model_list'] = os.listdir(ss['dapar']['feat_path'])
    st.rerun()
else :
    featu_path = load_data_from_npz(model_index = ss['upar']['current_model_index'])  
    with cols[0]:
        if len(ss['dapar']['X']) <= 0:
            st.info("Please select a dataset")
        else:   
            with st.container(border=True, height = 250):  
                st.text("Dimension") 
                st.info(str(ss['dapar']['X'].shape[0]) + '  imgs')
                st.info(str(ss['dapar']['X'].shape[1]) + '  feats') 
                

# main dashboard
if len(ss['dapar']['X']) > 0 :
   
    with cols[1]:
        with st.container(border=True, height = 250):   
            ca1, ca2 = st.columns([0.15, 0.8])
            with ca1:
                ss['upar']['skip_umap'] = st.checkbox("Skip UMAP")
            with ca2:
                _ = st.select_slider(label = "UMAP reduce dim", options=[2,4,8,16,32,64,128], disabled = ss['upar']['skip_umap'],
                                    key = "k_UMAP_dim", value = ss['upar']["umap_n_dims_red"], on_change=update_ss, args=["k_UMAP_dim", "umap_n_dims_red"])
                _ = st.select_slider(label = "UMAP nb neighbors", options=[2,5,10,15,20,30,40,50,75,100], disabled = ss['upar']['skip_umap'], 
                                key = "k_UMAP_n_neigh", value=ss['upar']["umap_n_neighbors"], on_change=update_ss, args=["k_UMAP_n_neigh", "umap_n_neighbors"])   
    
    #-------------------------------------------
    # computational block 1 (st-cached)
    X2D_scaled = dim_reduction_for_2D_plot(X = ss['dapar']['X'], n_neigh = ss['upar']['umap_n_neighbors'])
    X_scaled = dim_reduction_for_clustering(X = ss['dapar']['X'], n_neigh = ss['upar']['umap_n_neighbors'], n_dims_red = ss['upar']['umap_n_dims_red'], 
                                            skip_umap = ss['upar']['skip_umap'])
    #-------------------------------------------

    with cols[2]:
        with st.container(border=True, height = 250): 
            st.text("Dimension")  
            st.info(str(X_scaled.shape[0]) + '  imgs')
            st.info(str(X_scaled.shape[1]) + '  feats')

    with cols[3]:
        with st.container(border=True, height = 250): 
            eps_options = (10.0**(np.arange(-3.0, 2.5, 0.05))).round(3)
            _ = st.select_slider(label = "DBSCAN eps (good value depends on dims from UMAP)", options = eps_options, 
                                key = "k_dbscan_eps", value=ss['upar']["dbscan_eps"], on_change=update_ss, args=["k_dbscan_eps", "dbscan_eps"])
            _ = st.select_slider(label = "DBSCAN min samples", options=np.arange(5, 51, 5), 
                                key = "k_dbscan_min", value=ss['upar']["dbscan_min_samples"], on_change=update_ss, args=["k_dbscan_min", "dbscan_min_samples"])

    #-------------------------------------------
    # computational block 2 (st-cached)
    clusters_pred = perform_dbscan_clusterin(X = X_scaled, eps = ss['upar']['dbscan_eps'], min_samples = ss['upar']['dbscan_min_samples']) 
    num_unasigned = (clusters_pred == -1).sum()
    num_asigned = len(clusters_pred) - num_unasigned
    clusters_pred_str = np.array([format(a, '03d') for a in clusters_pred])
    df_true = make_sorted_df(cat = ss['dapar']['clusters_true'], cat_name = 'True class', X = X2D_scaled)
    df_pred = make_sorted_df(cat = clusters_pred_str, cat_name = 'Predicted cluster', X = X2D_scaled)
    fig01 = make_scatter_plot(df = df_true, cat_name = 'True class', title = "Ground truth")
    fig02 = make_scatter_plot(df = df_pred, cat_name = 'Predicted cluster', title = "Predicted clusters")
    # metrics 
    met_amui_sc = adjusted_mutual_info_score(labels_true = ss['dapar']['clusters_true'] , labels_pred = clusters_pred_str)
    met_rand_sc =        adjusted_rand_score(labels_true = ss['dapar']['clusters_true'] , labels_pred = clusters_pred_str)
    met_v_measu =            v_measure_score(labels_true = ss['dapar']['clusters_true'] , labels_pred = clusters_pred_str, beta=1.0)

    # st.dataframe(pd.crosstab(clusters_pred_str, ss['dapar']['clusters_true']))

    #-------------------------------------------

    with cols[4]:
        with st.container(border=True, height = 250): 
            st.text("Clustering metrics")
            coco = st.columns(2)
            coco[0].metric("N assigned ", num_asigned)
            coco[1].metric("Adj. Mutual Info Score " , format(round(met_amui_sc,2), '03.2f'))
            coco[1].metric("Adj. Rand Score " ,        format(round(met_rand_sc,2), '03.2f'))
   



    #-------------------------------------------
    # show plots 
    c01, c02 = st.columns(2)
    with c01:
        st.plotly_chart(fig01, use_container_width=False, theme=None)
    with c02:
        st.plotly_chart(fig02, use_container_width=False, theme=None)

    

    with st.container(border=True):   
        st.text("Plots are zoomable and categories can be selectively hidden/shown by click in legend.") 
        st.text("Noisy samples, i.e. those that were not assigned to a cluster, are given the label '-01'") 
        st.text("The numerical value of Cluster IDs is arbitrary and cannot be automatically linked to a true class, you have to assess the match graphically or with clustering metrics")
        st.text("adjusted_rand_score is a consensus measures between true and predicted clusters, values in [-0.5, 1]")  
        st.text("adjusted_mutual_info_score (AMI) is a consensus measures between true and predicted clusters, values in [~0, 1]")  
        st.markdown('''UMAP is a [stochastic algorithm](https://umap-learn.readthedocs.io/en/latest/reproducibility.html) and I intentionally did not fix a random seed -> 
                    you will observe small differences between runs''')

        
    st.info("Selected dataset: " + featu_path)        
     
        
  

   


