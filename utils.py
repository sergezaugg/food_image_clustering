#--------------------             
# Author : Serge Zaugg
# Description : Computation steps defined as functions here
#--------------------

import os
import streamlit as st
from streamlit import session_state as ss
import numpy as np
import gc
import pandas as pd 
import plotly.express as px
import umap.umap_ as umap
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

def update_ss(kname, ssname):
    """
    description : helper callback fun to implement statefull apps
    kname : key name of widget
    ssname : key name of variable in session state (ss)
    """
    ss["upar"][ssname] = ss[kname]      

@st.cache_data
def dim_reduction_for_2D_plot(X, n_neigh):
    """
    UMAP dim reduction for 2D plot 
    """
    reducer = umap.UMAP(
        n_neighbors = n_neigh, 
        n_components = 2, 
        metric = 'euclidean',
        n_jobs = -1
        )
    X2D_trans = reducer.fit_transform(X, ensure_all_finite=True)
    scaler = StandardScaler()
    X2D_scaled = scaler.fit_transform(X2D_trans)
    return(X2D_scaled)

@st.cache_data
def dim_reduction_for_clustering(X, n_neigh, n_dims_red, skip_umap = False):
    """
    UMAP dim reduction for clustering
    """
    scaler = StandardScaler()
    if skip_umap == True:
        X_scaled = scaler.fit_transform(X)
        return(X_scaled)
    else:    
        reducer = umap.UMAP(
            n_neighbors = n_neigh, 
            n_components = n_dims_red, 
            metric = 'euclidean',
            n_jobs = -1
            )
        X_trans = reducer.fit_transform(X, ensure_all_finite=True)
        X_scaled = scaler.fit_transform(X_trans)
        return(X_scaled)

@st.cache_data
def perform_dbscan_clusterin(X, eps, min_samples):
    """ 
    """
    clu = DBSCAN(eps = eps, min_samples = min_samples, metric='euclidean', n_jobs = 4) 
    clusters_pred = clu.fit_predict(X)
    return(clusters_pred)

@st.cache_data
def make_sorted_df(cat, cat_name, X):
    df = pd.DataFrame({ cat_name : cat, 'Dim-1' : X[:,0] , 'Dim-2' : X[:,1]})
    df = df.sort_values(by=cat_name)
    return(df)

@st.cache_data
def make_scatter_plot(df, cat_name, title = "not set"):
    fig = px.scatter(
        data_frame = df,
        x = 'Dim-1',
        y = 'Dim-2',
        color = cat_name,
        template='plotly_dark',
        height=900,
        width =1000,
        color_discrete_sequence = px.colors.qualitative.Light24,
        title = title,
        # labels = {'aaa', ""}
        )
    _ = fig.update_layout(margin=dict(t=30, b=450, l=15, r=15))
    _ = fig.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.1, xanchor="left", x=0.0))
    _ = fig.update_layout(showlegend=True,legend_title=None)
    _ = fig.update_layout(yaxis_title=None)
    _ = fig.update_layout(xaxis_title=None)
    _ = fig.update_xaxes(showline=True, linewidth=2, linecolor='white', mirror=True)
    _ = fig.update_yaxes(showline=True, linewidth=2, linecolor='white', mirror=True)

    return(fig)


@st.fragment
def show_cluster_details(conf_table):
    c03, c04 = st.columns([0.5, 0.5])
    with c03:
        st.text("Select a cluster ID")
        clu_id_list = conf_table.index  
        clu_selected = st.segmented_control(label = "Select a cluster ID", options = clu_id_list, selection_mode="single", default = clu_id_list[0], label_visibility="hidden")                
        clu_row = pd.DataFrame(conf_table.loc[clu_selected]  )
        clu_summary = (clu_row.loc[(clu_row!=0).any(axis=1)]).reset_index() 
        clu_summary.columns = ['True', 'Count']
        clu_summary = clu_summary.sort_values(by='Count', ascending = False)     
    with c04:
        st.text("Cluster content")
        st.dataframe(clu_summary, hide_index = True, height =800, use_container_width = True)
    