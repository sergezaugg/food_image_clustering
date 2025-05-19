



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
# from sklearn.metrics import v_measure_score, rand_score
# from sklearn.metrics.cluster import contingency_matrix


# fit_transform(X, y=None, ensure_all_finite=True, **kwargs)


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
def dim_reduction_for_clustering(X, n_neigh, n_dims_red):
    """
    UMAP dim reduction for clustering
    """
    reducer = umap.UMAP(
        n_neighbors = n_neigh, 
        n_components = n_dims_red, 
        metric = 'euclidean',
        n_jobs = -1
        )
    X_trans = reducer.fit_transform(X, ensure_all_finite=True)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_trans)
    return(X_scaled)

@st.cache_data
def perform_dbscan_clusterin(X, eps, min_samples):
    """ 
    """
    clu = DBSCAN(eps = eps, min_samples = min_samples, metric='euclidean', n_jobs = 8) 
    clusters_pred = clu.fit_predict(X)
    return(clusters_pred)

