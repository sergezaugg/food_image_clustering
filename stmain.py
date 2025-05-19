#--------------------             
# Author : Serge Zaugg
# Description : Main streamlit entry point
# run locally : streamlit run stmain.py
#--------------------

import streamlit as st
from streamlit import session_state as ss
import numpy as np

st.set_page_config(layout="wide")

if 'dapar' not in ss:
    ss['dapar'] = {
        'feat_path' : 'empty', 
        'X' : np.array([]),
        'clusters_true' : np.array([]),
        }
    
if 'upar' not in ss:
    ss['upar'] = {
        'umap_n_neighbors' : 10,
        'umap_n_dims_red' : 32,
        'dbscan_eps' : 0.501,
        'dbscan_min_samples' : 10,
        'model_list' : 'empty',
        'current_model_index' : 2, 
        }

p01 = st.Page("page01.py", title="Summary")
p02 = st.Page("page02.py", title="Analyse")

pg = st.navigation([p02, p01])

pg.run()

with st.sidebar:
    st.header(''':blue[**CLUSTER IMAGES WITH DNN FEATURES AND DIM REDUCTION**]''')
    st.text("v0.6.0 - under devel")
    st.markdown(''':blue[QUICK GUIDE]''')
    st.text("(1) Choose extracted features")
    st.text("(2) Set UMAP params")
    st.text("(3) Set DBSCAN params")
    st.text("(4) Explore scatterplots")
    st.title(""); st.title(""); st.title(""); 
    st.text("Needs quite some computation - UMAP run can take a few minutes, DBSCAN is much faster")
    st.title(""); st.title("")
    st.markdown(''':gray[RELATED TOPICS]''')
    st.page_link("https://ml-performance-metrics.streamlit.app/", label=":gray[ml-performance-metrics]")
    st.page_link("https://purenoisefeatures.streamlit.app", label=":gray[impact of pure-noise-features]")
