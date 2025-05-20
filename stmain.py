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
        'skip_umap' : False,
        'dbscan_eps' : 0.501,
        'dbscan_min_samples' : 10,
        'model_list' : 'empty',
        'current_model_index' : 2, 
        }

with st.sidebar:
    featu_path = ss['upar']['model_list'][ss['upar']['current_model_index']]
    st.info(featu_path) 
    st.header(''':blue[**CLUSTER IMAGES WITH DNN FEATURES AND DIM REDUCTION**]''')
    st.text("v0.6.2")
    st.markdown(''':blue[QUICK GUIDE]''')
    st.text("(1) Set UMAP params")
    st.text("(2) Set DBSCAN params")
    st.text("(3) Explore metrics & plots")
    st.markdown(":bulb: Plots are zoomable!")
    st.markdown(":bulb: Hide cats by click in legend!")
    st.markdown(":information_source: UMAP dim for plots always = 2")
    st.markdown(":information_source: UMAP dim for DBCAN can be > 2")
    st.title("")
    st.markdown(''':blue[COMPUTATION SPEED]''')
    st.text("UMAP run can take a few minutes")
    st.text("DBSCAN run takes a few seconds")
    st.title(""); 
    st.markdown(''':gray[RELATED TOPICS]''')
    st.page_link("https://ml-performance-metrics.streamlit.app/", label=":gray[ml-performance-metrics]")
    st.page_link("https://purenoisefeatures.streamlit.app", label=":gray[impact of pure-noise-features]")

p01 = st.Page("page01.py", title="Summary")
p02 = st.Page("page02.py", title="Analyse")
p03 = st.Page("page03.py", title="Select dataset")
pg = st.navigation([p03, p02, p01])
pg.run()

