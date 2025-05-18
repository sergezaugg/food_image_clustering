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
        'dbscan_eps' : 0.30,
        'dbscan_min_samples' : 10,
        }



p01 = st.Page("page01.py", title="Activate data")
p02 = st.Page("page02.py", title="Analyse")

pg = st.navigation([p01, p02])

pg.run()

