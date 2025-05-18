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
    
if 'dbscan' not in ss:
    ss['dbscan'] = {
        'eps' : 0.30,
        'min_samples' : 10,
        }

if 'umap' not in ss:
    ss['umap'] = {
        'n_dims_red' : 32,
        'n_neighbors' : 50,
        }

p01 = st.Page("page01.py", title="Activate data")
p02 = st.Page("page02.py", title="Analyse")

pg = st.navigation([p01, p02])

pg.run()

