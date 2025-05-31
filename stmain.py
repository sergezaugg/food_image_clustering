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
        'imgs_path' : 'empty',
        'npdata'    : 'empty',
        'dataset_name' :  'empty',
        'X' : np.array([]),
        'clusters_true' : np.array([]),
        'clusters_pred_str' : np.array([]),
        'im_filenames' : np.array([]),
        'li_npz' :'empty',
        }

if 'upar' not in ss:
    ss['upar'] = {
        'umap_n_neighbors' : 10,
        'umap_n_dims_red' : 16,
        'skip_umap' : False,
        'dbscan_eps' : 0.501,
        'dbscan_min_samples' : 20,
        }

with st.sidebar:
    st.info('Active data: ' + ss['dapar']['dataset_name'])
    st.header(''':blue[**CLUSTER IMAGES WITH DNN FEATURES AND DIM REDUCTION**]''')
    st.header("")
    st.markdown(''':blue[QUICK GUIDE]''')
    st.text("(1) Select a dataset")
    st.text("(2) Choose UMAP params")
    st.text("(3) Tune DBSCAN params")
    st.text("(4) Explore metrics & plots")
    st.header("")
    st.markdown(''':blue[COMPUTATION SPEED]''')
    st.text("UMAP can take a few minutes while DBSCAN takes a few seconds. Faster if computation already cached!")
    # logos an links
    st.header("")
    c1,c2=st.columns([80,200])
    c1.image(image='pics/z_logo_turqoise.png', width=65)
    c2.markdown(''':primary[v0.8.4]  
    :primary[Created by]
    :primary[[Serge Zaugg](https://github.com/sergezaugg)]''')
    st.logo(image='pics/z_logo_turqoise.png', size="large", link="https://github.com/sergezaugg")

p01 = st.Page("page01.py", title="Summary")
p02 = st.Page("page02.py", title="Analyse")
p03 = st.Page("page03.py", title="Select dataset")
pss = st.Page("./sandbox/page_ss.py", title="(Debug diagnostics)")
pg = st.navigation([p03, p02, p01])
pg.run()

