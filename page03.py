#--------------------             
# Author : Serge Zaugg
# Description : Utility functions used by other scripts
#--------------------

import os
import streamlit as st
from streamlit import session_state as ss
import numpy as np
import kagglehub

c00, c01  = st.columns([0.1, 0.18])

# First, get data or local path to data
if ss['dapar']['feat_path'] == 'empty' :
    st.text("Preparing data ...")
    # download the data from kaggle (https://www.kaggle.com/datasets/sezaugg/food-classification-features-v01)
    kgl_ds = "sezaugg/" + 'food-classification-features-v01' # link on Kaggle , is fixed
    kgl_path = kagglehub.dataset_download(kgl_ds, force_download = False) # get local path where downloaded
    ss['dapar']['feat_path'] = kgl_path
    ss['upar']['model_list'] = os.listdir(ss['dapar']['feat_path'])
    st.rerun()
else :
    with c00:
        with st.container(border=True, height = 230):   
            with st.form("form01", border=False):
                featu_path = st.selectbox("Select data with extracted features", options = ss['upar']['model_list'], index=ss['upar']['current_model_index'])
                submitted_1 = st.form_submit_button("Confirm", type = "primary")   
                if submitted_1:    
                    npzfile = np.load(os.path.join(ss['dapar']['feat_path'], featu_path))
                    ss['dapar']['X'] = npzfile['X']
                    ss['dapar']['clusters_true'] = npzfile['Y']
                    # update index for satefulness of the selectbox
                    ss['upar']['current_model_index'] =  ss['upar']['model_list'].index(featu_path)
                    st.rerun()
            st.text("")        
            st.write('Active: ', featu_path)        
      