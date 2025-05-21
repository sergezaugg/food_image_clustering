#--------------------             
# Author : Serge Zaugg
# Description : Utility functions used by other scripts
#--------------------

import os
import streamlit as st
from streamlit import session_state as ss
import numpy as np
import kagglehub
import gc
gc.collect()






from sklearn.model_selection import train_test_split


c00, c01  = st.columns([0.1, 0.18])

# First, get data into ss
if ss['dapar']['feat_path'] == 'empty' :
    st.text("Preparing data ...")
    # download the data from kaggle (https://www.kaggle.com/datasets/sezaugg/food-classification-features-v01)
    kgl_ds = "sezaugg/" + 'food-classification-features-v01' # link on Kaggle is fixed
    kgl_path = kagglehub.dataset_download(kgl_ds, force_download = False) # get local path where downloaded
    ss['dapar']['feat_path'] = kgl_path

    di = dict()
    for npz_finame in os.listdir(ss['dapar']['feat_path']):
        npzfile_full_path = os.path.join(ss['dapar']['feat_path'], npz_finame)
        npzfile = np.load(npzfile_full_path)

        # take a subset 
        X_train, X_test, Y_train, Y_test = train_test_split(npzfile['X'], npzfile['Y'], test_size=0.70, random_state=42)

        di[npz_finame] = {'X' : X_train , 'clusters_true' : Y_train }
    ss['dapar']['npdata'] = di
    gc.collect()
    st.rerun()
# Then, choose a dataset
else :
    with c00:
        with st.container(border=True, height = 200):   
            with st.form("form01", border=False):
                npz_finame = st.selectbox("Select data with extracted features", options = ss['dapar']['npdata'].keys())
                submitted_1 = st.form_submit_button("Confirm", type = "primary")   
                if submitted_1:
                    ss['dapar']['dataset_name']   = npz_finame 
                    ss['dapar']['X']              = ss['dapar']['npdata'][npz_finame]['X']  
                    ss['dapar']['clusters_true']  = ss['dapar']['npdata'][npz_finame]['clusters_true'] 
                    st.rerun()  # mainly to update sidebar   
        st.page_link("page02.py", label="Now let's analyse this data")                
    gc.collect()
   
      