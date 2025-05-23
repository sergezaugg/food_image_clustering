#--------------------             
# Author : Serge Zaugg
# Description : Utility functions used by other scripts
#--------------------

import os
import streamlit as st
from streamlit import session_state as ss
import numpy as np
import pandas as pd
import kagglehub
import gc
from sklearn.model_selection import train_test_split
gc.collect()


def get_short_class_name(a):
    """ a : a string"""
    # return("-".join(a.split("-")[0:2]))
    return("-".join(a.split("-")[0:1]))


c00, c01  = st.columns([0.1, 0.18])

# First, get data into ss
if ss['dapar']['feat_path'] == 'empty' :
    st.text("Preparing data ...")
    # download the data from kaggle (https://www.kaggle.com/datasets/sezaugg/food-classification-features-v01)

    kgl_ds = "sezaugg/" + 'food-classification-features-v01' # link on Kaggle is fixed

    
    kgl_path = kagglehub.dataset_download(kgl_ds, force_download = False) # get local path where downloaded
    ss['dapar']['feat_path'] = kgl_path
    ss['dapar']['imgs_path'] = os.path.join(ss['dapar']['feat_path'], 'train_images', 'train_images')
    di = dict()
    li_npz = [a for a in os.listdir(ss['dapar']['feat_path']) if '.npz' in a and 'Feat_from_' in a]
    for npz_finame in li_npz:
        npzfile_full_path = os.path.join(ss['dapar']['feat_path'], npz_finame)
        npzfile = np.load(npzfile_full_path)
        # take a subset 
        X_train, X_test, Y_train, Y_test, N_train, N_test, = train_test_split(npzfile['X'], npzfile['Y'], npzfile['N'], train_size=3000, random_state=6666, shuffle=True)
        di[npz_finame] = {'X' : X_train , 'clusters_true' : Y_train , 'im_filenames' : N_train}
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
                    # copy selected data into dedicated di 
                    ss['dapar']['dataset_name']   = npz_finame 
                    ss['dapar']['X']              = ss['dapar']['npdata'][npz_finame]['X']  
                    ss['dapar']['clusters_true']  = ss['dapar']['npdata'][npz_finame]['clusters_true'] 
                    ss['dapar']['im_filenames']  = ss['dapar']['npdata'][npz_finame]['im_filenames'] 
                    # simplify true class 
                    ss['dapar']['clusters_true'] = pd.Series(ss['dapar']['clusters_true']).apply(func= get_short_class_name).values
                    st.rerun()  # mainly to update sidebar   
        st.page_link("page02.py", label="Go to analysis")                
    gc.collect()
   
      