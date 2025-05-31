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
from utils import get_short_class_name
gc.collect()

c00, c01  = st.columns([0.1, 0.18])

# First, get data into ss
# download the data from kaggle (https://www.kaggle.com/datasets/sezaugg/food-classification-features-v01)
if ss['dapar']['feat_path'] == 'empty' :
    st.text("Preparing data ...")
    kgl_ds = "sezaugg/" + 'food-classification-features-v01' # prod
    kgl_path = kagglehub.dataset_download(kgl_ds, force_download = False) # get local path where downloaded
    ss['dapar']['feat_path'] = kgl_path
    ss['dapar']['imgs_path'] = os.path.join(ss['dapar']['feat_path'], 'train_images', 'train_images')
    ss['dapar']['li_npz'] = [a for a in os.listdir(ss['dapar']['feat_path']) if ('.npz' in a) and ('Feat_from_' in a or 'Pure_random_' in a)]
    ss['dapar']['li_npz'].sort() # sort option so we can the used index in st.selectbox
    st.rerun()
# Then, choose a dataset
else :
    with c00:
        with st.container(border=True, height = 200):   
            with st.form("form01", border=False):
                npz_finame = st.selectbox("Select data with extracted features", options = ss['dapar']['li_npz'], index = 6)
                submitted_1 = st.form_submit_button("Activate dataset", type = "primary")   
                if submitted_1:
                    npzfile_full_path = os.path.join(ss['dapar']['feat_path'], npz_finame)
                    npzfile = np.load(npzfile_full_path)
                    # sample a smaller random subset 
                    X, _, Y, _, N, _, = train_test_split(npzfile['X'], npzfile['Y'], npzfile['N'], train_size=6000, random_state=6666, shuffle=True)
                    # copy selected data into dedicated dict 
                    ss['dapar']['dataset_name']   = npz_finame 
                    ss['dapar']['X']              = X
                    ss['dapar']['clusters_true']  = Y 
                    ss['dapar']['im_filenames']   = N 
                    # simplify true class 
                    ss['dapar']['clusters_true'] = pd.Series(ss['dapar']['clusters_true']).apply(func= get_short_class_name).values
                    st.rerun()  # mainly to update sidebar   
                st.page_link("page02.py", label="Go to analysis")                
        
gc.collect()
        