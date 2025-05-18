#--------------------             
# Author : Serge Zaugg
# Description : Utility functions used by other scripts
#--------------------

import streamlit as st
import os 
from streamlit import session_state as ss
import gc
import kagglehub


# https://www.kaggle.com/datasets/sezaugg/food-classification-features-v01

c00, c01  = st.columns([0.1, 0.2])
with c00:
    with st.container(border=True) : 

        # user selects a kgl dataset
        st.text("Activate will fetch a Kaggle data set and load to memory")
     
        kgl_ds = "sezaugg/" + 'food-classification-features-v01'

        # Download kgl dataset (if not already available)
        with st.form("form01", border=False):
            submitted_1 = st.form_submit_button("Activate data set", type = "primary")   
            if submitted_1:    
                # download the data from kaggle 
                kgl_path = kagglehub.dataset_download(kgl_ds, force_download = False)
                # define path in ss to ge globally accessible 
                ss['dapar']['feat_path'] = kgl_path

                st.write(ss['dapar']['feat_path'])

                st.write(os.listdir(ss['dapar']['feat_path']))
                

      