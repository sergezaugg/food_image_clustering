#--------------------             
# Author : Serge Zaugg
# Description : Utility functions used by other scripts
#--------------------

import streamlit as st
import os 
from streamlit import session_state as ss
import gc



c00, c01  = st.columns([0.1, 0.2])
with c00:
    with st.container(border=True) : 

        # user selects a kgl dataset
        st.text("Activate will fetch a Kaggle data set and load to memory")
     
        # kgl_ds = "sezaugg/" + 'food-classification-features-v01'

        # with st.form("form01", border=False):
        #     submitted_1 = st.form_submit_button("Activate data set", type = "primary")   
        #     if submitted_1:    
        #         # download the data from kaggle 
        #         kgl_path = kagglehub.dataset_download(kgl_ds, force_download = False)
        #         ss['dapar']['feat_path'] = kgl_path

             
                

      