#--------------------             
# Author : Serge Zaugg
# Description : Utility functions used by other scripts
#--------------------

import streamlit as st
from streamlit import session_state as ss

c00, c01  = st.columns([0.1, 0.18])

with c00:
    with st.container(border=True) : 
        # st.header("Data flow chart")
        st.image(image = "pics/data_flow_chart_2.png", caption="Data flow chart", width=None, 
                 use_column_width=None, clamp=False, channels="RGB", output_format="auto", use_container_width=False)