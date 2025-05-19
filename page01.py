#--------------------             
# Author : Serge Zaugg
# Description : Utility functions used by other scripts
#--------------------

import streamlit as st
from streamlit import session_state as ss

c00, c01  = st.columns([0.1, 0.18])

with c00:
    with st.container(border=True) : 
        st.header("Data flow chart")
        st.image(image = "pics\data_flow_chart_2.png", caption="Data flow chart", width=None, 
                 use_column_width=None, clamp=False, channels="RGB", output_format="auto", use_container_width=False)
         
with c01:
    with st.container(border=True) : 
        
        # st.header("Summary")

        st.markdown(''' 

            ### Data
            * Data from Food Classification dataset published on Kaggle.
            * https://www.kaggle.com/datasets/bjoernjostein/food-classification
            * Over 9300 hand-annotated images with 61 classes

            ### Feature extraction (image to vector)
            * Features extracted with image classification models pre-trained with the Imagenet datataset
            * Details see here : https://docs.pytorch.org/vision/main/models.html
            * As output we used the last linear layer which outputs 1000 continuous features (ommited Softamx) 
            * These models were trained specifically for the Imagenet classes, so let's hope the feature are informative for our task

            ### Dimensionality reduction
            * We used Uniform Manifold Approximation and Projection ([UMAP](https://umap-learn.readthedocs.io)), a technique for general non-linear dimension reduction.        

            ### Clustering
            * Clustering done with [DBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html).
            * Here the focus is on clustering and label are not used during training
            * Labels are only used to assess the quality of clustering
                    
            ''')
     
 

      