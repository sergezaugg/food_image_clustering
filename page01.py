#--------------------             
# Author : Serge Zaugg
# Description : Utility functions used by other scripts
#--------------------

import streamlit as st
from streamlit import session_state as ss

c00, c01  = st.columns([0.2, 0.1])
with c00:
    with st.container(border=True) : 
        
        st.header("Summary")

        st.markdown(''' 

            ### Data
            * Data from Food Classification dataset published on Kaggle.
            * https://www.kaggle.com/datasets/bjoernjostein/food-classification
            * Over 9300 hand-annotated images with 61 classes

            ### Feature extraction (image to vector)
            * Features extracted with image classification models pre-trained with the Imagenet datataset
            * Details see here : https://docs.pytorch.org/vision/main/models.html
            * As output we used the last linear layer which outputs 1000 continuous features (ommited Softamx) 

            ### A bit of context
            * These models were trained specifically for the 1000 Imagenet classes, so let's hope the feature are informative for our task

            ### Clustering
            * Here the focus is on clustering (i.e. without using the labels)
            * Labels are only used to assess the quality of clustering
       
            ### LINKS  
            * The feature matrix dim-reduced with [UMAP](https://umap-learn.readthedocs.io). 
            * Clustering done with [DBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html).        
                    
            ''')
     
    
                

      