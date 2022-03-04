import streamlit as st
import pandas as pd
import numpy as np

from PIL import Image

st.set_page_config(layout="wide")

st.title("Experiment Results using Modified ResNet18 and Isolation Forest \
    from Scikit-learn") 

select_slide = st.sidebar.selectbox(
    "Which slide would you like to navigate to?",
    ("Modified ResNet18", "Isolation Forest", "Results")
)

if select_slide == "Modified ResNet18": 
    Ori_ResNet18_df= pd.read_csv('Original_ResNet18_Archi.csv')
    Modified_ResNet18_df= pd.read_csv('Modified_ResNet18_Archi.csv')
    ResNet_archi_AdaCos_image = Image.open('ResNet_archi_AdaCos.png')

    st.subheader("Modified ResNet18 from PyTorch as Feature Extractor") 

    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("Original ResNet18 Architecture from PyTorch:")
        st.table(Ori_ResNet18_df)
    
    with col2:
        st.write("Modified ResNet18 Architecture:")
        st.table(Modified_ResNet18_df)
    
    with col3:
        st.write("Modified ResNet Architecture \
            from the Team at the 3rd Place:")
        st.image(ResNet_archi_AdaCos_image)

    st.write("The team that got the 3rd place also utiized sub-cluster AdaCos \
        cosine-based softmax loss in the ResNet architecture. This will be \
        experimented with later.")
    st.write("After feature extraction, each audio file is converted into a \
        1*512 array.")


elif select_slide == "Isolation Forest":
    st.subheader("Settings for the Isolation Forest Models") 

    st.write("The training data is a 3009*512 array. Each row represents \
        an audio file.")
    
    st.write("Train the models using the parameter grid below:")
    IF_parameter_grid_df = pd.read_csv('IF parameter grid.csv')
    st.table(IF_parameter_grid_df)

    st.write("After training, the models will compute an anomaly score for \
        each row, which will be used for computing the AUC and pAUC values.")
    
elif select_slide == "Results":
    st.subheader("Best Results for Modified ResNet18 and Isolation Forest using Fan's Development Dataset") 
    result_df = pd.read_csv('Results.csv')
    col_ref = {'Baseline 1:': 'background-color: #ffec8c', 
            'Autoencoder': 'background-color: #ffec8c', 
            'Baseline 2:':'background-color: #c2f5ff',
            'MobileNetV2':'background-color: #c2f5ff'}
    st.table(result_df.style.apply(lambda x: pd.DataFrame(col_ref, \
        index=result_df.index, columns=result_df.columns).fillna(''), axis=None))
    
    st.subheader("Best Parameters for Isolation Forest:")
    st.write("n_estimators: 100")
    st.write("max_samples: 700")
    st.write("max_features: 10")
    st.write("contamination: 0.0001")


