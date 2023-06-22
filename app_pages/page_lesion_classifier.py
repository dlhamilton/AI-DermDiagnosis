import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd

from src.data_management import download_dataframe_as_csv
from src.machine_learning.predictive_analysis import (
                                                    load_model_and_predict,
                                                    resize_input_image,
                                                    plot_predictions_probabilities
                                                    )

def page_lesion_classifier_body():
    st.info(
        f"* The client is interested in telling what class a lesion is part of."
        )

    st.write(
        f"* You can download a set of lesions for live prediction. "
        f"You can download the images from [here](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000?select=HAM10000_images_part_2)."
        )

    st.write("---")

    image_shape = (75, 75)  # Example: (75, 75)
    model_path = 'outputs/modelling_evaluation_v5/lesion_classifier_model.h5'
    class_names = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']  # Example class names

    images_buffer = st.file_uploader('Upload skin lesion images. You may select more than one.',
                                    type=['png', 'jpeg', 'jpg'], accept_multiple_files=True)

    if images_buffer:
        df_report = pd.DataFrame(columns=["Name", "Result"])
        
        for image_buffer in images_buffer:
            img_pil = Image.open(image_buffer)
            
            st.info(f"Image: **{image_buffer.name}**")
            st.image(img_pil, caption=f"Uploaded Image")
            
            resized_img = resize_input_image(img_pil, image_shape)
            pred_proba, pred_class = load_model_and_predict(resized_img, model_path, class_names)
            plot_predictions_probabilities(pred_proba, pred_class, class_names)
            
            df_report = df_report.append({"Name": image_buffer.name, 'Result': pred_class}, ignore_index=True)
        
        if not df_report.empty:
            st.subheader("Analysis Report")
            st.table(df_report)
