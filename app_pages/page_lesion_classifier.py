import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd

from src.machine_learning.predictive_analysis import (
                                                    load_model_and_predict,
                                                    resize_input_image,
                                                    plot_predictions_probabilities
                                                    )

def page_lesion_classifier_body():
    """
    Displays the body of the lesion classifier page.

    The page provides information about classifying skin lesions and allows users to upload skin lesion images for live prediction.
    It also displays analysis reports for the uploaded images.

    Returns:
        None
    """

    st.info(
        f"* The client is interested in telling what class a lesion is part of."
        )

    st.write(
        f"* You can download a set of lesions for live prediction. "
        f"You can download the images from [here](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000?select=HAM10000_images_part_2)."
        )

    st.write("---")

    image_shape = (75, 75)
    model_path = 'outputs/modelling_evaluation_v5_small/lesion_classifier_model.h5'
    class_names = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

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
            
            descriptions = {
                "akiec": "Actinic keratoses and Intraepithelial Carcinoma / Bowen's disease (akiec): These are precancerous or early forms of skin cancer that can become malignant if left untreated.",
                "bcc": "Basal Cell Carcinoma (bcc): This is a type of skin cancer and is therefore malignant. It's one of the most common types of skin cancer.",
                "bkl": "Benign keratosis-like lesions (Solar Lentigines / Seborrheic Keratoses and Lichen Planus-like Keratoses, bkl): These are typically benign and are not cancerous.",
                "df": "Dermatofibroma (df): These are benign skin growths.",
                "mel": "Melanoma (mel): This is a type of skin cancer and is therefore malignant. Melanoma is the most dangerous type of skin cancer.",
                "nv": "Melanocytic Nevi (nv): These are commonly known as moles. They are typically benign but can become malignant, turning into melanoma.",
                "vasc": "Vascular lesions (Angiomas, Angiokeratomas, Pyogenic Granulomas, and Hemorrhage, vasc): These are typically benign, but like all skin conditions, any changes should be monitored for potential malignancy."
            }

            df_report = df_report.append({"Name": image_buffer.name, "Result": pred_class, "Description": descriptions.get(pred_class, "")},
                                         ignore_index=True)
            
            if pred_class in ["akiec", "bcc", "mel"]:
                st.error("Warning: It is recommended to get checked by a healthcare professional.")
            else:
                st.warning("Warning: It is recommended to monitor this and get it verified by a healthcare professional.")
        
        if not df_report.empty:
            st.subheader("Analysis Report")
            st.table(df_report)
