import os
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from scipy.special import softmax
import plotly.express as px


def load_model_and_predict(my_image, model_path, class_names):
    """
    Load and perform ML prediction over live images
    """
    model = load_model(model_path, compile=False)

    pred_logits = model.predict(my_image)[0]
    pred_proba = softmax(pred_logits)

    pred_class_index = np.argmax(pred_proba)
    pred_proba_max = np.max(pred_proba)

    target_map = {v: k for v, k in enumerate(class_names)}

    pred_class = target_map[pred_class_index]

    return pred_proba, pred_class


def plot_predictions_probabilities(pred_proba, pred_class, class_names):
    """
    Plot prediction probability results
    """
    prob_per_class = pd.DataFrame(
        data=pred_proba,
        index=class_names,
        columns=['Probability']
    )
    
    prob_per_class['Diagnostic'] = prob_per_class.index
    
    fig = px.bar(
        prob_per_class,
        x='Diagnostic',
        y='Probability',
        range_y=[0, 1],
        width=600, height=300, template='seaborn')
    st.plotly_chart(fig)


def resize_input_image(img, image_shape):
    """
    Reshape image to average image size
    """
    img_resized = img.resize((image_shape[1], image_shape[0]), Image.ANTIALIAS)
    my_image = np.expand_dims(img_resized, axis=0) / 255

    return my_image