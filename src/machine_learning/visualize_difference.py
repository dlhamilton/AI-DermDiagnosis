import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from PIL import Image
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns


def subset_image_label(X, y, label_to_display):
    """
    Subsets the images based on a specific label from the dataset.

    Args:
        X (np.ndarray): The input images.
        y (np.ndarray): The corresponding labels.
        label_to_display: The label to subset the images.

    Returns:
        np.ndarray: The subset of images with the specified label.

    """

    y = y.reshape(-1, 1, 1)
    boolean_mask = np.any(y == label_to_display, axis=1).reshape(-1)
    df = X[boolean_mask]
    return df

def plot_diff_bet_avg_image_labels(X, y, label_1, label_2):
    """
    Plots the difference between the average images of two specified labels.

    Args:
        X (np.ndarray): The input images.
        y (np.ndarray): The corresponding labels.
        label_1: The first label to compare.
        label_2: The second label to compare.

    Returns:
        matplotlib.figure.Figure: The generated figure containing the plot.

    """

    sns.set_style("white")
    
    # calculate mean from label1
    images_label = subset_image_label(X, y, label_1)
    label1_avg = np.mean(images_label, axis=0)
    
    # calculate mean from label2
    images_label = subset_image_label(X, y, label_2)
    label2_avg = np.mean(images_label, axis=0)
    
    # calculate difference and plot difference, avg label1 and avg label2
    difference_mean = label1_avg - label2_avg
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 10))
    axes[0].imshow(label1_avg, cmap='gray')
    axes[0].set_title(f'Average {label_1}')
    axes[1].imshow(label2_avg, cmap='gray')
    axes[1].set_title(f'Average {label_2}')
    axes[2].imshow(difference_mean, cmap='gray')
    axes[2].set_title(f'Difference image: Avg {label_1} & {label_2}')
    
    return fig
