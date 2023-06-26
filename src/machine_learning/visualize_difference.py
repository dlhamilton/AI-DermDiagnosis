import os
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from PIL import Image
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing import image


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

def plot_diff_bet_avg_image_labels(images_label_1, images_label_2, label_1, label_2):
    """
    Plots the difference between the average images of two specified labels.

    Args:
        images_label_1 (np.ndarray): The images belonging to label_1.
        images_label_2 (np.ndarray): The images belonging to label_2.
        label_1: The first label to compare.
        label_2: The second label to compare.

    Returns:
        matplotlib.figure.Figure: The generated figure containing the plot.

    """

    sns.set_style("white")

    label1_avg = np.mean(images_label_1, axis=0)
    label2_avg = np.mean(images_label_2, axis=0)

    difference_mean = label1_avg - label2_avg
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 10))
    axes[0].imshow(label1_avg, cmap='gray')
    axes[0].set_title(f'Average {label_1}')
    axes[1].imshow(label2_avg, cmap='gray')
    axes[1].set_title(f'Average {label_2}')
    axes[2].imshow(difference_mean, cmap='gray')
    axes[2].set_title(f'Difference image: Avg {label_1} & {label_2}')

    return fig


def load_image_as_array(my_data_dir, new_size=(50, 50), n_images_per_label=20):
    """
        Loads images from a directory and converts them into NumPy arrays.

        Args:
            my_data_dir (str): Path to the directory containing the images.
            new_size (tuple, optional): Size to which the images will be resized. Defaults to (50, 50).
            n_images_per_label (int, optional): Maximum number of images to load per label. Defaults to 20.

        Returns:
            tuple: A tuple containing the image arrays and corresponding labels.

        """
    X, y = np.array([], dtype='int'), np.array([], dtype='object')
    labels = os.listdir(my_data_dir)

    for label in labels:
        counter = 0
        for image_filename in os.listdir(my_data_dir + '/' + label):
            # n_images_per_label: we set a limit, since it may take too much time
            if counter < n_images_per_label:

                img = image.load_img(
                    my_data_dir + '/' + label + '/' + image_filename, target_size=new_size)
                if image.img_to_array(img).max() > 1:
                    img_resized = image.img_to_array(img) / 255
                else:
                    img_resized = image.img_to_array(img)

                X = np.append(X, img_resized).reshape(-1,
                                                      new_size[0], new_size[1], img_resized.shape[2])
                y = np.append(y, label)
                counter += 1

    return X, y