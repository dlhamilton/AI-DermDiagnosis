import streamlit as st
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import matplotlib.image as mpimg
from src.machine_learning.visualize_difference import (
                                                    plot_diff_bet_avg_image_labels
                                                    )

import itertools
import random

def page_lesion_exploration_body():
    st.write("### Lesion Visualizer")
    st.info(
        f"* The client is interested in having a study that visually "
        f"differentiates between the lesion classes.")
    

    if st.checkbox("informaion about each lesion class"):
      
      st.write("#### Actinic keratoses and Intraepithelial Carcinoma / Bowen's disease (akiec)")
      st.write(f"These are precancerous or early forms of skin cancer that can become malignant"
               f" if left untreated.\n")
      
      st.write("#### Basal Cell Carcinoma (bcc)")
      st.write(f"This is a type of skin cancer and is therefore malignant. It's one of the most"
               f" common types of skin cancer.\n")
      
      st.write("#### Benign keratosis-like lesions (Solar Lentigines / Seborrheic Keratoses and Lichen Planus-like Keratoses) (bkl)")
      st.write(f"These are typically benign and are not cancerous.\n")

      st.write("#### Dermatofibroma (df)")
      st.write(f"These are benign skin growths.\n")

      st.write("#### Melanoma (mel)")
      st.write(f"This is a type of skin cancer and is therefore malignant. Melanoma is the most"
               f" dangerous type of skin cancer.\n")
      
      st.write("#### Melanocytic Nevi (nv)")
      st.write(f"These are commonly known as moles. They are typically benign but can become "
               f"malignant, turning into melanoma.\n")
      
      st.write("#### Vascular lesions (Angiomas, Angiokeratomas, Pyogenic Granulomas, and Hemorrhage) (vasc)")
      st.write(f"These are typically benign, but like all skin conditions, any changes should"
               f" be monitored for potential malignancy. \n")
      
      st.write("---")


    version = 'v1'
    if st.checkbox("Difference between average and variability image"):
      
      avg_akiec= plt.imread(f"outputs/{version}/avg_var_akiec.png")
      avg_bcc = plt.imread(f"outputs/{version}/avg_var_bcc.png")
      avg_bkl = plt.imread(f"outputs/{version}/avg_var_bkl.png")
      avg_df = plt.imread(f"outputs/{version}/avg_var_df.png")
      avg_mel = plt.imread(f"outputs/{version}/avg_var_mel.png")
      avg_nv= plt.imread(f"outputs/{version}/avg_var_nv.png")
      avg_vasc = plt.imread(f"outputs/{version}/avg_var_vasc.png")

      st.warning(
        f"* We notice the average and variability images did not show "
        f"patterns where we could intuitively differentiate one from another. " 
        f"However, a small difference in the colour for mel and nv and a purplish"
        f" colour for vasc")

      st.image(avg_akiec, caption='Akiec - Average and Variability')
      st.image(avg_bcc, caption='Bcc - Average and Variability')
      st.image(avg_bkl, caption='Bkl - Average and Variability')
      st.image(avg_df, caption='Df - Average and Variability')
      st.image(avg_mel, caption='Mel - Average and Variability')
      st.image(avg_nv, caption='Nv - Average and Variability')
      st.image(avg_vasc, caption='Vasc - Average and Variability')

      st.write("---")


    if st.checkbox("Differences between average lesion types"):

      X = np.load('outputs/v1/X.npy', allow_pickle=True)
      y = np.load('outputs/v1/y.npy', allow_pickle=True)

      st.warning(
            f"* We notice this study didn't show "
            f"patterns where we could intuitively differentiate one from another.")

      unique_labels = np.unique(y)
      label_1 = st.selectbox("Select the first label", options=unique_labels, index=0)
      label_2 = st.selectbox("Select the second label", options=unique_labels, index=1)

      if label_1 and label_2:
        plot = plot_diff_bet_avg_image_labels(X, y, label_1, label_2)
        st.pyplot(plot)
      st.write("---")

    
    if st.checkbox("Differences between lesions colours"):
      st.warning(
        f"* We notice this study didn't show "
        f"patterns where we could intuitively differentiate one from another."
        f" There is a slightly darker colour for some of the lesion types")

      image_names = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
      selected_type_1= st.selectbox("Select type 1", options=image_names, index=0)
      selected_type_2 = st.selectbox("Select type 2", options=image_names, index=1)

      if selected_type_1 and selected_type_2:
        type1_path = os.path.join('outputs/v1/', f'col_dis_{selected_type_1}.png')
        type2_path = os.path.join('outputs/v1/', f'col_dis_{selected_type_2}.png')

        # Load images
        img1 = mpimg.imread(type1_path)
        img2 = mpimg.imread(type2_path)
        
        # Plot images side by side
        col1, col2 = st.columns(2)
        with col1:
            st.image(img1, caption=f'col_dis_{selected_type_1}', use_column_width=True)
        with col2:
            st.image(img2, caption=f'col_dis_{selected_type_2}', use_column_width=True)
      st.write("---")


    if st.checkbox("Montage of lesion shapes"):
      st.warning(
        f"* We notice this study shows there is a slight pattern"
        f" in shapes which differentiate between types of lesions.")

      image_names = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
      for image_name in image_names:
        image_path = os.path.join('outputs/v1/', f'thresholded_montage_{image_name}.png')
        st.image(image_path, caption=image_name, use_column_width=True)

    if st.checkbox("Image Montage"): 
      st.write("* To refresh the montage, click on the 'Create Montage' button")
      my_data_dir = 'inputs/skin_cancer_dataset/sorted_images'
      labels = os.listdir(my_data_dir+ '/validation')
      label_to_display = st.selectbox(label="Select label", options=labels, index=0)
      if st.button("Create Montage"):      
        image_montage(dir_path= my_data_dir + '/validation',
                      label_to_display=label_to_display,
                      nrows=8, ncols=3, figsize=(10,25))
      st.write("---")


def image_montage(dir_path, label_to_display, nrows, ncols, figsize=(15,10)):
  sns.set_style("white")
  labels = os.listdir(dir_path)

  # subset the class you are interested to display
  if label_to_display in labels:

    # checks if your montage space is greater than subset size
    # how many images in that folder
    images_list = os.listdir(dir_path+'/'+ label_to_display)
    if nrows * ncols < len(images_list):
      img_idx = random.sample(images_list, nrows * ncols)
    else:
      print(
          f"Decrease nrows or ncols to create your montage. \n"
          f"There are {len(images_list)} in your subset. "
          f"You requested a montage with {nrows * ncols} spaces")
      return
    

    # create list of axes indices based on nrows and ncols
    list_rows= range(0,nrows)
    list_cols= range(0,ncols)
    plot_idx = list(itertools.product(list_rows,list_cols))


    # create a Figure and display images
    fig, axes = plt.subplots(nrows=nrows,ncols=ncols, figsize=figsize)
    for x in range(0,nrows*ncols):
      img = mpimg.imread(dir_path + '/' + label_to_display + '/' + img_idx[x])
      img_shape = img.shape
      axes[plot_idx[x][0], plot_idx[x][1]].imshow(img)
      axes[plot_idx[x][0], plot_idx[x][1]].set_title(f"Width {img_shape[1]}px x Height {img_shape[0]}px")
      axes[plot_idx[x][0], plot_idx[x][1]].set_xticks([])
      axes[plot_idx[x][0], plot_idx[x][1]].set_yticks([])
    plt.tight_layout()
    
    st.pyplot(fig=fig)
    # plt.show()


  else:
    print("The label you selected doesn't exist.")
    print(f"The existing options are: {labels}")