import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.image import imread
from src.machine_learning.evaluate_clf import load_test_evaluation


def page_ml_performance_metrics():
    """
    Displays the performance metrics and evaluation results of a machine learning model.

    The page includes visualizations and statistics such as label frequencies in the train, validation, and test sets,
    model training history (accuracy and loss), general performance on the test set (loss and accuracy), confusion matrix,
    classification report, ROC curve, and precision-recall curve.

    Returns:
        None
    """
    
    version = 'v1'

    st.write("### Train, Validation and Test Set: Labels Frequencies")
    labels_distribution = plt.imread(f"outputs/modelling_evaluation_v2/labels_distribution.png")
    st.image(labels_distribution, caption='Labels Distribution on Train, Validation and Test Sets')
    st.write("---")

    st.write("### Model History")
    col1, col2 = st.columns(2)
    with col1: 
        model_acc = plt.imread(f"outputs/modelling_evaluation_v5/model_training_acc.png")
        st.image(model_acc, caption='Model Training Accuracy')
    with col2:
        model_loss = plt.imread(f"outputs/modelling_evaluation_v5/model_training_losses.png")
        st.image(model_loss, caption='Model Training Losses')
    st.write("---")

    st.write("### Generalised Performance on Test Set")
    st.dataframe(pd.DataFrame(load_test_evaluation(version), index=['Loss', 'Accuracy']))
    st.write("---")

    st.write("### Confusion Matrix")
    confusion_matrix = plt.imread(f"outputs/modelling_evaluation_v5/confusion_matrix.png")
    st.image(confusion_matrix, caption='Confusion Matrix')
    st.write("---")

    st.write("### Classification Report")
    file_path = 'outputs/modelling_evaluation_v5/classification_report.txt'
    try:
        with open(file_path, 'r') as file:
            content = file.read()
    except FileNotFoundError:
        content = "File not found"

    st.text("Classification Report:")
    st.text_area("", content, height=400)
    st.write("---")

    st.write("### ROC Curve")
    confusion_matrix = plt.imread(f"outputs/modelling_evaluation_v5/roc_curves.png")
    st.image(confusion_matrix, caption='Confusion Matrix')
    st.write("---")

    st.write("### Precision Recall Curve")
    confusion_matrix = plt.imread(f"outputs/modelling_evaluation_v5/precision_recall_curves.png")
    st.image(confusion_matrix, caption='Confusion Matrix')
    st.write("---")