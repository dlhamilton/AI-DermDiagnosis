import streamlit as st
from src.data_management import load_pkl_file


def load_test_evaluation(version):
    """
    Loads the evaluation results of the test set for a specific version of the model.

    Args:
        version (str): The version of the model for which to load the evaluation results.

    Returns:
        obj: The evaluation results loaded from the Pickle file.

    Example:
        evaluation_results = load_test_evaluation('v5')
    """
    return load_pkl_file(f'outputs/modelling_evaluation_v5/evaluation.pkl')