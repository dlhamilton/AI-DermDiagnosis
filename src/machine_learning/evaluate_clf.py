import streamlit as st
from src.data_management import load_pkl_file


def load_test_evaluation(version):
    return load_pkl_file(f'outputs/modelling_evaluation_v5/evaluation.pkl')