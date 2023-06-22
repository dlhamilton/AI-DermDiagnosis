import numpy as np
import pandas as pd
import os
import base64
from datetime import datetime
import joblib


def download_dataframe_as_csv(df):
    """
    Generates a download link for a Pandas DataFrame as a CSV file.

    Args:
        df (pd.DataFrame): The DataFrame to be downloaded.

    Returns:
        str: The HTML code for the download link.

    Example:
        df = pd.DataFrame(...)
        download_link = download_dataframe_as_csv(df)
    """

    datetime_now = datetime.now().strftime("%d%b%Y_%Hh%Mmin%Ss")
    csv = df.to_csv().encode()
    b64 = base64.b64encode(csv).decode()
    href = (
        f'<a href="data:file/csv;base64,{b64}" download="Report {datetime_now}.csv" '
        f'target="_blank">Download Report</a>'
    )
    return href


def load_pkl_file(file_path):
    """
    Loads a Pickle file (.pkl) from the specified path.

    Args:
        file_path (str): The path to the Pickle file.

    Returns:
        obj: The object loaded from the Pickle file.

    Example:
        model = load_pkl_file('model.pkl')
    """
    return joblib.load(filename=file_path)