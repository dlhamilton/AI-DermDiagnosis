import streamlit as st
import matplotlib.pyplot as plt


def page_project_hypothesis_body():
    """
    Displays the body of the project hypothesis and validation page.

    The page presents the project hypothesis and its potential impact on skin cancer diagnosis and treatment.
    It also highlights the studies conducted to validate the hypothesis and provides an overview of the AI-powered
    web application's role in expediting the diagnosis process.

    Returns:
        None
    """
    
    st.write("### Project Hypothesis and Validation")

    st.success(
        f"* Providing a confidence level for each prediction would allow users to gauge the urgency of a medical consultation. "
        f"More time will be needed to see if the confidence level is getting people to go and see consultants sooner. \n\n"

        f"* An Image Montage shows that typically a malignant lesion has a darker colour and a solid shape. "
        f"Average Image, Variability Image and Difference between Averages studies did reveal "
        f"a clear difference with malignant lesions. \n\n"

        f"* An AI-powered web application will expedite the skin cancer diagnosis process, leading to early "
        f"detection and better survival rates. "
        f"The use of the app has increased. More time will be needed to see if the early dectection is improving treatment. \n\n"

    )