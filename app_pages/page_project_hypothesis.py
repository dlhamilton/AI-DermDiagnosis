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

    st.info("The ML model and app have addressed and provided a postive result to all the business requirements.")
    
    st.write("### Project Hypothesis and Validation")

    st.success(
        f"Hypothesis 1\n"
        f"Providing a confidence level for each prediction would allow users to gauge the urgency of a medical consultation. "
        f"For example: telling someone that there is a 70% chance the lesion could be cancerous would get them to check the lesion sooner.\n"

        f"Validation Approach: The validation approach involves analyzing user feedback and statistical analysis to determine if the "
        f"provision of confidence levels influenced users' decisions for medical consultations. The findings from the analysis and studies"
        f" will provide insights into whether the hypothesis holds true.\n"

        f"Result: More time will be needed to see if the confidence level is getting people to go and see consultants sooner. "
        f"The feedback section in the app and speaking to consultants to see if there has been an increase in people getting lesions checked.\n\n"

        f"Hypothesis 2\n"
        f"An Image Montage shows that typically a malignant lesion has a darker colour and a solid shape. Average Image, Variability Image,"
        f" and Difference between Averages studies did reveal a clear difference with malignant lesions.\n"

        f"Validation Approach: The validation approach involves a combination of visual analysis, expert opinions, and statistical analysis "
        f"to validate the observed differences in color and shape between malignant and non-malignant lesions. The involvement of domain "
        f"experts and the collection of additional data contribute to the validation process.\n"

        f"Result: The comparison on the app showed this to be true, and that a dark lesion is more likely to be dangerous. Average Image, "
        f"Variability Image, and Difference between Averages studies did reveal a clear difference with malignant lesions.\n\n"

        f"Hypothesis 3\n"
        f"An AI-powered web application will expedite the skin cancer diagnosis process, leading to early detection and better survival rates.\n"

        f"Validation Approach: Monitor the usage metrics of the web application, including the number of unique users, number of images "
        f"uploaded, and time to consultation after receiving a prediction. Additionally, conduct surveys or interviews with healthcare "
        f"professionals to determine if they've seen improvements in the early detection and treatment of skin cancers. Positive feedback and "
        f"statistics would validate this hypothesis.\n"

        f"Result: The use of the app has increased. More time will be needed to see if the early detection is improving treatment and "
        f"increasing the survival rate.\n\n"
    )