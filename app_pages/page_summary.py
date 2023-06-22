import streamlit as st
import matplotlib.pyplot as plt


def page_summary_body():
    """
    Displays the body of the project summary page.

    The page provides a quick summary of the project, including general information about skin cancer,
    details about the project dataset (Skin Cancer MNIST: HAM10000), important considerations regarding the use
    of the AI model for skin cancer detection, and the way forward in terms of inclusivity for black skin.
    It also highlights the business requirements of the project.

    Returns:
        None
    """

    st.write("### Quick Project Summary")

    st.info(
        f"**General Information**\n"
        f"* Skin cancer is one of the most common forms of cancer globally, with melanoma "
        f"being the most dangerous type.\n"
        f"* Early detection is crucial in effectively treating skin cancer. Dermatologists "
        f"often use visual inspections of skin lesions and moles to determine if further testing "
        f"is needed.\n"
        f"* According to the [World Health Organization](https://www.who.int/uv/faq/skincancer/en/index1.html), "
        f"one in every three cancers diagnosed is a skin cancer, and 1 in every 5 Britons will develop "
        f"skin cancer by the age of 70.\n\n"
        f"**Project Dataset: Skin Cancer MNIST: HAM10000**\n"
        f"* The HAM10000 dataset is a collection of dermatoscopic images of common pigmented "
        f"skin lesions and is widely used for research in skin cancer classification and detection.\n"
        f"* It contains over 10,000 images that have been annotated by clinical experts with details "
        f"about the diagnosis, localization, and other metadata.\n"
        f"* The dataset aims to facilitate the training and validation of algorithms capable of "
        f"classifying skin lesions as either benign or malignant. Such algorithms have the potential "
        f"to support and improve the accuracy of diagnoses made by dermatologists.\n"
        f"* The images in this dataset represent seven different types of skin cancer, making it "
        f"a valuable resource for developing and evaluating machine learning models for multi-class "
        f"skin lesion classification.\n\n")
    st.error(
        f"**MUST READ**. \n\n It's important to note that while the AI model can assist with initial screening and diagnosis,"
        f" any prediction it makes must be verified by a healthcare professional to ensure accuracy and safety.")
    st.warning(
        f"Utilizing this dataset to create a model for early detection can have a significant "
        f"impact on reducing mortality rates and improving the quality of life for patients affected by skin cancer."
        f"**The Way Forward: Inclusivity for Black Skin**\n"
        f"* While the HAM10000 dataset provides a solid foundation for skin cancer detection models, "
        f"it is imperative to recognize the need for inclusivity and diversity in medical datasets.\n"
        f"* Skin cancer can manifest differently on black skin compared to white skin, and historically, "
        f"medical datasets have not been representative of the diversity in skin types.\n"
        f"* As part of responsible AI development, it is essential to ensure that models are trained on "
        f"data that represents all skin tones. This will enable the development of robust and unbiased "
        f"models that can effectively serve diverse populations.\n"
        f"* Researchers and developers should consider augmenting datasets like HAM10000 with additional "
        f"images representing black and brown skin tones. This is a critical step in developing models that "
        f"are equally effective in detecting skin cancer among individuals with darker skin and addressing "
        f"health disparities.")

    st.write(
        f"* For additional information, please visit and **read** the "
        f"[Project README file](https://github.com/dlhamilton/AI-DermDiagnosis/blob/main/README.md).")
    

    st.success(
        f"The project has 7 business requirements:\n"

        f"* Business Requirement 1: - The client aims to visually differentiate lesions. The model should be "
        f"capable of reaching an accuracy of at least 70%. \n"

        f"* Business Requirement 2: - The model should provide a confidence level for each prediction.\n"

        f"* Business Requirement 3 - If a skin lesion is predicted as malignant with high confidence, "
        f"the system should recommend immediate medical consultation. \n"

        f"* Business Requirement 4 - The project will deliver a web application where users can upload a "
        f"skin lesion image, and the system will provide a diagnosis, a confidence level of the prediction \n"

        f"* Business Requirement 5 - The AI model's insights should assist healthcare professionals "
        f"in making informed decisions about the treatment process. \n"

        f"* Answer business requirement 6 - "
        f"The model's performance will be evaluated using balanced performance metrics such as F1 Score aiming for scores above 0.7. \n"

        f"* Business Requirement 7 - The client is interested to have a study to visually differentiate between "
        f"lesions. \n"
        )