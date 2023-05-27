![CI logo](https://codeinstitute.s3.amazonaws.com/fullstack/ci_logo_small.png)

## Codeanywhere Template Instructions

Welcome,

This is the Code Institute student template for Codeanywhere. We have preinstalled all of the tools you need to get started. It's perfectly ok to use this template as the basis for your project submissions. Click the `Use this template` button above to get started.

You can safely delete the Gitpod Template Instructions section of this README.md file,  and modify the remaining paragraphs for your own project. Please do read the Gitpod Template Instructions at least once, though! It contains some important information about Gitpod and the extensions we use.

## How to use this repo

1. Use this template to create your GitHub project repo

1. Log into <a href="https://app.codeanywhere.com/" target="_blank" rel="noreferrer">CodeAnywhere</a> with your GitHub account.

1. On your Dashboard, click on the New Workspace button

1. Paste in the URL you copied from GitHub earlier

1. Click Create

1. Wait for the workspace to open. This can take a few minutes.

1. Open a new terminal and <code>pip3 install -r requirements.txt</code>

1. In the terminal type <code>pip3 install jupyter</code>

1. In the terminal type <code>jupyter notebook --NotebookApp.token='' --NotebookApp.password=''</code> to start the jupyter server.

1. Open port 8888 preview or browser

1. Open the jupyter_notebooks directory in the jupyter webpage that has opened and click on the notebook you want to open.

1. Click the button Not Trusted and choose Trust.

Note that the kernel says Python 3. It inherits from the workspace so it will be Python-3.8.12 as installed by our template. To confirm this you can use <code>! python --version</code> in a notebook code cell.

## Gitpod Reminders

To log into the Heroku toolbelt CLI:

1. Log in to your Heroku account and go to *Account Settings* in the menu under your avatar.
2. Scroll down to the *API Key* and click *Reveal*
3. Copy the key
4. In Gitpod, from the terminal, run `heroku_config`
5. Paste in your API key when asked

You can now use the `heroku` CLI program - try running `heroku apps` to confirm it works. This API key is unique and private to you so do not share it. If you accidentally make it public then you can create a new one with *Regenerate API Key*.

## user Stories

[View Project](https://github.com/users/dlhamilton/projects/)

### Must Have

- [#1](https://github.com/dlhamilton/AI-DermDiagnosis/issues/1) As a data scientist, I must create a machine learning model that can differentiate between benign and malignant skin lesions, so that the system can provide accurate diagnoses.
- [#2](https://github.com/dlhamilton/AI-DermDiagnosis/issues/2) As a user, I need a web application where I can upload an image of a skin lesion and receive an instant predition, so I can assess the urgency of medical consultation.
- [#3](https://github.com/dlhamilton/AI-DermDiagnosis/issues/3) As a user, I need the system to provide a confidence level with each prediction, so I can understand how certain the model is about its diagnosis.
- [#4](https://github.com/dlhamilton/AI-DermDiagnosis/issues/4) As a healthcare professional, I need the AI model to recommend immediate medical consultation if a skin lesion is predicted as malignant with high confidence, so I can expedite the treatment process.

### Should Have

- [#5](https://github.com/dlhamilton/AI-DermDiagnosis/issues/5) As a data scientist, I should implement a clustering algorithm to identify common characteristics associated with benign or malignant conditions, to improve the machine learning model's understanding and prediction accuracy.
- [#6](https://github.com/dlhamilton/AI-DermDiagnosis/issues/6) As a user, I should receive information about the associated cluster when I upload an image, so I can learn more about the nature of the skin lesion.

### Could Have

- [#7](https://github.com/dlhamilton/AI-DermDiagnosis/issues/7) As a user, I could have access to a database of example skin lesion images within the app, so I can compare my skin lesion with others.
- [#8](https://github.com/dlhamilton/AI-DermDiagnosis/issues/8) As a healthcare professional, I could have access to the details of the machine learning model's prediction process, so I can better understand how the AI system reached its conclusion.

### Won't Have this Time

- [#9](https://github.com/dlhamilton/AI-DermDiagnosis/issues/9) As a user, I won't have the ability to use the app for definitive medical diagnosis, as the app serves to provide an additional layer of information and should not replace professional medical advice.
- [#10](https://github.com/dlhamilton/AI-DermDiagnosis/issues/10) As a data scientist, I won't develop a feature for users to track changes in their skin lesions over time within the app, as it falls outside the scope of the current project.

## Epics

### Epic 1: Machine Learning Model Development

#### Must Have

- As a data scientist, I must create a machine learning model that can differentiate between benign and malignant skin lesions, so that the system can provide accurate diagnoses.

#### Should Have

- As a data scientist, I should implement a clustering algorithm to identify common characteristics associated with benign or malignant conditions, to improve the machine learning model's understanding and prediction accuracy.
  
### Epic 2: Web Application Design and Deployment

#### Must Have

- As a user, I need a web application where I can upload an image of a skin lesion and receive an instant prediction, so I can assess the urgency of medical consultation.

#### Could Have

- As a user, I could have access to a database of example skin lesion images within the app, so I can compare my skin lesion with others.

### Epic 3: Confidence Level and Medical Consultation Recommendation

#### Must Have

- As a user, I need the system to provide a confidence level with each prediction, so I can understand how certain the model is about its diagnosis.
- As a healthcare professional, I need the AI model to recommend immediate medical consultation if a skin lesion is predicted as malignant with high confidence, so I can expedite the treatment process.
-

### Epic 4: Cluster Identification and Display

#### Should Have

- As a user, I should receive information about the associated cluster when I upload an image, so I can learn more about the nature of the skin lesion.

### Epic 5: Information Accessibility for Healthcare Professionals

#### Could Have

- As a healthcare professional, I could have access to the details of the machine learning model's prediction process, so I can better understand how the AI system reached its conclusion.

### Epic 6: Limitations and Future Enhancements

#### Won't Have this Time

- As a user, I won't have the ability to use the app for definitive medical diagnosis, as the app serves to provide an additional layer of information and should not replace professional medical advice.
- As a data scientist, I won't develop a feature for users to track changes in their skin lesions over time within the app, as it falls outside the scope of the current project.

## Dataset Content

- The dataset is sourced from [Kaggle](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000). We then created a fictitious user story where predictive analytics can be applied in a real project in the workplace.
- The dataset contains dermatoscopic images from different populations, acquired and stored by different modalities. The dataset consists of 10015 dermatoscopic images which can serve as a training set for machine learning purposes. The data set have a collection of all important diagnostic categories in the realm of pigmented lesions: Actinic keratoses and intraepithelial carcinoma / Bowen's disease (akiec), basal cell carcinoma (bcc), benign keratosis-like lesions (solar lentigines / seborrheic keratoses and lichen-planus like keratoses, bkl), dermatofibroma (df), melanoma (mel), melanocytic nevi (nv) and vascular lesions (angiomas, angiokeratomas, pyogenic granulomas and hemorrhage, vasc).

### Data set breakdown

#### categories of skin lesions

- Actinic keratoses and Intraepithelial Carcinoma / Bowen's disease (akiec): These are precancerous or early forms of skin cancer that can become malignant if left untreated.
- Basal Cell Carcinoma (bcc): This is a type of skin cancer and is therefore malignant. It's one of the most common types of skin cancer.
- Benign keratosis-like lesions (Solar Lentigines / Seborrheic Keratoses and Lichen Planus-like Keratoses, bkl): These are typically benign and are not cancerous.
- Dermatofibroma (df): These are benign skin growths.
- Melanoma (mel): This is a type of skin cancer and is therefore malignant. Melanoma is the most dangerous type of skin cancer.
- Melanocytic Nevi (nv): These are commonly known as moles. They are typically benign but can become malignant, turning into melanoma.
- Vascular lesions (Angiomas, Angiokeratomas, Pyogenic Granulomas, and Hemorrhage, vasc): These are typically benign, but like all skin conditions, any changes should be monitored for potential malignancy.

#### categories of diagnosis

- Histopathology (histo): This is the most definitive diagnosis method and involves examining cells or tissues under a microscope to check for abnormalities. When a lesion is confirmed by histopathology, it means a biopsy was performed, and the lesion was examined in a lab. If a user receives a high-confidence prediction that aligns with this method, it's essential to consult a healthcare professional immediately for further examination and treatment.

- Follow-up examination (follow_up): This diagnosis method involves monitoring the lesion over time to check for changes. If the AI system indicates a need for a follow-up based on changes in the lesion, it would be wise to consult a healthcare professional for an in-person evaluation.

- Expert consensus (consensus): This diagnosis is determined when a group of dermatologists agrees on the nature of the lesion after examining it. If the prediction matches with an expert consensus, consulting a healthcare professional for an in-person examination is recommended.

- In-vivo confocal microscopy (confocal): This non-invasive imaging technique provides a real-time "optical biopsy" of the skin, allowing for a diagnosis without the need for a physical biopsy. If a high-confidence prediction aligns with this method, it should be taken seriously, and an in-person consultation with a healthcare professional is necessary.

In all cases, regardless of the diagnostic method, a prediction by the AI should be considered as a suggestion or an additional tool for skin cancer screening, not a definitive diagnosis. A healthcare professional should always be consulted for accurate diagnosis and treatment recommendations.

## Business Requirements

- AI-Derm is a healthcare startup that uses artificial intelligence for dermatological diagnoses. It is spearheading a project to develop an AI-powered application that can help healthcare professionals and individuals identify skin cancer at an early stage. The primary aim is to distinguish malignant skin lesions from benign ones, providing an immediate diagnosis and confidence level, which would recommend further medical consultation if necessary.

- Currently, the process of diagnosing skin cancer is often time-consuming and requires expert analysis. A dermatologist might need to examine the suspicious lesion visually, perform a dermoscopic analysis, and in some cases, conduct a biopsy to definitively diagnose if the skin lesion is malignant. This process can take from days to weeks, depending on various factors such as the healthcare system's efficiency and the need for further tests.

- However, early detection of skin cancer, particularly malignant melanoma, is crucial for effective treatment. When identified at an early stage, the survival rate for melanoma is significantly high. However, if the cancer has spread to other parts of the body, the survival rate drops dramatically. This underlines the importance of rapid and accurate diagnosis.

- The proposed AI model could provide instant predictions with associated confidence levels, significantly reducing the time required to identify potential malignant skin lesions. This immediacy could lead to earlier consultations and interventions, increasing the chances of successful treatment. It's important to note that while the AI model can assist with initial screening and diagnosis, any prediction it makes must be verified by a healthcare professional to ensure accuracy and safety.

  - The client is interested in creating a machine learning model that can visually differentiate a benign skin lesion from a malignant one using the lesion images.

  - The model should provide a confidence level for each prediction, indicating how certain the model is about its diagnosis.

  - If a skin lesion is predicted as malignant with high confidence, the system should recommend immediate medical consultation.

  - Use clustering algorithms to identify patterns within the skin lesion images, helping to understand characteristics commonly associated with benign or malignant conditions.

  - Deliver a web application where users can upload a skin lesion image, and the system will provide a diagnosis, confidence level of the prediction, and associated cluster.

  - The AI model's insights should be used to assist healthcare professionals in making informed decisions on the treatment process.

This solution aims to augment the decision-making process for dermatologists and bring about a transformative change in the early detection and treatment of skin cancer.

## Hypothesis and how to validate?

### Hypothesis 1

A clustering algorithm can identify patterns and characteristics commonly associated with benign or malignant conditions, providing an additional layer of information and insights.

**Validation Approach**: After applying the clustering algorithm on the dataset, review the clusters' properties manually. If clusters show distinct characteristics aligning with benign or malignant classes, the hypothesis is validated.

### Hypothesis 2

Providing a confidence level for each prediction would allow users to gauge the urgency of a medical consultation.

**Validation Approach**: Collect user feedback or conduct user testing to determine whether the confidence level helps users make informed decisions about seeking medical advice. High user satisfaction and increased rate of medical consultations following high-confidence malignant predictions would validate this hypothesis.

### Hypothesis 3

An AI-powered web application will expedite the skin cancer diagnosis process, leading to early detection and better survival rates.

**Validation Approach**: Monitor the usage metrics of the web application, including the number of unique users, number of images uploaded, and time to consultation after receiving a prediction. Additionally, conduct surveys or interviews with healthcare professionals to determine if they've seen improvements in the early detection and treatment of skin cancers. Positive feedback and statistics would validate this hypothesis.

## The rationale to map the business requirements to the Data Visualisations and ML tasks

- List your business requirements and a rationale to map them to the Data Visualisations and ML tasks

## ML Business Case

- In the previous bullet, you potentially visualised an ML task to answer a business requirement. You should frame the business case using the method we covered in the course

## Dashboard Design

- List all dashboard pages and their content, either blocks of information or widgets, like buttons, checkboxes, images, or any other item that your dashboard library supports.

- Later, during the project development, you may revisit your dashboard plan to update a given feature (for example, at the beginning of the project you were confident you would use a given plot to display an insight but subsequently you used another plot type).

## Unfixed Bugs

- You will need to mention unfixed bugs and why they were not fixed. This section should include shortcomings of the frameworks or technologies used. Although time can be a significant variable to consider, paucity of time and difficulty understanding implementation is not a valid reason to leave bugs unfixed.

## Deployment

### Heroku

- The App live link is: <https://YOUR_APP_NAME.herokuapp.com/>
- Set the runtime.txt Python version to a [Heroku-20](https://devcenter.heroku.com/articles/python-support#supported-runtimes) stack currently supported version.
- The project was deployed to Heroku using the following steps.

1. Log in to Heroku and create an App
2. At the Deploy tab, select GitHub as the deployment method.
3. Select your repository name and click Search. Once it is found, click Connect.
4. Select the branch you want to deploy, then click Deploy Branch.
5. The deployment process should happen smoothly if all deployment files are fully functional. Click now the button Open App on the top of the page to access your App.
6. If the slug size is too large then add large files not required for the app to the .slugignore file.

## Main Data Analysis and Machine Learning Libraries

- Here you should list the libraries you used in the project and provide an example(s) of how you used these libraries.

## Credits

- In this section, you need to reference where you got your content, media and extra help from. It is common practice to use code from other repositories and tutorials, however, it is important to be very specific about these sources to avoid plagiarism.
- You can break the credits section up into Content and Media, depending on what you have included in your project.

### Content

- The text for the Home page was taken from Wikipedia Article A
- Instructions on how to implement form validation on the Sign-Up page was taken from [Specific YouTube Tutorial](https://www.youtube.com/)
- The icons in the footer were taken from [Font Awesome](https://fontawesome.com/)

### Media

- The photos used on the home and sign-up page are from This Open-Source site
- The images used for the gallery page were taken from this other open-source site

## Acknowledgements (optional)

- Thank the people that provided support through this project.
