![Banner](readme_images/banner.png)

## Table of Contents

1. [User Stories](#user-stories)
2. [Epics](#epics)
3. [Dataset Content](#dataset-content)
4. [Business Requirements](#business-requirements)
5. [Hypothesis and validation](#hypothesis-and-how-to-validate)
6. [The model](#the-model)
7. [Implementation of the Business Requirements](#the-rationale-to-map-the-business-requirements-to-the-data-visualizations-and-ml-tasks)
8. [ML Business case](#ml-business-case)
9. [Dashboard design](#dashboard-design)
10. [Bugs](#bugs)
11. [CRISP DM Process](#crisp-dm-process)
12. [Bugs](#bugs)
13. [Deployment](#deployment)
14. [Data Analysis and Machine Learning Libraries](#data-analysis-and-machine-learning-libraries)
15. [Credits](#credits)
16. [Acknowledgements](#Acknowledgements)

---

### Deployed version at [AI-DermDiagnosis](https://ai-dermdiagnosis-75d8dba881ea.herokuapp.com/)

---

## User Stories

[View Project](https://github.com/users/dlhamilton/projects/)

### Must Have User Stories

- [#1](https://github.com/dlhamilton/AI-DermDiagnosis/issues/1) As a data scientist, I must create a machine learning model that can differentiate between benign and malignant skin lesions, so that the system can provide accurate diagnoses.
- [#2](https://github.com/dlhamilton/AI-DermDiagnosis/issues/2) As a user, I need a web application where I can upload an image of a skin lesion and receive an instant predition, so I can assess the urgency of medical consultation.
- [#3](https://github.com/dlhamilton/AI-DermDiagnosis/issues/3) As a user, I need the system to provide a confidence level with each prediction, so I can understand how certain the model is about its diagnosis.
- [#4](https://github.com/dlhamilton/AI-DermDiagnosis/issues/4) As a healthcare professional, I need the AI model to recommend immediate medical consultation if a skin lesion is predicted as malignant with high confidence, so I can expedite the treatment process.

### Should Have User Stories

- [#5](https://github.com/dlhamilton/AI-DermDiagnosis/issues/5) As a data scientist, I should implement a clustering algorithm to identify common characteristics associated with benign or malignant conditions, to improve the machine learning model's understanding and prediction accuracy.
- [#6](https://github.com/dlhamilton/AI-DermDiagnosis/issues/6) As a user, I should receive information about the associated cluster when I upload an image, so I can learn more about the nature of the skin lesion.

### Could Have User Stories

- [#7](https://github.com/dlhamilton/AI-DermDiagnosis/issues/7) As a user, I could have access to a database of example skin lesion images within the app, so I can compare my skin lesion with others.
- [#8](https://github.com/dlhamilton/AI-DermDiagnosis/issues/8) As a healthcare professional, I could have access to the details of the machine learning model's prediction process, so I can better understand how the AI system reached its conclusion.

### Won't Have this Time User Stories

- [#9](https://github.com/dlhamilton/AI-DermDiagnosis/issues/9) As a user, I won't have the ability to use the app for definitive medical diagnosis, as the app serves to provide an additional layer of information and should not replace professional medical advice.
- [#10](https://github.com/dlhamilton/AI-DermDiagnosis/issues/10) As a data scientist, I won't develop a feature for users to track changes in their skin lesions over time within the app, as it falls outside the scope of the current project.

---

## Epics

### Epic 1: Machine Learning Model Development

#### Epic 1 - Must Have

- As a data scientist, I must create a machine learning model that can differentiate between benign and malignant skin lesions, so that the system can provide accurate diagnoses.

#### Epic 1 - Should Have

- As a data scientist, I should implement a clustering algorithm to identify common characteristics associated with benign or malignant conditions, to improve the machine learning model's understanding and prediction accuracy.
  
### Epic 2: Web Application Design and Deployment

#### Epic 2 - Must Have

- As a user, I need a web application where I can upload an image of a skin lesion and receive an instant prediction, so I can assess the urgency of medical consultation.

#### Epic 2 - Could Have

- As a user, I could have access to a database of example skin lesion images within the app, so I can compare my skin lesion with others.

### Epic 3: Confidence Level and Medical Consultation Recommendation

#### Epic 3 - Must Have

- As a user, I need the system to provide a confidence level with each prediction, so I can understand how certain the model is about its diagnosis.
- As a healthcare professional, I need the AI model to recommend immediate medical consultation if a skin lesion is predicted as malignant with high confidence, so I can expedite the treatment process.
-

### Epic 4: Cluster Identification and Display

#### Epic 4 - Should Have

- As a user, I should receive information about the associated cluster when I upload an image, so I can learn more about the nature of the skin lesion.

### Epic 5: Information Accessibility for Healthcare Professionals

#### Epic 5 - Could Have

- As a healthcare professional, I could have access to the details of the machine learning model's prediction process, so I can better understand how the AI system reached its conclusion.

### Epic 6: Limitations and Future Enhancements

#### Epic 6 - Won't Have this Time

- As a user, I won't have the ability to use the app for definitive medical diagnosis, as the app serves to provide an additional layer of information and should not replace professional medical advice.
- As a data scientist, I won't develop a feature for users to track changes in their skin lesions over time within the app, as it falls outside the scope of the current project.

---

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

---

## Business Requirements

- AI-Derm is a healthcare startup that uses artificial intelligence for dermatological diagnoses. It is spearheading a project to develop an AI-powered application that can help healthcare professionals and individuals identify skin cancer at an early stage. The primary aim is to distinguish malignant skin lesions from benign ones, providing an immediate diagnosis and confidence level, which would recommend further medical consultation if necessary.

- Currently, the process of diagnosing skin cancer is often time-consuming and requires expert analysis. A dermatologist might need to examine the suspicious lesion visually, perform a dermoscopic analysis, and in some cases, conduct a biopsy to definitively diagnose if the skin lesion is malignant. This process can take from days to weeks, depending on various factors such as the healthcare system's efficiency and the need for further tests.

- However, early detection of skin cancer, particularly malignant melanoma, is crucial for effective treatment. When identified at an early stage, the survival rate for melanoma is significantly high. However, if the cancer has spread to other parts of the body, the survival rate drops dramatically. This underlines the importance of rapid and accurate diagnosis.

- The proposed AI model could provide instant predictions with associated confidence levels, significantly reducing the time required to identify potential malignant skin lesions. This immediacy could lead to earlier consultations and interventions, increasing the chances of successful treatment. It's important to note that while the AI model can assist with initial screening and diagnosis, any prediction it makes must be verified by a healthcare professional to ensure accuracy and safety.

  - **Business Requirement 1**: The client aims to visually differentiate lesions. The model should be capable of reaching an accuracy of at least 70%.

  - **Business Requirement 2**: The model should provide a confidence level for each prediction.

  - **Business Requirement 3**: If a skin lesion is predicted as malignant with high confidence, the system should recommend immediate medical consultation.

  - **Business Requirement 4**: The project will deliver a web application where users can upload a skin lesion image, and the system will provide a diagnosis, a confidence level of the prediction.

  - **Business Requirement 5**: The AI model's insights should assist healthcare professionals in making informed decisions about the treatment process.

  - **Business Requirement 6**: The model's performance will be evaluated using balanced performance metrics such as F1 Score aiming for scores above 0.7.

  - **Business Requirement 7**: The client is interested to have a study to visually differentiate between lesions.

This solution aims to augment the decision-making process for dermatologists and bring about a transformative change in the early detection and treatment of skin cancer.

---

## Hypothesis and how to validate

### Hypothesis 1

Providing a confidence level for each prediction would allow users to gauge the urgency of a medical consultation. For example: telling someone that there is a 70% chance the lesion could be cancerous would get them to check the lesion sooner. 

**Validation Approach**: The validation approach involves analyzing user feedback and statistical analysis to determine if the provision of confidence levels influenced users' decisions for medical consultations. The findings from the analysis and studies will provide insights into whether the hypothesis holds true.

**Result**: More time will be needed to see if the confidence level is getting people to go and see consultants sooner. the feedback section in the app and speaking to consultans to see if there has been a increase of people getting lesions checked. 

### Hypothesis 2

An Image Montage shows that typically a malignant lesion has a darker colour and a solid shape. Average Image, Variability Image and Difference between Averages studies did reveal a clear difference with malignant lesions.

**Validation Approach**: The validation approach involves a combination of visual analysis, expert opinions, and statistical analysis to validate the observed differences in color and shape between malignant and non-malignant lesions. The involvement of domain experts and the collection of additional data contribute to the validation process.

**Result**: The compansion on the app showed this to be true and that a dark lesion is more likely to be dangerous. Average Image, Variability Image and Difference between Averages studies did reveal a clear difference with malignant lesions.

### Hypothesis 3

An AI-powered web application will expedite the skin cancer diagnosis process, leading to early detection and better survival rates.

**Validation Approach**: Monitor the usage metrics of the web application, including the number of unique users, number of images uploaded, and time to consultation after receiving a prediction. Additionally, conduct surveys or interviews with healthcare professionals to determine if they've seen improvements in the early detection and treatment of skin cancers. Positive feedback and statistics would validate this hypothesis.

**Result**: The use of the app has increased. More time will be needed to see if the early dectection is improving treatment and increasing the survival rate. 

---

## The Model
The model has 1 input layer, Multiple Convolutional and Pooling Layers in the base model (Xception), 1 Flatten Layer, 1 Dense Layer, 1 Output Layer.

Base Model: The base model is initialized using the Xception architecture. Xception is a deep convolutional neural network (CNN) model that has shown excellent performance on image classification tasks. It is pre-trained on the ImageNet dataset, which provides a strong foundation for feature extraction.

Trainable Layers: By default, all layers in the Xception base model are set to be trainable. This means that during the training process, the weights of these layers will be updated based on the data specific to the skin lesion classification task.

Sequential Model: The Sequential model is initialized to build the classification model on top of the base model.

Flatten Layer: The Flatten layer is added to convert the multi-dimensional output of the base model into a one-dimensional feature vector, which can be input to the following dense layers.

Dense Layers: Two dense layers are added after the Flatten layer to learn high-level representations from the extracted features.

Dropout Layer: The Dropout layer with a dropout rate of 0.5 is added to mitigate overfitting. It randomly sets a fraction of input units to 0 during training, which helps prevent the model from relying too much on specific features.

Batch Normalization Layer: The Batch Normalization layer is added to normalize the activations of the previous layer, which helps stabilize and speed up the training process.

Activation Function: The ReLU (Rectified Linear Unit) activation function is chosen as the activation function for the dense layers. ReLU is commonly used in deep learning models as it introduces non-linearity and helps the model learn complex patterns.

Output Layer: The output layer consists of a Dense layer with a number of units equal to the number of classes (num_classes) in the skin lesion classification task. The model uses a softmax activation function to output the predicted probabilities for each class.

Loss Function: The model is compiled with the SparseCategoricalCrossentropy loss function, which is suitable for multi-class classification problems. It computes the cross-entropy loss between the true labels and the predicted probabilities.

Optimizer: The Adam optimizer is chosen as the optimizer for training the model. Adam is an adaptive learning rate optimization algorithm that combines the benefits of the RMSprop and AdaGrad algorithms. It helps accelerate the training process and converges to a good solution efficiently.

Metrics: The model is evaluated using the accuracy metric, which calculates the proportion of correctly classified samples during training and evaluation.

### Trial and Error
In addition to the current iteration, there were four previous iterations of the model, each documented in separate notebooks. These iterations employed different techniques to address the challenges of imbalanced data, including SMOTE upsampling, downsampling, and the use of different models.

The purpose of these iterations was to explore various approaches and evaluate their effectiveness in improving the model's performance. Each iteration represents a unique attempt to tackle the imbalanced data issue and enhance the model's ability to accurately classify skin lesions.

By employing techniques such as SMOTE upsampling and downsampling, the previous iterations aimed to address the class imbalance problem by either generating synthetic samples or reducing the number of majority class samples. These techniques were used in conjunction with different models to examine their impact on model performance.

Overall, these iterations demonstrate a comprehensive exploration of techniques and models, reflecting a deep understanding of the challenges posed by imbalanced data and a systematic approach to finding effective solutions.

---

## The rationale to map the business requirements to the Data Visualisations and ML tasks

### Business Requirement 1

- **Business Requirement 1**: The client aims to visually differentiate lesions. The model should be capable of reaching an accuracy of at least 70%.

**Rationale**: 
- We will display the "mean" and "standard deviation" images for each lesion type.
- We will display the difference between each classes average image.
- We will display an image montage for each lesion class.

### Business Requirement 2

- **Business Requirement 2**: The model should provide a confidence level for each prediction.

**Rationale**: 
- This requirement links directly to the model's output interpretation. Most classification models can offer a probability or confidence score along with the class prediction. 
- Understanding the distribution of these confidence scores can be achieved via data visualisations like histograms or density plots.

### Business Requirement 3

- **Business Requirement 3**: If a skin lesion is predicted as malignant with high confidence, the system should recommend immediate medical consultation.

**Rationale**: 
- This requirement involves the application of a decision rule on the model's output. It doesn't directly correspond to a specific ML task or data visualisation,
- its effectiveness can be validated through techniques like Precision-Recall, which offer a trade-off visualisation between recall (sensitivity) and precision for different threshold settings.

### Business Requirement 4

- **Business Requirement 4**: The project will deliver a web application where users can upload a skin lesion image, and the system will provide a diagnosis, a confidence level of the prediction.

**Rationale**: 
- While not a direct ML task, this requirement relates to deploying the model within a user-accessible application.
- Displaying the model's results, including the diagnosis, associated confidence level.
- Show the data visualisations in a user-friendly interface.

### Business Requirement 5

- **Business Requirement 5**: The AI model's insights should assist healthcare professionals in making informed decisions about the treatment process.

**Rationale**: 
- This requirement pertains to the application of the ML model's results in a real-world healthcare context. 
- Clear and comprehensible data visualisations are crucial for this requirement.
- Show the confidence levels, for each lesion type that can aid healthcare professionals.

### Business Requirement 6

- **Business Requirement 6**: The model's performance will be evaluated using balanced performance metrics such as F1 Score aiming for scores above 0.7.

**Rationale**: 
- These performance metrics are integral to assess the model's performance in a balanced manner, especially in the case of imbalanced datasets. 
- Precision-Recall  provide visualisations to comprehend the performance of the model at various threshold settings. 
- The F1 Score is a harmonic mean of precision and recall, providing a singular metric that balances both these important measures.

### Business Requirement 7

- **Business Requirement 7**: The client is interested to have a study to visually differentiate between lesions.

**Rationale**: 
- We will display an image montage for all the types of lesions, 
- We will display the "mean" and "standard deviation" images and display the difference between an average of each lesion type.

---

## ML Business Case

- We want an ML model to predict the lesion class, based on historical image data. It is a supervised model, a 7-class, single-label, classification model.
- Our ideal outcome is to provide the medical team and patients a faster and more reliable diagnostic for lesion detection.
- The model success metrics are
- - Accuracy of 65% or above on the test set.
- The model output is defined as a flag, indicating what type of lesion and the associated probability of being part of that class. The user will take a picture of the lesion and upload the picture to the App. The prediction is made on the fly (not in batches).
- Heuristics: The current diagnostic needs an experienced staff and detailed inspection to distinguish the type of lesion and that has a 70% accuracy. Further inspections and processes are needed to get the exact type of lesion. On top of that, some specific hospital facilities need more trained staff and expertise and are typically understaffed. There is also lokng waits to see a GP for a lesion check.
- The training data to fit the model come from the International Skin Imaging Collaboration (ISIC). This dataset contains about 10+ thousand images. We have extracted the images from Kaggle.
Train data - target: type class; features: all images

- What is the business objective requiring a ML solution?

The business objective is to develop an AI-powered application that assists in the early identification of skin cancer. The AI model will differentiate between malignant and benign skin lesions based on images, thereby providing a rapid diagnosis and recommending further medical consultation when necessary.

- Can traditional data analysis be used?

Traditional data analysis can be used to some extent, such as finding correlations or identifying patterns. However, given the complexity of image analysis and the need for predictive capabilities, machine learning, specifically deep learning methods, would likely be more efficient and effective.

- Does the customer need a dashboard or an API endpoint?

The customer needs a web application where users can upload an image of a skin lesion and receive an immediate prediction. This would likely be implemented through a combination of a user-friendly dashboard that interact with the underlying machine learning model.

- What does success look like?

Success would be a functional application that can accurately differentiate between benign and malignant skin lesions, provide confidence levels for its predictions, and suggest further medical consultation when necessary. The solution should be reliable enough to assist healthcare professionals in their decision-making process, leading to earlier and more effective treatment of skin cancer.

- Can you break down the project into epics and user stories?

Yes epics and user stories for this project are shown above.

- Ethical or privacy concerns?

Yes, this project does have ethical and privacy concerns. It's crucial to ensure that all skin lesion images are anonymised and that user data is securely stored and handled. Consent should be obtained before using any personal data, and the app should comply with relevant data protection regulations. Also, the solution should not replace professional medical advice; it's meant to assist in the initial screening process.

- What level of prediction performance is needed?

Given the serious implications of false negatives (missing a malignant lesion) and false positives (misidentifying a benign lesion as malignant), a high level of prediction performance is needed. The model should strive for high sensitivity (minimising false negatives) at ideally over 90% and specificity (minimising false positives) above 85%.

- What are the project input and intended outputs?

The input for the project will be images of skin lesions. The intended output will be a diagnosis (benign or malignant), a confidence level associated with the prediction, and in the case of a high-confidence malignant prediction, a recommendation for immediate medical consultation.

- Does the data suggest a particular model?

Given the nature of the data (images) and the task (classification), a Convolutional Neural Network (CNN) is a suitable model as it has proven to be effective for image classification tasks. Depending on the results of initial experiments, more advanced models or ensemble methods might be considered.

- How will the customer benefit?

The customer will benefit from a faster and more efficient initial screening process for skin cancer. The application can provide immediate feedback, allowing for potential early detection and treatment. It should also reduce the workload for healthcare professionals by serving as a first step in the diagnosis process.

### Possible Heuristics

Image Preprocessing: Images may need to be resized or normalised to ensure consistency across the dataset. This can involve scaling the images to have the same width and height and normalising the pixel values.

Class Imbalance: If the dataset has many more examples of one class (e.g., benign) than another (e.g., malignant), it may be necessary to balance the classes using techniques like oversampling the minority class, undersampling the majority class, or using a combination of both (SMOTE).

Splitting the Data: A standard heuristic in machine learning is to split the available data into training, validation, and test sets. A common split is 70% for training, 10% for validation, and 10% for testing.

Model Selection: Start with simpler models and progress to more complex ones if necessary. This heuristic helps to prevent overfitting and reduces computational resources.

Confidence Thresholds: Establish a threshold for determining when the model's prediction confidence is high enough to recommend immediate medical consultation. This can be determined by assessing the model's performance at various threshold levels on a validation set.

Error Analysis: If the model's performance is not satisfactory, perform error analysis to understand the types of mistakes the model is making. This can often provide insights into what can be improved.

Model Interpretability: Even though more complex models like deep learning might provide better performance, simpler models, or the use of interpretability techniques might be preferred in this case, given the importance of the healthcare professionals being able to understand and trust the model's predictions.

Data Privacy: Given the sensitive nature of medical images, ensure all data is handled in compliance with relevant privacy laws and regulations. Images should be anonymised, and any personally identifiable information should be removed.

---

## Dashboard Design

- List all dashboard pages and their content, either blocks of information or widgets, like buttons, checkboxes, images, or any other item that your dashboard library supports.

- Later, during the project development, you may revisit your dashboard plan to update a given feature (for example, at the beginning of the project you were confident you would use a given plot to display an insight but subsequently you used another plot type).

### Pages

**Project Summary** - This page provides an overall summary of the project. Showing the outline of the system, information about the data set, the business requirments and a link to this read me.

- Project Description: Brief text block summarizing the project, the goals, and the proposed AI solution.
- System Outline: Visual schematic or flow diagram of the overall system architecture.
- Data Set Information: Summary statistics, metadata, and other relevant details about the data set being used.
- Business Requirements: A concise list of the business requirements guiding the project.
- Read Me Link: A clickable button or hyperlink leading to a more detailed project documentation.

**Lesion Exploration / Clusters Page** -  This page offers visualisations of the data clusters identified in the skin lesion images. It can display a scatter plot or other visualization of the clusters and give details about the characteristics associated with each cluster.

- Data Clusters Visualization: A scatter plot or similar visualization showing the different data clusters in the skin lesion images.
- Cluster Details: Text blocks or interactive tooltips providing additional information about each cluster, such as common characteristics or typical images.
- Cluster Navigation: Dropdown menu or similar widget to select and explore individual clusters in detail.

**Upload page and Results** - This page allows users (either individuals or healthcare professionals) to upload a skin lesion image for diagnosis. After the image is processed, the diagnosis, confidence level, associated cluster will be displayed and an explanation of what the cluster means.

- Image Upload Interface: A form or button allowing users to upload skin lesion images for analysis.
- Diagnosis Display: A text block or similar element showing the diagnosis once the image has been processed.
- Confidence Level Display: A bar chart, gauge, or similar visualization showing the confidence level associated with the diagnosis.
- Cluster Association: A text block indicating which cluster the image belongs to and what it implies.

**Project Hypothesis and Validation** - This page contains information related to the initial assumptions or hypotheses made about the project, and how those hypotheses are being validated or have been validated.

- Hypotheses List: A bulleted or numbered list of the initial project hypotheses.
- Validation Approach: A text block explaining how each hypothesis is or will be validated.
- Validation Results: Charts, graphs, or other visualizations showing the results of the validation process.
- Conclusions and Next Steps: A text block summarizing the validation results and outlining recommended next steps.

**ML Performance Page** - This page presents detailed metrics on the performance of the AI model. It might include confusion matrix, ROC curve, precision-recall curve, F1 score, AUC-ROC value, etc. It can also include a comparison of performance metrics over different versions of the model.

- Model Metrics: Various visualizations (like confusion matrix, ROC curve, precision-recall curve) showing detailed performance metrics for the AI model.
- Model Comparison: A table or graph comparing the performance of different model versions or configurations.
- Model Interpretation: A text block explaining what these metrics mean in terms of model performance and quality.

**Feedback / Reporting Page** - This page allows users to provide feedback on the system's performance and report any issues. It could be a form where users can rate the system, leave comments, or report false positives/negatives. It will also show future features for the system.

- User Feedback Form: A form where users can rate their experience with the system, leave comments, or report false positives or negatives.
- Reporting Mechanism: A separate form or section where users can report technical issues or other problems with the system.
- Future Features: A text block outlining upcoming features or improvements planned for the system.

---

## Bugs

- You will need to mention unfixed bugs and why they were not fixed. This section should include shortcomings of the frameworks or technologies used. Although time can be a significant variable to consider, paucity of time and difficulty understanding implementation is not a valid reason to leave bugs unfixed.

---

## CRISP DM Process
TODO

---

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

---

## Data Analysis and Machine Learning Libraries

- Here you should list the libraries you used in the project and provide an example(s) of how you used these libraries.

---

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

---

## Acknowledgements

- Thank the people that provided support through this project.
