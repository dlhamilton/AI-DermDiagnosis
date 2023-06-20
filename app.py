import streamlit as st
from app_pages.multipage import MultiPage

# load pages scripts
from app_pages.page_summary import page_summary_body
from app_pages.page_lesion_exploration import page_lesion_exploration_body
from app_pages.page_lesion_classifier import page_lesion_classifier_body
from app_pages.page_project_hypothesis import page_project_hypothesis_body
from app_pages.page_ml_performance import page_ml_performance_metrics
from app_pages.page_reporting import page_reporting

app = MultiPage(app_name="AI-Derm")  # Create an instance of the app

# Add your app pages here using .add_page()
app.add_page("Quick Project Summary", page_summary_body)
app.add_page("Lesion Exploration", page_lesion_exploration_body)
app.add_page("Lesion Classifier ", page_lesion_classifier_body)
app.add_page("Project Hypothesis", page_project_hypothesis_body)
app.add_page("ML Performance Metrics", page_ml_performance_metrics)
app.add_page("Feedback / Reporting", page_reporting)

app.run()  # Run the app