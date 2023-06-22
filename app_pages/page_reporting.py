import streamlit as st

def page_reporting():
    """
    Displays the feedback and reporting page of the application.

    The page includes a user feedback form for rating the experience and leaving comments, a reporting mechanism
    for describing technical issues or problems with the system, and a section for showcasing future features
    that are currently being worked on.

    Returns:
        None
    """
    
    # Title of the app
    st.title("Feedback / Reporting Page")
    
    # Subheader
    st.subheader("We value your feedback and are constantly working to improve the system.")
    
    # User Feedback Form
    st.markdown("## User Feedback Form")
    st.markdown("Please rate your experience and let us know your thoughts on how we can improve.")
    
    rating = st.selectbox("Rate your experience", ["Select", "Excellent", "Good", "Average", "Poor", "Very Poor"])
    comment = st.text_area("Leave your comment")
    
    if st.button("Submit Feedback"):
        st.success("Thank you for your feedback!")
        
    # Reporting Mechanism
    st.markdown("## Reporting Mechanism")
    st.markdown("Report technical issues or problems with the system below.")
    
    issue = st.text_area("Describe the issue")
    
    if st.button("Report Issue"):
        st.success("Your issue has been reported. Thank you!")
    
    # Future Features
    st.markdown("## Future Features")
    st.markdown("""
        We are excited to share some of the features we are currently working on:
        
        - Improved accuracy in predictions.
        - Support for additional skin types and conditions.
        - Enhanced user interface for easier navigation.
        - Option to consult a dermatologist through the platform.
        - Integration with health applications to track skin health over time.
    """)