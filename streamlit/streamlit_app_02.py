import streamlit as st

st.set_page_config(page_title="Higgs Boson Detection")

# Sidebar for navigation
import streamlit as st

st.sidebar.title("Navigate")
page = st.sidebar.radio(
    "Select a Section",
    [
        "Home",
        "Instructions",
        "What is Higgs Boson?",
        "Documentation",
        "Predict the Particle",
        "Feedback"
    ]
)

# Home Page
if page == "Home":
    st.title("Higgs Boson (God Particle) Detection")
    st.write("""
    This project focuses on detecting the presence of the Higgs Boson particle using machine learning models.
    The dataset used is from the [Higgs UCI Dataset](https://www.kaggle.com/datasets/erikbiswas/higgs-uci-dataset).

    You will explore particle properties and predict if an event is a signal (Higgs) or background (noise).
    """)

# Instructions Page
elif page == "Instructions":
    st.title("Instructions")
    st.write("""
    1. Load the dataset.
    2. Select a machine learning model (like Random Forest or XGBoost).
    3. Train the model on the features.
    4. Predict whether a particle event is Higgs Boson or background.
    5. Evaluate the model's performance.

    Keep an eye on metrics like accuracy, precision, and recall!
    """)

# What is Higgs Boson Page
elif page == "What is Higgs Boson?":
    st.title("What is Higgs Boson?")
    st.write("""
    The Higgs Boson, often called the "God Particle," is a fundamental particle associated with the Higgs field.
    It explains why other particles have mass.

    Discovered in 2012 at CERN, the Higgs Boson confirmed a missing piece of the Standard Model of particle physics.
    It's not magical — but without it, the universe wouldn't have formed the way it did.
    """)

# Predict Higgs Boson Page
elif page == "Predict the Particle":
    st.title("Predict Higgs Boson")
    st.write(
        """
        (Coming soon...)  
        Here, you'll be able to upload data and predict whether the event involves a Higgs Boson or not!
        """
    )

# Documentation Page
elif page == "Documentation":
    st.title("Documentation")
    st.write(
        """
        - **Dataset**: Higgs UCI Dataset from Kaggle  
        - **Goal**: Classify particle physics events as signal (Higgs) or background.  
        - **Model**: Machine Learning based classifier.
        """
    )

elif page == "Feedback":
    st.title("Feedback")
    feedback_text = st.text_area("Share your thoughts about the app:")
    if st.button("Submit Feedback"):
        st.success("Thank you for your feedback!")