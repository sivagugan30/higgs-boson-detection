import streamlit as st
import base64

st.set_page_config(page_title="Higgs Boson Detection", layout="wide")

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


# Set background image or color dynamically based on the page
def set_background(image_path):
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{base64_image}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
# Home Page
if page == "Home":

    set_background(image_path='data/pic2.jpg')

    st.title("Higgs Boson (God Particle) Detection")

# Instructions Page
elif page == "Instructions":
    st.write("""
    ### About the Project:
    This Project is your ultimate companion for exploring the Higgs Boson particle, offering insights through interactive visualizations, scientific explanations, and predictive models.

    ### Sections:
    
    - **What is Higgs Boson?:** Discover the scientific significance of the Higgs Boson and its role in particle physics.
    
    - **Predict the Particle:** Run predictions on particle data to determine the presence of the Higgs Boson using a trained model.
    
    - **Documentation:** Dive into the technical details, data sources, and methodologies used in the model.
    
    - **Feedback:** Share your thoughts and suggestions to help improve the Project experience.
    
    - **Home:** Overview of the Higgs Boson Project and what you can explore.

    Navigate the Project using the sidebar, and immerse yourself in the exciting world of particle physics!
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
