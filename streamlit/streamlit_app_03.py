import streamlit as st

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["God Particle Info", "Higgs Boson Detector"])

# Page 1: God Particle Info
if page == "God Particle Info":
    st.title("What is the God Particle: The Higgs Boson?")
    st.write("")
    st.write("""
    The Higgs Boson is a tiny particle that gives other particles their mass. 
    Often called the 'God Particle,' it helps explain why things in our universe have weight and isn't just floating around as energy.
    """)
    st.write("")

    video_url = "https://youtu.be/Ltx2QhRBdHk"
    st.video(video_url)

    st.header("Further Reading and Articles")
    st.write("""
    Here are some articles and resources to dive deeper into the fascinating world of the Higgs Boson:

    - [Higgs boson](https://en.wikipedia.org/wiki/Higgs_boson)
    - [The Higgs boson: a landmark discovery - ATLAS Experiment](https://atlas.cern/Discover/Physics/Higgs#:~:text=The%20discovery%20of%20the%20Higgs%20boson%20opened%20a%20whole%20new,and%20no%20strong%20force%20interaction.)
    - [The Higgs boson, ten years after its discovery](https://home.cern/news/press-release/physics/higgs-boson-ten-years-after-its-discovery)
    - [How did we discover the Higgs boson?](https://home.cern/science/physics/higgs-boson/how)
    """)

# Page 2: Higgs Boson Detector
elif page == "Higgs Boson Detector":
    import pandas as pd
    import joblib
    data = pd.read_csv("E:\higgs-boson-detection\data\stratified_dataset_cleaned.csv")
    st.title("Higgs Boson Detector: Predict the Unknown!")

    st.write("")
    st.write("""
    Welcome to the Higgs Boson Detector!

    Here, you can input event features from particle collisions and predict whether the event represents a **Signal (Higgs Boson)** or a **Background** event using a trained machine learning model.
    """)

    st.header("Enter Particle Collision Data")

    # Sliders for inputs
    m_bb = st.slider(
        "m_bb - Invariant mass of two b-tagged jets",
        float(data['m_bb'].min()),
        float(data['m_bb'].max()),
    )

    m_wwbb = st.slider(
        "m_wwbb - Combined mass of W boson and two b-tagged jets",
        float(data['m_wwbb'].min()),
        float(data['m_wwbb'].max()),
    )

    m_jjj = st.slider(
        "m_jjj - Mass of three jets",
        float(data['m_jjj'].min()),
        float(data['m_jjj'].max()),
    )

    missing_energy_magnitude = st.slider(
        "Missing Energy Magnitude",
        float(data['missing_energy_magnitude'].min()),
        float(data['missing_energy_magnitude'].max()),
    )

    jet1_pt = st.slider(
        "jet1_pt - Transverse momentum of leading jet",
        float(data['jet1_pt'].min()),
        float(data['jet1_pt'].max()),
    )

    st.write("---")

    # Model selection
    st.header("Select Machine Learning Model")
    model_choice = st.selectbox("Choose a model:", ("Logistic Regression", "Random Forest", "XGBoost", "Support Vector Machine"))

    # Load models
    model_paths = {
        "Logistic Regression": "E:/higgs-boson-detection/data/logistic_regression_model.pkl",
        "Random Forest": "E:/higgs-boson-detection/data/random_forest_model.pkl",
        "XGBoost": "E:/higgs-boson-detection/data/xgboost_model.pkl",
        "Support Vector Machine": "E:/higgs-boson-detection/data/svm_model.pkl"
    }

    st.write("---")

    if st.button("Predict"):
        # Load the selected model
        with open(model_paths[model_choice], "rb") as file:
            model = joblib.load(file)

        # Prepare input data
        random_row = data.sample(1).drop(columns=['Label']).values.flatten()

        # Update the 5 features with user inputs
        feature_names = list(data.drop(columns=['Label']).columns)

        # Mapping feature index
        feature_index_mapping = {
            'm_bb': feature_names.index('m_bb'),
            'm_wwbb': feature_names.index('m_wwbb'),
            'm_jjj': feature_names.index('m_jjj'),
            'missing_energy_magnitude': feature_names.index('missing_energy_magnitude'),
            'jet1_pt': feature_names.index('jet1_pt')
        }

        random_row[feature_index_mapping['m_bb']] = m_bb
        random_row[feature_index_mapping['m_wwbb']] = m_wwbb
        random_row[feature_index_mapping['m_jjj']] = m_jjj
        random_row[feature_index_mapping['missing_energy_magnitude']] = missing_energy_magnitude
        random_row[feature_index_mapping['jet1_pt']] = jet1_pt

        # Reshape for prediction
        final_input = random_row.reshape(1, -1)

        # Make prediction
        prediction = model.predict(final_input)[0]

        # Display result
        if prediction == 1:
            st.success(f"Prediction using **{model_choice}**: 🚀 **Signal (Higgs Boson)** Detected!")
        else:
            st.error(f"Prediction using **{model_choice}**: ❌ **Background** Event Detected.")

        import shap
        import matplotlib.pyplot as plt
        import streamlit.components.v1 as components


        # Function to render SHAP plots in Streamlit
        def st_shap(plot, height=None):
            shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
            components.html(shap_html, height=height)


        # Background data
        X_background = data.drop(columns=['Label'])
        features = X_background.columns.to_list()

        label_names = ["Background", "Signal"]
        #st.header("Model Explanations")

        if model_choice == "Random Forest" or model_choice == "XGBoost":
            st.subheader("Heading")
            from sklearn.tree import plot_tree
            model = joblib.load('E:/higgs-boson-detection/data/tree_basic.pkl')
            individual_tree = model.estimators_[0]
            fig, ax = plt.subplots(figsize=(16, 10))
            plot_tree(individual_tree, feature_names=features, class_names=label_names, filled=True, ax=ax)
            st.pyplot(fig)

        elif model_choice == "Logistic Regression":
            st.subheader("Feature Impact (SHAP Analysis)")

            # Create a Linear Explainer for Logistic Regression
            explainer = shap.LinearExplainer(model, X_background, feature_perturbation="interventional")

            # Calculate SHAP values
            shap_values = explainer.shap_values(final_input)

            # Create SHAP Explanation object for waterfall
            explanation = shap.Explanation(
                values=shap_values[0],
                base_values=explainer.expected_value,
                data=final_input.flatten(),
                feature_names=X_background.columns
            )

            # Force plot
            st_shap(shap.force_plot(
                explainer.expected_value,
                shap_values[0],
                feature_names=X_background.columns
            ))

            st.subheader("Waterfall Plot (Feature Contributions)")
            fig, ax = plt.subplots()
            shap.plots.waterfall(explanation, show=False)
            st.pyplot(fig)

        elif model_choice == "Support Vector Machine":
            st.warning("SHAP is computationally expensive for SVM. Skipping explanation.")
        else:
            st.warning("SHAP not available for this model type.")



