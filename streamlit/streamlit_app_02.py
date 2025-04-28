import streamlit as st
import base64
import pandas as pd
import plotly.graph_objects as go
import plotly.figure_factory as ff
import ast

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
    # Load the model metrics from the CSV file
    df_metrics = pd.read_csv('data/model_metrics.csv', index_col=0)

    # List of model names
    model_names = df_metrics.columns.tolist()

    st.header("Model Comparison")

    # 1. Baseline Model (Dropdown)
    baseline_model = st.selectbox("Select Baseline Model:", options=model_names,
                                  index=model_names.index('Logistic Regression'))

    # 2. Novel Model (Radio Buttons)
    novel_model_options = [model for model in model_names if model != baseline_model]
    novel_model = st.radio("Select Novel Model to Compare:", options=novel_model_options)

    st.markdown(
        f"<p style='font-size: 18px; text-align: center;'>"
        f"<span style='color: rgba(0, 123, 255, 1);'>{baseline_model}</span>    vs    "
        f"<span style='color: rgba(158, 42, 47, 1);'>{novel_model}</span></p>",
        unsafe_allow_html=True
    )

    # Plot Radar Chart comparing the two models
    metrics = df_metrics.index.tolist()

    baseline_values = df_metrics[baseline_model].values
    novel_values = df_metrics[novel_model].values

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=baseline_values,
        theta=metrics,
        fill='toself',
        name=baseline_model,
        marker=dict(size=8, color='rgba(0, 0, 255, 1)')
    ))

    fig.add_trace(go.Scatterpolar(
        r=novel_values,
        theta=metrics,
        fill='toself',
        name=novel_model,
        marker=dict(size=8, color='rgba(200, 0, 0, 0.5)')
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0.6, 0.8]),
        ),
        showlegend=True,
        template='plotly_dark',
        title=f"Model Performance Metrics"
    )

    st.plotly_chart(fig, use_container_width=True)

    # ---- Load ROC Curve Data from CSV ----
    roc_df = pd.read_csv('data/roc_curves.csv')

    # Convert FPR and TPR columns from string to list
    roc_df['FPR'] = roc_df['FPR'].apply(ast.literal_eval)
    roc_df['TPR'] = roc_df['TPR'].apply(ast.literal_eval)

    # Create a Plotly figure for ROC curves
    fig_roc = go.Figure()

    # Filter for baseline and novel model data
    baseline_df = roc_df[roc_df['Model'] == baseline_model]
    novel_df = roc_df[roc_df['Model'] == novel_model]

    # Plot baseline model in blue
    fpr_baseline = baseline_df['FPR'].values[0]
    tpr_baseline = baseline_df['TPR'].values[0]
    fig_roc.add_trace(go.Scatter(
        x=fpr_baseline, y=tpr_baseline,
        mode='lines+markers',
        name=baseline_model,
        line=dict(color='rgba(0, 0, 255, 1)', width=3),
    ))

    # Plot novel model in red
    fpr_novel = novel_df['FPR'].values[0]
    tpr_novel = novel_df['TPR'].values[0]
    fig_roc.add_trace(go.Scatter(
        x=fpr_novel, y=tpr_novel,
        mode='lines+markers',
        name=novel_model,
        line=dict(color='rgba(200, 0, 0, 1)', width=3),
    ))

    # Add the diagonal line for random guessing (FPR = TPR)
    fig_roc.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        line=dict(color='gray', dash='dash'),
        showlegend=False
    ))

    # Update layout of the plot
    fig_roc.update_layout(
        template='plotly_dark',
        title="ROC Curve Comparison",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        width=800,
        height=600,
        showlegend=True
    )

    # Show the ROC curve plot in Streamlit
    st.plotly_chart(fig_roc, use_container_width=True)

    # ---- CONFUSION MATRICES ----
    conf_matrix_df = pd.read_csv('data/confusion_matrices.csv')

    # Get confusion matrices
    baseline_cm = conf_matrix_df[conf_matrix_df['Model'] == baseline_model][['TN', 'FP', 'FN', 'TP']].values[0]
    novel_cm = conf_matrix_df[conf_matrix_df['Model'] == novel_model][['TN', 'FP', 'FN', 'TP']].values[0]

    # Reshape into 2x2
    baseline_matrix = [[baseline_cm[0], baseline_cm[1]],
                       [baseline_cm[2], baseline_cm[3]]]

    novel_matrix = [[novel_cm[0], novel_cm[1]],
                    [novel_cm[2], novel_cm[3]]]

    # Create Plotly confusion matrices
    fig_baseline = ff.create_annotated_heatmap(
        z=baseline_matrix,
        x=['Predicted: 0', 'Predicted: 1'],
        y=['Actual: 0', 'Actual: 1'],
        colorscale='Blues',
        showscale=True
    )

    fig_novel = ff.create_annotated_heatmap(
        z=novel_matrix,
        x=['Predicted: 0', 'Predicted: 1'],
        y=['Actual: 0', 'Actual: 1'],
        colorscale='Reds',
        showscale=True
    )

    # Update layout for dark mode
    fig_baseline.update_layout(
        template="plotly_dark",
        title=f"{baseline_model}",
        title_font=dict(weight='normal'),
        title_font_size=12,
        title_x=0.4  # Centers the title
    )

    fig_novel.update_layout(
        template="plotly_dark",
        title=f"{novel_model}",
        title_font=dict(weight='normal'),
        title_font_size=12,
        title_x=0.4  # Centers the title
    )

    st.markdown("##### Confusion Matrix")
    # Display plots side by side
    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(fig_baseline, use_container_width=True)

    with col2:
        st.plotly_chart(fig_novel, use_container_width=True)


elif page == "Feedback":
    st.title("Feedback")
    feedback_text = st.text_area("Share your thoughts about the app:")
    if st.button("Submit Feedback"):
        st.success("Thank you for your feedback!")
