import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Correct way: create actual tab objects
tab1, tab2 = st.tabs(['IDA', 'EDA'])
























# Use each tab properly
with tab1:
    st.title("Initial Data Analysis (IDA)")

    # Section 1: Data Collection
    st.header("1. Data Collection")

    st.markdown("""
    - **Dataset Source**: [Higgs UCI Dataset on Kaggle](https://www.kaggle.com/datasets/erikbiswas/higgs-uci-dataset)
    - **Description**: This is a classification problem to distinguish between a signal process that produces Higgs bosons and a background process that does not.
    """)

    # Section 2: Data Overview
    st.header("2. Data Overview")

    # Abstract / Dataset Information
    st.subheader("Dataset Information")
    st.markdown("""
    - The data has been produced using **Monte Carlo simulations**.
    - The first **21 features** (columns 2–22) are **kinematic properties** measured by particle detectors in the accelerator.
    - The last **seven features** are **high-level features** derived from the first 21 features by physicists to help discriminate between signal and background.
    - There is interest in using deep learning methods to avoid manually engineering features.
    - Benchmark results using Bayesian Decision Trees and 5-layer Neural Networks are presented in the original paper.
    - Note: In the original dataset, the last 500,000 examples were used as a test set.
    """)

    # Feature Description Table
    st.subheader("Feature Terminologies")

    # Prepare feature descriptions
    feature_descriptions = {
        "Label": "Target Variable: 1 = signal (Higgs boson detected), 0 = background (no Higgs boson).",
        "lepton_pT": "Transverse momentum of the lepton (electron or muon) — momentum perpendicular to the beam.",
        "lepton_eta": "Pseudorapidity of the lepton — a measure related to the angle of the particle’s emission (higher = more forward direction).",
        "lepton_phi": "Azimuthal angle of the lepton — angle around the beam axis (like longitude).",
        "missing_energy_magnitude": "Magnitude of missing transverse energy — energy imbalance suggesting an invisible particle (like a neutrino).",
        "missing_energy_phi": "Azimuthal angle of missing transverse energy — direction where energy is “missing.”",
        "jet1_pt": "Transverse momentum of the first jet (jet = a spray of particles from quarks or gluons).",
        "jet1_eta": "Pseudorapidity of the first jet.",
        "jet1_phi": "Azimuthal angle of the first jet.",
        "jet1_btag": "b-tagging score of the first jet — likelihood that this jet comes from a b-quark (important in Higgs decays).",
        "jet2_pt": "Transverse momentum of the second jet.",
        "jet2_eta": "Pseudorapidity of the second jet.",
        "jet2_phi": "Azimuthal angle of the second jet.",
        "jet2_btag": "b-tagging score of the second jet.",
        "jet3_pt": "Transverse momentum of the third jet.",
        "jet3_eta": "Pseudorapidity of the third jet.",
        "jet3_phi": "Azimuthal angle of the third jet.",
        "jet3_btag": "b-tagging score of the third jet.",
        "jet4_pt": "Transverse momentum of the fourth jet.",
        "jet4_eta": "Pseudorapidity of the fourth jet.",
        "jet4_phi": "Azimuthal angle of the fourth jet.",
        "jet4_btag": "b-tagging score of the fourth jet.",
        "m_jj": "Invariant mass of the two leading jets — total mass if two jets were one particle.",
        "m_jjj": "Invariant mass of three leading jets — important for reconstructing possible particle decays.",
        "m_lv": "Invariant mass of lepton + missing energy system — helps find W boson decays.",
        "m_jlv": "Invariant mass of leading jet + lepton + missing energy system — reconstructs possible top quark or Higgs events.",
        "m_bb": "Invariant mass of two b-tagged jets — crucial because Higgs often decays into two b quarks (H → bb̄).",
        "m_wbb": "Invariant mass of W boson candidate (lepton + missing energy) + b-tagged jets.",
        "m_wwbb": "Invariant mass of two W bosons + two b-jets system — entire final state; useful to identify complex decays like Higgs-top events."
    }

    # Convert to dataframe
    features_df = pd.DataFrame(list(feature_descriptions.items()), columns=["Feature Name", "Meaning"])

    # Display the feature table
    st.dataframe(features_df, use_container_width=True)

    # Display First 10 Rows of Dataset
    # Load the stratified 30k dataset

    file_path = 'data/stratified_dataset.csv'

    df = pd.read_csv(file_path)

    # Section: First 10 Rows
    st.subheader("Data table")
    st.dataframe(df.head(10))



    # Section: Data Reduction
    st.header("3. Data Reduction (Stratified Sampling)")

    # Full Dataset (10M rows) proportions (hardcoded)
    full_labels = ['Signal (1)', 'Background (0)']
    full_values = [0.52992, 0.47008]

    # Sampled Dataset (30k rows) proportions (calculated from df)
    file_path = 'data/stratified_dataset.csv'
    df = pd.read_csv(file_path)

    sampled_counts = df['Label'].value_counts(normalize=True).sort_index()
    sampled_labels = ['Background (0)', 'Signal (1)']
    sampled_values = [sampled_counts[0.0], sampled_counts[1.0]]

    # --- Create Combined Subplots Figure ---
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type':'domain'}, {'type':'domain'}]],
        subplot_titles=["Before Sampling: 10M rows", "After Sampling: 30k rows"]
    )

    # Add first pie chart (Before Sampling)
    fig.add_trace(go.Pie(
        labels=full_labels,
        values=full_values,
        name="Full Dataset",
        hole=0.4
    ), 1, 1)

    # Add second pie chart (After Sampling)
    fig.add_trace(go.Pie(
        labels=sampled_labels,
        values=sampled_values,
        name="Sampled Dataset",
        hole=0.4
    ), 1, 2)

    # Update Layout
    fig.update_layout(
        height=500,
        width=900,
        title_text="Comparison of Label Distribution Before and After Stratified Sampling",
        showlegend=True
    )

    # --- Display Combined Plotly Figure ---
    st.plotly_chart(fig, use_container_width=False)

    # --- Add a compact note ---
    st.write("Due to computational resource constraints, the original 10 million row dataset was reduced to 30,000 rows using stratified sampling, while preserving the original class distribution.")

    # Section 4: Data Quality Assessment
    st.header("4. Data Quality Assessment")


    # -------------------------------
    # 4.1 Missingness Check
    st.subheader("4.1 Missingness Check")

    st.markdown("""
    A heatmap showing missing values is displayed below.
    """)

    # Missingness Heatmap
    fig_missing, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap='viridis', ax=ax)
    ax.set_title('Missing Values Heatmap')

    st.pyplot(fig_missing)

    # Conclusion for Missingness
    st.write("No missing values detected in the dataset.")

    # -------------------------------
    # 4.2 Duplicate Check
    st.subheader("4.2 Duplicate Check")

    # Simulate duplicates for visualization (optional if needed)
    df_with_duplicates = pd.concat([df, df.sample(300, random_state=42)], ignore_index=True)

    # --- Count duplicates BEFORE removing
    num_duplicates_before = df_with_duplicates.duplicated().sum()

    # Heatmap showing duplicates only (Before Cleaning)
    fig_before_dup, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(df_with_duplicates.duplicated(keep=False).to_frame().T, cbar=False, cmap='coolwarm', ax=ax)
    ax.set_title(f'Duplicates Before Cleaning: {num_duplicates_before}')

    # Display only BEFORE cleaning heatmap
    st.pyplot(fig_before_dup)

    st.write(f"Removed {num_duplicates_before} duplicate rows to ensure data consistency.")

    # -------------------------------
    # Now: Real Duplicate Removal on actual df (NOT simulated one)
    num_real_duplicates = df.duplicated().sum()

    # Remove real duplicates
    df_cleaned = df.drop_duplicates()

    # Save the cleaned dataset
    output_path = 'data/stratified_dataset_cleaned.csv'
    df_cleaned.to_csv(output_path, index=False)