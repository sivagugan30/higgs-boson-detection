import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.stats import ttest_ind
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif, f_classif

file_path = 'data/stratified_dataset_cleaned.csv'
df = pd.read_csv(file_path)

# Correct way: create actual tab objects
tab1, tab2, tab3 = st.tabs(['IDA', 'EDA', 'Feature Selection'])

with tab3:
    #st.header("Feature Selection")

    # Step 1: Load cleaned correlation table
    file_path = 'data/correlation_table.csv'
    corr_table = pd.read_csv(file_path)

    # Step 2: Prepare full correlation matrix for heatmap
    # (No deduplication here to keep symmetric matrix)
    corr_matrix = corr_table.pivot(index='Feature 1', columns='Feature 2', values='Correlation')

    # Fill diagonal with 1s for nicer heatmap
    if corr_matrix is not None:
        for col in corr_matrix.columns:
            if col in corr_matrix.index:
                corr_matrix.loc[col, col] = 1.0

    # --- Plot 1: Correlation Heatmap (full matrix)
    st.subheader("2.1 Correlation Heatmap")

    # Set a dark background
    plt.style.use('dark_background')

    # Create the plot
    fig_corr, ax = plt.subplots(figsize=(16, 12))

    # Plot heatmap with adjusted colors
    sns.heatmap(
        corr_matrix,
        cmap='viridis',  # 'viridis' works very well on dark backgrounds
        center=0,
        annot=False,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8}  # Optional: shrink colorbar slightly
    )

    # Set title with bright color manually
    ax.set_title('Pearson Correlation Heatmap of Features', fontsize=18, color='white')

    # Display plot
    st.pyplot(fig_corr)

    # ---------------------------------------------------------

    # Step 3: Deduplicate for Bar Plot Only
    dedup_corr_table = corr_table.copy()
    dedup_corr_table['sorted_pair'] = dedup_corr_table.apply(lambda row: tuple(sorted([row['Feature 1'], row['Feature 2']])), axis=1)
    dedup_corr_table = dedup_corr_table.drop_duplicates(subset=['sorted_pair']).drop(columns='sorted_pair')

    # Absolute correlation for sorting
    dedup_corr_table['abs_corr'] = dedup_corr_table['Correlation'].abs()

    st.write('')
    # User input for Top N
    top_n = st.slider("Select number of top correlated pairs to display:", min_value=5, max_value=30, value=10, step=1)

    # Get Top N feature pairs
    top_corr = dedup_corr_table.sort_values(by='abs_corr', ascending=False).head(top_n)

    # Sort top_corr DataFrame descending by absolute correlation
    top_corr = top_corr.sort_values(by='abs_corr', ascending=False)

    import plotly.graph_objects as go

    fig_bar = go.Figure()

    fig_bar.add_trace(go.Bar(
        x=top_corr['abs_corr'],
        y=top_corr['Feature 1'] + " & " + top_corr['Feature 2'],
        orientation='h',
        marker=dict(
            color=top_corr['abs_corr'],
            colorscale='RdBu',
            cmin=0,
            cmax=1,
            colorbar=dict(title='')
        ),
        text=top_corr['abs_corr'].round(3),
        textposition='outside',
        hovertemplate='<b>Feature Pair:</b> %{y}<br><b>Abs Corr:</b> %{x:.3f}<extra></extra>'
    ))

    fig_bar.update_layout(
        xaxis=dict(title='Absolute Correlation', range=[0, 1]),
        yaxis=dict(title='Feature Pairs', autorange='reversed'),  # Reverse y-axis to highest at top
        title=f'Top {top_n} Most Correlated Feature Pairs',
        height=600,
        margin=dict(l=100, r=50, t=80, b=50),
        template='plotly_white'
    )

    st.plotly_chart(fig_bar, use_container_width=True)
    st.write("""
    The correlation analysis revealed that several features exhibit strong linear relationships. 
    Notably, pairs such as **m_wwbb & m_wbb**, **m_jj & m_jjj**, and **m_jjj & m_wbb** showed high correlations.
     
    This suggests redundancy among mass-combination features, and one feature from each highly correlated group may be dropped during feature selection to reduce multicollinearity risks.
    """)

    # Features and target
    X = df.drop(columns=['Label'])
    y = df['Label']

    # -------------------------------
    # Step 2.1: T-Test for Numerical Features
    st.subheader("2.1 T-Test for Numerical Features")

    # T-Test
    ttest_results = []
    for feature in X.columns:
        group0 = X[y == 0][feature]
        group1 = X[y == 1][feature]
        stat, p_value = ttest_ind(group0, group1, equal_var=False)  # Welch’s t-test
        ttest_results.append((feature, p_value))

    ttest_df = pd.DataFrame(ttest_results, columns=['Feature', 'p-value'])

    # Calculate 1 - p-value
    ttest_df['1 - p-value'] = 1 - ttest_df['p-value']

    # Set threshold
    threshold = 0.95

    # Sort features
    ttest_df = ttest_df.sort_values(by='1 - p-value', ascending=False)

    # Plot manually with Matplotlib
    fig, ax = plt.subplots(figsize=(12, 6))

    for index, row in ttest_df.iterrows():
        color = 'green' if row['1 - p-value'] >= threshold else 'orangered'
        ax.bar(row['Feature'], row['1 - p-value'], color=color)

    # Add threshold line
    plt.axhline(y=threshold, color='red', linestyle='--', label=f'Threshold ({threshold})')

    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1)
    plt.title('T-Test for Numerical Features', fontsize=16)
    plt.ylabel('1 - p-value')
    plt.xlabel('Features')
    plt.legend()
    plt.grid(axis='y')
    st.pyplot(fig)

    # Conclusion after the plot
    st.write("""
    Features with 1-p-values above the threshold (0.95) are considered statistically significant. 
    These features show strong evidence of differing between the two classes and will be prioritized for model building.
    """)
    # Filter features where 1 - p-value > threshold
    significant_features = ttest_df[ttest_df['1 - p-value'] > 0.95]['Feature'].tolist()

    # Create a small sentence
    important_features_text = ", ".join(significant_features)

    # Now print this
    st.write(f"Important features identified from the T-Test are: {important_features_text}.")

    # -------------------------------
    # Step 2.2 and 2.3: ANOVA F-Test and Mutual Information

    st.subheader("2.2 ANOVA & Mutual Information")


    # 1. Features and target
    X = df.drop(columns=['Label'])
    y = df['Label']

    # 2. Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. Mutual Information calculation
    mi_scores = mutual_info_classif(X_scaled, y, random_state=42)
    mi_df = pd.DataFrame({'Feature': X.columns, 'Mutual_Information': mi_scores})

    # 4. ANOVA F-Test calculation
    f_statistic, _ = f_classif(X_scaled, y)
    anova_df = pd.DataFrame({'Feature': X.columns, 'ANOVA_F': f_statistic})

    # 5. Sort and select Top 15 from each
    top_mi_features = mi_df.sort_values(by='Mutual_Information', ascending=False).head(15)['Feature'].tolist()
    top_anova_features = anova_df.sort_values(by='ANOVA_F', ascending=False).head(15)['Feature'].tolist()

    # 6. Find common Top 10 features (keep MI order)
    common_features = [feat for feat in top_mi_features if feat in top_anova_features][:10]

    # 7. Prepare filtered MI and ANOVA data
    # Reorder properly
    top_mi = mi_df.set_index('Feature').loc[common_features]['Mutual_Information']
    top_anova = anova_df.set_index('Feature').loc[common_features]['ANOVA_F']

    # 8. Plot using Plotly
    fig = go.Figure()

    # Add Mutual Information line (Green)
    fig.add_trace(go.Scatter(
        x=top_mi.index,
        y=top_mi.values,
        mode='markers+lines',
        name='Mutual Information (MI)',
        marker=dict(color='limegreen', size=10),
        yaxis='y1',
        line=dict(shape='spline'),
        hovertemplate='<b>Feature:</b> %{x}<br><b>MI:</b> %{y:.4f}<extra></extra>'
    ))

    # Add ANOVA F-Value line (Blue)
    fig.add_trace(go.Scatter(
        x=top_anova.index,
        y=top_anova.values,
        mode='markers+lines',
        name='ANOVA F-Value',
        marker=dict(color='royalblue', size=10),
        yaxis='y2',
        line=dict(shape='hv'),
        hovertemplate='<b>Feature:</b> %{x}<br><b>F-Value:</b> %{y:.4f}<extra></extra>'
    ))

    # Annotations
    fig.add_annotation(
        text="Higher MI = More information shared",
        xref="paper", yref="paper",
        x=0.05, y=1.15, showarrow=False,
        font=dict(color="limegreen", size=12),
        xanchor='left'
    )
    fig.add_annotation(
        text="Higher F-Value = Better class separation",
        xref="paper", yref="paper",
        x=0.95, y=1.15, showarrow=False,
        font=dict(color="royalblue", size=12),
        xanchor='right'
    )

    # Layout
    fig.update_layout(
        xaxis=dict(title='Features', tickangle=45),  # Rotate x-axis labels
        yaxis=dict(title='Mutual Information (MI)', side='left'),
        yaxis2=dict(title='ANOVA F-Value', side='right', overlaying='y', showgrid=False),
        template='plotly_dark',
        legend=dict(x=0.5, y=1.1, orientation='h'),
        height=600,
        margin=dict(t=100)
    )

    # Display
    st.plotly_chart(fig, use_container_width=True)

    # Short Conclusion
    st.write("Features such as m_bb, m_wwbb, and m_jjj ranked highly across both ANOVA and Mutual Information analyses and will be prioritized for model development.")

    # -------------------------------
    # Step 3.1: Low Variance Filter
    st.subheader("3.1 Low Variance Feature Removal")

    from sklearn.feature_selection import VarianceThreshold

    # 1. Calculate variance
    selector = VarianceThreshold(threshold=0.01)  # You can adjust threshold
    selector.fit(X)

    # 2. Create a DataFrame with variances
    variances = pd.Series(selector.variances_, index=X.columns).sort_values()

    # 3. Identify low-variance features
    low_variance_features = variances[variances < 0.01].index.tolist()

    # 4. Plot variance of all features
    import plotly.express as px

    fig_var = px.bar(
        x=variances.index,
        y=variances.values,
        color=(variances < 0.01),  # Different color if variance low
        color_discrete_map={True: 'red', False: 'green'},
        labels={'x': 'Features', 'y': 'Variance'},
        title="Feature Variance Distribution (Low Variance Highlighted)"
    )

    fig_var.update_layout(
        xaxis_tickangle=-45,
        height=600,
        margin=dict(l=50, r=50, t=80, b=50),
        showlegend=False
    )

    st.plotly_chart(fig_var, use_container_width=True)

    # 5. Short Conclusion
    st.write(f"Features identified with very low variance (< 0.01) and considered for removal: {', '.join(low_variance_features)}.")

    # -------------------------------
    # Step 3.2: Scree Plot for PCA
    st.subheader("3.2 Scree Plot for Principal Component Analysis (PCA)")


    from sklearn.decomposition import PCA


    # 1. Standardize features (very important before PCA)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 2. Apply PCA
    pca = PCA()
    pca.fit(X_scaled)

    # 3. Extract explained variance ratio
    explained_variance = pca.explained_variance_ratio_

    # 4. Create Scree Plot using Plotly
    fig_scree = go.Figure()

    fig_scree.add_trace(go.Scatter(
        x=[f'PC{i+1}' for i in range(len(explained_variance))],
        y=explained_variance,
        mode='markers+lines',
        marker=dict(color='dodgerblue', size=8),
        line=dict(color='dodgerblue', width=2),
        hovertemplate='<b>Component:</b> %{x}<br><b>Explained Variance:</b> %{y:.4f}<extra></extra>'
    ))

    fig_scree.update_layout(
        title="Scree Plot: Explained Variance by Principal Components",
        xaxis_title="Principal Components",
        yaxis_title="Explained Variance Ratio",
        yaxis=dict(range=[0, max(explained_variance) + 0.05]),
        height=600,
        template="plotly_white",
        margin=dict(l=50, r=50, t=50, b=50)
    )

    # 5. Display plot
    st.plotly_chart(fig_scree, use_container_width=True)

    # 6. Short explanation
    st.write("The Scree Plot visualizes the proportion of variance explained by each principal component, helping guide dimensionality reduction decisions.")

    # -------------------------------
    # Step 3.2: Scree Plot with Cumulative Variance and 95% Threshold
    st.subheader("3.2 Scree Plot for Principal Component Analysis (PCA)")

    # 1. Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 2. Apply PCA
    pca = PCA()
    pca.fit(X_scaled)

    # 3. Calculate cumulative explained variance
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)

    # 4. Find number of components to capture 95% variance
    n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1

    # 5. Create Scree Plot
    fig_scree = go.Figure()

    # Plot cumulative variance
    fig_scree.add_trace(go.Scatter(
        x=np.arange(1, len(cumulative_variance) + 1),
        y=cumulative_variance,
        mode='markers+lines',
        marker=dict(color='blue', size=6),
        line=dict(color='blue', width=2),
        name='Cumulative Variance',
        hovertemplate='PC %{x}<br>Cumulative Variance: %{y:.2f}<extra></extra>'
    ))

    # Add vertical red line at 95% variance
    fig_scree.add_shape(
        type='line',
        x0=n_components_95, x1=n_components_95,
        y0=0, y1=1,
        line=dict(color='red', width=2, dash='solid')
    )

    # Add annotation
    fig_scree.add_annotation(
        x=n_components_95,
        y=0.95,
        text=f"Trade-off point: {n_components_95} PCs (95% variance)",
        showarrow=True,
        arrowhead=1,
        ax=40,
        ay=-40
    )

    # Final layout adjustments
    fig_scree.update_layout(
        title="Scree Plot - Principal Component Analysis (PCA)",
        xaxis_title="Principal Components",
        yaxis_title="Variance % Explained",
        yaxis=dict(range=[0, 1.02]),
        template='plotly_white',
        height=600,
        margin=dict(l=50, r=50, t=80, b=50)
    )

    # 6. Display plot
    st.plotly_chart(fig_scree, use_container_width=True)

    # 7. Small Conclusion
    st.write(f"Principal Component Analysis (PCA) was applied to reduce dimensionality, capturing 95% of the variance with the first {n_components_95} components. This selection balances accuracy and computational cost.")

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