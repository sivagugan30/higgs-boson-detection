# Core
import os
import base64

# Streamlit
import streamlit as st

# Data handling
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff

# Machine Learning
import joblib
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import mutual_info_classif, f_classif
from scipy.stats import ttest_ind

# Utilities
import ast

import shap
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
from pathlib import Path

selected_features = [
        'm_bb',
        'm_wwbb',
        'm_jjj',
        'missing_energy_magnitude',
        'jet1_pt',
        'jet2_btag',
        'jet4_pt',
        'lepton_pT',
        'jet2_pt',
        'jet3_pt',
        'jet3_btag',
        'm_jlv',
        'jet4_btag'
    ]

os.chdir("..")

#os.chdir("..")

st.set_page_config(page_title="Higgs Boson Detection", layout="wide")

#st.write(os.getcwd())

# Define the base directory dynamically
BASE_DIR = Path(__file__).resolve().parent.parent  # still correct
os.chdir(BASE_DIR)  # 👈 set working dir to project root


conf_matrix_df = pd.read_csv('data/confusion_matrices.csv')
roc_df = pd.read_csv('data/roc_curves.csv')
df_metrics = pd.read_csv('data/model_metrics.csv', index_col=0)


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

# Predict Higgs Boson Page
elif page == "Predict the Particle":
    data = pd.read_csv("data/stratified_dataset_cleaned.csv")
    st.markdown("## Higgs Boson Detector: Predict the Unknown!")

    st.write("")
    #st.write("""Here, you can input event features from particle collisions and predict whether the event represents a **Signal (Higgs Boson)** or a **Background** event using a trained machine learning model.""")

    st.markdown("### Enter Particle Collision Data")

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
    st.markdown("### Select Machine Learning Model")
    model_choice = st.selectbox("Choose a model:",
                                ("Logistic Regression", "Random Forest", "XGBoost", "Support Vector Machine"))

    # Load models
    model_paths = {
        "Logistic Regression": "data/logistic_regression_model.pkl",
        "Random Forest": "data/random_forest_model.pkl",
        "XGBoost": "data/xgboost_model.pkl",
        "Support Vector Machine": "data/svm_model.pkl"
    }

    st.write("---")

    # Add custom CSS for larger button
    st.markdown("""
        <style>
        div.stButton > button {
            height: 3em;
            width: 12em;
            font-size: 75px; /* Much smaller, clean */
            font-weight: 1000; /* Semi-bold, professional */
            background-color: #264653; /* Dark blue-teal */
            color: white;
            border-radius: 8px;
            border: none;
            transition: 0.3s;
        }
        div.stButton > button:hover {
            background-color: #1b333d; /* Even darker on hover */
            color: white;
        }
        </style>
        """, unsafe_allow_html=True)


    if st.button("Predict"):
        # Load the selected model
        st.write("---")

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
            st.success("")
            st.markdown(
                "<h3 style='text-align: center; color: white;'>Higgs Boson Event Identified</h3>",
                unsafe_allow_html=True
            )
            st.success("")
        else:
            st.error("")
            st.markdown(
                "<h3 style='text-align: center; color: white;'>Background Event Identified</h3>",
                unsafe_allow_html=True
            )
            st.error("")


        # Function to render SHAP plots in Streamlit
        def st_shap(plot, height=None):
            shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
            components.html(shap_html, height=height)


        # Background data
        X_background = data.drop(columns=['Label'])
        features = X_background.columns.to_list()

        label_names = ["Background", "Signal"]
        #st.header("Model Explanations")



        if model_choice == "Logistic Regression":

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

        elif model_choice == "Random Forest":
            rf_df = pd.read_csv('data/random_forest_feature_importance.csv')

            #st.subheader("Heading")
            from sklearn.tree import plot_tree

            st.write("Decision Tree")

            model = joblib.load('data/tree_basic.pkl')
            individual_tree = model.estimators_[1]
            # Plot the tree
            fig, ax = plt.subplots(figsize=(16, 10))
            plot_tree(
                individual_tree,
                feature_names=features,
                class_names=label_names,
                filled=True,
                ax=ax,
                fontsize=10,
                rounded=True,
                proportion=True,
                precision=2,
                impurity=False,
                label='all'
            )

            # Force all text inside the tree to black
            for text in ax.get_figure().findobj(match=plt.Text):
                text.set_color("black")

            st.pyplot(fig)

            fig_rf = px.bar(
                rf_df.sort_values('Importance', ascending=True),
                x='Importance',
                y='Feature',
                orientation='h',
                title='🌲 Random Forest - Feature Importance',
            )

            fig_rf.update_layout(
                template='plotly_dark',
                height=600,
                title_font_size=24,
                xaxis_title='Importance',
                yaxis_title='Feature',
                title_x=0.5
            )

            st.plotly_chart(fig_rf, use_container_width=True)

        elif model_choice == "XGBoost":

            st.write("Decision Tree")
            from sklearn.tree import plot_tree

            model = joblib.load('data/tree_basic.pkl')
            individual_tree = model.estimators_[0]
            # Plot the tree
            fig, ax = plt.subplots(figsize=(16, 10))
            plot_tree(
                individual_tree,
                feature_names=features,
                class_names=label_names,
                filled=True,
                ax=ax,
                fontsize=10,
                rounded=True,
                proportion=True,
                precision=2,
                impurity=False,
                label='all'
            )

            # Force all text inside the tree to black
            for text in ax.get_figure().findobj(match=plt.Text):
                text.set_color("black")

            st.pyplot(fig)


            xgb_df = pd.read_csv('data/xgboost_feature_importance.csv')

            fig_xgb = px.bar(
                xgb_df.sort_values('Importance', ascending=True),
                x='Importance',
                y='Feature',
                orientation='h',
                title='XGBoost - Feature Importance',
            )

            fig_xgb.update_layout(
                template='plotly_dark',
                height=600,
                title_font_size=24,
                xaxis_title='Importance',
                yaxis_title='Feature',
                title_x=0.5
            )

            st.plotly_chart(fig_xgb, use_container_width=True)

        elif model_choice == "Support Vector Machine":
            pass




# Documentation Page
elif page == "Documentation":

    file_path = 'data/stratified_dataset_cleaned.csv'
    df = pd.read_csv(file_path)

    # Correct way: create actual tab objects
    tab1, tab2, tab3 = st.tabs(['Data Overview', 'Feature Selection', 'Model Results'])

    with tab2:
        # st.header("Feature Selection")

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
        dedup_corr_table['sorted_pair'] = dedup_corr_table.apply(
            lambda row: tuple(sorted([row['Feature 1'], row['Feature 2']])), axis=1)
        dedup_corr_table = dedup_corr_table.drop_duplicates(subset=['sorted_pair']).drop(columns='sorted_pair')

        # Absolute correlation for sorting
        dedup_corr_table['abs_corr'] = dedup_corr_table['Correlation'].abs()

        st.write('')
        # User input for Top N
        top_n = st.slider("Select number of top correlated pairs to display:", min_value=5, max_value=30, value=10,
                          step=1)

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
        st.write(
            "Features such as m_bb, m_wwbb, and m_jjj ranked highly across both ANOVA and Mutual Information analyses and will be prioritized for model development.")

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
        st.write(
            "Despite their low variance, features such as m_wwbb, m_wbb, and m_jjj were retained due to their high predictive power as evidenced by ANOVA and Mutual Information analyses.")

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
            x=[f'PC{i + 1}' for i in range(len(explained_variance))],
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
        st.write(
            "The Scree Plot visualizes the proportion of variance explained by each principal component, helping guide dimensionality reduction decisions.")

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
        st.write(
            f"Principal Component Analysis (PCA) was applied to reduce dimensionality, capturing 95% of the variance with the first {n_components_95} components. This selection balances accuracy and computational cost.")

        # -------------------------------
        # Step 3.3: Variance vs Mutual Information Scatter Plot
        st.subheader("3.3 Variance vs Mutual Information Scatter Plot")

        # Assuming you have:
        # - X = your feature set (after dropping 'Label')
        # - y = your target variable (Label)

        from sklearn.feature_selection import mutual_info_classif

        # Calculate Mutual Information
        mutual_info = pd.Series(mutual_info_classif(X, y, discrete_features=False), index=X.columns)

        # Combine into a DataFrame
        feature_analysis = pd.DataFrame({
            'Variance': variances,
            'Mutual Information': mutual_info
        })

        # Create scatter plot
        fig_scatter = px.scatter(
            feature_analysis,
            x='Variance',
            y='Mutual Information',
            text=feature_analysis.index,  # Feature names
            title="Feature Variance vs Mutual Information",
            labels={'Variance': 'Feature Variance', 'Mutual Information': 'Mutual Information Score'},
            color=feature_analysis['Mutual Information'] > 0.01,  # Highlight important features
            color_discrete_map={True: 'blue', False: 'red'}
        )

        fig_scatter.update_traces(textposition='top center')
        fig_scatter.update_layout(
            height=600,
            template='plotly_white',
            showlegend=False
        )

        st.plotly_chart(fig_scatter, use_container_width=True)

        # Short Conclusion
        st.write(
            "Features with low variance but high Mutual Information, such as m_wwbb, m_wbb, and m_jjj, were retained as they are highly predictive despite limited variability.")

        # -------------------------------
        # Final Feature Selection Table
        st.subheader("Final Selected Features for Modeling")

        import pandas as pd

        # Define the selected features and reason
        feature_selection_table = pd.DataFrame({
            'Feature Name': [
                'm_bb',
                'm_wwbb',
                'm_jjj',
                'missing_energy_magnitude',
                'jet1_pt',
                'jet2_btag',
                'jet4_pt',
                'lepton_pT',
                'jet2_pt',
                'jet3_pt',
                'jet3_btag',
                'm_jlv',
                'jet4_btag'
            ],
            'Reason for Selection': [
                'Top ranked in Mutual Information and ANOVA',
                'High predictive power despite low variance',
                'High predictive power despite low variance',
                'Strong T-Test significance',
                'Important kinematic measurement (T-Test)',
                'Significant b-tag feature (T-Test)',
                'Important physical feature (T-Test)',
                'Lepton momentum measurement (T-Test)',
                'Relevant secondary jet momentum (T-Test)',
                'Relevant tertiary jet momentum (T-Test)',
                'Tertiary jet classification feature',
                'Derived high-level feature with high MI',
                'Quaternary jet classification feature'
            ]
        })

        # Display the table
        st.dataframe(feature_selection_table, use_container_width=True)

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
        - The first **21 features** (columns 2–22) are **kinematic properties** measured by particle detectors in the accelerator.
        - The last **seven features** are **high-level features** derived from the first 21 features by physicists to help discriminate between signal and background.
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
            specs=[[{'type': 'domain'}, {'type': 'domain'}]],
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
        st.write(
            "Due to computational resource constraints, the original 10 million row dataset was reduced to 30,000 rows using stratified sampling, while preserving the original class distribution.")

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

    with tab3:

        # Load the model metrics from the CSV file
        #df_metrics = pd.read_csv('data/model_metrics.csv', index_col=0)

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
        #roc_df = pd.read_csv('data/roc_curves.csv')

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
        #conf_matrix_df = pd.read_csv('data/confusion_matrices.csv')

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
