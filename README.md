Streamlit link: https://higgs-boson-detection-01.streamlit.app/

# Higgs Boson Classification

This project tackles the binary classification of Higgs boson events using simulated data from the ATLAS experiment. The primary objective is to distinguish rare signal events (presence of Higgs boson) from background noise using rigorous feature selection techniques and powerful machine learning models.

## 📊 Dataset
- Source: Simulated ATLAS dataset
- Size: 10+ million rows, reduced to 30,000 using stratified sampling
- Features: 28 numerical features (kinematic & high-level physics features), 1 binary target (`Label`: 1 = signal, 0 = background)

## 🔬 Feature Engineering
- **Statistical tests**: T-test, ANOVA
- **Information theory**: Mutual Information
- Selected features enhanced class separability and reduced dimensionality.

## ⚙️ Models Used
- Logistic Regression (Baseline)
- Random Forest
- Support Vector Machine (SVM)
- XGBoost (Best performer with >70% accuracy)

## 🧪 Training Strategy
- 80-20 Train-Test Split
- 5-fold Cross-Validation
- Evaluation Metrics: Accuracy, Precision, Recall, F1-score

## 📈 Results
- XGBoost outperformed all other models with superior metrics across the board.
- Ensemble models proved effective in handling high-dimensional physics data.
