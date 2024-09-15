import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report

def show_about_page():
    # Load the model results
    model_results = joblib.load('rain_prediction_model_results.pkl')

    # Title for the Streamlit app
    st.title("Rainfall Prediction Model Performance")

    # Description for the app
    st.write("""
        This app displays the performance of various models used for predicting rainfall in the next 2 hours.
        You can view the accuracy, classification report, confusion matrix, and ROC curve for all models below.
    """)

    # Select a model to view performance
    model_name = st.selectbox("Select Model", list(model_results.keys()))

    # Retrieve model data
    model_data = model_results[model_name]

    # Display accuracy
    st.write(f"## Model: {model_name}")
    st.write(f"**Accuracy:** {model_data['accuracy']:.4f}")

    # Display classification report
    st.write("**Classification Report:**")
    class_report = model_data['classification_report']
    df = pd.DataFrame(class_report).T
    st.dataframe(df, width=800)

    # Display confusion matrix
    st.write("**Confusion Matrix:**")
    conf_matrix = model_data['confusion_matrix']
    conf_matrix_df = pd.DataFrame(conf_matrix, index=['Actual False', 'Actual True'], columns=['Predicted False', 'Predicted True'])
    st.write(conf_matrix_df)

    # Plot ROC curve for the current model
    st.write("**ROC Curve:**")
    roc_data = model_data['roc_data']
    plt.figure(figsize=(8, 6))
    for i in range(len(roc_data['fpr'])):
        plt.plot(roc_data['fpr'][i], roc_data['tpr'][i], label=f'Fold {i+1} (AUC = {roc_data["roc_auc"][i]:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc="lower right")
    st.pyplot(plt)
    plt.clf()

show_about_page()

# def main():
#     st.sidebar.title("Navigation")
#     page = st.sidebar.radio("Go to", ["Prediction", "About"])

#     if page == "Prediction":
#         show_prediction_page()
#     elif page == "About":
#         show_about_page()

# if __name__ == "__main__":
#     main()
