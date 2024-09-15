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

def show_prediction_page():
    model_files = {
        'RandomForest': 'rain_prediction_pipeline_RandomForest.pkl',
        'GradientBoosting': 'rain_prediction_pipeline_GradientBoosting.pkl',
        'LogisticRegression': 'rain_prediction_pipeline_LogisticRegression.pkl',
        'KNeighbors': 'rain_prediction_pipeline_KNeighbors.pkl',
        'SVC': 'rain_prediction_pipeline_SVC.pkl',
        'DecisionTree': 'rain_prediction_pipeline_DecisionTree.pkl'
    }

    models = {name: joblib.load(file) for name, file in model_files.items()}

    # Load historical data for visualization (if available)
    historical_data = pd.read_csv("Rainfall Prediction dataset.csv")

    # Streamlit UI
    st.title('Rainfall Prediction Dashboard')

    # Sidebar for input features
    st.sidebar.header('Input Features')
    def user_input_features():
        data = {
            'Inside Temp - C': st.sidebar.number_input('Inside Temp - C', value=20),
            'High Inside Temp - C': st.sidebar.number_input('High Inside Temp - C', value=25),
            'Low Inside Temp - C': st.sidebar.number_input('Low Inside Temp - C', value=15),
            'Inside Hum - %': st.sidebar.number_input('Inside Hum - %', value=50),
            'High Inside Hum - %': st.sidebar.number_input('High Inside Hum - %', value=60),
            'Low Inside Hum - %': st.sidebar.number_input('Low Inside Hum - %', value=40),
            'Inside Dew Point - C': st.sidebar.number_input('Inside Dew Point - C', value=10),
            'Inside Heat Index - C': st.sidebar.number_input('Inside Heat Index - C', value=20),
            'Barometer - in Hg': st.sidebar.number_input('Barometer - in Hg', value=29.9),
            'High Bar - in Hg': st.sidebar.number_input('High Bar - in Hg', value=30.0),
            'Low Bar - in Hg': st.sidebar.number_input('Low Bar - in Hg', value=29.8),
            'Absolute Pressure - in Hg': st.sidebar.number_input('Absolute Pressure - in Hg', value=29.9),
            'Temp - C': st.sidebar.number_input('Temp - C', value=20),
            'High Temp - C': st.sidebar.number_input('High Temp - C', value=25),
            'Low Temp - C': st.sidebar.number_input('Low Temp - C', value=15),
            'Hum - %': st.sidebar.number_input('Hum - %', value=50),
            'High Hum - %': st.sidebar.number_input('High Hum - %', value=60),
            'Low Hum - %': st.sidebar.number_input('Low Hum - %', value=40),
            'Dew Point - C': st.sidebar.number_input('Dew Point - C', value=10),
            'High Dew Point - C': st.sidebar.number_input('High Dew Point - C', value=15),
            'Low Dew Point - C': st.sidebar.number_input('Low Dew Point - C', value=5),
            'Wet Bulb - C': st.sidebar.number_input('Wet Bulb - C', value=15),
            'High Wet Bulb - C': st.sidebar.number_input('High Wet Bulb - C', value=20),
            'Low Wet Bulb - C': st.sidebar.number_input('Low Wet Bulb - C', value=10),
            'Avg Wind Speed - km/h': st.sidebar.number_input('Avg Wind Speed - km/h', value=10),
            'High Wind Speed - km/h': st.sidebar.number_input('High Wind Speed - km/h', value=15),
            'Prevailing Wind Direction': st.sidebar.selectbox('Prevailing Wind Direction', ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']),
            'High Wind Direction': st.sidebar.selectbox('High Wind Direction', ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']),
            'Wind Chill - C': st.sidebar.number_input('Wind Chill - C', value=5),
            'Low Wind Chill - C': st.sidebar.number_input('Low Wind Chill - C', value=0),
            'Heat Index - C': st.sidebar.number_input('Heat Index - C', value=25),
            'High Heat Index - C': st.sidebar.number_input('High Heat Index - C', value=30),
            'THW Index - C': st.sidebar.number_input('THW Index - C', value=25),
            'High THW Index - C': st.sidebar.number_input('High THW Index - C', value=30),
            'Low THW Index - C': st.sidebar.number_input('Low THW Index - C', value=20),
            'Wind Run - km': st.sidebar.number_input('Wind Run - km', value=100),
            'THSW Index - C': st.sidebar.number_input('THSW Index - C', value=30),
            'High THSW Index - C': st.sidebar.number_input('High THSW Index - C', value=35),
            'Low THSW Index - C': st.sidebar.number_input('Low THSW Index - C', value=25),
            'Rain - mm': st.sidebar.number_input('Rain - mm', value=0),
            'High Rain Rate - mm/h': st.sidebar.number_input('High Rain Rate - mm/h', value=1),
            'Solar Rad - W/m^2': st.sidebar.number_input('Solar Rad - W/m^2', value=200),
            'High Solar Rad - W/m^2': st.sidebar.number_input('High Solar Rad - W/m^2', value=250),
            'UV Index': st.sidebar.number_input('UV Index', value=2),
            'High UV Index': st.sidebar.number_input('High UV Index', value=3),
            'UV Dose - MEDs': st.sidebar.number_input('UV Dose - MEDs', value=0.5),
            'Heating Degree Days': st.sidebar.number_input('Heating Degree Days', value=5),
            'Cooling Degree Days': st.sidebar.number_input('Cooling Degree Days', value=10),
        }
        return pd.DataFrame(data, index=[0])

    input_data = user_input_features()
    input_data['Rain_Lag_1'] = 0
    input_data['Rain_Lag_2'] = 0

    # Predict and display results for all models
    if st.button('Predict'):
        results = {}
        for model_name, model in models.items():
            try:
                prediction = model.predict(input_data)
                prediction_text = 'Rain' if prediction[0] else 'No Rain'
                results[model_name] = prediction_text
            except Exception as e:
                results[model_name] = f"Error: {e}"
        
        st.header('Prediction Results')
        for model_name, prediction in results.items():
            st.markdown(f"**{model_name}:** <span style='color: {'red' if prediction == 'Rain' else 'green'}'>{prediction}</span>", unsafe_allow_html=True)

    # Show a sample of the historical data
    st.header('Historical Data Sample')
    st.write(historical_data.head())

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
