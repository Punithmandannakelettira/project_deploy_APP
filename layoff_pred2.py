import streamlit as st
import pickle
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_option_menu import option_menu  # Import the option menu library
import os





# Load the saved models and components for future forecast
with open('trend_arima_model.pkl', 'rb') as f:
    model_trend_full = pickle.load(f)

with open('residual_arima_model.pkl', 'rb') as f:
    model_resid_full = pickle.load(f)

with open('seasonal_component.pkl', 'rb') as f:
    seasonal_full = pickle.load(f)

# Load the trained Random Forest model for attrition
with open('random_forest_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)

with open('label_encoders.pkl', 'rb') as f:
    encoders = pickle.load(f)
    
if "forecast_data" not in st.session_state:
    st.session_state.forecast_data = None
if "attrition_prediction" not in st.session_state:
    st.session_state.attrition_prediction = None
if "bulk_data_summary" not in st.session_state:
    st.session_state.bulk_data_summary = None
    

def chatbot_response(user_input):
    # Basic chatbot logic (extendable with ML/NLP models)
    responses = {
        "forecast": "The future forecasting predicts layoff trends using STL ARIMA. Please check the 'Future Forecasting' menu.",
        "attrition": "Attrition prediction uses the Random Forest model to assess the likelihood of employee attrition.",
        "upload": "You can upload CSV files in the 'Upload & Analyze' menu for custom analysis.",
        "summary": "This Help Desk summarizes insights and findings from all the menus for your convenience.",
    }
    for key, response in responses.items():
        if key in user_input.lower():
            return response
    return "I'm sorry, I didn't understand that. Please try asking about 'forecast', 'attrition', 'upload', or 'summary'."

def main():
    st.title("Role of Artificial Intelligence in Evolving the Workforce in the IT Sector")

    # Add the option menu to the sidebar
    with st.sidebar:
        menu = option_menu(
            "Menu",
            ["Future Forecasting", "Attrition Prediction", "Bulk Attrition Prediction","Help Desk"],
            icons=["graph-up", "people", "upload", "question-circle"],
            menu_icon="robot",
            default_index=0,
            styles={
                "container": {"padding": "5px"},
                "icon": {"color": "#aad8e8"},
                "nav-link": {"font-size": "16px"},
            }
        )

    if menu == "Future Forecasting":
        st.title("Future Forecasting of No. of Layoffs")
        st.write("This section forecasts future trends using STL ARIMA.")

        # User input for number of future steps
        future_steps = st.number_input(
            "Enter the number of future steps to forecast:", min_value=1, max_value=12, value=12
        )

        # Generate forecast
        trend_forecast_full = model_trend_full.forecast(steps=future_steps)
        residual_forecast_full = model_resid_full.forecast(steps=future_steps)

        future_forecast_log = trend_forecast_full + seasonal_full[:future_steps] + residual_forecast_full
        future_forecast_values = np.expm1(future_forecast_log)
        future_forecast_values = np.clip(future_forecast_values, a_min=0, a_max=None).astype(int)


        # Display forecast
        data = pd.read_csv('real_data.csv')

        data['Date'] = pd.to_datetime(data['Date'])  # Convert 'Date' to datetime
        data.set_index('Date', inplace=True)  # Set 'Date' as the index
        future_dates = pd.date_range(data.index[-1], periods=future_steps + 1, freq='W')[1:]
        forecast_df = pd.DataFrame({
            'Date': future_dates,
            'Forecasted_Laid_Off_Count': future_forecast_values
        })
        st.session_state.forecast_data={"data":forecast_df,"plot":plt}
        st.write("Forecasted Values:")
        st.write(forecast_df.drop(columns=["Date"]))

        # Plot forecast
        # Create a Matplotlib figure and axis
        fig, ax = plt.subplots(figsize=(12, 6))


        ax.plot(data.index, data['Laid_Off_Count'], label='Observed', color='blue')


        ax.plot(future_dates, future_forecast_values, label='Forecast (Future)', color='red', linestyle='-')


        ax.set_title('STL-ARIMA Forecast')
        ax.set_xlabel('Date')
        ax.set_ylabel('Laid Off Count')

# Add legend and grid
        ax.legend()
        ax.grid()

# Display the chart
        st.write("Chart")
        st.pyplot(fig)
        st.session_state.forecast_data={"data":forecast_df,"plot":fig}
    elif menu == "Attrition Prediction":
        st.title("Attrition Prediction")
        st.header("Feature Importance for Attrition Prediction")
        feature_importance_df = pd.DataFrame({
            'Feature': [
                'MonthlyIncome', 'Age', 'YearsAtCompany', 'DistanceFromHome',
                'NumCompaniesWorked', 'JobSatisfaction', 'EnvironmentSatisfaction',
                'EducationField', 'MaritalStatus', 'Education', 'WorkLifeBalance', 'Department'
            ],
            'Importance': [
                0.196761, 0.13211, 0.111539, 0.106697, 0.076737, 0.064058,
                0.059736, 0.058995, 0.056235, 0.050554, 0.050467, 0.036103
            ]
        })

        # Display the table in Streamlit
        st.write("Feature Importance Table:")
        st.write(feature_importance_df)

        # Plot the feature importance using Matplotlib
        fig ,ax =plt.subplots(figsize=(8, 4))
        
        ax.bar(feature_importance_df['Feature'], feature_importance_df['Importance'], color='skyblue')
        ax.set_title('Feature Importance for Attrition Prediction')
        ax.set_ylabel('Importance')
        ax.set_xlabel('Features')
        ax.tick_params(axis='x', rotation=90)

        # Display the graph in Streamlit
        st.write("Feature Importance Bar Chart:")
        st.pyplot(fig)
        
        st.header("Attrition Prediction Using Random Forest Model")
        marital_status_options = ["Single", "Married"]
        education_field_options = ['Life Sciences', 'Other', 'Medical', 'Marketing',
                                   'Technical Degree', 'Human Resources']

        # Input fields for numerical columns
        monthly_income = st.number_input("Monthly Income", min_value=0, step=1000)
        age = st.number_input("Age", min_value=18, step=1)
        years_at_company = st.number_input("Years at Company", min_value=0, step=1)
        distance_from_home = st.number_input("Distance From Home (km)", min_value=0, step=1)
        num_companies_worked = st.number_input("Number of Companies Worked", min_value=0, step=1)
        job_satisfaction = st.slider("Job Satisfaction (1-4)", min_value=1, max_value=4)
        environment_satisfaction = st.slider("Environment Satisfaction (1-4)", min_value=1, max_value=4)
        education = st.slider("Education (1-5)", min_value=1, max_value=5)
        work_life_balance = st.slider("Work Life Balance (1-4)", min_value=1, max_value=4)

        # Input fields for categorical columns
        education_field = st.selectbox("Education Field", education_field_options)
        marital_status = st.selectbox("Marital Status", marital_status_options)

        # Preprocess the inputs
        input_data = {
            "MonthlyIncome": monthly_income,
            "Age": age,
            "YearsAtCompany": years_at_company,
            "DistanceFromHome": distance_from_home,
            "NumCompaniesWorked": num_companies_worked,
            "JobSatisfaction": job_satisfaction,
            "EnvironmentSatisfaction": environment_satisfaction,
            "EducationField": education_field,
            "MaritalStatus": marital_status,
            "Education": education,
            "WorkLifeBalance": work_life_balance
        }

        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])

        # Encode categorical columns
        for col in ["EducationField", "MaritalStatus"]:
            encoder = encoders.get(col)
            if encoder:
                input_df[col] = encoder.transform(input_df[col])

        # Make prediction
        if st.button("Predict Attrition"):
            prediction = rf_model.predict(input_df)[0]
            prediction_proba = rf_model.predict_proba(input_df)[0]

            if prediction == 1:
                st.error(f"The employee is likely to attrite with a probability of {prediction_proba[1]*100:.2f}%({prediction_proba[1]:.2f}).")
            else:
                st.success(f"The employee is unlikely to attrite with a probability of {prediction_proba[0]*100:.2f}%({prediction_proba[1]:.2f}).")
        st.session_state.attrition_prediction = {
              "data": input_df if 'input_df' in locals() else None,
               "plot": fig if 'fig' in locals() else None,
               "pred": prediction if 'prediction' in locals() else None,
               "pred_proba": prediction_proba if 'prediction_proba' in locals() else None
              }
    elif menu == "Bulk Attrition Prediction":
        st.title("Bulk Attrition prediction")
        st.write("Upload a CSV file to view its data and basic statistics and predict attrition")

        # File uploader
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            # Read the uploaded file
            df = pd.read_csv(uploaded_file)
            # Display the data
            st.write("Uploaded Data:")
            st.dataframe(df)
       
            st.write("Basic Statistics:")
            st.write(df.describe())
            # Plot 1: Count of Attrition (existing plot)
            sns.set_theme(style="whitegrid")  # Optional: set a Seaborn theme
            fig1, ax1 = plt.subplots(figsize=(8, 6))  # Create a figure and axis for the first plot
            sns.countplot(data=df, x='Attrition', palette='viridis', ax=ax1)  # Pass `ax` to the plot
            ax1.set_title("Attrition Count")  # Set title for the plot
            st.write("Attrition Count Plot:")
            st.pyplot(fig1)
           
            # Calculate Attrition Percentage
            if 'Attrition' in df.columns:
                  total_employees = df.shape[0]
                  attrite_count = df[df['Attrition'] == 'Yes'].shape[0]
                  attrition_percentage = int((attrite_count / total_employees) * 100)
        
                  st.markdown(
                      f"""
                       <div style="background-color: #4c62ec; padding: 10px; border-radius: 5px; border: 1px solid #ddd;">
                       <strong>Percentage of Employees that would attrite:</strong> 
                       <span style="color: #0E0E0E; font-weight: bold;">{attrition_percentage}%</span>
                      </div>
                      """,
                      unsafe_allow_html=True)
            else:
                  st.error("Column 'Attrition' not found in the dataset.")
            attrition_yes = df[df['Attrition'] == 'Yes']
            st.dataframe(attrition_yes)
            if 'Department' in attrition_yes.columns:  # Ensure 'Department' column exists
                fig2, ax2 = plt.subplots(figsize=(8, 6))  # Create a figure and axis for the second plot
                sns.countplot(data=attrition_yes, x='Department', palette='pastel', ax=ax2)
                ax2.set_title("Department Distribution for Attrited Employees")
                ax2.set_xlabel("Department")
                ax2.set_ylabel("Count")
                ax2.tick_params(axis='x', rotation=45)  # Rotate x-axis labels for better readability
                st.write("Department Distribution Plot:")
                st.pyplot(fig2)
            st.session_state.bulk_data_summary={"data1":df if 'df' in locals() else None,
                                                "plotatcount":fig1 if 'fig1' in locals() else None,
                                                "value":attrition_percentage if 'attrition_percentage' in locals() else None,
                                                "data2":attrition_yes if 'attrition_yes' in locals() else None,
                                                "plotdepartcount":fig2 if 'fig2' in locals() else None}  
    elif menu == "Help Desk":
        st.title("Help Desk")
        st.write("Interact with our chatbot or see a summary of findings and insights.")
    
        # Section to display stored session state values dynamically
        def display_session_data(session_key, data_key, description):
            if session_key in st.session_state and st.session_state[session_key] is not None:
                st.write(f"### {description}")
                for key, value in st.session_state[session_key].items():
                    if isinstance(value, pd.DataFrame):
                        st.write(f"**{key}** (Data):")
                        st.dataframe(value)
                    elif isinstance(value, plt.Figure):
                        st.write(f"**{key}** (Graph):")
                        st.pyplot(value)
                    else:
                        # Apply custom style to the values
                        st.markdown(f"""
                            <div style="background-color: #4c62ec; color: white; padding: 10px; border-radius: 5px; margin-top: 5px;">
                                <strong>{key}:</strong> {value}
                            </div>
                        """, unsafe_allow_html=True)
            else:
                st.warning(f"No data available for {description}.")
    
        
    
        # Chatbot Section
        st.subheader("Chatbot")
        user_input = st.text_input(
            "Ask a question about forecasting, attrition, or analysis:")
    
        if user_input:
            # Sample chatbot logic for interpreting user input
            def chatbot_response(input_text):
                if "forecast" in input_text.lower():
                    return "Showing forecast analysis data."
                elif "attrition" in input_text.lower():
                    return "Displaying attrition prediction results."
                elif "bulk" in input_text.lower():
                    return "Providing bulk attrition analysis insights."
                else:
                    return "I'm here to assist with your questions about forecasting, attrition, or analysis."
    
            response = chatbot_response(user_input)
            st.write("Chatbot Response:", response)
    
            # Context-aware responses based on input
            if "forecast" in user_input.lower():
                if st.session_state.forecast_data is not None:
                    st.write("Forecast Data:")
                    st.dataframe(st.session_state.forecast_data.get(
                        "data", "No data available"))
                    st.write("Forecast Plot:")
                    st.pyplot(st.session_state.forecast_data.get(
                        "plot", "No plot available"))
                else:
                    st.warning("Forecast data is not available.")
            elif "attrition" in user_input.lower():
                if st.session_state.attrition_prediction is not None:
                    st.write("Attrition Prediction Data:")
                    display_session_data(
                        "attrition_prediction",
                        ["data", "plot", "pred", "pred_proba"],
                        "Attrition Prediction Results"
                    )
                else:
                    st.warning("Attrition prediction data is not available.")
            elif "bulk" in user_input.lower():
                if st.session_state.bulk_data_summary is not None:
                    st.write("Bulk Data Summary:")
                    display_session_data(
                        "bulk_data_summary",
                        ["data1", "plotatcount", "value", "data2", "plotdepartcount"],
                        "Bulk Data Analysis"
                    )
                else:
                    st.warning("Bulk data summary is not available.")
            else:
                st.write(
                    "Feel free to ask specific questions like 'forecast', 'attrition', or 'bulk analysis'.")
    
                                                                               
                
                        

        # Summary section
        st.subheader("Summary of Findings and Insights")
        st.write("- **Future Forecasting:** The model predicts layoff trends using STL-ARIMA with seasonal decomposition.")
        st.write("- **Attrition Prediction:** The Random Forest model helps identify employees at risk of leaving based on input features.")
        st.write("- **Upload & Analyze:** Analyze your custom datasets by uploading a CSV file for automated processing.")     
                
            
    
           


if __name__ == "__main__":
    main()
