# 41072025_Churning_Customers

# Churn Prediction App

**Overview**
This repository contains code for a Churn Prediction App that utilizes a machine learning model to predict customer churn. The model is trained on a dataset containing customer information, and the app provides a user-friendly interface for predicting whether a customer is likely to churn or stay.

**Features**
- Predicts customer churn based on input data.
- Streamlit app for easy user interaction.
- Trained machine learning model with a focus on accuracy.
- Utilizes a StandardScaler for data scaling.

**Technologies Used**
- Python
- Jupyter Notebook
- TensorFlow
- Scikit-learn
- Streamlit

**Files**
- `Godwin_Gbeve_Assignment_3.ipynb`: Jupyter Notebook with data processing, exploration, and model training.
- `model.pkl`: Pickle file containing the trained machine learning model.
- `scaler.pkl`: Pickle file containing the StandardScaler used for input data scaling.
- `requirements.txt`: List of required Python packages for running the app.
- `app.py`: Streamlit app code for user interaction and predictions.

**How to Use**
1. Install the required packages by running `pip install -r requirements.txt`.
2. Run the Streamlit app using the command `streamlit run app.py`.
3. Input customer information (Total Charges, Monthly Charges, Tenure, Contract, Payment Method).
4. Receive the churn prediction result.

**Deployment**
The app is deployed using Streamlit, offering a simple and intuitive interface for users. The trained model (`model.pkl`) and the StandardScaler (`scaler.pkl`) are loaded within the app for making accurate predictions.

**Acknowledgments**
This project is part of an assignment, covering data cleaning, exploration, and the development of a machine learning model for predicting customer churn.

**Instructions**
Feel free to explore the Jupyter Notebook (`Godwin_Gbeve_Assignment_3.ipynb`) for a detailed walkthrough of the data processing and model training steps.

**Future Enhancements**
Possible future enhancements include:
- Improving the model's interpretability.
- Adding more features for better prediction accuracy.
- Enhancing the app's visual appearance and user experience.

For any questions or issues, please contact the repository owner.
