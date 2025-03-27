# Cancer_Prediction_GUI

## Overview

This project is a Breast Cancer Prediction GUI built using machine learning and Streamlit. It is designed to assist medical professionals in predicting whether a diagnosis is necessary based on patient data. The model is trained using logistic regression and deployed using GitHub and Streamlit Cloud services.

## Key Features

- Machine Learning Model: Logistic Regression with an accuracy of 97.02%.
- User-Friendly Interface: A graphical user interface (GUI) built with Streamlit.
- Data Preprocessing: Standardized features for improved model performance.
- Efficient Deployment: Model and scaler converted to binary files using pickle to avoid redundant training.
- Visualization: Graphical representation of the dataset and results.

## Dataset Details

- Source: The dataset contains 32 columns and 569 observations.
- Preprocessing Steps:
  - Checked for missing values and duplicates (none found).
  - Removed unnecessary columns (ID and unnamed columns).
  - Separated independent and dependent variables.
  - Split data into training and testing sets using train_test_split.
  - Standardized variables using StandardScaler.

## Model Development

- Data Inspection: Verified dataset structure and cleaned unnecessary columns.
- Splitting Data: Used train_test_split to create training and testing datasets.
- Feature Scaling: Applied StandardScaler to normalize the feature values.
- Training the Model: Used LogisticRegression from sklearn.linear_model.
- Model Evaluation: Achieved 97.02% accuracy.
- Serialization: Saved the trained model and scaler as binary files using pickle for later use.

## GUI Development

- Built with: Streamlit (streamlit package in Python).
- Layout:
  - Sidebar Panel: User inputs for prediction.
  - Main Panel:
   - Two columns: One for graphical representation of data (4x larger) and another for displaying predictions.
- Functionality:
 - Loads the trained model and scaler from binary files.

[Live Demo](https://cancerpredictiongui-6msaybcztmbzfiqz9qh6ms.streamlit.app/
)


## Deployment
- Hosted on Streamlit Cloud for easy access.
- Version control and collaboration managed through GitHub.


