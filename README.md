# Wildfire Characteristics Prediction App

## Overview
The Wildfire Characteristics Prediction App is a Streamlit-based web application designed to predict wildfire characteristics such as fire size, duration, suppression cost, and occurrence based on various environmental and geographical factors. This app leverages machine learning models, specifically XGBoost, to provide accurate predictions that can help in wildfire management and prevention efforts.

## Features
- **User-Friendly Interface**: Easily upload your own datasets in Excel format and interact with the application through a clean and intuitive interface.
- **Data Processing**: The app performs one-hot encoding for categorical variables, handles missing values, and removes outliers based on the Interquartile Range (IQR) method.
- **Model Training and Prediction**: The application uses XGBoost for regression and classification tasks to predict fire characteristics based on user inputs.
- **Prediction Results**: Provides predicted values for fire size, duration, suppression cost, and fire occurrence.

## Installation
To run the app locally, follow these steps:

1. Clone the repository or download the code files.
   ```bash
   git clone <repository-url>
   cd <directory-name>
