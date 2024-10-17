import pandas as pd
import numpy as np
import streamlit as st
import xgboost as xgb
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split
import joblib
import openpyxl

# Title for the app
st.title('Wildfire Characteristics Prediction App')

# File upload section for users to upload their own dataset
st.subheader('Upload Dataset')
uploaded_file = st.file_uploader("Upload your Excel dataset", type=["xlsx"])

# Load the dataset either from the uploaded file or use the default one
if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file)
        st.write("User data loaded successfully")
    except Exception as e:
        st.error(f"Error loading the file: {e}")
        st.stop()

# Display the first few rows of the dataset
st.write(df.head())

# One-hot encoding for categorical columns
categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

# Identify target columns based on dataset characteristics
# Assuming target columns will contain the words 'Size', 'Duration', 'Cost', and 'Occurrence'
target_columns = [col for col in df.columns if 'Size' in col or 'Duration' in col or 'Cost' in col or 'Occurrence' in col]

# Convert 'Fire Occurrence' (if it exists) to binary values: Yes=1, No=0
if 'Fire Occurrence' in df.columns:
    df['Fire Occurrence'] = df['Fire Occurrence'].map({'Yes': 1, 'No': 0})

# Handle missing values
if df.isnull().sum().sum() > 0:
    if st.checkbox("Replace missing values with column mean?", value=True):
        df.fillna(df.mean(), inplace=True)
        st.write("Missing values handled")

# Identifying and removing outliers using IQR
if st.checkbox("Remove outliers?"):
    for column in df.select_dtypes(include=[np.number]).columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    st.write("Outliers removed")

# Check the columns after one-hot encoding
st.write("Columns after one-hot encoding:", df.columns.tolist())

# Identify regression and classification targets
target_regression = [col for col in target_columns if 'Occurrence' not in col]
target_classification = next((col for col in target_columns if 'Occurrence' in col), None)

# Define features based on remaining DataFrame columns
features = [col for col in df.columns if col not in target_columns]

X = df[features]
y_regression = df[target_regression] if target_regression else None
y_classification = df[target_classification] if target_classification else None

# Split data into training and testing sets if regression targets exist
if y_regression is not None and y_classification is not None:
    X_train, X_test, y_train_reg, y_test_reg, y_train_clf, y_test_clf = train_test_split(
        X, y_regression, y_classification, test_size=0.2, random_state=30)

    # XGBoost Regressor for Fire Size, Duration, and Suppression Cost
    xgb_regressor = xgb.XGBRegressor(objective='reg:squarederror', random_state=30)
    xgb_regressor.fit(X_train, y_train_reg)

    # XGBoost Classifier for Fire Occurrence
    xgb_classifier = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=30)
    xgb_classifier.fit(X_train, y_train_clf)

    # App Title and Inputs
    st.subheader('Input Features for Prediction')

    # Feature Inputs (using sliders and dropdowns)
    numeric_inputs = {col: st.slider(col, min_value=float(df[col].min()), max_value=float(df[col].max()), value=float(df[col].mean())) for col in features if df[col].dtype in [np.int64, np.float64]}
    
    categorical_inputs = {col: st.selectbox(col, options=df[col].unique()) for col in features if df[col].dtype == 'object'}

    # Combine user inputs into a feature array
    user_input = np.array([[numeric_inputs[col] if col in numeric_inputs else 0 for col in features]])

    # Predict Fire Characteristics (using pre-trained models)
    if st.button('Predict Wildfire Characteristics'):
        fire_size_pred = xgb_regressor.predict(user_input)
        fire_duration_pred = xgb_regressor.predict(user_input)
        suppression_cost_pred = xgb_regressor.predict(user_input)
        fire_occurrence_pred = xgb_classifier.predict(user_input)

        # Display predictions
        st.subheader('Prediction Results:')
        if target_regression:
            st.write(f"Predicted Fire Size (hectares): {fire_size_pred[0][0]:.2f}")  # Accessing the first element in the array
            st.write(f"Predicted Fire Duration (hours): {fire_duration_pred[0][0]:.2f}")  # Accessing the first element in the array
            st.write(f"Predicted Suppression Cost ($): {suppression_cost_pred[0][0]:.2f}")  # Accessing the first element in the array)

        if target_classification:
            fire_occurrence_result = 'Yes' if fire_occurrence_pred[0] == 1 else 'No'
            st.write(f"Fire Occurrence: {fire_occurrence_result}")

