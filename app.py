import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    r2_score,
    mean_squared_error,
    mean_absolute_error,
)
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer, LabelEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import tempfile
import zipfile
import shutil

st.set_page_config(page_title="AutoML Dashboard", layout="wide")

st.title("📊 AutoML + CNN Dashboard with Auto Data Cleaning & Model Selection")

tab1, tab2 = st.tabs(["Tabular Data (CSV)", "Image Classification (CNN)"])

# Utility: Automatic data cleaning & preprocessing
def clean_and_prepare_data(df, target, task_type):
    st.info("Automatic data cleaning & preprocessing started...")
    
    X = df.drop(target, axis=1)
    y = df[target]
    
    # Report basic info
    st.write("Dataset shape:", df.shape)
    st.write("Target column:", target)
    st.write("Target type:", y.dtype)
    
    # Detect missing values
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) > 0:
        st.warning(f"Missing values detected in columns:\n{missing}")
    else:
        st.success("No missing values detected!")
    
    # Fill missing values
    for col in X.columns:
        if X[col].isnull().any():
            if X[col].dtype == object:
                mode_val = X[col].mode()[0]
                X[col].fillna(mode_val, inplace=True)
            else:
                mean_val = X[col].mean()
                X[col].fillna(mean_val, inplace=True)
    if y.isnull().any():
        if task_type == "Classification":
            y.fillna(y.mode()[0], inplace=True)
        else:
            y.fillna(y.mean(), inplace=True)
    
    # Encode categorical features in X
    cat_cols = X.select_dtypes(include=["object", "category"]).columns
    if len(cat_cols) > 0:
        st.info(f"Encoding categorical features: {list(cat_cols)}")
        X = pd.get_dummies(X, columns=cat_cols)
    else:
        st.info("No categorical features to encode in X")
    
    # Encode target y if classification
    if task_type == "Classification":
        if y.dtype == object or y.dtype.name == 'category':
            st.info("Encoding target variable")
            y = LabelEncoder().fit_transform(y)
    
    return X, y

# Utility: Automatic algorithm selection
def auto_select_model(task_type):
    if task_type == "Classification":
        st.info("Using RandomForestClassifier by default for classification")
        return RandomForestClassifier(random_state=42)
    else:
        st.info("Using RandomForestRegressor by default for regression")
        return RandomForestRegressor(random_state=42)

# Utility: Show data analysis
def data_analysis(df):
    st.subheader("Dataset Analysis")

    st.write("Basic info:")
    buffer = []
    df.info(buf=buffer)
    st.text('\n'.join(buffer))

    st.write("Descriptive statistics:")
    st.write(df.describe(include='all'))

    st.subheader("Missing values")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) > 0:
        st.write(missing)
    else:
        st.write("No missing values.")

    st.subheader("Data types")
    st.write(df.dtypes)

    # Correlation heatmap for numeric columns
    numeric_cols = df.select_dtypes(include=np.number).columns
    if len(numeric_cols) > 1:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)
    else:
        st.info("Not enough numeric columns for correlation heatmap.")

# Tab 1: Tabular Data
with tab1:
    uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Preview of Dataset", df.head())

        data_analysis(df)

        target = st.selectbox("Select Target Column", df.columns)
        if target:
            # Ask task type automatically (classification if target categorical or less unique vals)
            if df[target].dtype == object or df[target].nunique() < 20:
                task_type = "Classification"
            else:
                task_type = "Regression"
            st.info(f"Detected task type: {task_type}")

            X, y = clean_and_prepare_data(df, target, task_type)

            # Option to scale features
            scale_data = st.checkbox("Scale features with StandardScaler", value=True)
            if scale_data:
                scaler = StandardScaler()
                X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
                st.success("Features scaled.")

            # Train/test split
            test_size = st.slider("Test set size (%)", 10, 50, 20)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size / 100, random_state=42
            )

            # Automatic model selection
            model = auto_select_model(task_type)

            # Option: Hyperparameter tuning (simple grid for RF)
            tune = st.checkbox("Perform hyperparameter tuning (Random Forest default)")
            if tune:
                param_grid = {
                    'n_estimators': [50, 100],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5]
                }
                st.info("Starting GridSearchCV tuning...")
                grid_search = GridSearchCV(model, param_grid, cv=3, n_jobs=-1)
                grid_search.fit(X_train, y_train)
                model = grid_search.best_estimator_
                st.success(f"Tuning complete. Best params: {grid_search.best_params_}")

            # Train model
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Show results
            st.subheader("Model Performance Metrics")
            if task_type == "Classification":
                acc = accuracy_score(y_test, y_pred)
                st.write(f"Accuracy: {acc:.3f}")
                st.text("Classification Report:")
                st.text(classification_report(y_test, y_pred))
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                st.pyplot(fig)
            else:
                r2 = r2_score(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                st.write(f"R^2 Score: {r2:.3f}")
                st.write(f"Mean Squared Error: {mse:.3f}")
                st.write(f"Mean Absolute Error: {mae:.3f}")

            # Save model option
            save_model = st.checkbox("Save trained model")
            if save_model:
                import joblib
                model_filename = st.text_input("Model filename (e.g., model.joblib)", "model.joblib")
                if st.button("Save Model"):
                    joblib.dump(model, model_filename)
                    st.success(f"Model saved as {model_filename}")

            # Load model option
            st.subheader("Or load a saved model")
            model_file = st.file_uploader("Upload model file", type=["joblib"])
            if model_file:
                import joblib
                loaded_model = joblib.load(model_file)
                st.success("Model loaded!")
                # You can add prediction functionality with loaded model if you want

# Tab 2: Image Classification with CNN
with tab2:
    st.subheader("Upload a ZIP of your image dataset (train/val folders inside)")
    zip_file = st.file_uploader("Upload ZIP File of Image Dataset", type=["zip"])

    if zip_file:
        temp_dir = tempfile.mkdtemp()
        zip_path = os.path.join(temp_dir, "dataset.zip")
        with open(zip_path, "wb") as f:
            f.write(zip_file.read())

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)

        st.success("Dataset extracted successfully!")

        img_height = st.slider("Image Height", 64, 224, 128)
        img_width = st.slider("Image Width", 64, 224, 128)
        batch_size = st.slider("Batch Size", 8, 64, 32)
        epochs = st.slider("Epochs", 1, 20, 5)

        train_path = os.path.join(temp_dir, "train")
        val_path = os.path.join(temp_dir, "val")

        if os.path.exists(train_path) and os.path.exists(val_path):
            datagen = ImageDataGenerator(rescale=1.0 / 255)
            train_gen = datagen.flow_from_directory(
                train_path, target_size=(img_height, img_width), batch_size=batch_size, class_mode='categorical'
            )
            val_gen = datagen.flow_from_directory(
                val_path, target_size=(img_height, img_width), batch_size=batch_size, class_mode='categorical'
            )

            # Build CNN model
            model = Sequential(
                [
                    Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
                    MaxPooling2D(2, 2),
                    Conv2D(64, (3, 3), activation='relu'),
                    MaxPooling2D(2, 2),
                    Flatten(),
                    Dense(128, activation='relu'),
                    Dropout(0.5),
                    Dense(train_gen.num_classes, activation='softmax'),
                ]
            )

            model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

            history = model.fit(train_gen, validation_data=val_gen, epochs=epochs)

            st.success("Model training completed!")

            st.subheader("Training Metrics")
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            ax1.plot(history.history['accuracy'], label='Train Acc')
            ax1.plot(history.history['val_accuracy'], label='Val Acc')
            ax1.legend()
            ax1.set_title("Accuracy")

            ax2.plot(history.history['loss'], label='Train Loss')
            ax2.plot(history.history['val_loss'], label='Val Loss')
            ax2.legend()
            ax2.set_title("Loss")

            st.pyplot(fig)

        else:
            st.error("Train/Val folder structure not found inside ZIP.")

