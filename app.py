import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    r2_score, mean_absolute_error, mean_squared_error,
    precision_score, recall_score, f1_score
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import tempfile
import zipfile

st.set_page_config(page_title="AutoML Dashboard", layout="wide")

st.title("📊 AutoML + CNN Dashboard")

tab1, tab2 = st.tabs(["Tabular Data (CSV)", "Image Classification (CNN)"])

# ------------------ TABULAR DATA ------------------
with tab1:
    uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Preview of Dataset", df.head())

        target = st.selectbox("Select Target Column", df.columns)

        if target:
            # Determine task type
            if pd.api.types.is_numeric_dtype(df[target]):
                task = st.radio("Task type:", ("Regression", "Classification"))
                if task == "Classification":
                    # If numeric target but classification, convert to categorical
                    y = df[target].astype(str)
                else:
                    y = df[target]
            else:
                task = "Classification"
                y = df[target].astype(str)

            X = df.drop(target, axis=1)

            # Encode categorical features if any
            for col in X.select_dtypes(include=['object']).columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))

            # Encode target if classification and non-numeric
            if task == "Classification" and not pd.api.types.is_numeric_dtype(y):
                le_target = LabelEncoder()
                y = le_target.fit_transform(y)

            # Model selection
            st.subheader("Select Model")
            if task == "Classification":
                model_name = st.selectbox("Choose a classifier", 
                                          ["Random Forest", "Logistic Regression", "Support Vector Machine"])
            else:
                model_name = st.selectbox("Choose a regressor", 
                                          ["Random Forest", "Linear Regression", "Support Vector Regression"])

            # Train/test split
            test_size = st.slider("Test set size (%)", 5, 50, 20)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42)

            # Initialize model
            if model_name == "Random Forest":
                if task == "Classification":
                    model = RandomForestClassifier(random_state=42)
                else:
                    model = RandomForestRegressor(random_state=42)
            elif model_name == "Logistic Regression":
                model = LogisticRegression(max_iter=1000)
            elif model_name == "Support Vector Machine":
                model = SVC()
            elif model_name == "Linear Regression":
                model = LinearRegression()
            elif model_name == "Support Vector Regression":
                model = SVR()
            else:
                st.error("Unsupported model selected!")
                st.stop()

            # Fit model
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            st.subheader("Model Performance")

            if task == "Classification":
                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                
                st.success(f"Accuracy: {acc:.3f}")
                st.text(f"Precision (weighted): {prec:.3f}")
                st.text(f"Recall (weighted): {rec:.3f}")
                st.text(f"F1 Score (weighted): {f1:.3f}")

                st.text("Classification Report:")
                st.text(classification_report(y_test, y_pred, zero_division=0))

                st.text("Confusion Matrix:")
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                st.pyplot(fig)

            else:
                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)

                st.success(f"R² Score: {r2:.3f}")
                st.text(f"Mean Absolute Error (MAE): {mae:.3f}")
                st.text(f"Mean Squared Error (MSE): {mse:.3f}")
                st.text(f"Root Mean Squared Error (RMSE): {rmse:.3f}")

                # Plot true vs predicted
                fig, ax = plt.subplots()
                ax.scatter(y_test, y_pred, alpha=0.7)
                ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
                ax.set_xlabel("Actual")
                ax.set_ylabel("Predicted")
                ax.set_title("Actual vs Predicted")
                st.pyplot(fig)

# ------------------ IMAGE DATA / CNN ------------------
with tab2:
    st.subheader("Upload a ZIP of your image dataset (train/val split inside)")
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
            datagen = ImageDataGenerator(rescale=1./255)
            train_gen = datagen.flow_from_directory(train_path, target_size=(img_height, img_width), batch_size=batch_size, class_mode='categorical')
            val_gen = datagen.flow_from_directory(val_path, target_size=(img_height, img_width), batch_size=batch_size, class_mode='categorical')

            model = Sequential([
                Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
                MaxPooling2D(2, 2),
                Conv2D(64, (3, 3), activation='relu'),
                MaxPooling2D(2, 2),
                Flatten(),
                Dense(128, activation='relu'),
                Dropout(0.5),
                Dense(train_gen.num_classes, activation='softmax')
            ])

            model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

            history = model.fit(train_gen, validation_data=val_gen, epochs=epochs)

            st.success("Model Training Completed!")

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
            st.error("Train/Val folder structure not found in ZIP.")
