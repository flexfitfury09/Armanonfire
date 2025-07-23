import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import shutil

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
            X = df.drop(target, axis=1)
            y = df[target]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model = RandomForestClassifier()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            st.success(f"Accuracy: {acc:.2f}")

            st.text("Classification Report")
            st.text(classification_report(y_test, y_pred))

            st.text("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            st.pyplot(fig)

# ------------------ IMAGE DATA / CNN ------------------
with tab2:
    st.subheader("Upload a ZIP of your image dataset (train/val split inside)")
    zip_file = st.file_uploader("Upload ZIP File of Image Dataset", type=["zip"])

    if zip_file:
        import zipfile
        import tempfile

        temp_dir = tempfile.mkdtemp()
        zip_path = os.path.join(temp_dir, "dataset.zip")
        with open(zip_path, "wb") as f:
            f.write(zip_file.read())
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)

        # Assume folders inside extracted dir: train/ and val/
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
