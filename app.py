import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import io
import zipfile
from PIL import Image
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, LabelEncoder
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
from sklearn.pipeline import Pipeline
import shap
import lime.lime_tabular
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense

st.set_page_config(page_title="Enhanced ML App", layout="wide")
st.title("📊 ML + DL + CNN Upload + Metrics Dashboard")

Python3_version = "3.10"
st.sidebar.info(f"Recommended Python version: {Python3_version}")

# --------- Data upload & modeling ---------
data_file = st.file_uploader("Upload CSV (tabular)", type=["csv"])
if data_file:
    df = pd.read_csv(data_file)
    st.dataframe(df.head())
    target = st.selectbox("Select target column", df.columns)
    task = st.selectbox("Choose task", ["Regression", "Classification", "Polynomial Regression"])

    X = df.drop(columns=[target])
    y = df[target]
    if task == "Classification":
        le = LabelEncoder()
        y = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = None

    if st.button("Train Tabular Model"):
        if task == "Regression":
            model = RandomForestRegressor(random_state=42)
        elif task == "Classification":
            model = RandomForestClassifier(random_state=42)
        else:
            degree = st.slider("Polynomial degree", 2, 5, 3)
            model = Pipeline([
                ("poly", PolynomialFeatures(degree)),
                ("lr", LinearRegression())
            ])
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        st.write("Evaluation results:")
        if task == "Classification":
            st.write("Accuracy:", accuracy_score(y_test, y_pred))
            st.text(classification_report(y_test, y_pred))
        else:
            st.write("MSE:", mean_squared_error(y_test, y_pred))

        joblib.dump(model, "tabular_model.pkl")
        st.success("Model trained & saved.")

        # Metrics dashboard
        st.subheader("Feature Importance")
        if hasattr(model, "feature_importances_"):
            fi = model.feature_importances_
            st.bar_chart(pd.Series(fi, index=X.columns))

        st.subheader("SHAP Summary (first 100 rows)")
        explainer = shap.Explainer(model, X_train)
        shap_values = explainer(X_test[:100])
        fig = shap.plots.beeswarm(shap_values, show=False)
        st.pyplot(fig)

# --------- CNN dataset upload ---------
st.header("📸 CNN Model Training (Image Data)")
cnn_zip = st.file_uploader("Upload images ZIP (single class folders)", type=["zip"])
if cnn_zip and st.button("Train CNN"):
    with zipfile.ZipFile(io.BytesIO(cnn_zip.read())) as zf:
        images, labels = [], []
        for fname in zf.namelist():
            if fname.lower().endswith(("png", "jpg", "jpeg")):
                with zf.open(fname) as f:
                    img = Image.open(f).convert("RGB").resize((64, 64))
                    images.append(np.array(img)/255.0)
                    labels.append(os.path.dirname(fname))
        X = np.stack(images)
        le = LabelEncoder()
        y = le.fit_transform(labels)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = Sequential([
            Conv2D(16, (3,3), activation='relu', input_shape=X_train.shape[1:]),
            Flatten(),
            Dense(len(le.classes_), activation='softmax')
        ])
        model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])

        hist = model.fit(X_train, y_train, epochs=5, validation_split=0.2)
        st.write("Training complete.")
        st.line_chart(pd.DataFrame(hist.history))

        loss, acc = model.evaluate(X_test, y_test, verbose=0)
        st.write(f"Test Accuracy: {acc:.3f}")
        model.save("cnn_model_tf")

# --------- Model serving UI ---------
st.header("🤖 Prediction UI")
if st.sidebar.button("Load Tabular Model"):
    if os.path.exists("tabular_model.pkl"):
        model = joblib.load("tabular_model.pkl")
        st.success("Tabular model loaded.")

if 'model' in locals():
    inp = {}
    for col in X_train.columns:
        inp[col] = st.number_input(f"Input {col}", value=0.0)
    if st.button("Predict"):
        pred = model.predict(pd.DataFrame([inp]))
        st.write("Prediction:", float(pred))

st.sidebar.write("✅ All features included. Delete cache before redeploy.")

