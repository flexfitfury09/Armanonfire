import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, mean_squared_error
from sklearn.pipeline import Pipeline

import shap
import lime.lime_tabular
import matplotlib.pyplot as plt
import seaborn as sns

# TensorFlow for DL
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, SimpleRNN

st.set_page_config(layout="wide")

# --- UPLOAD
st.title("All-in-One ML/DL App 🧠")

data_file = st.file_uploader("Upload CSV", type=["csv"])
if data_file:
    df = pd.read_csv(data_file)
    st.dataframe(df.head())

    target = st.selectbox("Select target", df.columns)
    task = st.selectbox("Task type", ["Classification", "Regression"])

    if task == "Classification":
        model = RandomForestClassifier()
        df[target] = df[target].astype("category").cat.codes
    else:
        model = RandomForestRegressor()

    # Train/test split
    X = df.drop(columns=[target])
    y = df[target]

    if y.ndim == 1:
        y = y.to_frame()

    test_size = st.slider("Test size", 0.1, 0.5, 0.2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    # --- TRAIN
    if st.button("Train Model"):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        if task == "Classification":
            st.text("Classification Report:")
            st.text(classification_report(y_test, y_pred))
        else:
            st.text("Regression MSE:")
            st.text(mean_squared_error(y_test, y_pred))

        joblib.dump(model, "model.pkl")
        st.success("Model trained and saved ✅")

        # SHAP Explainability
        explainer = shap.Explainer(model, X_train)
        shap_values = explainer(X_test[:100])

        st.subheader("SHAP Summary Plot")
        fig = shap.plots.beeswarm(shap_values, show=False)
        st.pyplot(bbox_inches='tight')

        # LIME
        st.subheader("LIME Explanation")
        lime_exp = lime.lime_tabular.LimeTabularExplainer(
            training_data=np.array(X_train),
            feature_names=X.columns,
            mode="regression" if task == "Regression" else "classification"
        )
        i = np.random.randint(0, X_test.shape[0])
        exp = lime_exp.explain_instance(X_test.iloc[i], model.predict)
        st.write(exp.as_list())

    # --- POLYNOMIAL REGRESSION
    st.subheader("Polynomial Regression")
    poly_degree = st.slider("Degree", 2, 5)
    if st.button("Run Poly Regression"):
        pipe = Pipeline([
            ("poly", PolynomialFeatures(degree=poly_degree)),
            ("scale", StandardScaler()),
            ("reg", LinearRegression())
        ])
        pipe.fit(X_train, y_train)
        st.write("MSE:", mean_squared_error(y_test, pipe.predict(X_test)))

# --- DEEP LEARNING DEMOS
st.header("🧠 Deep Learning Demos")

if st.checkbox("Show CNN Demo"):
    st.write("CNN expects 2D input. Simulated with random image data.")
    cnn = Sequential([
        Conv2D(16, kernel_size=3, activation='relu', input_shape=(28, 28, 1)),
        Flatten(),
        Dense(1, activation='sigmoid')
    ])
    cnn.compile(optimizer='adam', loss='binary_crossentropy')
    st.code(cnn.summary())

if st.checkbox("Show RNN Demo"):
    st.write("RNN expects sequences. Simulated with random sequence data.")
    rnn = Sequential([
        SimpleRNN(10, input_shape=(5, 1)),
        Dense(1)
    ])
    rnn.compile(optimizer='adam', loss='mse')
    st.code(rnn.summary())
