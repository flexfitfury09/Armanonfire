import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, mean_absolute_error
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.multioutput import MultiOutputRegressor
import joblib
import shap
import lime
import lime.lime_tabular
import io
import os
import threading
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional

# Delayed TensorFlow import to avoid startup lag
def import_tf():
    global tf
    import tensorflow as tf
    return tf

# --- FastAPI app for model serving ---
api = FastAPI()
model_storage = {}  # dict to hold models in memory keyed by 'model_name'

class PredictRequest(BaseModel):
    features: List[float]
    model_name: Optional[str] = "default"

@api.post("/predict")
def predict(request: PredictRequest):
    model_name = request.model_name
    if model_name not in model_storage:
        return {"error": f"Model '{model_name}' not loaded."}
    model = model_storage[model_name]
    try:
        x = np.array(request.features).reshape(1, -1)
        # TF model detection
        if hasattr(model, 'predict') and hasattr(model, 'save'):
            # keras model
            pred = model.predict(x).flatten().tolist()
        else:
            pred = model.predict(x).tolist()
        return {"prediction": pred}
    except Exception as e:
        return {"error": str(e)}

def run_api():
    uvicorn.run(api, host="0.0.0.0", port=8000)

# --- Streamlit App ---
pd.options.mode.chained_assignment = None

st.set_page_config(page_title="Advanced ML & DL + API Serving App", layout='wide')
st.title("🧠 ML + DL + Explainability + Model Saving + FastAPI Serving")

uploaded_file = st.file_uploader("Upload CSV dataset", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("### Data Preview")
    st.dataframe(data.head())

    if st.checkbox("Show Dataset Info"):
        buffer = io.StringIO()
        data.info(buf=buffer)
        st.text(buffer.getvalue())

    if st.checkbox("Show Summary Statistics"):
        st.write(data.describe())

    if st.checkbox("Show Correlation Heatmap"):
        fig, ax = plt.subplots()
        sns.heatmap(data.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    # Select target & task
    target_column = st.selectbox("Select Target Variable", options=data.columns)

    task = st.selectbox("Select Task", options=[
        "Regression",
        "Polynomial Regression",
        "Classification",
        "Multi-output Regression",
        "Deep Learning (MLP)",
        "Deep Learning (CNN - for images)",
        "Deep Learning (RNN - for sequences)"
    ])

    features = data.drop(columns=[target_column])

    if task == "Classification" and data[target_column].dtype == "object":
        le = LabelEncoder()
        data[target_column] = le.fit_transform(data[target_column])

    if task == "Multi-output Regression":
        target_columns = st.text_input("Enter comma separated target columns for multi-output regression", value=target_column)
        target_columns_list = [x.strip() for x in target_columns.split(",") if x.strip() in data.columns]
        if len(target_columns_list) < 2:
            st.error("Please select at least two valid target columns for multi-output regression.")
        else:
            target_column = target_columns_list

    numeric_features = features.select_dtypes(include=np.number).columns.tolist()
    categorical_features = features.select_dtypes(exclude=np.number).columns.tolist()

    if categorical_features:
        for cat_col in categorical_features:
            features[cat_col] = features[cat_col].astype("category").cat.codes

    scaler = StandardScaler()
    features_scaled = pd.DataFrame(scaler.fit_transform(features[numeric_features]), columns=numeric_features)
    if categorical_features:
        features_scaled = pd.concat([features_scaled, features[categorical_features]], axis=1)

    if task != "Multi-output Regression":
        X_train, X_test, y_train, y_test = train_test_split(features_scaled, data[target_column], test_size=0.2, random_state=42)
    else:
        X_train, X_test, y_train, y_test = train_test_split(features_scaled, data[target_column], test_size=0.2, random_state=42)

    model = None
    model_name = "default"

    # --- MODEL TRAINING ---

    if task == "Regression":
        model_type = st.selectbox("Select Regression Model", options=["Linear Regression", "Random Forest Regressor"])
        if model_type == "Linear Regression":
            model = LinearRegression()
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
        else:
            model = RandomForestRegressor(random_state=42)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

        st.write(f"R2 Score: {r2_score(y_test, preds):.4f}")
        st.write(f"RMSE: {mean_squared_error(y_test, preds, squared=False):.4f}")
        st.write(f"MAE: {mean_absolute_error(y_test, preds):.4f}")
        epsilon = 1e-10
        mape = np.mean(np.abs((y_test - preds) / (y_test + epsilon))) * 100
        st.write(f"MAPE (%): {mape:.2f}")

        fig, ax = plt.subplots()
        ax.scatter(y_test, preds, alpha=0.7)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        ax.set_xlabel("True Values")
        ax.set_ylabel("Predictions")
        ax.set_title("True vs Predicted")
        st.pyplot(fig)

    elif task == "Polynomial Regression":
        degree = st.slider("Polynomial Degree", min_value=2, max_value=5, value=2)
        poly = PolynomialFeatures(degree=degree)
        X_train_poly = poly.fit_transform(X_train)
        X_test_poly = poly.transform(X_test)

        model = LinearRegression()
        model.fit(X_train_poly, y_train)
        preds = model.predict(X_test_poly)

        st.write(f"Polynomial Degree: {degree}")
        st.write(f"R2 Score: {r2_score(y_test, preds):.4f}")
        st.write(f"RMSE: {mean_squared_error(y_test, preds, squared=False):.4f}")
        st.write(f"MAE: {mean_absolute_error(y_test, preds):.4f}")
        epsilon = 1e-10
        mape = np.mean(np.abs((y_test - preds) / (y_test + epsilon))) * 100
        st.write(f"MAPE (%): {mape:.2f}")

        fig, ax = plt.subplots()
        ax.scatter(y_test, preds, alpha=0.7)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        ax.set_xlabel("True Values")
        ax.set_ylabel("Predictions")
        ax.set_title("Polynomial Regression True vs Predicted")
        st.pyplot(fig)

    elif task == "Classification":
        model_type = st.selectbox("Select Classification Model", options=["Logistic Regression", "Random Forest Classifier"])
        if model_type == "Logistic Regression":
            model = LogisticRegression(max_iter=200)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
        else:
            model = RandomForestClassifier(random_state=42)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

        st.write(f"Accuracy: {accuracy_score(y_test, preds):.4f}")
        st.text(classification_report(y_test, preds))

    elif task == "Multi-output Regression":
        model_type = st.selectbox("Select Model for Multi-output Regression", options=["Random Forest Regressor", "Linear Regression"])
        if model_type == "Random Forest Regressor":
            base_model = RandomForestRegressor(random_state=42)
        else:
            base_model = LinearRegression()
        model = MultiOutputRegressor(base_model)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        r2s = [r2_score(y_test.iloc[:, i], preds[:, i]) for i in range(len(target_column))]
        rmses = [mean_squared_error(y_test.iloc[:, i], preds[:, i], squared=False) for i in range(len(target_column))]
        st.write("R2 Scores for each output:")
        for i, col in enumerate(target_column):
            st.write(f"{col}: {r2s[i]:.4f}")
        st.write("RMSE for each output:")
        for i, col in enumerate(target_column):
            st.write(f"{col}: {rmses[i]:.4f}")

    elif task.startswith("Deep Learning"):
        tf = import_tf()
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, LSTM, Embedding, Dropout
        from tensorflow.keras.utils import to_categorical
        import tensorflow.keras.backend as K

        # Prepare data for DL:
        if task == "Deep Learning (MLP)":
            # MLP for tabular data
            if task == "Classification":
                n_classes = len(np.unique(y_train))
                y_train_dl = to_categorical(y_train, num_classes=n_classes)
                y_test_dl = to_categorical(y_test, num_classes=n_classes)
            else:
                y_train_dl = y_train
                y_test_dl = y_test

            model = Sequential([
                Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
                Dropout(0.2),
                Dense(64, activation='relu'),
                Dropout(0.2),
                Dense(n_classes if task=="Classification" else 1, activation='softmax' if task=="Classification" else 'linear')
            ])

            model.compile(optimizer='adam',
                          loss='categorical_crossentropy' if task=="Classification" else 'mse',
                          metrics=['accuracy'] if task=="Classification" else ['mse'])

            model.fit(X_train, y_train_dl, epochs=20, batch_size=32, verbose=0)
            preds_prob = model.predict(X_test)
            if task == "Classification":
                preds = preds_prob.argmax(axis=1)
                st.write(f"Accuracy: {accuracy_score(y_test, preds):.4f}")
                st.text(classification_report(y_test, preds))
            else:
                preds = preds_prob.flatten()
                st.write(f"R2 Score: {r2_score(y_test, preds):.4f}")
                st.write(f"RMSE: {mean_squared_error(y_test, preds, squared=False):.4f}")

        elif task == "Deep Learning (CNN - for images)":
            st.info("Expecting image data in shape (samples, height, width, channels). Your CSV must have flattened images or feature vectors.")
            # For demonstration, reshape features as 28x28 grayscale image, if possible
            img_size = 28
            expected_size = img_size * img_size
            if X_train.shape[1] != expected_size:
                st.error(f"Features count ({X_train.shape[1]}) doesn't match expected flattened image size {expected_size}. Cannot train CNN.")
            else:
                X_train_img = X_train.values.reshape(-1, img_size, img_size, 1)
                X_test_img = X_test.values.reshape(-1, img_size, img_size, 1)
                if task == "Classification":
                    n_classes = len(np.unique(y_train))
                    y_train_dl = to_categorical(y_train, num_classes=n_classes)
                    y_test_dl = to_categorical(y_test, num_classes=n_classes)
                else:
                    y_train_dl = y_train
                    y_test_dl = y_test

                model = Sequential([
                    Conv2D(32, (3,3), activation='relu', input_shape=(img_size,img_size,1)),
                    MaxPooling2D((2,2)),
                    Conv2D(64, (3,3), activation='relu'),
                    MaxPooling2D((2,2)),
                    Flatten(),
                    Dense(64, activation='relu'),
                    Dense(n_classes if task=="Classification" else 1, activation='softmax' if task=="Classification" else 'linear')
                ])

                model.compile(optimizer='adam',
                              loss='categorical_crossentropy' if task=="Classification" else 'mse',
                              metrics=['accuracy'] if task=="Classification" else ['mse'])

                model.fit(X_train_img, y_train_dl, epochs=20, batch_size=32, verbose=0)
                preds_prob = model.predict(X_test_img)
                if task == "Classification":
                    preds = preds_prob.argmax(axis=1)
                    st.write(f"Accuracy: {accuracy_score(y_test, preds):.4f}")
                    st.text(classification_report(y_test, preds))
                else:
                    preds = preds_prob.flatten()
                    st.write(f"R2 Score: {r2_score(y_test, preds):.4f}")
                    st.write(f"RMSE: {mean_squared_error(y_test, preds, squared=False):.4f}")

        elif task == "Deep Learning (RNN - for sequences)":
            st.info("Expecting sequence data in shape (samples, timesteps, features). Your CSV must be reshaped appropriately.")
            # Simplified: reshape to (samples, timesteps=10, features=features/10)
            seq_len = 10
            if X_train.shape[1] % seq_len != 0:
                st.error(f"Number of features ({X_train.shape[1]}) not divisible by {seq_len}, can't reshape for RNN.")
            else:
                n_features = X_train.shape[1] // seq_len
                X_train_seq = X_train.values.reshape(-1, seq_len, n_features)
                X_test_seq = X_test.values.reshape(-1, seq_len, n_features)
                if task == "Classification":
                    n_classes = len(np.unique(y_train))
                    y_train_dl = to_categorical(y_train, num_classes=n_classes)
                    y_test_dl = to_categorical(y_test, num_classes=n_classes)
                else:
                    y_train_dl = y_train
                    y_test_dl = y_test

                model = Sequential([
                    LSTM(64, input_shape=(seq_len, n_features)),
                    Dense(32, activation='relu'),
                    Dense(n_classes if task=="Classification" else 1, activation='softmax' if task=="Classification" else 'linear')
                ])

                model.compile(optimizer='adam',
                              loss='categorical_crossentropy' if task=="Classification" else 'mse',
                              metrics=['accuracy'] if task=="Classification" else ['mse'])

                model.fit(X_train_seq, y_train_dl, epochs=20, batch_size=32, verbose=0)
                preds_prob = model.predict(X_test_seq)
                if task == "Classification":
                    preds = preds_prob.argmax(axis=1)
                    st.write(f"Accuracy: {accuracy_score(y_test, preds):.4f}")
                    st.text(classification_report(y_test, preds))
                else:
                    preds = preds_prob.flatten()
                    st.write(f"R2 Score: {r2_score(y_test, preds):.4f}")
                    st.write(f"RMSE: {mean_squared_error(y_test, preds, squared=False):.4f}")

    # --- MODEL SAVING & LOADING ---
    st.markdown("---")
    st.header("Model Persistence & Serving")

    save_path = st.text_input("Enter path to save model (e.g. model.pkl or model.h5)", value="model.pkl")

    if st.button("Save Model"):
        try:
            if "tensorflow" in str(type(model)).lower():
                # TF model saving
                model.save(save_path)
                st.success(f"TensorFlow model saved to {save_path}")
            else:
                joblib.dump(model, save_path)
                st.success(f"Model saved to {save_path}")
            # Also load into API model_storage dict for serving
            if "tensorflow" in str(type(model)).lower():
                model_storage[model_name] = model
            else:
                model_storage[model_name] = joblib.load(save_path)
        except Exception as e:
            st.error(f"Failed to save model: {e}")

    load_path = st.text_input("Enter path to load model", value="model.pkl")

    if st.button("Load Model"):
        try:
            if load_path.endswith(".h5"):
                tf = import_tf()
                model_storage[model_name] = tf.keras.models.load_model(load_path)
                st.success(f"TensorFlow model loaded from {load_path}")
            else:
                model_storage[model_name] = joblib.load(load_path)
                st.success(f"Model loaded from {load_path}")
        except Exception as e:
            st.error(f"Failed to load model: {e}")

    # --- SHAP / LIME Explainability ---
    st.markdown("---")
    st.header("Explainability (SHAP/LIME)")

    if model is not None:
        explainer_type = st.selectbox("Choose Explainer", options=["SHAP", "LIME"])

        if explainer_type == "SHAP":
            try:
                if hasattr(model, "predict"):
                    explainer = shap.Explainer(model, X_train)
                    shap_values = explainer(X_test)
                    st.set_option('deprecation.showPyplotGlobalUse', False)
                    shap.summary_plot(shap_values, X_test, show=False)
                    st.pyplot(bbox_inches='tight')
                else:
                    st.warning("Model does not support SHAP explainer.")
            except Exception as e:
                st.error(f"SHAP failed: {e}")

        elif explainer_type == "LIME":
            try:
                explainer = lime.lime_tabular.LimeTabularExplainer(
                    training_data=np.array(X_train),
                    feature_names=X_train.columns,
                    class_names=[str(i) for i in np.unique(y_train)] if task == "Classification" else None,
                    mode='classification' if task == "Classification" else 'regression'
                )
                idx = st.number_input("Instance index to explain", min_value=0, max_value=len(X_test)-1, value=0)
                exp = explainer.explain_instance(X_test.iloc[idx].values, model.predict)
                st.write(exp.as_list())
                fig = exp.as_pyplot_figure()
                st.pyplot(fig)
            except Exception as e:
                st.error(f"LIME failed: {e}")

# Run FastAPI server in separate thread alongside Streamlit
if __name__ == "__main__":
    threading.Thread(target=run_api, daemon=True).start()
