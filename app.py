import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import load_boston, load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, LabelEncoder
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.pipeline import make_pipeline
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv1D, Flatten, SimpleRNN, Embedding
from tensorflow.keras.utils import to_categorical
import shap
import lime
import lime.lime_tabular
import joblib
import os
import tempfile
from fastapi import FastAPI
import uvicorn
import threading
import pandas_profiling

st.set_page_config(page_title="Advanced ML App with DL + Explainability + API", layout="wide")

st.title("🚀 Advanced ML App: Regression, Classification, DL, Explainability & API")

# --- Sidebar ---
st.sidebar.header("Select Task")
task = st.sidebar.selectbox("Choose task", ["Regression", "Classification", "Deep Learning Regression", "Deep Learning Classification", "AutoEDA", "Explainability (SHAP/LIME)", "API Server"])

# Global temp directory for saving/loading models
model_dir = tempfile.gettempdir()

# --- Regression Task ---
def regression_task():
    st.header("Regression - Boston Housing Dataset")
    data = load_boston()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="PRICE")
    
    st.write("Data Sample:")
    st.dataframe(X.head())
    
    # Select regression model
    model_type = st.selectbox("Choose regression model", ["Linear Regression", "Polynomial Regression", "Ridge Regression", "Multi-Output Regression (Ridge)"])
    
    test_size = st.slider("Test set size", 0.1, 0.5, 0.2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    if model_type == "Linear Regression":
        model = LinearRegression()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mse = mean_squared_error(y_test, preds)
        st.write(f"Mean Squared Error: {mse:.4f}")
        joblib.dump(model, os.path.join(model_dir, "linear_regression.pkl"))
        st.success("Model saved!")
    
    elif model_type == "Polynomial Regression":
        degree = st.slider("Polynomial degree", 2, 5, 3)
        pipeline = make_pipeline(PolynomialFeatures(degree), LinearRegression())
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        mse = mean_squared_error(y_test, preds)
        st.write(f"Mean Squared Error: {mse:.4f}")
        joblib.dump(pipeline, os.path.join(model_dir, "poly_regression.pkl"))
        st.success("Model saved!")
    
    elif model_type == "Ridge Regression":
        alpha = st.slider("Alpha (Regularization)", 0.1, 10.0, 1.0)
        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mse = mean_squared_error(y_test, preds)
        st.write(f"Mean Squared Error: {mse:.4f}")
        joblib.dump(model, os.path.join(model_dir, "ridge_regression.pkl"))
        st.success("Model saved!")
    
    elif model_type == "Multi-Output Regression (Ridge)":
        # For demo: Create multi-output y by adding noise to original y
        y_multi = pd.DataFrame({
            "PRICE": y,
            "PRICE_PLUS_NOISE": y + np.random.normal(0, 5, len(y))
        })
        X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(X, y_multi, test_size=test_size, random_state=42)
        model = MultiOutputRegressor(Ridge(alpha=1.0))
        model.fit(X_train_m, y_train_m)
        preds = model.predict(X_test_m)
        mse1 = mean_squared_error(y_test_m["PRICE"], preds[:, 0])
        mse2 = mean_squared_error(y_test_m["PRICE_PLUS_NOISE"], preds[:, 1])
        st.write(f"MSE for PRICE: {mse1:.4f}")
        st.write(f"MSE for PRICE_PLUS_NOISE: {mse2:.4f}")
        joblib.dump(model, os.path.join(model_dir, "multioutput_ridge.pkl"))
        st.success("Multi-output model saved!")

# --- Classification Task ---
def classification_task():
    st.header("Classification - Iris Dataset")
    data = load_iris()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="Species")
    target_names = data.target_names
    
    st.write("Data Sample:")
    st.dataframe(X.head())
    
    # Encode target for clarity
    y_encoded = y
    
    test_size = st.slider("Test set size", 0.1, 0.5, 0.2)
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=test_size, random_state=42)
    
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    st.write(f"Accuracy: {acc:.4f}")
    joblib.dump(model, os.path.join(model_dir, "logistic_regression.pkl"))
    st.success("Model saved!")

# --- Deep Learning Regression ---
def dl_regression():
    st.header("Deep Learning Regression - Boston Housing")
    data = load_boston()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)
    
    test_size = st.slider("Test set size", 0.1, 0.5, 0.2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = Sequential([
        Dense(64, activation="relu", input_shape=(X_train_scaled.shape[1],)),
        Dense(32, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X_train_scaled, y_train, epochs=20, verbose=0)
    
    preds = model.predict(X_test_scaled).flatten()
    mse = mean_squared_error(y_test, preds)
    st.write(f"DL Model Mean Squared Error: {mse:.4f}")
    
    model.save(os.path.join(model_dir, "dl_regression_model"))
    joblib.dump(scaler, os.path.join(model_dir, "scaler.pkl"))
    st.success("DL regression model and scaler saved!")

# --- Deep Learning Classification ---
def dl_classification():
    st.header("Deep Learning Classification - Iris")
    data = load_iris()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    
    test_size = st.slider("Test set size", 0.1, 0.5, 0.2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # One-hot encode targets
    y_train_cat = to_categorical(y_train)
    y_test_cat = to_categorical(y_test)
    
    model = Sequential([
        Dense(64, activation="relu", input_shape=(X_train_scaled.shape[1],)),
        Dense(32, activation="relu"),
        Dense(len(np.unique(y)), activation="softmax")
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(X_train_scaled, y_train_cat, epochs=30, verbose=0)
    
    loss, acc = model.evaluate(X_test_scaled, y_test_cat, verbose=0)
    st.write(f"DL Model Accuracy: {acc:.4f}")
    
    model.save(os.path.join(model_dir, "dl_classification_model"))
    joblib.dump(scaler, os.path.join(model_dir, "scaler_classification.pkl"))
    st.success("DL classification model and scaler saved!")

# --- AutoEDA ---
def auto_eda():
    st.header("AutoEDA with pandas-profiling")
    uploaded_file = st.file_uploader("Upload a CSV file for EDA", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write(df.head())
        profile = pandas_profiling.ProfileReport(df, minimal=True)
        st_profile = st.components.v1.html(profile.to_html(), height=800, scrolling=True)
        
# --- Explainability with SHAP & LIME ---
def explainability():
    st.header("Model Explainability with SHAP and LIME")
    
    st.info("This demo will train a simple model and explain predictions with SHAP and LIME")
    
    data = load_boston()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    explainer_type = st.selectbox("Choose explainer", ["SHAP", "LIME"])
    index = st.slider("Select test instance index", 0, len(X_test)-1, 0)
    instance = X_test.iloc[index]
    
    if explainer_type == "SHAP":
        explainer = shap.Explainer(model.predict, X_train)
        shap_values = explainer(X_test)
        st.write("SHAP force plot for selected instance")
        shap_html = shap.plots.force(shap_values[index], matplotlib=False)
        st_shap_html(shap_html)
    
    else:
        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=np.array(X_train),
            feature_names=X_train.columns,
            mode='regression'
        )
        exp = explainer.explain_instance(
            data_row=instance.values,
            predict_fn=model.predict,
            num_features=5
        )
        st.write("LIME explanation:")
        st.write(exp.as_list())
        fig = exp.as_pyplot_figure()
        st.pyplot(fig)

# Helper function to display SHAP plots in Streamlit
def st_shap_html(shap_html):
    import streamlit.components.v1 as components
    components.html(shap_html.data, height=300)

# --- FastAPI Model Serving ---

app_fastapi = FastAPI()

# Load model for API serving
saved_model_path = os.path.join(model_dir, "dl_regression_model")
if os.path.exists(saved_model_path):
    dl_model = load_model(saved_model_path)
    scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
else:
    dl_model = None
    scaler = None

@app_fastapi.get("/")
def root():
    return {"message": "ML Model Serving API Running"}

@app_fastapi.post("/predict")
def predict(features: list):
    if dl_model is None or scaler is None:
        return {"error": "Model not available. Train DL regression model first."}
    features_np = np.array(features).reshape(1, -1)
    scaled = scaler.transform(features_np)
    pred = dl_model.predict(scaled)
    return {"prediction": float(pred[0][0])}

def run_fastapi():
    uvicorn.run(app_fastapi, host="0.0.0.0", port=8000)

# Streamlit main control
if task == "Regression":
    regression_task()
elif task == "Classification":
    classification_task()
elif task == "Deep Learning Regression":
    dl_regression()
elif task == "Deep Learning Classification":
    dl_classification()
elif task == "AutoEDA":
    auto_eda()
elif task == "Explainability (SHAP/LIME)":
    explainability()
elif task == "API Server":
    st.header("FastAPI Model Serving")
    st.write("This will run a FastAPI server in a background thread on port 8000.")
    if st.button("Start API Server"):
        st.info("Starting FastAPI server on port 8000...")
        threading.Thread(target=run_fastapi, daemon=True).start()
        st.success("FastAPI server started. You can access http://localhost:8000 in your browser.")

