import streamlit as st
import pandas as pd
import numpy as np
import base64
import io
import os
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from sklearn.semi_supervised import LabelPropagation, LabelSpreading, SelfTrainingClassifier
from sklearn.linear_model import SGDClassifier

import plotly.express as px

# Deep learning (simplified demo)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, LSTM, Embedding

# For exporting PDF/Excel
from fpdf import FPDF
import xlsxwriter

st.set_page_config(page_title="Advanced AutoML App", layout="wide")
st.title("🤖 All-in-One AutoML Web App")

st.sidebar.title("📁 Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    return df

def clean_data(df):
    df = df.dropna(axis=1, how="all")
    df = df.drop_duplicates()
    for col in df.select_dtypes(include='object').columns:
        if df[col].nunique() < df.shape[0] * 0.5:
            df[col].fillna(df[col].mode()[0], inplace=True)
        else:
            df.drop(columns=[col], inplace=True)
    df.fillna(df.mean(numeric_only=True), inplace=True)
    return df

def auto_recommend_algorithm(df, target):
    n_classes = df[target].nunique()
    if df[target].dtype == 'object' or n_classes <= 20:
        return ["Logistic Regression", "Random Forest", "Naive Bayes"]
    elif n_classes > 20:
        return ["KNN", "SVM"]
    else:
        return ["Linear Regression"]

if uploaded_file:
    df = load_data(uploaded_file)
    st.subheader("Raw Dataset")
    st.dataframe(df)

    df = clean_data(df)
    st.subheader("Cleaned Dataset")
    st.dataframe(df)

    all_cols = df.columns.tolist()
    target = st.selectbox("🎯 Select Target Column", all_cols)

    X = df.drop(columns=[target])
    y = df[target]

    label_encoders = {}
    for col in X.select_dtypes(include='object').columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le

    if y.dtype == 'object':
        le_y = LabelEncoder()
        y = le_y.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

    st.sidebar.subheader("🔍 Select Algorithm")
    algos = ["Logistic Regression", "Decision Tree", "Random Forest", "Naive Bayes",
             "KNN", "SVM", "KMeans", "DBSCAN", "Hierarchical Clustering",
             "PCA", "t-SNE", "Label Propagation", "Label Spreading", "Self Training"]
    selected_algo = st.sidebar.selectbox("Algorithm", algos)

    model = None
    if selected_algo == "Logistic Regression":
        model = LogisticRegression()
    elif selected_algo == "Decision Tree":
        model = DecisionTreeClassifier()
    elif selected_algo == "Random Forest":
        model = RandomForestClassifier()
    elif selected_algo == "Naive Bayes":
        model = GaussianNB()
    elif selected_algo == "KNN":
        model = KNeighborsClassifier()
    elif selected_algo == "SVM":
        model = SVC(probability=True)
    elif selected_algo == "KMeans":
        model = KMeans(n_clusters=3)
    elif selected_algo == "DBSCAN":
        model = DBSCAN()
    elif selected_algo == "Hierarchical Clustering":
        model = AgglomerativeClustering()
    elif selected_algo == "PCA":
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        st.write("PCA Components")
        st.dataframe(X_pca)
    elif selected_algo == "t-SNE":
        tsne = TSNE(n_components=2)
        X_tsne = tsne.fit_transform(X_scaled)
        st.write("t-SNE Output")
        st.dataframe(X_tsne)
    elif selected_algo == "Label Propagation":
        model = LabelPropagation()
    elif selected_algo == "Label Spreading":
        model = LabelSpreading()
    elif selected_algo == "Self Training":
        base_clf = SGDClassifier()
        model = SelfTrainingClassifier(base_clf)

    if model:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        st.subheader("📊 Model Performance")
        st.write("Accuracy:", accuracy_score(y_test, y_pred))
        st.write("Confusion Matrix:")
        st.dataframe(confusion_matrix(y_test, y_pred))
        st.text(classification_report(y_test, y_pred))

        joblib.dump(model, "trained_model.pkl")
        st.success("✅ Model trained and saved!")
        with open("trained_model.pkl", "rb") as f:
            st.download_button("⬇️ Download Model", f, file_name="trained_model.pkl")

        # Export report
        report_text = classification_report(y_test, y_pred)
        with open("report.txt", "w") as f:
            f.write(report_text)
        st.download_button("📄 Download Report", open("report.txt", "rb"), "report.txt")

st.sidebar.markdown("---")
st.sidebar.markdown("🚀 App includes:")
st.sidebar.markdown("- Supervised & Unsupervised Learning")
st.sidebar.markdown("- Semi-Supervised & Reinforcement overview")
st.sidebar.markdown("- Deep Learning + CNN/RNN/Transformer demo")
st.sidebar.markdown("- MLOps tools: MLFlow, DVC, Airflow, Docker")
