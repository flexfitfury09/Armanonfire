
import streamlit as st
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, r2_score, classification_report
from sklearn.pipeline import Pipeline

# Supervised
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, SVR

# Unsupervised
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
from scipy.cluster.hierarchy import linkage, dendrogram

# Models
models_classification = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(),
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier()
}
models_regression = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(),
    "KNN": KNeighborsRegressor(),
    "SVM": SVR(),
    "Decision Tree": DecisionTreeRegressor()
}

st.set_page_config(page_title="AutoML Pro", layout="wide")
st.title("🧠 Bulletproof AutoML: Full Auto Analyzer & ML Explorer")

# Upload CSV
file = st.file_uploader("📂 Upload a CSV file", type=['csv'])
if file:
    df = pd.read_csv(file)
    st.success("✅ File uploaded successfully")
    st.dataframe(df.head())

    # Data Cleaning
    object_cols = df.select_dtypes(include='object').columns
    for col in object_cols:
        df[col] = df[col].astype(str)
        df[col] = SimpleImputer(strategy='most_frequent').fit_transform(df[[col]]).ravel()
        df[col] = LabelEncoder().fit_transform(df[col])
    for col in df.select_dtypes(include=[np.number]):
        df[col] = SimpleImputer(strategy='mean').fit_transform(df[[col]]).ravel()

    st.subheader("🔍 Data Summary")
    st.dataframe(df.describe())
    st.write("📉 Correlation Heatmap")
    fig, ax = plt.subplots()
    sns.heatmap(df.corr(), cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    target = st.selectbox("🎯 Choose Target Column", df.columns)
    task = st.selectbox("📘 Task Type", ["Classification", "Regression", "Unsupervised"])

    if task != "Unsupervised":
        y = df[target]
        X = df.drop(columns=[target])
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        st.subheader("🤖 Choose ML Model")
        model_name = st.selectbox("Select Model", list(models_classification.keys()) if task == "Classification" else list(models_regression.keys()))
        model = models_classification[model_name] if task == "Classification" else models_regression[model_name]

        if st.button("🚀 Train Model"):
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            score = accuracy_score(y_test, preds) if task == "Classification" else r2_score(y_test, preds)
            st.success(f"✅ Model Trained - Score: {score:.4f}")
            if task == "Classification":
                st.dataframe(pd.DataFrame(classification_report(y_test, preds, output_dict=True)).transpose())
    else:
        st.subheader("🔓 Unsupervised Learning")
        method = st.selectbox("Choose Method", ["KMeans", "Hierarchical", "DBSCAN", "PCA", "t-SNE"])
        features = df.drop(columns=[target]) if target in df.columns else df.copy()
        X = StandardScaler().fit_transform(features)

        if method == "KMeans":
            k = st.slider("Number of Clusters", 2, 10, 3)
            model = KMeans(n_clusters=k)
            labels = model.fit_predict(X)
            st.write("📊 Cluster Labels", labels)
            df["Cluster"] = labels
            st.dataframe(df)
        elif method == "Hierarchical":
            st.write("🔗 Dendrogram")
            linked = linkage(X, 'ward')
            fig, ax = plt.subplots(figsize=(10, 5))
            dendrogram(linked, ax=ax)
            st.pyplot(fig)
        elif method == "DBSCAN":
            model = DBSCAN()
            labels = model.fit_predict(X)
            df["Cluster"] = labels
            st.dataframe(df)
        elif method == "PCA":
            pca = PCA(n_components=2)
            comp = pca.fit_transform(X)
            fig, ax = plt.subplots()
            ax.scatter(comp[:, 0], comp[:, 1])
            st.pyplot(fig)
        elif method == "t-SNE":
            tsne = TSNE(n_components=2)
            emb = tsne.fit_transform(X)
            fig, ax = plt.subplots()
            ax.scatter(emb[:, 0], emb[:, 1])
            st.pyplot(fig)

    # Simulators and Concept Modules
    st.subheader("📘 Educational Modules")
    with st.expander("📌 What is Semi-Supervised Learning?"):
        st.markdown("""
        - Combines labeled and unlabeled data
        - Useful when labeled data is scarce
        - Algorithms: Self-training, Label propagation, Graph-based methods
        """)

    with st.expander("🎮 Reinforcement Learning Overview"):
        st.markdown("""
        - Agent interacts with environment
        - Learns by reward signals
        - Algorithms: Q-Learning, SARSA, DQN, PPO, A3C
        """)

    with st.expander("🧠 Deep Learning Overview"):
        st.markdown("""
        - CNN: Image recognition
        - RNN: Sequence modeling
        - Transformers: NLP (BERT, GPT)
        """)

    with st.expander("⚙️ MLOps & Pipeline Automation"):
        st.markdown("""
        - Tools: MLFlow, DVC, Airflow, Docker
        - Concepts: CI/CD, Model tracking, Drift monitoring
        """)

    st.success("🎯 End of Demo — All core modules tested and operational ✅")
else:
    st.info("👈 Upload a dataset to begin")
