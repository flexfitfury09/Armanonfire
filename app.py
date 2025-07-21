import streamlit as st
import pandas as pd
import numpy as np
import joblib
import io
import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import accuracy_score, r2_score, classification_report
from sklearn.cluster import KMeans
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
from sklearn.inspection import PartialDependenceDisplay

# --- Page Configuration ---
st.set_page_config(page_title="Definitive AutoML Toolkit", page_icon="🚀", layout="wide")

# --- Session State Initialization ---
st.session_state.setdefault('analysis_complete', False)
st.session_state.setdefault('results', {})

# --- Main Analysis Pipeline Function ---
def run_analysis_pipeline(df, config):
    results = {}
    target_column = config['target_column']
    task = config['task']
    
    # --- EDA ---
    profile = ProfileReport(df, title=f"Automated EDA Report", explorative=True)
    results['eda_report'] = profile

    # --- Preprocessing ---
    if task in ["Regression", "Classification"]:
        y = df[target_column].copy()
        X = df.drop(columns=[target_column]).copy()

        if task == "Classification":
            le = LabelEncoder()
            y = le.fit_transform(y.astype(str))
            results['label_encoder'] = le
    else: # Clustering
        X = df.copy()
        y = None

    for col in X.columns:
        if X[col].dtype == 'object' or X[col].dtype.name == 'category':
            X[col] = X[col].astype(str)
            imputer = SimpleImputer(strategy='most_frequent')
            X[col] = imputer.fit_transform(X[[col]]).flatten()
            X[col] = LabelEncoder().fit_transform(X[col])
        else:
            imputer = SimpleImputer(strategy='median')
            X[col] = imputer.fit_transform(X[[col]]).flatten()
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    results['scaler'] = scaler
    results['features'] = X.columns.tolist()
    results['cleaned_data'] = pd.DataFrame(X_scaled, columns=X.columns)
    if y is not None:
        results['cleaned_data'][target_column] = y

    # --- Model Training ---
    if task in ["Regression", "Classification"]:
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        
        models = {
            "RandomForest": RandomForestRegressor() if task == "Regression" else RandomForestClassifier(),
            "XGBoost": XGBRegressor() if task == "Regression" else XGBClassifier(),
            "LightGBM": LGBMRegressor() if task == "Regression" else LGBMClassifier()
        }
        
        leaderboard = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            score = r2_score(y_test, preds) if task == "Regression" else accuracy_score(y_test, preds)
            leaderboard[name] = score
        
        results['leaderboard'] = pd.DataFrame(list(leaderboard.items()), columns=['Model', 'Score']).sort_values('Score', ascending=False).reset_index(drop=True)
        best_model_name = results['leaderboard']['Model'].iloc[0]
        best_model = models[best_model_name].fit(X_train, y_train)

        results.update({"best_model": best_model, "X_test_df": pd.DataFrame(X_test, columns=X.columns), "y_test": y_test})

        # --- Explainability ---
        explainer = shap.Explainer(best_model, X_train)
        shap_values = explainer(X_test)
        results.update({"explainer": explainer, "shap_values": shap_values})

    elif task == "Clustering":
        kmeans = KMeans(n_clusters=config['n_clusters'], random_state=42, n_init='auto')
        clusters = kmeans.fit_predict(X_scaled)
        results['model'] = kmeans
        results['clusters'] = clusters
        df['cluster'] = clusters
        results['clustered_data'] = df
        
    return results

# --- UI Sidebar ---
with st.sidebar:
    st.title("🚀 Definitive AutoML Toolkit")
    st.header("1. Upload Data")
    uploaded_file = st.file_uploader("Upload your CSV", type="csv")

    if uploaded_file:
        df = load_data(uploaded_file)
        st.header("2. Choose Analysis Type")
        task = st.selectbox("Select Task", ["🎯 Classification", "📈 Regression", "🧩 Clustering"])

        config = {"task": task.split(" ")[1]}
        if task != "🧩 Clustering":
            config['target_column'] = st.selectbox("Select Target Column", df.columns)
        else:
            config['n_clusters'] = st.slider("Number of Clusters", 2, 10, 3)

        if st.button("🚀 LAUNCH ANALYSIS", use_container_width=True, type="primary"):
            st.session_state.analysis_complete = False
            with st.spinner("Executing a full, robust analysis... Please wait."):
                st.session_state.results = run_analysis_pipeline(df, config)
                st.session_state.analysis_complete = True
            st.success("Analysis Complete!")
            st.rerun()

# --- Main Page Display ---
if not st.session_state.analysis_complete:
    st.info("Configure your analysis in the sidebar and click 'Launch Analysis' to begin.")
else:
    res = st.session_state.results
    st.header(f"Analysis Dashboard: {st.session_state.results['task']}")
    
    tab_list = ["📊 Data Profile", "🏆 Model Performance", "🧠 Explainability", "📦 Assets"]
    if st.session_state.results['task'] == 'Clustering':
        tab_list = ["📊 Data Profile", "🧩 Clustering Results", "📦 Assets"]
    
    tabs = st.tabs(tab_list)

    with tabs[0]:
        st.header("Automated Exploratory Data Analysis Report")
        st_profile_report(res['eda_report'])

    if res['task'] in ["Classification", "Regression"]:
        with tabs[1]:
            st.header("Model Competition Leaderboard")
            st.dataframe(res['leaderboard'])
            st.subheader(f"Best Model Performance: {res['best_model'].__class__.__name__}")
            y_pred = res['best_model'].predict(res['X_test_df'])
            if res['task'] == 'Classification':
                st.dataframe(pd.DataFrame(classification_report(res['y_test'], y_pred, target_names=res['label_encoder'].classes_, output_dict=True)).transpose())
            else:
                st.metric("R-squared (R²)", f"{r2_score(res['y_test'], y_pred):.4f}")

        with tabs[2]:
            st.header("Model Explainability (XAI)")
            st.subheader("SHAP Summary Plot (Global Feature Importance)")
            summary_fig, ax = plt.subplots()
            shap.summary_plot(res['shap_values'], res['X_test_df'], show=False)
            st.pyplot(summary_fig)

            st.subheader("Partial Dependence Plots (How a feature affects predictions)")
            feature = st.selectbox("Select a feature to analyze:", res['features'])
            pdp_fig, ax = plt.subplots()
            PartialDependenceDisplay.from_estimator(res['best_model'], res['X_test_df'], [feature], ax=ax)
            st.pyplot(pdp_fig)

        with tabs[3]:
            st.header("Downloadable Assets")
            model_bytes = io.BytesIO()
            joblib.dump(res['best_model'], model_bytes)
            st.download_button("⬇️ Download Best Model (.joblib)", data=model_bytes, file_name="best_model.joblib")
            st.download_button("⬇️ Download Cleaned Data (.csv)", data=res['cleaned_data'].to_csv(index=False).encode('utf-8'), file_name="cleaned_data.csv")
    
    if res['task'] == "Clustering":
        with tabs[1]:
            st.header("Clustering Results")
            st.dataframe(res['clustered_data'])

        with tabs[2]:
            st.header("Downloadable Assets")
            st.download_button("⬇️ Download Clustered Data (.csv)", data=res['clustered_data'].to_csv(index=False).encode('utf-8'), file_name="clustered_data.csv")
