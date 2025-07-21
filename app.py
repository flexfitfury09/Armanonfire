import streamlit as st
import pandas as pd
import numpy as np
import joblib
import io
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import accuracy_score, r2_score, classification_report
from sklearn.cluster import KMeans
from sklearn.inspection import PartialDependenceDisplay

# --- Page Configuration ---
st.set_page_config(page_title="Performance AutoML Platform", page_icon="⚡️", layout="wide")

# --- Helper Functions ---
@st.cache_data
def load_data(file):
    return pd.read_csv(file, encoding='utf-8')

def run_analysis_pipeline(df, config):
    results = {}
    task = config['task']
    
    with st.spinner("Step 1/3: Profiling & Cleaning Data..."):
        # Manual EDA
        eda = {'description': df.describe(), 'missing_values': df.isnull().sum().to_frame('Missing Values')}
        numeric_cols = df.select_dtypes(include=np.number)
        if len(numeric_cols.columns) > 1:
            fig, ax = plt.subplots(); sns.heatmap(numeric_cols.corr(), ax=ax, cmap='viridis'); eda['correlation_heatmap'] = fig
        results['eda_report'] = eda

        # Preprocessing
        if task in ["Regression", "Classification"]:
            target_column = config['target_column']
            y = df[target_column].copy()
            X = df.drop(columns=[target_column]).copy()
            if task == "Classification":
                le = LabelEncoder(); y = le.fit_transform(y.astype(str)); results['label_encoder'] = le
        else:
            X = df.copy(); y = None
        
        original_features = X.columns.tolist()
        for col in original_features:
            if X[col].dtype == 'object' or X[col].dtype.name == 'category':
                X[col] = X[col].astype(str)
                X[col] = SimpleImputer(strategy='most_frequent').fit_transform(X[[col]]).flatten()
                X[col] = LabelEncoder().fit_transform(X[col])
            else:
                X[col] = SimpleImputer(strategy='median').fit_transform(X[[col]]).flatten()
        
        scaler = StandardScaler(); X_scaled = scaler.fit_transform(X)
        results.update({'scaler': scaler, 'features': original_features})

    with st.spinner("Step 2/3: Training & Competing Models..."):
        if task in ["Regression", "Classification"]:
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42)
            models = {
                "LightGBM": LGBMRegressor(random_state=42) if task == "Regression" else LGBMClassifier(random_state=42),
                "RandomForest": RandomForestRegressor(random_state=42) if task == "Regression" else RandomForestClassifier(random_state=42)
            }
            leaderboard = {}
            for name, model in models.items():
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                score = r2_score(y_test, preds) if task == "Regression" else accuracy_score(y_test, preds)
                leaderboard[name] = score
            results['leaderboard'] = pd.DataFrame(list(leaderboard.items()), columns=['Model', 'Score']).sort_values('Score', ascending=False)
            best_model_name = results['leaderboard']['Model'].iloc[0]
            best_model = models[best_model_name]
            results.update({"best_model": best_model, "X_train": X_train, "X_test": X_test, "y_test": y_test})
        
        elif task == "Clustering":
            kmeans = KMeans(n_clusters=config['n_clusters'], random_state=42, n_init='auto')
            clusters = kmeans.fit_predict(X_scaled)
            df['cluster'] = clusters; results['clustered_data'] = df
            
    with st.spinner("Step 3/3: Generating Final Report..."):
        plt.close('all') # Close all open figures
        
    return results

# --- UI Sidebar ---
st.sidebar.title("⚡️ Performance AutoML")
st.sidebar.header("1. Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload your CSV", type="csv")

if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False

if uploaded_file:
    df = load_data(uploaded_file)
    st.sidebar.header("2. Configure Analysis")
    task_options = ["🎯 Classification", "📈 Regression", "🧩 Clustering"]
    task = st.sidebar.selectbox("Select Task", task_options)
    
    config = {"task": task.split(" ")[1]}
    if task != "🧩 Clustering":
        config['target_column'] = st.sidebar.selectbox("Select Target Column", df.columns)
    else:
        config['n_clusters'] = st.sidebar.slider("Number of Clusters", 2, 10, 3)

    if st.sidebar.button("🚀 LAUNCH ANALYSIS", use_container_width=True, type="primary"):
        st.session_state.results = run_analysis_pipeline(df, config)
        st.session_state.analysis_complete = True
else:
    st.session_state.analysis_complete = False

# --- Main Page Display ---
if st.session_state.analysis_complete:
    res = st.session_state.results
    st.header(f"Analysis Dashboard: {res['task']}")
    
    tab_list = ["📊 Data Profile", "🏆 Model Performance", "🧠 Explainability", "📦 Assets"]
    if res['task'] == 'Clustering': tab_list = ["📊 Data Profile", "🧩 Clustering Results"]
    
    tabs = st.tabs(tab_list)

    with tabs[0]:
        st.header("Data Profile")
        st.dataframe(res['eda_report']['description'])
        if 'correlation_heatmap' in res['eda_report']:
            st.subheader("Correlation Heatmap"); st.pyplot(res['eda_report']['correlation_heatmap'])

    if res['task'] != 'Clustering':
        with tabs[1]:
            st.header("Model Leaderboard")
            st.dataframe(res['leaderboard'])
            y_pred = res['best_model'].predict(res['X_test'])
            if res['task'] == 'Classification':
                st.dataframe(pd.DataFrame(classification_report(res['y_test'], y_pred, target_names=res['label_encoder'].classes_, output_dict=True)).transpose())
            else:
                st.metric("R-squared (R²)", f"{r2_score(res['y_test'], y_pred):.4f}")
        with tabs[2]:
            st.header("Model Explainability")
            st.subheader("Partial Dependence Plots")
            feature = st.selectbox("Select feature to analyze:", res['features'])
            pdp_fig, ax = plt.subplots(); PartialDependenceDisplay.from_estimator(res['best_model'], res['X_train'], [res['features'].index(feature)], feature_names=res['features'], ax=ax); st.pyplot(pdp_fig)
        with tabs[3]:
            st.header("Downloadable Assets")
            model_bytes = io.BytesIO(); joblib.dump(res['best_model'], model_bytes)
            st.download_button("⬇️ Download Model", data=model_bytes, file_name="model.joblib")
    else:
        with tabs[1]:
            st.header("Clustering Results")
            st.dataframe(res['clustered_data'])
else:
    st.info("Upload a dataset and launch the analysis from the sidebar to begin.")
