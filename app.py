import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import io
import shap

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score, classification_report

# --- App Configuration ---
st.set_page_config(page_title="The Final Answer", page_icon="✅", layout="wide")

# --- Session State ---
st.session_state.setdefault('analysis_done', False)
st.session_state.setdefault('results', {})

# --- Sidebar UI ---
with st.sidebar:
    st.title("✅ The Final Answer")
    st.header("1. Upload Data")
    uploaded_file = st.file_uploader("Upload your CSV", type="csv")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.header("2. Configure & Run")
        target_column = st.selectbox("Select Target Column", df.columns)

        if st.button("🚀 Launch Analysis", use_container_width=True, type="primary"):
            st.session_state.analysis_done = False

            # --- PARANOID PRE-FLIGHT CHECK ---
            with st.spinner("Executing Pre-flight Checks..."):
                try:
                    target_series = df[target_column].dropna()
                    if pd.api.types.is_numeric_dtype(target_series) and target_series.nunique() >= 15:
                        task = "📈 Regression"
                    else:
                        task = "🎯 Classification"
                    st.session_state.task = task
                except Exception as e:
                    st.error(f"Pre-flight check failed: {e}")
                    st.stop()
            st.success(f"Check Complete! Task auto-detected: **{task}**")

            # --- MAIN PROCESSING ---
            with st.spinner("Executing a full, robust analysis... Please wait."):
                y = df[target_column].copy()
                X = df.drop(columns=[target_column]).copy()

                if task == "🎯 Classification":
                    le = LabelEncoder()
                    y = le.fit_transform(y.astype(str))
                    st.session_state.results['label_encoder'] = le

                for col in X.columns:
                    if X[col].dtype == 'object' or X[col].dtype.name == 'category':
                        X[col] = X[col].astype(str) # Ensure all are strings before imputing
                        imputer = SimpleImputer(strategy='most_frequent')
                        X[col] = imputer.fit_transform(X[[col]]).flatten()
                        X[col] = LabelEncoder().fit_transform(X[col])
                    else: # Numeric
                        imputer = SimpleImputer(strategy='median')
                        X[col] = imputer.fit_transform(X[[col]]).flatten()
                
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

                model = RandomForestRegressor(random_state=42) if task == "📈 Regression" else RandomForestClassifier(random_state=42)
                model.fit(X_train, y_train)

                explainer = shap.Explainer(model, X_train)
                shap_values = explainer(X_test)

                st.session_state.results.update({
                    "model": model, "scaler": scaler, "features": X.columns.tolist(),
                    "X_test_df": pd.DataFrame(X_test, columns=X.columns), "y_test": y_test,
                    "explainer": explainer, "shap_values": shap_values,
                })
                st.session_state.analysis_done = True
            st.success("Analysis Complete!")
            st.rerun()

# --- Main Page Display ---
if not st.session_state.analysis_done:
    st.info("Upload a dataset and launch the analysis from the sidebar.")
else:
    res = st.session_state.results
    st.header(f"Analysis Dashboard: {st.session_state.task}")
    
    tab1, tab2, tab3 = st.tabs(["🏆 Performance", "🧠 Explainability", "💡 Simulator & Assets"])

    with tab1:
        st.subheader("Model Evaluation")
        y_pred = res['model'].predict(res['X_test_df'])
        if st.session_state.task == '🎯 Classification':
            st.metric("Accuracy", f"{accuracy_score(res['y_test'], y_pred):.4f}")
            report = classification_report(res['y_test'], y_pred, target_names=res['label_encoder'].classes_, output_dict=True)
            st.dataframe(pd.DataFrame(report).transpose())
        else:
            st.metric("R-squared (R²)", f"{r2_score(res['y_test'], y_pred):.4f}")

    with tab2:
        st.subheader("SHAP Model Explanations")
        st.write("Understand *why* the model makes its predictions.")
        summary_fig, ax = plt.subplots()
        shap.summary_plot(res['shap_values'], res['X_test_df'], show=False)
        st.pyplot(summary_fig)
        
        st.write("**Deconstruct a Single Prediction:**")
        row_idx = st.selectbox("Select a row to explain:", res['X_test_df'].index)
        st.dataframe(res['X_test_df'].iloc[[row_idx]])
        decision_fig = shap.decision_plot(res['explainer'].expected_value, res['shap_values'][row_idx], res['X_test_df'].iloc[row_idx], show=False, new_base_value=True)
        st.pyplot(decision_fig)

    with tab3:
        st.subheader("What-If Scenario Simulator")
        input_data = {}
        for col in res['features']:
            input_data[col] = st.number_input(f"Input for '{col}'", value=0.0, key=f"sim_{col}")
        if st.button("Predict Scenario"):
            input_df = pd.DataFrame([input_data])
            input_scaled = res['scaler'].transform(input_df)
            prediction = res['model'].predict(input_scaled)
            if st.session_state.task == "🎯 Classification":
                prediction = res['label_encoder'].inverse_transform(prediction)
            st.success(f"**Simulated Prediction:** {prediction[0]}")
            
        st.subheader("Downloadable Assets")
        model_bytes = io.BytesIO()
        joblib.dump(res['model'], model_bytes)
        st.download_button("⬇️ Download Model (.joblib)", data=model_bytes, file_name="final_model.joblib")
