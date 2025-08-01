# import streamlit as st
# import pandas as pd
# import numpy as np
# import os, tempfile, zipfile, joblib
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
# from sklearn.linear_model import LinearRegression, LogisticRegression
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.metrics import (
#     classification_report, confusion_matrix, accuracy_score,
#     r2_score, mean_squared_error, mean_absolute_error
# )
# from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer, LabelEncoder
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.models import Sequential, load_model
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
# from tensorflow.keras.optimizers import Adam

# st.set_page_config(page_title="AutoML + CNN Dashboard", layout="wide")
# st.title("📊 AutoML + CNN Dashboard w/ Auto Cleaning & Docker Support")

# tab1, tab2 = st.tabs(["📈 Tabular CSV Data", "📷 Image Classification (CNN)"])

# # -- Utilities --
# def clean_prepare(df, target, is_class):
#     st.info("Starting data cleaning & preprocessing…")
#     X = df.drop(target, axis=1)
#     y = df[target]
#     missing = df.isnull().sum()
#     if missing.any():
#         st.warning("Missing data detected and auto-filled!")
#     X = X.select_dtypes(include=[np.number]).fillna(X.mean())
#     if is_class:
#         y = y.fillna(y.mode()[0])
#     else:
#         y = y.fillna(y.mean())
#     X = pd.get_dummies(X)
#     st.success("Features one-hot encoded.")
#     if is_class and (y.dtype == object or not np.issubdtype(y.dtype, np.number)):
#         y = LabelEncoder().fit_transform(y)
#         st.success("Target label-encoded.")
#     return X, y

# def auto_model(is_class):
#     if is_class:
#         st.info("Using default RandomForestClassifier")
#         return RandomForestClassifier(random_state=42)
#     else:
#         st.info("Using default RandomForestRegressor")
#         return RandomForestRegressor(random_state=42)

# # -- Tab 1: Tabular data --
# with tab1:
#     csv = st.file_uploader("Upload CSV", type=["csv"])
#     if csv:
#         df = pd.read_csv(csv)
#         st.write("Data preview", df.head())
#         if st.checkbox("Show data analysis"):
#             st.write(df.info(), df.describe(), df.dtypes)
#         target = st.text_input("Enter target column name")
#         if target and target in df.columns:
#             is_class = df[target].dtype == object or df[target].nunique() < 20
#             X, y = clean_prepare(df, target, is_class)
#             if st.checkbox("Scale features"):
#                 X = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns)
#             tr = st.slider("Test size (%)", 10, 50, 20)
#             X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=tr/100, random_state=42)
#             model = auto_model(is_class)
#             if st.checkbox("Enable GridSearch (RF only)"):
#                 params = {'n_estimators':[50,100], 'max_depth':[None,10]}
#                 gs = GridSearchCV(model, params, cv=3, n_jobs=-1)
#                 gs.fit(X_train, y_train)
#                 model = gs.best_estimator_
#                 st.success(f"Best params: {gs.best_params_}")
#             model.fit(X_train, y_train)
#             pred = model.predict(X_test)

#             st.subheader("Results")
#             if is_class:
#                 st.write("Accuracy:", accuracy_score(y_test, pred))
#                 st.text(classification_report(y_test, pred))
#                 fig, ax = plt.subplots()
#                 sns.heatmap(confusion_matrix(y_test, pred), annot=True, fmt="d", ax=ax)
#                 st.pyplot(fig)
#             else:
#                 st.write("R²:", r2_score(y_test, pred))
#                 st.write("MSE:", mean_squared_error(y_test, pred))
#                 fig, ax = plt.subplots()
#                 ax.scatter(y_test, pred, alpha=0.6); st.pyplot(fig)

#             if st.checkbox("Save trained model"):
#                 fname = st.text_input("Filename", "model.joblib")
#                 joblib.dump(model, fname)
#                 st.success(f"Saved to {fname}")

#             loaded = st.file_uploader("Load a .joblib model")
#             if loaded:
#                 mod = joblib.load(loaded)
#                 st.success("Loaded model ready!")

# # -- Tab 2: CNN for image classification --
# with tab2:
#     zip_file = st.file_uploader("Upload ZIP (with train/val folders)", type=["zip"])
#     if zip_file:
#         tmp = tempfile.mkdtemp()
#         zname = os.path.join(tmp, "z.zip")
#         open(zname, "wb").write(zip_file.getbuffer())
#         zipfile.ZipFile(zname).extractall(tmp)
#         img_h = st.slider("Image height", 64, 224, 128)
#         img_w = st.slider("Image width", 64, 224, 128)
#         bs = st.slider("Batch size", 8, 64, 32)
#         ep = st.slider("Epochs", 1, 20, 5)
#         tp, vp = os.path.join(tmp,"train"), os.path.join(tmp,"val")
#         if os.path.isdir(tp) and os.path.isdir(vp):
#             datagen = ImageDataGenerator(rescale=1.0/255)
#             trg = datagen.flow_from_directory(tp, target_size=(img_h,img_w), batch_size=bs, class_mode="categorical")
#             vlg = datagen.flow_from_directory(vp, target_size=(img_h,img_w), batch_size=bs, class_mode="categorical")
#             model = Sequential([
#                 Conv2D(32,(3,3),activation='relu',input_shape=(img_h,img_w,3)),
#                 MaxPooling2D(2,2),
#                 Flatten(), Dense(128,activation='relu'), Dropout(0.5),
#                 Dense(trg.num_classes, activation="softmax")
#             ])
#             model.compile(Adam(), loss="categorical_crossentropy", metrics=["accuracy"])
#             hist = model.fit(trg, validation_data=vlg, epochs=ep)
#             st.success("Training completed")
#             fig, ax = plt.subplots(1,2, figsize=(10,4))
#             ax[0].plot(hist.history['accuracy'], label='train_acc')
#             ax[0].plot(hist.history['val_accuracy'], label='val_acc')
#             ax[1].plot(hist.history['loss'], label='train_loss')
#             ax[1].plot(hist.history['val_loss'], label='val_loss')
#             ax[0].legend(); ax[1].legend(); st.pyplot(fig)
#             if st.checkbox("Save CNN model"):
#                 fname = st.text_input("Filename", "model.h5")
#                 model.save(fname); st.success(f"Saved {fname}")
#         else:
#             st.error("Missing 'train/' or 'val/' folder in your zip")



# Second App.py code 




# import streamlit as st
# import pandas as pd
# import numpy as np
# import os, tempfile, zipfile, joblib, io
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
# from sklearn.linear_model import LinearRegression, LogisticRegression
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.metrics import (
#     classification_report, confusion_matrix, accuracy_score,
#     r2_score, mean_squared_error, mean_absolute_error,
#     f1_score, recall_score, precision_score
# )
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
# from tensorflow.keras.optimizers import Adam
# import nbformat
# from nbformat.v4 import new_notebook, new_code_cell

# st.set_page_config(page_title="AutoML + CNN Dashboard", layout="wide")
# st.title("📊 AutoML + CNN Dashboard w/ Auto Cleaning & Docker Support")

# tab1, tab2 = st.tabs(["📈 Tabular CSV Data", "📷 Image Classification (CNN)"])

# def clean_prepare(df, target, is_class):
#     X = df.drop(target, axis=1)
#     y = df[target]
#     X = X.select_dtypes(include=[np.number]).fillna(X.mean())
#     y = y.fillna(y.mode()[0] if is_class else y.mean())
#     X = pd.get_dummies(X)
#     if is_class and (y.dtype == object or not np.issubdtype(y.dtype, np.number)):
#         y = LabelEncoder().fit_transform(y)
#     return X, y

# def auto_model(is_class):
#     return RandomForestClassifier(random_state=42) if is_class else RandomForestRegressor(random_state=42)

# def generate_notebook(code_string):
#     nb = new_notebook()
#     nb.cells.append(new_code_cell(code_string))
#     return nb

# with tab1:
#     csv = st.file_uploader("Upload CSV", type=["csv"])
#     if csv:
#         df = pd.read_csv(csv)
#         st.write("Data preview", df.head())
#         if st.checkbox("Show data analysis"):
#             st.write(df.describe())
#             st.write(df.dtypes)

#         target = st.text_input("Enter target column name")
#         if target and target in df.columns:
#             is_class = df[target].dtype == object or df[target].nunique() < 20
#             X, y = clean_prepare(df, target, is_class)

#             if st.checkbox("Scale features"):
#                 X = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns)

#             test_size = st.slider("Test size (%)", 10, 50, 20)
#             X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42)

#             model = auto_model(is_class)
#             if st.checkbox("Enable GridSearch (RF only)"):
#                 params = {'n_estimators': [50, 100], 'max_depth': [None, 10]}
#                 gs = GridSearchCV(model, params, cv=3, n_jobs=-1)
#                 gs.fit(X_train, y_train)
#                 model = gs.best_estimator_
#                 st.success(f"Best parameters: {gs.best_params_}")

#             model.fit(X_train, y_train)
#             pred_train = model.predict(X_train)
#             pred_test = model.predict(X_test)

#             st.subheader("📊 Model Performance")
#             if is_class:
#                 metrics = {
#                     "Train Accuracy": accuracy_score(y_train, pred_train),
#                     "Test Accuracy": accuracy_score(y_test, pred_test),
#                     "F1 Score": f1_score(y_test, pred_test, average='weighted'),
#                     "Recall Score": recall_score(y_test, pred_test, average='weighted'),
#                     "Precision Score": precision_score(y_test, pred_test, average='weighted')
#                 }
#                 st.json(metrics)
#                 st.text(classification_report(y_test, pred_test))
#                 fig, ax = plt.subplots()
#                 sns.heatmap(confusion_matrix(y_test, pred_test), annot=True, fmt="d", ax=ax)
#                 st.pyplot(fig)
#             else:
#                 metrics = {
#                     "Train R²": r2_score(y_train, pred_train),
#                     "Test R²": r2_score(y_test, pred_test),
#                     "MAE": mean_absolute_error(y_test, pred_test),
#                     "MSE": mean_squared_error(y_test, pred_test)
#                 }
#                 st.json(metrics)
#                 fig, ax = plt.subplots()
#                 ax.scatter(y_test, pred_test, alpha=0.6)
#                 st.pyplot(fig)

#             df_metrics = pd.DataFrame([metrics])

#             # Excel download (fixed)
#             excel_buffer = io.BytesIO()
#             with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
#                 df_metrics.to_excel(writer, index=False)
#             excel_buffer.seek(0)
#             st.download_button("📥 Download Excel Report", data=excel_buffer,
#                                file_name="report.xlsx",
#                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

#             # CSV download
#             csv_buffer = df_metrics.to_csv(index=False).encode('utf-8')
#             st.download_button("📥 Download CSV Report", csv_buffer, file_name="report.csv", mime="text/csv")

#             # Jupyter notebook download
#             code_str = f"""
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

# df = pd.read_csv("{csv.name}")
# X = df.drop("{target}", axis=1).select_dtypes(include=[float, int]).fillna(df.mean())
# y = df["{target}"].fillna(df["{target}"].mode()[0])
# from sklearn.preprocessing import LabelEncoder
# y = LabelEncoder().fit_transform(y)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size={test_size}/100, random_state=42)
# model = RandomForestClassifier(random_state=42)
# model.fit(X_train, y_train)
# pred = model.predict(X_test)
# print("Accuracy:", accuracy_score(y_test, pred))
# print("F1 Score:", f1_score(y_test, pred, average='weighted'))
# print("Recall:", recall_score(y_test, pred, average='weighted'))
# print("Precision:", precision_score(y_test, pred, average='weighted'))
#             """.strip()
#             nb = generate_notebook(code_str)
#             with tempfile.NamedTemporaryFile(delete=False, suffix=".ipynb") as tmp_nb:
#                 nbformat.write(nb, tmp_nb.name)
#                 with open(tmp_nb.name, 'rb') as f:
#                     st.download_button("📘 Download Jupyter Notebook", f.read(), file_name="report.ipynb")

#             if st.checkbox("💾 Save trained model"):
#                 fname = st.text_input("Filename", "model.joblib")
#                 if fname:
#                     joblib.dump(model, fname)
#                     st.success(f"Model saved to {fname}")

# # CNN TAB (unchanged, working)
# with tab2:
#     zip_file = st.file_uploader("Upload ZIP (with train/val folders)", type=["zip"])
#     if zip_file:
#         tmp = tempfile.mkdtemp()
#         zip_path = os.path.join(tmp, "images.zip")
#         with open(zip_path, "wb") as f:
#             f.write(zip_file.getbuffer())
#         zipfile.ZipFile(zip_path).extractall(tmp)

#         img_h = st.slider("Image height", 64, 224, 128)
#         img_w = st.slider("Image width", 64, 224, 128)
#         bs = st.slider("Batch size", 8, 64, 32)
#         ep = st.slider("Epochs", 1, 20, 5)

#         tp, vp = os.path.join(tmp, "train"), os.path.join(tmp, "val")
#         if os.path.isdir(tp) and os.path.isdir(vp):
#             datagen = ImageDataGenerator(rescale=1.0 / 255)
#             trg = datagen.flow_from_directory(tp, target_size=(img_h, img_w), batch_size=bs, class_mode="categorical")
#             vlg = datagen.flow_from_directory(vp, target_size=(img_h, img_w), batch_size=bs, class_mode="categorical")

#             model = Sequential([
#                 Conv2D(32, (3, 3), activation='relu', input_shape=(img_h, img_w, 3)),
#                 MaxPooling2D(2, 2),
#                 Flatten(),
#                 Dense(128, activation='relu'),
#                 Dropout(0.5),
#                 Dense(trg.num_classes, activation="softmax")
#             ])
#             model.compile(optimizer=Adam(), loss="categorical_crossentropy", metrics=["accuracy"])
#             hist = model.fit(trg, validation_data=vlg, epochs=ep)

#             st.success("Training completed ✅")

#             fig, ax = plt.subplots(1, 2, figsize=(12, 4))
#             ax[0].plot(hist.history['accuracy'], label='Train Accuracy')
#             ax[0].plot(hist.history['val_accuracy'], label='Val Accuracy')
#             ax[0].set_title('Accuracy')
#             ax[0].legend()

#             ax[1].plot(hist.history['loss'], label='Train Loss')
#             ax[1].plot(hist.history['val_loss'], label='Val Loss')
#             ax[1].set_title('Loss')
#             ax[1].legend()

#             st.pyplot(fig)

#             if st.checkbox("💾 Save CNN model"):
#                 fname = st.text_input("Filename", "cnn_model.h5")
#                 if fname:
#                     model.save(fname)
#                     st.success(f"Saved model to: {fname}")
#         else:
#             st.error("Missing 'train/' or 'val/' folder inside the ZIP.")

import streamlit as st
import pandas as pd
import numpy as np
import os, io, json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, VotingClassifier, VotingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score, KFold
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    r2_score, mean_squared_error, mean_absolute_error,
    f1_score, recall_score, precision_score, roc_curve, auc,
    log_loss
)
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
import nbformat
from nbformat.v4 import new_notebook, new_code_cell
from joblib import dump, load
import shap

# --- Streamlit Page Configuration ---
st.set_page_config(page_title="Advanced AutoML Dashboard", layout="wide")
st.title("📊 Advanced AutoML Dashboard for Tabular Data")

# --- Helper Functions ---
@st.cache_data(show_spinner="Loading data...")
def load_data(csv_file):
    """Loads CSV data and caches it."""
    return pd.read_csv(csv_file)

@st.cache_data(show_spinner="Cleaning and preparing data...")
def clean_prepare(df, target, is_class, impute_strategy_num):
    """
    Cleans and prepares the DataFrame for machine learning.
    This function is cached to avoid re-running on every widget interaction.
    """
    X = df.drop(target, axis=1).copy()
    y = df[target].copy()

    # Special handling for 'TotalCharges'
    if 'TotalCharges' in X.columns:
        X['TotalCharges'] = pd.to_numeric(X['TotalCharges'], errors='coerce')
        # Fill NA after conversion
        if impute_strategy_num == "Mean":
            X['TotalCharges'] = X['TotalCharges'].fillna(X['TotalCharges'].mean())
        else:
            X['TotalCharges'] = X['TotalCharges'].fillna(X['TotalCharges'].median())

    # Imputation for numerical features
    num_cols = X.select_dtypes(include=np.number).columns
    if impute_strategy_num == "Mean":
        X[num_cols] = X[num_cols].fillna(X[num_cols].mean())
    else: # Median
        X[num_cols] = X[num_cols].fillna(X[num_cols].median())

    # Imputation for categorical features
    cat_cols = X.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        if X[col].isnull().any():
            X[col] = X[col].fillna(X[col].mode()[0])

    # One-hot encoding
    X = pd.get_dummies(X, drop_first=True)

    original_y_name = y.name
    original_y_index = y.index

    le = None
    if is_class:
        if y.isnull().any():
            y = y.fillna(y.mode()[0])
        if y.dtype == object or not np.issubdtype(y.dtype, np.number):
            le = LabelEncoder()
            y_transformed = le.fit_transform(y)
            y = pd.Series(y_transformed, index=original_y_index, name=original_y_name)
    else:
        if y.isnull().any():
            y = y.fillna(y.mean())
        y = pd.Series(y, index=original_y_index, name=original_y_name)
    
    return X, y, le

def get_model_and_params(model_name, is_class):
    """
    Returns the selected model, default parameters, and a GridSearch/RandomSearch
    parameter grid for the selected model.
    """
    model, param_grid = None, {}
    if is_class:
        if model_name == "RandomForestClassifier":
            model = RandomForestClassifier(random_state=42)
            param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]}
        elif model_name == "LogisticRegression":
            model = LogisticRegression(random_state=42, max_iter=1000)
            param_grid = {'C': [0.1, 1, 10]}
        elif model_name == "VotingClassifier":
            estimators = [('rf', RandomForestClassifier(random_state=42)), ('lr', LogisticRegression(random_state=42, max_iter=1000))]
            model = VotingClassifier(estimators=estimators, voting='hard')
        elif model_name == "XGBClassifier":
            model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
            param_grid = {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1]}
        elif model_name == "SVC":
            model = SVC(random_state=42, probability=True)
            param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
    else:  # Regression
        if model_name == "RandomForestRegressor":
            model = RandomForestRegressor(random_state=42)
            param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]}
        elif model_name == "LinearRegression":
            model = LinearRegression()
        elif model_name == "VotingRegressor":
            estimators = [('rf', RandomForestRegressor(random_state=42)), ('lr', LinearRegression())]
            model = VotingRegressor(estimators=estimators)
        elif model_name == "XGBRegressor":
            model = XGBRegressor(random_state=42)
            param_grid = {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1]}
    
    return model, param_grid

def detect_outliers_iqr(df, threshold=1.5):
    """Detects and returns a list of rows with outliers using the IQR method."""
    outlier_indices = []
    for col in df.select_dtypes(include=np.number).columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index
        outlier_indices.extend(outliers)
    return list(set(outlier_indices))

def generate_notebook(code_string):
    """Generates an .ipynb file from a code string."""
    nb = new_notebook()
    nb.cells.append(new_code_cell(code_string))
    return nb

# --- Main App Logic ---
st.header("📊 Advanced AutoML Dashboard for Tabular Data")
csv = st.file_uploader("Upload CSV", type=["csv"])

# Initialize session state for caching results
if 'le_classes' not in st.session_state: st.session_state['le_classes'] = None
if 'metrics' not in st.session_state: st.session_state['metrics'] = {}
if 'model' not in st.session_state: st.session_state['model'] = None
if 'trained' not in st.session_state: st.session_state['trained'] = False
if 'X' not in st.session_state: st.session_state['X'] = None
if 'y' not in st.session_state: st.session_state['y'] = None
if 'X_test' not in st.session_state: st.session_state['X_test'] = None
if 'y_test' not in st.session_state: st.session_state['y_test'] = None

if csv:
    df = load_data(csv)
    st.subheader("1. Data Preview & Analysis")
    st.write(df.head())

    with st.expander("Show detailed data analysis"):
        st.subheader("Data Description")
        st.write(df.describe())
        st.subheader("Data Types")
        st.write(df.dtypes)
        st.subheader("Missing Values Count")
        st.write(df.isnull().sum())

        st.subheader("Distribution of Numerical Features")
        numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
        if numerical_cols:
            for col in numerical_cols:
                fig, ax = plt.subplots()
                sns.histplot(df[col].dropna(), kde=True, ax=ax)
                ax.set_title(f'Distribution of {col}')
                st.pyplot(fig)


    target = st.text_input("Enter target column name", "Churn", key="target_col_input")
    if target and target in df.columns:
        
        st.subheader("2. Advanced Data Preprocessing")
        col1, col2, col3 = st.columns(3)
        with col1:
            impute_strategy_num = st.selectbox("Numerical Imputation", ["Mean", "Median"])
        with col2:
            remove_outliers = st.checkbox("Remove Outliers (IQR Method)")
        with col3:
            poly_features_active = st.checkbox("Generate Polynomial Features")
        
        try:
            is_class = df[target].dtype == object or df[target].nunique() < 20
            X, y, le = clean_prepare(df, target, is_class, impute_strategy_num)
            
            if remove_outliers:
                with st.spinner("Detecting and removing outliers..."):
                    outlier_indices = detect_outliers_iqr(X)
                    if outlier_indices:
                        X = X.drop(outlier_indices).reset_index(drop=True)
                        y = y.drop(outlier_indices).reset_index(drop=True)
                        st.info(f"Removed {len(outlier_indices)} outliers.")
                    else:
                        st.info("No outliers were detected.")
            
            if poly_features_active:
                with st.spinner("Generating polynomial features..."):
                    poly = PolynomialFeatures(degree=2, include_bias=False)
                    X_poly = poly.fit_transform(X.select_dtypes(include=np.number))
                    poly_feature_names = poly.get_feature_names_out(X.select_dtypes(include=np.number).columns)
                    X = pd.DataFrame(X_poly, columns=poly_feature_names)
                    st.info("Polynomial features generated.")
            
            st.session_state['X'] = X
            st.session_state['y'] = y
            if le:
                st.session_state['le_classes'] = le.classes_
            else:
                st.session_state['le_classes'] = None

            st.write("Cleaned & Preprocessed Features (X) Preview:")
            st.write(X.head())
            st.write("Cleaned & Preprocessed Target (y) Preview:")
            st.write(y.head())
            st.write(f"Problem Type: {'Classification' if is_class else 'Regression'}")
            
            st.subheader("3. Model Configuration & Training")
            
            model_options_class = ["RandomForestClassifier", "LogisticRegression", "XGBClassifier", "SVC", "VotingClassifier"]
            model_options_reg = ["RandomForestRegressor", "LinearRegression", "XGBRegressor", "VotingRegressor"]
            model_options = model_options_class if is_class else model_options_reg
            selected_model_name = st.selectbox("Choose a model", model_options, key="model_selector")
            
            scaling_applied = st.checkbox("Scale features (using StandardScaler)", key="scaling_checkbox")
            if scaling_applied:
                scaler = StandardScaler()
                X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
            else:
                X_scaled = X
            
            model, default_param_grid = get_model_and_params(selected_model_name, is_class)
            
            search_method = None
            custom_params = {}
            if default_param_grid:
                search_method = st.radio("Choose hyperparameter tuning method:", ["None", "GridSearch", "RandomSearch"], key="search_method_radio")
                if search_method != "None":
                    with st.expander(f"Customize {search_method} Parameters"):
                        st.write("Define your own parameter grid in JSON format.")
                        st.write(f"Example for {selected_model_name}:")
                        st.json(default_param_grid)
                        custom_param_json = st.text_area("Enter JSON string for parameters:", json.dumps(default_param_grid), height=150)
                        try:
                            custom_params = json.loads(custom_param_json)
                        except json.JSONDecodeError:
                            st.error("Invalid JSON format. Please check your syntax.")
                            custom_params = default_param_grid
            
            cross_val_active = st.checkbox("Enable K-Fold Cross-Validation", key="cv_checkbox")
            n_splits = 5
            if cross_val_active:
                n_splits = st.slider("Number of folds (k)", 2, 10, 5, key="k_folds")
                st.info(f"Cross-Validation is enabled with {n_splits} folds. This will take longer to run and requires significant CPU resources.")
            else:
                test_size = st.slider("Test size (%)", 10, 50, 20, key="test_size_slider")

            if st.button("Run Model Training"):
                st.session_state['trained'] = False
                st.session_state['metrics'] = {}
                st.session_state['model'] = None
                st.session_state['X_test'] = None
                st.session_state['y_test'] = None

                with st.spinner(f"Training {selected_model_name}... This may take a while."):
                    metrics = {}
                    
                    if not cross_val_active:
                        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size/100, random_state=42)
                        st.session_state['X_test'] = X_test
                        st.session_state['y_test'] = y_test

                        if search_method == "GridSearch":
                            with st.spinner("Starting GridSearchCV..."):
                                gs = GridSearchCV(model, custom_params, cv=3, n_jobs=1, verbose=1)
                                gs.fit(X_train, y_train)
                                model = gs.best_estimator_
                                st.success(f"Best parameters found by GridSearchCV: {gs.best_params_}")
                                st.write(f"Best score: {gs.best_score_:.4f}")
                        elif search_method == "RandomSearch":
                            with st.spinner("Starting RandomizedSearchCV..."):
                                rs = RandomizedSearchCV(model, custom_params, n_iter=10, cv=3, n_jobs=1, random_state=42, verbose=1)
                                rs.fit(X_train, y_train)
                                model = rs.best_estimator_
                                st.success(f"Best parameters found by RandomizedSearchCV: {rs.best_params_}")
                                st.write(f"Best score: {rs.best_score_:.4f}")
                        else:
                            model.fit(X_train, y_train)
                        
                        st.success(f"{selected_model_name} trained successfully!")
                        
                        pred_train = model.predict(X_train)
                        pred_test = model.predict(X_test)
                        
                        st.subheader("📊 Model Performance (on Test Set)")
                        
                        if is_class:
                            metrics = {
                                "Train Accuracy": accuracy_score(y_train, pred_train),
                                "Test Accuracy": accuracy_score(y_test, pred_test),
                                "F1 Score (Weighted)": f1_score(y_test, pred_test, average='weighted', zero_division=0),
                                "Recall Score (Weighted)": recall_score(y_test, pred_test, average='weighted', zero_division=0),
                                "Precision Score (Weighted)": precision_score(y_test, pred_test, average='weighted', zero_division=0)
                            }
                            if hasattr(model, "predict_proba"):
                                try:
                                    y_pred_proba = model.predict_proba(X_test)
                                    metrics["Log Loss"] = log_loss(y_test, y_pred_proba)
                                    if len(np.unique(y)) == 2:
                                        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba[:, 1])
                                        metrics["AUC"] = auc(fpr, tpr)
                                except Exception as e:
                                    st.warning(f"Could not compute some classification metrics: {e}")

                            st.json(metrics)
                            st.subheader("Classification Report")
                            report = classification_report(y_test, pred_test, target_names=st.session_state.get('le_classes'), zero_division=0)
                            st.text(report)

                            st.subheader("Confusion Matrix")
                            fig_cm, ax_cm = plt.subplots()
                            labels = st.session_state.get('le_classes')
                            sns.heatmap(confusion_matrix(y_test, pred_test), annot=True, fmt="d", ax=ax_cm, xticklabels=labels, yticklabels=labels)
                            ax_cm.set_xlabel('Predicted')
                            ax_cm.set_ylabel('True')
                            st.pyplot(fig_cm)
                            
                            if hasattr(model, "predict_proba") and len(np.unique(y)) == 2:
                                st.subheader("ROC Curve")
                                fig_roc, ax_roc = plt.subplots()
                                ax_roc.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {metrics["AUC"]:.2f})')
                                ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                                ax_roc.set_title('Receiver Operating Characteristic (ROC) Curve')
                                ax_roc.legend(loc="lower right")
                                st.pyplot(fig_roc)
                            
                        else: # Regression metrics
                            metrics = {
                                "Train R²": r2_score(y_train, pred_train),
                                "Test R²": r2_score(y_test, pred_test),
                                "MAE (Mean Absolute Error)": mean_absolute_error(y_test, pred_test),
                                "MSE (Mean Squared Error)": mean_squared_error(y_test, pred_test),
                                "RMSE (Root Mean Squared Error)": np.sqrt(mean_squared_error(y_test, pred_test))
                            }
                            st.json(metrics)
                            st.subheader("Actual vs Predicted Plot")
                            fig_reg, ax_reg = plt.subplots()
                            ax_reg.scatter(y_test, pred_test, alpha=0.6)
                            ax_reg.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
                            ax_reg.set_xlabel('Actual Values')
                            ax_reg.set_ylabel('Predicted Values')
                            ax_reg.set_title('Actual vs Predicted Values')
                            st.pyplot(fig_reg)
                    
                    else: # Cross-Validation
                        with st.spinner(f"Running {n_splits}-fold cross-validation..."):
                            kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
                            
                            if is_class:
                                cv_scores = cross_val_score(model, X_scaled, y, cv=kf, scoring='accuracy', n_jobs=1)
                                st.write(f"Cross-Validation Accuracy Scores: {cv_scores}")
                                st.success(f"Average Accuracy: {np.mean(cv_scores):.4f} (Standard Deviation: {np.std(cv_scores):.4f})")
                                metrics = {"Average CV Accuracy": np.mean(cv_scores), "CV Std Dev": np.std(cv_scores)}

                            else:
                                cv_scores = cross_val_score(model, X_scaled, y, cv=kf, scoring='r2', n_jobs=1)
                                cv_mse = -cross_val_score(model, X_scaled, y, cv=kf, scoring='neg_mean_squared_error', n_jobs=1)
                                st.write(f"Cross-Validation R² Scores: {cv_scores}")
                                st.write(f"Cross-Validation MSE Scores: {cv_mse}")
                                st.success(f"Average R²: {np.mean(cv_scores):.4f} (Standard Deviation: {np.std(cv_scores):.4f})")
                                metrics = {"Average CV R²": np.mean(cv_scores), "CV R² Std Dev": np.std(cv_scores), "Average CV MSE": np.mean(cv_mse)}
                            
                            st.info("Detailed plots are not available with cross-validation as there is no single test set.")

                    # Common output section after both CV and train-test split
                    if hasattr(model, 'feature_importances_') and selected_model_name not in ["SVC", "LinearRegression", "LogisticRegression"]:
                        st.subheader("Feature Importance")
                        feature_importance = pd.DataFrame({
                            'feature': X.columns,
                            'importance': model.feature_importances_
                        }).sort_values('importance', ascending=False)
                        fig_fi, ax_fi = plt.subplots(figsize=(10, 6))
                        sns.barplot(x='importance', y='feature', data=feature_importance.head(20), ax=ax_fi)
                        ax_fi.set_title('Top 20 Feature Importance')
                        ax_fi.set_xlabel('Importance')
                        ax_fi.set_ylabel('Feature')
                        st.pyplot(fig_fi)
                    
                    st.session_state['model'] = model
                    st.session_state['metrics'] = metrics
                    st.session_state['trained'] = True

        except Exception as e:
            st.error(f"An error occurred during processing: {e}")
            st.info("Please check your target column name and data quality. For complex models, consider using a smaller dataset or fewer features.")
    elif target:
        st.error("Target column not found in the dataset. Please enter a valid column name.")

if st.session_state['trained']:
    st.subheader("4. Model Explainability (SHAP)")
    if st.session_state.get('X_test') is not None and st.session_state.get('y_test') is not None and \
       selected_model_name not in ["VotingClassifier", "VotingRegressor", "SVC", "LinearRegression"]:
        
        st.write("SHAP can be very memory-intensive. Consider a smaller sample size for large datasets.")
        shap_sample_size = st.slider("Select SHAP sample size", 1, min(1000, len(st.session_state['X_test'])), 100)
        X_test_sample = st.session_state['X_test'].sample(n=shap_sample_size, random_state=42)
        
        try:
            with st.spinner(f"Calculating SHAP values for {shap_sample_size} samples..."):
                if "XGB" in selected_model_name or "RandomForest" in selected_model_name:
                    explainer = shap.TreeExplainer(st.session_state['model'])
                else:
                    explainer = shap.Explainer(st.session_state['model'].predict, X_test_sample)

                shap_values = explainer.shap_values(X_test_sample)
                
                st.write("SHAP Summary Plot")
                fig_shap, ax_shap = plt.subplots(figsize=(10, 6))
                if is_class:
                    shap.summary_plot(shap_values[1], X_test_sample, show=False)
                else:
                    shap.summary_plot(shap_values, X_test_sample, show=False)
                st.pyplot(fig_shap)

                st.write("SHAP Waterfall Plot for a single prediction")
                sample_index = st.slider("Select a test sample to explain", 0, len(X_test_sample)-1, 0)
                fig_waterfall, ax_waterfall = plt.subplots(figsize=(10, 6))
                if is_class:
                    shap_obj = explainer(X_test_sample)
                    shap.plots.waterfall(shap_obj[sample_index, :, 1], show=False)
                else:
                    shap_obj = explainer(X_test_sample)
                    shap.plots.waterfall(shap_obj[sample_index], show=False)
                st.pyplot(fig_waterfall)

        except Exception as e:
            st.warning(f"Could not generate SHAP plots: {e}. SHAP is not supported for this model or with this configuration. Try adjusting the sample size.")
            st.info("SHAP is not supported for Voting Classifiers/Regressors or SVC.")
    else:
        st.info("SHAP plots are available for most models with a defined test set. They are not currently supported for Voting Classifiers/Regressors or SVC.")

    st.subheader("5. Download Outputs")
    
    # Download Report
    X = st.session_state['X']
    y = st.session_state['y']
    cleaned_df = X.copy()
    cleaned_df[target] = y
    df_metrics = pd.DataFrame([st.session_state['metrics']])

    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
        cleaned_df.to_excel(writer, sheet_name="Cleaned_Data", index=False)
        df_metrics.to_excel(writer, sheet_name="Performance_Metrics", index=False)
    excel_buffer.seek(0)
    st.download_button(
        label="📥 Download Excel Report (Cleaned Data + Metrics)",
        data=excel_buffer,
        file_name="report_with_data.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    # Download Jupyter Notebook
    notebook_code_lines = [
        "import pandas as pd", "import numpy as np", "import matplotlib.pyplot as plt", "import seaborn as sns",
        "from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV, KFold",
        "from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, VotingClassifier, VotingRegressor",
        "from sklearn.linear_model import LogisticRegression, LinearRegression",
        "from sklearn.svm import SVC", "from xgboost import XGBClassifier, XGBRegressor",
        "from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, r2_score, mean_absolute_error, mean_squared_error, classification_report, confusion_matrix, roc_curve, auc",
        "from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures",
        "from joblib import dump, load", "import shap", "",
        f"print('--- AutoML Pipeline Execution ---')",
        "",
        f"# Load data (assuming the CSV is in the same directory)",
        f"df = pd.read_csv('{csv.name}')",
        "",
        "# --- Data Cleaning and Preparation ---",
        f"target_col = '{target}'",
        f"is_classification_problem = df[target_col].dtype == object or df[target_col].nunique() < 20",
        "",
        "X = df.drop(target_col, axis=1).copy()",
        "y = df[target_col].copy()",
        "",
        "# Special handling for 'TotalCharges'",
        "if 'TotalCharges' in X.columns:",
        "    X['TotalCharges'] = pd.to_numeric(X['TotalCharges'], errors='coerce')",
        "    # Fill NA after conversion",
        f"    impute_strategy_num = '{impute_strategy_num}'",
        "    if impute_strategy_num == 'Mean':",
        "        X['TotalCharges'] = X['TotalCharges'].fillna(X['TotalCharges'].mean())",
        "    else:",
        "        X['TotalCharges'] = X['TotalCharges'].fillna(X['TotalCharges'].median())",
        "",
        "# Imputation for numerical features",
        "num_cols = X.select_dtypes(include=np.number).columns",
        "if impute_strategy_num == 'Mean':",
        "    X[num_cols] = X[num_cols].fillna(X[num_cols].mean())",
        "else:",
        "    X[num_cols] = X[num_cols].fillna(X[num_cols].median())",
        "",
        "# Imputation for categorical features",
        "cat_cols = X.select_dtypes(include=['object', 'category']).columns",
        "for col in cat_cols:",
        "    if X[col].isnull().any():",
        "        X[col] = X[col].fillna(X[col].mode()[0])",
        "",
        "# One-hot encoding",
        "X = pd.get_dummies(X, drop_first=True)",
        "",
        "original_y_name = y.name",
        "original_y_index = y.index",
        "le = None",
        "if is_classification_problem:",
        "    if y.dtype == object or not np.issubdtype(y.dtype, np.number):",
        "        le = LabelEncoder()",
        "        y_transformed = le.fit_transform(y)",
        "        y = pd.Series(y_transformed, index=original_y_index, name=original_y_name)",
        "",
        "# --- Outlier Removal (optional) ---",
        f"if {remove_outliers}:",
        "    def detect_outliers_iqr(df, threshold=1.5):",
        "        outlier_indices = []",
        "        for col in df.select_dtypes(include=np.number).columns:",
        "            Q1 = df[col].quantile(0.25)",
        "            Q3 = df[col].quantile(0.75)",
        "            IQR = Q3 - Q1",
        "            lower_bound = Q1 - threshold * IQR",
        "            upper_bound = Q3 + threshold * IQR",
        "            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index",
        "            outlier_indices.extend(outliers)",
        "        return list(set(outlier_indices))",
        "    outlier_indices = detect_outliers_iqr(X)",
        "    if outlier_indices:",
        "        X = X.drop(outlier_indices).reset_index(drop=True)",
        "        y = y.drop(outlier_indices).reset_index(drop=True)",
        "        print(f'Removed {len(outlier_indices)} outliers.')",
        "    else:",
        "        print('No outliers were detected.')",
        "",
        "# --- Feature Engineering (Polynomial Features) ---",
        f"if {poly_features_active}:",
        "    poly = PolynomialFeatures(degree=2, include_bias=False)",
        "    X_poly = poly.fit_transform(X.select_dtypes(include=np.number))",
        "    poly_feature_names = poly.get_feature_names_out(X.select_dtypes(include=np.number).columns)",
        "    X = pd.DataFrame(X_poly, columns=poly_feature_names)",
        "    print('Polynomial features generated.')",
        "",
        f"print('Cleaned Features (X) Shape:', X.shape)",
        f"print('Cleaned Target (y) Shape:', y.shape)",
        "",
        "# --- Feature Scaling (if applied) ---",
        f"scaling_applied = {scaling_applied}",
        "if scaling_applied:",
        "    scaler = StandardScaler()",
        "    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)",
        "    print('Features scaled.')",
        "else:",
        "    X_scaled = X",
        "",
        f"# --- Model Selection & Training ({selected_model_name}) ---",
        f"selected_model_name = '{selected_model_name}'",
        "model = None",
        "if is_classification_problem:",
        "    if selected_model_name == 'RandomForestClassifier': model = RandomForestClassifier(random_state=42)",
        "    elif selected_model_name == 'LogisticRegression': model = LogisticRegression(random_state=42, max_iter=1000)",
        "    elif selected_model_name == 'VotingClassifier': model = VotingClassifier(estimators=[('rf', RandomForestClassifier(random_state=42)), ('lr', LogisticRegression(random_state=42, max_iter=1000))], voting='hard')",
        "    elif selected_model_name == 'XGBClassifier': model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)",
        "    elif selected_model_name == 'SVC': model = SVC(random_state=42, probability=True)",
        "else:",
        "    if selected_model_name == 'RandomForestRegressor': model = RandomForestRegressor(random_state=42)",
        "    elif selected_model_name == 'LinearRegression': model = LinearRegression()",
        "    elif selected_model_name == 'VotingRegressor': model = VotingRegressor(estimators=[('rf', RandomForestRegressor(random_state=42)), ('lr', LinearRegression())])",
        "    elif selected_model_name == 'XGBRegressor': model = XGBRegressor(random_state=42)",
        "",
    ]
    
    if search_method == "GridSearch":
        notebook_code_lines.extend([
            f"# --- Hyperparameter Tuning (GridSearchCV) ---",
            f"params = {json.dumps(custom_params)}",
            "print('Running GridSearchCV...')",
            f"X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size={test_size/100}, random_state=42)",
            "gs = GridSearchCV(model, params, cv=3, n_jobs=-1, verbose=1)",
            "gs.fit(X_train, y_train)",
            "model = gs.best_estimator_",
            "print(f'Best parameters found: {gs.best_params_}')",
            "print(f'Best score: {gs.best_score_:.4f}')",
            "",
        ])
    elif search_method == "RandomSearch":
        notebook_code_lines.extend([
            f"# --- Hyperparameter Tuning (RandomizedSearchCV) ---",
            f"params = {json.dumps(custom_params)}",
            "print('Running RandomizedSearchCV...')",
            f"X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size={test_size/100}, random_state=42)",
            "rs = RandomizedSearchCV(model, params, n_iter=10, cv=3, n_jobs=-1, random_state=42, verbose=1)",
            "rs.fit(X_train, y_train)",
            "model = rs.best_estimator_",
            "print(f'Best parameters found: {rs.best_params_}')",
            "print(f'Best score: {rs.best_score_:.4f}')",
            "",
        ])
    else: # No hyperparameter search
        notebook_code_lines.extend([
            f"# --- Train-Test Split & Final Evaluation ---",
            f"test_split_ratio = {test_size/100}",
            "X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_split_ratio, random_state=42)",
            "",
            "model.fit(X_train, y_train)",
            "pred_test = model.predict(X_test)",
            "",
        ])

    notebook_code_lines.extend([
        "print('\\nFinal Model Evaluation on Test Set:')",
        "if is_classification_problem:",
        "    print('Test Accuracy:', accuracy_score(y_test, pred_test))",
        "    print('F1 Score (Weighted):', f1_score(y_test, pred_test, average='weighted', zero_division=0))",
        "    print('\\nClassification Report:')",
        "    print(classification_report(y_test, pred_test, target_names=le.classes_ if le else None, zero_division=0))",
        "else:",
        "    print('Test R²:', r2_score(y_test, pred_test))",
        "    print('MAE (Mean Absolute Error):', mean_absolute_error(y_test, pred_test))",
        "    print('RMSE (Root Mean Squared Error):', np.sqrt(mean_squared_error(y_test, pred_test)))",
        "",
        "if hasattr(model, 'feature_importances_'):",
        "    feature_importance = pd.DataFrame({'feature': X.columns, 'importance': model.feature_importances_}).sort_values('importance', ascending=False)",
        "    print('\\nFeature Importance (Top 10):')",
        "    print(feature_importance.head(10))",
        "",
    ])

    notebook_code_str = "\n".join(notebook_code_lines)
    nb = generate_notebook(notebook_code_str)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".ipynb", mode="w") as tmp_nb:
        nbformat.write(nb, tmp_nb)
        tmp_nb.flush()
    
    with open(tmp_nb.name, 'rb') as f:
        st.download_button("📘 Download Jupyter Notebook (Full Pipeline)", f.read(), file_name="full_ml_pipeline.ipynb")
    os.remove(tmp_nb.name)


    if st.checkbox("💾 Save trained model"):
        fname = st.text_input("Filename to save model (e.g., model.joblib)", "model.joblib", key="model_save_filename")
        if fname and st.session_state['model'] is not None:
            try:
                dump(st.session_state['model'], fname)
                st.success(f"Model saved to {fname} successfully!")
            except Exception as e:
                st.error(f"Error saving model: {e}")
