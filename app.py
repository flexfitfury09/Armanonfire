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
import os, tempfile, zipfile, joblib, io
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    r2_score, mean_squared_error, mean_absolute_error,
    f1_score, recall_score, precision_score, roc_curve, auc
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import nbformat
from nbformat.v4 import new_notebook, new_code_cell

st.set_page_config(page_title="AutoML + CNN Dashboard", layout="wide")
st.title("📊 AutoML + CNN Dashboard w/ Auto Cleaning & Docker Support")

tab1, tab2 = st.tabs(["📈 Tabular CSV Data", "📷 Image Classification (CNN)"])

def clean_prepare(df, target, is_class):
    """
    Cleans and prepares the DataFrame for machine learning.
    Handles numerical imputation, one-hot encoding for categorical features,
    and target variable encoding.
    """
    # Create copies to avoid SettingWithCopyWarning
    X = df.drop(target, axis=1).copy()
    y = df[target].copy()

    # Convert 'TotalCharges' to numeric, coercing errors to NaN
    # This is a common issue with this dataset where 'TotalCharges' might have non-numeric strings
    if 'TotalCharges' in X.columns:
        X['TotalCharges'] = pd.to_numeric(X['TotalCharges'], errors='coerce')

    # Impute missing numerical values with the mean
    for col in X.select_dtypes(include=np.number).columns:
        if X[col].isnull().any():
            X[col] = X[col].fillna(X[col].mean())

    # One-hot encode all categorical columns (object and category dtypes)
    # pd.get_dummies automatically handles both numerical and categorical,
    # converting categorical to one-hot and leaving numerical as is.
    # drop_first=True to avoid multicollinearity
    X = pd.get_dummies(X, drop_first=True)

    # Impute missing values in the target variable
    if is_class:
        if y.isnull().any():
            y = y.fillna(y.mode()[0]) # For classification, fill with mode
        # Ensure target is encoded if it's an object type and classification
        if y.dtype == object or not np.issubdtype(y.dtype, np.number):
            le = LabelEncoder()
            y = le.fit_transform(y)
            st.session_state['le_classes'] = le.classes_ # Store classes for later use (e.g., confusion matrix labels)
    else:
        if y.isnull().any():
            y = y.fillna(y.mean()) # For regression, fill with mean

    return X, y

def auto_model(model_name, is_class):
    """Returns the selected machine learning model."""
    if is_class:
        if model_name == "RandomForestClassifier":
            return RandomForestClassifier(random_state=42)
        elif model_name == "LogisticRegression":
            return LogisticRegression(random_state=42, max_iter=1000)
    else: # Regression
        if model_name == "RandomForestRegressor":
            return RandomForestRegressor(random_state=42)
        elif model_name == "LinearRegression":
            return LinearRegression()
    return None # Should not happen with proper selection


def generate_notebook(code_string):
    nb = new_notebook()
    nb.cells.append(new_code_cell(code_string))
    return nb

with tab1:
    st.header("📊 Tabular Data Analysis & Modeling")
    csv = st.file_uploader("Upload CSV", type=["csv"])
    if 'le_classes' not in st.session_state:
        st.session_state['le_classes'] = None

    if csv:
        df = pd.read_csv(csv)
        st.subheader("Data Preview")
        st.write(df.head())

        if st.checkbox("Show detailed data analysis"):
            st.subheader("Data Description")
            st.write(df.describe())
            st.subheader("Data Types")
            st.write(df.dtypes)
            st.subheader("Missing Values Count")
            st.write(df.isnull().sum())

            st.subheader("Value Counts for Categorical Features")
            for col in df.select_dtypes(include=['object', 'category']).columns:
                st.write(f"**{col}**:")
                st.write(df[col].value_counts())

            st.subheader("Distribution of Numerical Features")
            numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
            if numerical_cols:
                for col in numerical_cols:
                    fig, ax = plt.subplots()
                    sns.histplot(df[col].dropna(), kde=True, ax=ax)
                    ax.set_title(f'Distribution of {col}')
                    st.pyplot(fig)


        target = st.text_input("Enter target column name", key="target_col_input")
        if target and target in df.columns:
            # Determine if it's a classification or regression problem
            # Assume classification if target is object or has few unique values
            is_class = df[target].dtype == object or df[target].nunique() < 20

            try:
                X, y = clean_prepare(df, target, is_class)
                st.write("Cleaned Features (X) Preview:")
                st.write(X.head())
                st.write("Cleaned Target (y) Preview:")
                st.write(y.head())
                st.write(f"Problem Type: {'Classification' if is_class else 'Regression'}")

                if st.checkbox("Scale features (using StandardScaler)"):
                    scaler = StandardScaler()
                    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
                    st.write("Scaled Features (X) Preview:")
                    st.write(X.head())

                test_size = st.slider("Test size (%)", 10, 50, 20, key="test_size_slider")
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42)

                st.subheader("Model Selection")
                model_options = ["RandomForestClassifier", "LogisticRegression"] if is_class else ["RandomForestRegressor", "LinearRegression"]
                selected_model_name = st.selectbox("Choose a model", model_options, key="model_selector")

                model = auto_model(selected_model_name, is_class)

                if selected_model_name in ["RandomForestClassifier", "RandomForestRegressor"] and st.checkbox("Enable GridSearch (for Random Forest only)"):
                    params = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]}
                    st.info("Running GridSearchCV. This might take some time...")
                    gs = GridSearchCV(model, params, cv=3, n_jobs=-1, verbose=1)
                    gs.fit(X_train, y_train)
                    model = gs.best_estimator_
                    st.success(f"Best parameters found by GridSearchCV: {gs.best_params_}")
                    st.write(f"Best score: {gs.best_score_:.4f}")

                st.subheader("Model Training")
                model.fit(X_train, y_train)
                st.success(f"{selected_model_name} trained successfully!")

                pred_train = model.predict(X_train)
                pred_test = model.predict(X_test)

                st.subheader("📊 Model Performance")
                metrics = {}
                if is_class:
                    metrics = {
                        "Train Accuracy": accuracy_score(y_train, pred_train),
                        "Test Accuracy": accuracy_score(y_test, pred_test),
                        "F1 Score (Weighted)": f1_score(y_test, pred_test, average='weighted'),
                        "Recall Score (Weighted)": recall_score(y_test, pred_test, average='weighted'),
                        "Precision Score (Weighted)": precision_score(y_test, pred_test, average='weighted')
                    }
                    st.json(metrics)
                    st.subheader("Classification Report")
                    st.text(classification_report(y_test, pred_test, target_names=st.session_state['le_classes'] if st.session_state['le_classes'] is not None else None))

                    st.subheader("Confusion Matrix")
                    fig_cm, ax_cm = plt.subplots()
                    sns.heatmap(confusion_matrix(y_test, pred_test), annot=True, fmt="d", ax=ax_cm,
                                xticklabels=st.session_state['le_classes'], yticklabels=st.session_state['le_classes'])
                    ax_cm.set_xlabel('Predicted')
                    ax_cm.set_ylabel('True')
                    st.pyplot(fig_cm)

                    st.subheader("ROC Curve")
                    if hasattr(model, "predict_proba"):
                        y_pred_proba = model.predict_proba(X_test)[:, 1]
                        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
                        roc_auc = auc(fpr, tpr)

                        fig_roc, ax_roc = plt.subplots()
                        ax_roc.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
                        ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                        ax_roc.set_xlim([0.0, 1.0])
                        ax_roc.set_ylim([0.0, 1.05])
                        ax_roc.set_xlabel('False Positive Rate')
                        ax_roc.set_ylabel('True Positive Rate')
                        ax_roc.set_title('Receiver Operating Characteristic (ROC) Curve')
                        ax_roc.legend(loc="lower right")
                        st.pyplot(fig_roc)
                    else:
                        st.info("ROC Curve not available for this model (no predict_proba).")


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

                if selected_model_name in ["RandomForestClassifier", "RandomForestRegressor"] and hasattr(model, 'feature_importances_'):
                    st.subheader("Feature Importance")
                    feature_importance = pd.DataFrame({
                        'feature': X.columns,
                        'importance': model.feature_importances_
                    }).sort_values('importance', ascending=False)

                    fig_fi, ax_fi = plt.subplots(figsize=(10, 6))
                    sns.barplot(x='importance', y='feature', data=feature_importance, ax=ax_fi)
                    ax_fi.set_title('Feature Importance')
                    ax_fi.set_xlabel('Importance')
                    ax_fi.set_ylabel('Feature')
                    st.pyplot(fig_fi)

                # Prepare full cleaned dataset with target for Excel
                # Ensure the target column is named correctly in cleaned_df for excel export
                cleaned_df = X.copy()
                cleaned_df[target] = y # Add encoded/cleaned target back for export

                # Create dataframe for metrics for Excel sheet
                df_metrics = pd.DataFrame([metrics])

                # Excel download with multiple sheets: Cleaned data + Metrics
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

                # CSV download (metrics only)
                csv_buffer_metrics = df_metrics.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download CSV Report (Metrics only)", csv_buffer_metrics, file_name="report_metrics.csv", mime="text/csv")


                # Jupyter notebook download (metrics calculation snippet)
                # This snippet is for a basic RandomForestClassifier, can be expanded if needed
                code_str = f"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np

# Load data (assuming the CSV is in the same directory)
df = pd.read_csv("{csv.name}")

# --- Data Cleaning and Preparation (as in Streamlit app) ---
target_col = "{target}"
is_classification_problem = df[target_col].dtype == object or df[target_col].nunique() < 20

X = df.drop(target_col, axis=1).copy()
y = df[target_col].copy()

if 'TotalCharges' in X.columns:
    X['TotalCharges'] = pd.to_numeric(X['TotalCharges'], errors='coerce')

for col in X.select_dtypes(include=np.number).columns:
    if X[col].isnull().any():
        X[col] = X[col].fillna(X[col].mean())

X = pd.get_dummies(X, drop_first=True)

if is_classification_problem:
    if y.isnull().any():
        y = y.fillna(y.mode()[0])
    if y.dtype == object or not np.issubdtype(y.dtype, np.number):
        le = LabelEncoder()
        y = le.fit_transform(y)
else:
    if y.isnull().any():
        y = y.fillna(y.mean())

# --- Feature Scaling (if applied in Streamlit app) ---
# Uncomment and run if you selected 'Scale features' in the app
# scaler = StandardScaler()
# X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# --- Train-Test Split ---
test_split_ratio = {test_size}/100
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split_ratio, random_state=42)

# --- Model Training and Evaluation ---
print(f"--- Model: {selected_model_name} ---")

if is_classification_problem:
    model = RandomForestClassifier(random_state=42) if "{selected_model_name}" == "RandomForestClassifier" else LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    pred_test = model.predict(X_test)

    print("\\nClassification Metrics:")
    print("Test Accuracy:", accuracy_score(y_test, pred_test))
    print("F1 Score (Weighted):", f1_score(y_test, pred_test, average='weighted'))
    print("Recall Score (Weighted):", recall_score(y_test, pred_test, average='weighted'))
    print("Precision Score (Weighted):", precision_score(y_test, pred_test, average='weighted'))
    print("\\nClassification Report:")
    print(classification_report(y_test, pred_test))

else: # Regression
    model = RandomForestRegressor(random_state=42) if "{selected_model_name}" == "RandomForestRegressor" else LinearRegression()
    model.fit(X_train, y_train)
    pred_test = model.predict(X_test)

    print("\\nRegression Metrics:")
    print("Test R²:", r2_score(y_test, pred_test))
    print("MAE (Mean Absolute Error):", mean_absolute_error(y_test, pred_test))
    print("MSE (Mean Squared Error):", mean_squared_error(y_test, pred_test))
    print("RMSE (Root Mean Squared Error):", np.sqrt(mean_squared_error(y_test, pred_test)))

# Feature Importance (for tree-based models)
if hasattr(model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    print("\\nFeature Importance:")
    print(feature_importance.head())
                """.strip()
                nb = generate_notebook(code_str)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".ipynb") as tmp_nb:
                    nbformat.write(nb, tmp_nb.name)
                    with open(tmp_nb.name, 'rb') as f:
                        st.download_button("📘 Download Jupyter Notebook (Full Pipeline)", f.read(), file_name="full_ml_pipeline.ipynb")
                os.remove(tmp_nb.name) # Clean up temp file


                if st.checkbox("💾 Save trained model"):
                    fname = st.text_input("Filename to save model (e.g., model.joblib)", "model.joblib", key="model_save_filename")
                    if fname:
                        try:
                            joblib.dump(model, fname)
                            st.success(f"Model saved to `{fname}` successfully!")
                        except Exception as e:
                            st.error(f"Error saving model: {e}")
            except Exception as e:
                st.error(f"An error occurred during processing: {e}")
                st.info("Please check your target column name and data quality.")
        elif target:
            st.error("Target column not found in the dataset. Please enter a valid column name.")

with tab2:
    st.header("📷 Image Classification (CNN)")
    zip_file = st.file_uploader("Upload ZIP (with 'train/' and 'val/' folders containing image categories)", type=["zip"])
    if zip_file:
        tmp = tempfile.mkdtemp()
        zip_path = os.path.join(tmp, "images.zip")
        with open(zip_path, "wb") as f:
            f.write(zip_file.getbuffer())
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(tmp)

        img_h = st.slider("Image height", 64, 224, 128, key="img_h_slider")
        img_w = st.slider("Image width", 64, 224, 128, key="img_w_slider")
        bs = st.slider("Batch size", 8, 64, 32, key="batch_size_slider")
        ep = st.slider("Epochs", 1, 20, 5, key="epochs_slider")

        tp, vp = os.path.join(tmp, "train"), os.path.join(tmp, "val")
        if os.path.isdir(tp) and os.path.isdir(vp):
            st.info(f"Train directory: {tp}")
            st.info(f"Validation directory: {vp}")
            try:
                datagen = ImageDataGenerator(rescale=1.0 / 255)
                trg = datagen.flow_from_directory(
                    tp,
                    target_size=(img_h, img_w),
                    batch_size=bs,
                    class_mode="categorical",
                    shuffle=True
                )
                vlg = datagen.flow_from_directory(
                    vp,
                    target_size=(img_h, img_w),
                    batch_size=bs,
                    class_mode="categorical",
                    shuffle=False
                )

                st.write(f"Found {trg.num_classes} classes in training data: {trg.class_indices}")
                st.write(f"Found {vlg.num_classes} classes in validation data: {vlg.class_indices}")

                if trg.num_classes == 0 or vlg.num_classes == 0:
                    st.error("No classes found in one or both directories. Ensure subfolders represent classes.")
                else:
                    model = Sequential([
                        Conv2D(32, (3, 3), activation='relu', input_shape=(img_h, img_w, 3)),
                        MaxPooling2D(2, 2),
                        Conv2D(64, (3, 3), activation='relu'), # Added another conv layer
                        MaxPooling2D(2, 2),
                        Flatten(),
                        Dense(128, activation='relu'),
                        Dropout(0.5),
                        Dense(trg.num_classes, activation="softmax")
                    ])
                    model.compile(optimizer=Adam(), loss="categorical_crossentropy", metrics=["accuracy"])
                    st.subheader("CNN Model Summary")
                    model.summary(print_fn=lambda x: st.text(x)) # Display model summary in Streamlit

                    st.info("Starting CNN training...")
                    hist = model.fit(trg, validation_data=vlg, epochs=ep)
                    st.success("CNN Training completed ✅")

                    st.subheader("CNN Training History")
                    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
                    ax[0].plot(hist.history['accuracy'], label='Train Accuracy')
                    ax[0].plot(hist.history['val_accuracy'], label='Val Accuracy')
                    ax[0].set_title('Accuracy')
                    ax[0].set_xlabel('Epoch')
                    ax[0].set_ylabel('Accuracy')
                    ax[0].legend()

                    ax[1].plot(hist.history['loss'], label='Train Loss')
                    ax[1].plot(hist.history['val_loss'], label='Val Loss')
                    ax[1].set_title('Loss')
                    ax[1].set_xlabel('Epoch')
                    ax[1].set_ylabel('Loss')
                    ax[1].legend()

                    st.pyplot(fig)

                    if st.checkbox("💾 Save CNN model"):
                        fname_cnn = st.text_input("Filename to save CNN model (e.g., cnn_model.h5)", "cnn_model.h5", key="cnn_model_save_filename")
                        if fname_cnn:
                            try:
                                model.save(fname_cnn)
                                st.success(f"CNN Model saved to `{fname_cnn}` successfully!")
                            except Exception as e:
                                st.error(f"Error saving CNN model: {e}")
            except Exception as e:
                st.error(f"An error occurred during CNN processing: {e}")
                st.info("Please ensure your ZIP file has 'train/' and 'val/' directories with image subfolders.")
        else:
            st.error("Missing 'train/' or 'val/' folder inside the ZIP file. Please ensure your ZIP structure is like: `your_zip.zip/train/category1/image.jpg` and `your_zip.zip/val/category1/image.jpg`.")
        # Clean up temporary directory
        import shutil
        shutil.rmtree(tmp)
