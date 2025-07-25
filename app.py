import streamlit as st
import pandas as pd
import numpy as np
import os, tempfile, zipfile, joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    r2_score, mean_squared_error, mean_absolute_error
)
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer, LabelEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

st.set_page_config(page_title="AutoML + CNN Dashboard", layout="wide")
st.title("📊 AutoML + CNN Dashboard w/ Auto Cleaning & Docker Support")

tab1, tab2 = st.tabs(["📈 Tabular CSV Data", "📷 Image Classification (CNN)"])

# -- Utilities --
def clean_prepare(df, target, is_class):
    st.info("Starting data cleaning & preprocessing…")
    X = df.drop(target, axis=1)
    y = df[target]
    missing = df.isnull().sum()
    if missing.any():
        st.warning("Missing data detected and auto-filled!")
    X = X.select_dtypes(include=[np.number]).fillna(X.mean())
    if is_class:
        y = y.fillna(y.mode()[0])
    else:
        y = y.fillna(y.mean())
    X = pd.get_dummies(X)
    st.success("Features one-hot encoded.")
    if is_class and (y.dtype == object or not np.issubdtype(y.dtype, np.number)):
        y = LabelEncoder().fit_transform(y)
        st.success("Target label-encoded.")
    return X, y

def auto_model(is_class):
    if is_class:
        st.info("Using default RandomForestClassifier")
        return RandomForestClassifier(random_state=42)
    else:
        st.info("Using default RandomForestRegressor")
        return RandomForestRegressor(random_state=42)

# -- Tab 1: Tabular data --
with tab1:
    csv = st.file_uploader("Upload CSV", type=["csv"])
    if csv:
        df = pd.read_csv(csv)
        st.write("Data preview", df.head())
        if st.checkbox("Show data analysis"):
            st.write(df.info(), df.describe(), df.dtypes)
        target = st.text_input("Enter target column name")
        if target and target in df.columns:
            is_class = df[target].dtype == object or df[target].nunique() < 20
            X, y = clean_prepare(df, target, is_class)
            if st.checkbox("Scale features"):
                X = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns)
            tr = st.slider("Test size (%)", 10, 50, 20)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=tr/100, random_state=42)
            model = auto_model(is_class)
            if st.checkbox("Enable GridSearch (RF only)"):
                params = {'n_estimators':[50,100], 'max_depth':[None,10]}
                gs = GridSearchCV(model, params, cv=3, n_jobs=-1)
                gs.fit(X_train, y_train)
                model = gs.best_estimator_
                st.success(f"Best params: {gs.best_params_}")
            model.fit(X_train, y_train)
            pred = model.predict(X_test)

            st.subheader("Results")
            if is_class:
                st.write("Accuracy:", accuracy_score(y_test, pred))
                st.text(classification_report(y_test, pred))
                fig, ax = plt.subplots()
                sns.heatmap(confusion_matrix(y_test, pred), annot=True, fmt="d", ax=ax)
                st.pyplot(fig)
            else:
                st.write("R²:", r2_score(y_test, pred))
                st.write("MSE:", mean_squared_error(y_test, pred))
                fig, ax = plt.subplots()
                ax.scatter(y_test, pred, alpha=0.6); st.pyplot(fig)

            if st.checkbox("Save trained model"):
                fname = st.text_input("Filename", "model.joblib")
                joblib.dump(model, fname)
                st.success(f"Saved to {fname}")

            loaded = st.file_uploader("Load a .joblib model")
            if loaded:
                mod = joblib.load(loaded)
                st.success("Loaded model ready!")

# -- Tab 2: CNN for image classification --
with tab2:
    zip_file = st.file_uploader("Upload ZIP (with train/val folders)", type=["zip"])
    if zip_file:
        tmp = tempfile.mkdtemp()
        zname = os.path.join(tmp, "z.zip")
        open(zname, "wb").write(zip_file.getbuffer())
        zipfile.ZipFile(zname).extractall(tmp)
        img_h = st.slider("Image height", 64, 224, 128)
        img_w = st.slider("Image width", 64, 224, 128)
        bs = st.slider("Batch size", 8, 64, 32)
        ep = st.slider("Epochs", 1, 20, 5)
        tp, vp = os.path.join(tmp,"train"), os.path.join(tmp,"val")
        if os.path.isdir(tp) and os.path.isdir(vp):
            datagen = ImageDataGenerator(rescale=1.0/255)
            trg = datagen.flow_from_directory(tp, target_size=(img_h,img_w), batch_size=bs, class_mode="categorical")
            vlg = datagen.flow_from_directory(vp, target_size=(img_h,img_w), batch_size=bs, class_mode="categorical")
            model = Sequential([
                Conv2D(32,(3,3),activation='relu',input_shape=(img_h,img_w,3)),
                MaxPooling2D(2,2),
                Flatten(), Dense(128,activation='relu'), Dropout(0.5),
                Dense(trg.num_classes, activation="softmax")
            ])
            model.compile(Adam(), loss="categorical_crossentropy", metrics=["accuracy"])
            hist = model.fit(trg, validation_data=vlg, epochs=ep)
            st.success("Training completed")
            fig, ax = plt.subplots(1,2, figsize=(10,4))
            ax[0].plot(hist.history['accuracy'], label='train_acc')
            ax[0].plot(hist.history['val_accuracy'], label='val_acc')
            ax[1].plot(hist.history['loss'], label='train_loss')
            ax[1].plot(hist.history['val_loss'], label='val_loss')
            ax[0].legend(); ax[1].legend(); st.pyplot(fig)
            if st.checkbox("Save CNN model"):
                fname = st.text_input("Filename", "model.h5")
                model.save(fname); st.success(f"Saved {fname}")
        else:
            st.error("Missing 'train/' or 'val/' folder in your zip")

