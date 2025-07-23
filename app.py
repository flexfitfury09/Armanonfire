import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, r2_score, mean_squared_error
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import zipfile
import tempfile

st.set_page_config(page_title="AutoML Dashboard", layout="wide")

st.title("📊 AutoML + CNN Dashboard with Advanced Features")

tab1, tab2 = st.tabs(["Tabular Data (CSV)", "Image Classification (CNN)"])

# ------------------ TABULAR DATA ------------------
with tab1:
    uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Preview of Dataset", df.head())

        target = st.selectbox("Select Target Column", df.columns)
        task_type = st.selectbox("Select Task Type", ["Classification", "Regression"])

        if target and task_type:
            X = df.drop(target, axis=1)
            y = df[target]

            # Handle categorical features by simple encoding (if needed)
            X = pd.get_dummies(X)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            st.write("Select Model")
            if task_type == "Classification":
                model_name = st.selectbox("Model", ["Random Forest", "Logistic Regression", "SVM", "Decision Tree"])
            else:
                model_name = st.selectbox("Model", ["Random Forest Regressor", "Linear Regression", "SVR", "Decision Tree Regressor"])

            use_grid = st.checkbox("Use Hyperparameter Tuning (Grid Search)")

            def get_model_and_params(name, task):
                if task == "Classification":
                    if name == "Random Forest":
                        model = RandomForestClassifier(random_state=42)
                        params = {'n_estimators':[50,100], 'max_depth':[None,10,20]}
                    elif name == "Logistic Regression":
                        model = LogisticRegression(max_iter=1000)
                        params = {'C':[0.01, 0.1, 1, 10]}
                    elif name == "SVM":
                        model = SVC()
                        params = {'C':[0.1,1,10], 'kernel':['linear', 'rbf']}
                    elif name == "Decision Tree":
                        model = DecisionTreeClassifier(random_state=42)
                        params = {'max_depth':[None,5,10,20]}
                else:
                    if name == "Random Forest Regressor":
                        model = RandomForestRegressor(random_state=42)
                        params = {'n_estimators':[50,100], 'max_depth':[None,10,20]}
                    elif name == "Linear Regression":
                        model = LinearRegression()
                        params = {}
                    elif name == "SVR":
                        model = SVR()
                        params = {'C':[0.1,1,10], 'kernel':['linear', 'rbf']}
                    elif name == "Decision Tree Regressor":
                        model = DecisionTreeRegressor(random_state=42)
                        params = {'max_depth':[None,5,10,20]}
                return model, params

            model, param_grid = get_model_and_params(model_name, task_type)

            if use_grid and param_grid:
                st.write("Performing Grid Search for hyperparameter tuning...")
                grid = GridSearchCV(model, param_grid, cv=3, n_jobs=-1, verbose=0)
                grid.fit(X_train, y_train)
                best_model = grid.best_estimator_
                st.success(f"Best Params: {grid.best_params_}")
            else:
                model.fit(X_train, y_train)
                best_model = model

            y_pred = best_model.predict(X_test)

            st.subheader("Model Performance Metrics")

            if task_type == "Classification":
                acc = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average='weighted')
                st.write(f"Accuracy: {acc:.4f}")
                st.write(f"F1 Score: {f1:.4f}")

                st.text("Classification Report")
                st.text(classification_report(y_test, y_pred))

                st.text("Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                st.pyplot(fig)

            else:
                r2 = r2_score(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                st.write(f"R² Score: {r2:.4f}")
                st.write(f"Mean Squared Error: {mse:.4f}")

                fig, ax = plt.subplots()
                ax.scatter(y_test, y_pred, alpha=0.6)
                ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
                ax.set_xlabel("Actual")
                ax.set_ylabel("Predicted")
                ax.set_title("Actual vs Predicted")
                st.pyplot(fig)

# ------------------ IMAGE DATA / CNN ------------------
with tab2:
    st.subheader("Upload a ZIP of your image dataset (train/val split inside)")
    zip_file = st.file_uploader("Upload ZIP File of Image Dataset", type=["zip"])

    if zip_file:
        temp_dir = tempfile.mkdtemp()
        zip_path = os.path.join(temp_dir, "dataset.zip")
        with open(zip_path, "wb") as f:
            f.write(zip_file.read())
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)

        st.success("Dataset extracted successfully!")

        img_height = st.slider("Image Height", 64, 224, 128)
        img_width = st.slider("Image Width", 64, 224, 128)
        batch_size = st.slider("Batch Size", 8, 64, 32)
        epochs = st.slider("Epochs", 1, 20, 5)

        train_path = os.path.join(temp_dir, "train")
        val_path = os.path.join(temp_dir, "val")

        if os.path.exists(train_path) and os.path.exists(val_path):
            datagen = ImageDataGenerator(rescale=1./255)
            train_gen = datagen.flow_from_directory(train_path, target_size=(img_height, img_width), batch_size=batch_size, class_mode='categorical')
            val_gen = datagen.flow_from_directory(val_path, target_size=(img_height, img_width), batch_size=batch_size, class_mode='categorical')

            st.write("Select CNN Architecture")
            arch = st.selectbox("Architecture", ["Simple CNN", "Deeper CNN"])

            model = Sequential()
            if arch == "Simple CNN":
                model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)))
                model.add(MaxPooling2D(2, 2))
                model.add(Flatten())
                model.add(Dense(128, activation='relu'))
                model.add(Dropout(0.5))
                model.add(Dense(train_gen.num_classes, activation='softmax'))
            else:
                model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)))
                model.add(MaxPooling2D(2, 2))
                model.add(Conv2D(64, (3, 3), activation='relu'))
                model.add(MaxPooling2D(2, 2))
                model.add(Conv2D(128, (3,3), activation='relu'))
                model.add(MaxPooling2D(2,2))
                model.add(Flatten())
                model.add(Dense(256, activation='relu'))
                model.add(Dropout(0.5))
                model.add(Dense(train_gen.num_classes, activation='softmax'))

            model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

            early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

            history = model.fit(train_gen, validation_data=val_gen, epochs=epochs, callbacks=[early_stop])

            st.success("Model Training Completed!")

            st.subheader("Training Metrics")

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            ax1.plot(history.history['accuracy'], label='Train Acc')
            ax1.plot(history.history['val_accuracy'], label='Val Acc')
            ax1.legend()
            ax1.set_title("Accuracy")

            ax2.plot(history.history['loss'], label='Train Loss')
            ax2.plot(history.history['val_loss'], label='Val Loss')
            ax2.legend()
            ax2.set_title("Loss")

            st.pyplot(fig)

        else:
            st.error("Train/Val folder structure not found in ZIP.")
