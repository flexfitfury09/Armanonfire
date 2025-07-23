import streamlit as st
import pandas as pd
import numpy as np
import joblib
from io import BytesIO

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                             mean_squared_error, mean_absolute_error, r2_score)

from sklearn.linear_model import LogisticRegression, LinearRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, SVR

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.semi_supervised import LabelPropagation, LabelSpreading, SelfTrainingClassifier

import shap
import lime
import lime.lime_tabular

import plotly.express as px
import matplotlib.pyplot as plt

# Deep learning import with safe fallback
try:
    import tensorflow as tf
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Dense, Conv2D, Flatten, LSTM, Reshape
    TF_AVAILABLE = True
except Exception as e:
    TF_AVAILABLE = False
    tf_error_msg = str(e)

st.set_page_config(page_title="Advanced AutoML App", layout="wide")
st.title("🤖 AutoML + Regression + DL + Explainability Toolbox")

# File upload
uploaded_file = st.sidebar.file_uploader("📁 Upload CSV dataset", type="csv")

@st.cache_data
def load_data(file):
    return pd.read_csv(file)

def clean_data(df):
    df = df.dropna(axis=1, how="all").drop_duplicates()
    for c in df.select_dtypes('object'):
        if df[c].nunique() < df.shape[0]*0.5:
            df[c].fillna(df[c].mode()[0], inplace=True)
        else:
            df.drop(columns=[c], inplace=True)
    df.fillna(df.mean(numeric_only=True), inplace=True)
    return df

# Main logic
if uploaded_file:
    df = load_data(uploaded_file)
    st.subheader("🔍 Raw Dataset")
    st.dataframe(df)
    df = clean_data(df)
    st.subheader("🧹 Cleaned Dataset")
    st.dataframe(df)

    # EDA
    st.subheader("📊 Automated EDA")
    st.write("Shape:", df.shape)
    st.write(df.dtypes)
    st.write(df.isna().sum())
    fig, axes = plt.subplots(1, min(5, df.select_dtypes('number').shape[1]), figsize=(15,3))
    for ax, col in zip(axes, df.select_dtypes('number').columns[:5]):
        df[col].hist(ax=ax); ax.set_title(col)
    st.pyplot(fig)

    # Multi-target selection
    all_cols = df.columns.tolist()
    targets = st.multiselect("Select Target Column(s)", all_cols)
    if not targets:
        st.warning("Please select at least one target column.")
        st.stop()

    X = df.drop(columns=targets)
    y = df[targets]

    # Label-encode objects
    for c in X.select_dtypes('object').columns:
        X[c] = LabelEncoder().fit_transform(X[c])
    # Multi-output regression only numeric, but classification handles single target
    is_regression = y.shape[1] > 1 or np.issubdtype(y.dtypes[0], np.number)

    if is_regression and y.shape[1] == 1 and y[targets[0]].nunique() <= 20:
        # single numeric target but few distinct => classification
        is_regression = False

    # Encode classification target
    if not is_regression:
        y = LabelEncoder().fit_transform(y[targets[0]].values)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y.values, test_size=0.3, random_state=42)

    # Choose model
    algo = st.sidebar.selectbox("🔍 Select Algorithm",
        ["Logistic Regression","Decision Tree","Random Forest","Naive Bayes","KNN","SVM",
         "Linear Regression","Decision Tree Regressor","RandomForestRegressor","SVR",
         "KMeans","DBSCAN","Hierarchical Clustering","PCA","t-SNE",
         "Label Propagation","Label Spreading","Self Training"]
    )

    model, show_perf = None, False

    # Supervised/semi-supervised
    if algo in ["Logistic Regression","Decision Tree","Random Forest","Naive Bayes","KNN","SVM",
                "Linear Regression","Decision Tree Regressor","RandomForestRegressor","SVR",
                "Label Propagation","Label Spreading","Self Training"]:
        if algo=="Logistic Regression":
            model = LogisticRegression()
        elif algo=="Decision Tree":
            model = DecisionTreeClassifier()
        elif algo=="Random Forest":
            model = RandomForestClassifier()
        elif algo=="Naive Bayes":
            model = GaussianNB()
        elif algo=="KNN":
            model = KNeighborsClassifier()
        elif algo=="SVM":
            model = SVC(probability=True)
        elif algo=="Linear Regression":
            model = LinearRegression()
        elif algo=="Decision Tree Regressor":
            model = DecisionTreeRegressor()
        elif algo=="RandomForestRegressor":
            model = RandomForestRegressor()
        elif algo=="SVR":
            model = SVR()
        elif algo=="Label Propagation":
            model = LabelPropagation()
        elif algo=="Label Spreading":
            model = LabelSpreading()
        elif algo=="Self Training":
            model = SelfTrainingClassifier(SGDClassifier())

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        show_perf = True

    # Unsupervised and dimension reduction
    elif algo=="KMeans":
        clusters = KMeans(n_clusters=3).fit_predict(X_scaled)
        fig = px.scatter(pd.DataFrame(X_scaled, columns=X.columns), x=X.columns[0],y=X.columns[1],color=clusters, title="KMeans")
        st.plotly_chart(fig)
    elif algo=="DBSCAN":
        st.write(set(DBSCAN().fit_predict(X_scaled)))
    elif algo=="Hierarchical Clustering":
        st.write(set(AgglomerativeClustering(n_clusters=3).fit_predict(X_scaled)))
    elif algo=="PCA":
        pcs = PCA(n_components=2).fit_transform(X_scaled)
        st.plotly_chart(px.scatter(x=pcs[:,0],y=pcs[:,1], title="PCA"))
    elif algo=="t-SNE":
        ts = TSNE(n_components=2).fit_transform(X_scaled)
        st.plotly_chart(px.scatter(x=ts[:,0],y=ts[:,1], title="t-SNE"))

    # Display performance & metrics
    if show_perf:
        st.subheader("📊 Performance Metrics")
        if is_regression:
            if y_train.ndim > 1:
                # Multi-output regression
                mse = mean_squared_error(y_test, y_pred, multioutput='raw_values')
                st.write("MSE per target:", mse)
            else:
                st.write("MAE:", mean_absolute_error(y_test, y_pred))
                st.write("MSE:", mean_squared_error(y_test, y_pred))
                st.write("R²:", r2_score(y_test, y_pred))
        else:
            st.write("Accuracy:", accuracy_score(y_test, y_pred))
            st.dataframe(confusion_matrix(y_test, y_pred))
            st.text(classification_report(y_test, y_pred))

        st.write("Cross‑val score:", cross_val_score(model, X_scaled, y_train, cv=5))

        # Save model
        joblib.dump(model, "model.pkl")
        st.download_button("⬇️ Download Model", open("model.pkl","rb"), file_name="model.pkl")

        # Explainability
        if not is_regression and algo in ["Random Forest", "Decision Tree"]:
            expl = shap.TreeExplainer(model)
            vals = expl.shap_values(X_test)
            shap.summary_plot(vals, X_test, feature_names=X.columns, show=False)
            st.pyplot(bbox_inches='tight')
        elif not is_regression:
            expl = lime.lime_tabular.LimeTabularExplainer(X_train, feature_names=X.columns,
                                                          class_names=[str(c) for c in np.unique(y_train)],
                                                          discretize_continuous=True)
            i= np.random.randint(X_test.shape[0])
            exp = expl.explain_instance(X_test[i], model.predict_proba if hasattr(model,"predict_proba") else model.predict, 
                                        num_features=5)
            st.components.v1.html(exp.as_html(), height=350)

    # Deep learning demos
    st.sidebar.markdown("---")
    st.sidebar.subheader("🧠 Deep Learning Demo")
    dl_mode = st.sidebar.selectbox("DL Demo", ["None", 
                                               is_regression and "DL Regression Toy" or None,
                                               not is_regression and "CNN Toy",
                                               not is_regression and "RNN Toy"])
    if dl_mode and dl_mode!="None":
        if not TF_AVAILABLE:
            st.error("TensorFlow unavailable.\n" + tf_error_msg)
        else:
            st.success("Running " + dl_mode)
            if dl_mode=="CNN Toy":
                Xc = X_scaled[:, :10].reshape(-1,2,5,1)
                yc = tf.keras.utils.to_categorical(y_train if not is_regression else y_train)
                model_dl = Sequential([
                    Conv2D(8,(2,2),activation='relu',input_shape=(2,5,1)),
                    Flatten(),
                    Dense(16,activation='relu'),
                    Dense(yc.shape[1],activation='softmax') if not is_regression else Dense(1)
                ])
            elif dl_mode=="RNN Toy":
                Xt = X_scaled.reshape(-1, X_scaled.shape[1], 1)
                yc = tf.keras.utils.to_categorical(y_train) if not is_regression else y_train
                model_dl = Sequential([
                    LSTM(16, input_shape=(Xt.shape[1],1)),
                    Dense(yc.shape[1] if not is_regression else 1)
                ])
            elif dl_mode=="DL Regression Toy":
                Xt = X_scaled
                yc = y_train
                model_dl = Sequential([
                    Dense(32, activation='relu', input_shape=(Xt.shape[1],)),
                    Dense(16, activation='relu'),
                    Dense(yc.shape[1] if yc.ndim>1 else 1)
                ])
            loss = 'mse' if is_regression else 'categorical_crossentropy'
            model_dl.compile(optimizer='adam', loss=loss, metrics=['mse'] if is_regression else ['accuracy'])
            history = model_dl.fit(Xc if 'Xc' in locals() else Xt, yc, epochs=5, validation_split=0.2, verbose=0)
            st.line_chart(history.history)
            st.write("Final metrics:", history.history)

