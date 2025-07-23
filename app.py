import streamlit as st
import pandas as pd
import numpy as np
import joblib
from io import BytesIO

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.semi_supervised import LabelPropagation, LabelSpreading, SelfTrainingClassifier

import shap
import lime
import lime.lime_tabular

import plotly.express as px
import matplotlib.pyplot as plt

try:
    import tensorflow as tf
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Dense, Conv2D, Flatten, LSTM, Embedding, Reshape
    TF_AVAILABLE = True
except Exception as e:
    TF_AVAILABLE = False
    tf_error_msg = str(e)

st.set_page_config(page_title="Advanced AutoML App", layout="wide")
st.title("🤖 Full AutoML + DL + Explainability App")

st.sidebar.title("📁 Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

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

if uploaded_file:
    df = load_data(uploaded_file)
    st.subheader("✅ Raw Dataset")
    st.dataframe(df)

    df = clean_data(df)
    st.subheader("🧹 Cleaned Dataset")
    st.dataframe(df)

    st.subheader("📊 AutoEDA Summary")
    st.write("Shape:", df.shape)
    st.write("Dtypes:\n", df.dtypes)
    st.write("Missing values:\n", df.isna().sum())
    st.write("Numeric feature histograms:")
    fig, axes = plt.subplots(1, min(5, df.select_dtypes('number').shape[1]), figsize=(15,3))
    for ax, col in zip(axes, df.select_dtypes('number').columns[:5]):
        df[col].hist(ax=ax)
        ax.set_title(col)
    st.pyplot(fig)

    all_cols = df.columns.tolist()
    target = st.selectbox("🎯 Select Target Column", all_cols)
    X = df.drop(columns=[target])
    y = df[target]

    for c in X.select_dtypes('object').columns:
        X[c] = LabelEncoder().fit_transform(X[c])
    if y.dtype == 'object':
        y = LabelEncoder().fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

    st.sidebar.subheader("🔍 Choose Algorithm")
    algos = ["Logistic Regression", "Decision Tree", "Random Forest", "Naive Bayes",
             "KNN", "SVM", "KMeans", "DBSCAN", "Hierarchical Clustering",
             "PCA", "t-SNE", "Label Propagation", "Label Spreading", "Self Training"]
    algo = st.sidebar.selectbox("Algorithm", algos)

    model = None
    show_perf = False
    if algo in ["Logistic Regression","Decision Tree","Random Forest","Naive Bayes","KNN","SVM",
                "Label Propagation","Label Spreading","Self Training"]:
        if algo == "Logistic Regression":
            model = LogisticRegression()
        elif algo == "Decision Tree":
            model = DecisionTreeClassifier()
        elif algo == "Random Forest":
            model = RandomForestClassifier()
        elif algo == "Naive Bayes":
            model = GaussianNB()
        elif algo == "KNN":
            model = KNeighborsClassifier()
        elif algo == "SVM":
            model = SVC(probability=True)
        elif algo == "Label Propagation":
            model = LabelPropagation()
        elif algo == "Label Spreading":
            model = LabelSpreading()
        elif algo == "Self Training":
            base = SGDClassifier()
            model = SelfTrainingClassifier(base)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        show_perf = True

    elif algo == "KMeans":
        clusters = KMeans(n_clusters=3).fit_predict(X_scaled)
        fig = px.scatter(pd.DataFrame(X_scaled, columns=X.columns),
                         x=X.columns[0], y=X.columns[1], color=clusters, title="KMeans Clustering")
        st.plotly_chart(fig)

    elif algo == "DBSCAN":
        labels = DBSCAN().fit_predict(X_scaled)
        st.write("DBSCAN labels:", set(labels))

    elif algo == "Hierarchical Clustering":
        labels = AgglomerativeClustering(n_clusters=3).fit_predict(X_scaled)
        st.write("Hierarchical Clustering labels:", set(labels))

    elif algo == "PCA":
        pcs = PCA(n_components=2).fit_transform(X_scaled)
        fig = px.scatter(x=pcs[:,0], y=pcs[:,1], title="PCA 2D Projection")
        st.plotly_chart(fig)

    elif algo == "t-SNE":
        ts = TSNE(n_components=2).fit_transform(X_scaled)
        fig = px.scatter(x=ts[:,0], y=ts[:,1], title="t-SNE 2D Projection")
        st.plotly_chart(fig)

    if show_perf:
        st.subheader("📊 Performance")
        st.write("Accuracy:", accuracy_score(y_test, y_pred))
        st.write("Confusion Matrix")
        st.dataframe(confusion_matrix(y_test, y_pred))
        st.text(classification_report(y_test, y_pred))

        scores = cross_val_score(model, X_scaled, y, cv=5)
        st.write("Cross‑val scores:", scores, "| Mean:", scores.mean())

        joblib.dump(model, "model.pkl")
        st.download_button("⬇️ Download Model", open("model.pkl", "rb"), file_name="model.pkl")

        if algo == "Random Forest":
            fig = px.bar(x=X.columns, y=model.feature_importances_,
                         title="Feature Importances")
            st.plotly_chart(fig)

            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
            shap.summary_plot(shap_values, X_test, feature_names=X.columns, show=False)
            st.pyplot(bbox_inches='tight')

        else:
            explainer = lime.lime_tabular.LimeTabularExplainer(
                X_train, feature_names=X.columns,
                class_names=[str(c) for c in np.unique(y)], discretize_continuous=True
            )
            idx = np.random.randint(0, X_test.shape[0])
            exp = explainer.explain_instance(X_test[idx], model.predict_proba, num_features=5)
            st.write(f"LIME Explanation for sample #{idx}")
            st.components.v1.html(exp.as_html(), height=350)

    st.sidebar.markdown("---")
    st.sidebar.subheader("🧠 Deep Learning Demo")
    dl = st.sidebar.selectbox("DL Demo:", ["None", "CNN Toy", "RNN Toy"])

    if dl != "None":
        if not TF_AVAILABLE:
            st.error("❌ TensorFlow is unavailable.\nDetails:\n" + tf_error_msg)
        else:
            st.success("✅ TensorFlow loaded—running demo!")
            if dl == "CNN Toy":
                Xc = X_scaled[:, :10].reshape(-1, 2, 5, 1)
                yc = tf.keras.utils.to_categorical(y, num_classes=len(np.unique(y)))
                m = Sequential([
                    Conv2D(8, (2,2), activation='relu', input_shape=(2,5,1)),
                    Flatten(),
                    Dense(16, activation='relu'),
                    Dense(yc.shape[1], activation='softmax')
                ])
            else:
                Xt = X_scaled.reshape(-1, X_scaled.shape[1], 1)
                yt = tf.keras.utils.to_categorical(y)
                m = Sequential([
                    LSTM(16, input_shape=(Xt.shape[1],1)),
                    Dense(yt.shape[1], activation='softmax')
                ])

            m.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            hist = m.fit(Xc if dl=="CNN Toy" else Xt,
                         yc if dl=="CNN Toy" else yt,
                         epochs=5, validation_split=0.2, verbose=0)
            fig = px.line(
                x=list(range(len(hist.history['loss']))),
                y=[hist.history['loss'], hist.history['val_loss']],
                labels={'x':'Epoch','value':'Loss'},
                title="Training & Validation Loss"
            )
            st.plotly_chart(fig)
            st.write("Final Accuracy:", hist.history['accuracy'][-1],
                     "Val Accuracy:", hist.history['val_accuracy'][-1])
