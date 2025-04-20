import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
from collections import Counter

import warnings
import os

warnings.filterwarnings("ignore")

st.set_page_config(layout="wide")
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "Home",
    "Data Exploration",
    "Model Training",
    "Fraud Detector",
    "Müşteri Kontrolü"
])

@st.cache_data
def load_data():
    for root, dirs, files in os.walk(os.path.dirname(os.path.abspath(__file__))):
        for file in files:
            if file.endswith(".csv"):
                csv_path = os.path.join(root, file)
                return pd.read_csv(csv_path)
    st.error("Proje dizininde yüklenebilecek bir .csv dosyası bulunamadı!")
    st.stop()

data2 = load_data()
data2 = data2.copy()
data2['customer_id'] = data2.index  # Tüm veri için customer_id ekle

# 50000 gözlem örneklem alma
if len(data2) > 50000:
    data = data2.sample(n=50000, random_state=42).copy()
else:
    data = data2.copy()
data['customer_id'] = data.index  # Örneklem için customer_id ekle

# Sütun adını karşı repo ile uyumlu hale getir
if "fraud_bool" in data.columns and "fraud" not in data.columns:
    data = data.rename(columns={"fraud_bool": "fraud"})

if "fraud_bool" in data2.columns and "fraud" not in data2.columns:
    data2 = data2.rename(columns={"fraud_bool": "fraud"})

def preprocess_data(df):
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = df.select_dtypes(include='object').columns

    imputer = SimpleImputer(strategy='median')
    data_numeric_imputed = pd.DataFrame(imputer.fit_transform(df[numeric_cols]), columns=numeric_cols)

    label_encoders = {}
    data_categorical_encoded = pd.DataFrame()
    for column in categorical_cols:
        le = LabelEncoder()
        data_categorical_encoded[column] = le.fit_transform(df[column].astype(str))
        label_encoders[column] = le

    data_imputed = pd.concat([data_numeric_imputed, data_categorical_encoded], axis=1)
    return data_imputed, label_encoders

data_imputed, label_encoders = preprocess_data(data)
X = data_imputed.drop("fraud", axis=1)
y = data_imputed["fraud"]
numeric_data = data.select_dtypes(include=[np.number])

if page == "Home":
    st.markdown("""
    <div style="display:flex;align-items:center;justify-content:space-between;">
        <div style="flex:1">
            <h1 style="color:#095484;font-weight:bold;">
                🏦 Makine Öğrenimi ile Sahtecilik Tespiti (Fraud Detection)
            </h1>
            <div style="font-size:18px;">
                <b>
                Makine öğrenimi ve gelişmiş analizlerle sahte işlemleri tespit edin, finansal güvenliğinizi artırın!
                </b>
                <ul>
                    <li>🔍 <b>Etkileşimli Veri Analizi:</b> Verinizi keşfedin ve temel istatistikleri görselleştirin.</li>
                    <li>🤖 <b>Çoklu Model ve Hiperparametre Seçimi:</b> Farklı algoritmaları ve ayarları deneyin.</li>
                    <li>⚖️ <b>SMOTE ile Akıllı Dengeleme:</b> Dengesiz veride daha adil sonuçlar alın.</li>
                    <li>📊 <b>Performans Karşılaştırmaları:</b> ROC, doğruluk, precision, recall ve daha fazlası.</li>
                    <li>📝 <b>Kendi İşlemini Test Et:</b> Sahtecilik tahminini anında öğren!</li>
                </ul>
            </div>
        </div>
        <div style="flex:1;display:flex;justify-content:center;">
            <img src="app_images/fraud_detection.jpg" alt="Fraud Detection Cycle" style="width:300px;height:auto;">
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Toplam İşlem", f"{len(data):,} (örneklem)")
    with col2:
        st.metric("Sahtecilik Oranı", f"%{100 * data['fraud'].mean():.2f}")

    st.markdown("""
    <style>
    .stMarkdown ul {
        margin-bottom: 0;
    }
    </style>
    """, unsafe_allow_html=True)

    st.success(
        "Bu uygulama, MIUUL Data Science Bootcamp final projesi kapsamında geliştirilmiştir. "
        "Geri bildirimleriniz ve katkılarınız için GitHub üzerinden iletişime geçebilirsiniz."
    )

elif page == "Data Exploration":
    st.title("Data Exploration")
    st.subheader("Raw Data")
    with st.expander("Show raw data"):
        st.write(data.head())
        st.write(f"Shape: {data.shape}")
    st.subheader("Distribution of Class Labels")
    st.write(data["fraud"].value_counts())
    fig1, ax1 = plt.subplots()
    sns.countplot(x="fraud", data=data, ax=ax1)
    ax1.set_title("Fraud Distribution")
    st.pyplot(fig1)

    st.subheader("Correlation Heatmap")
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    if len(numeric_data.columns) > 1:
        corr = numeric_data.corr()
        sns.heatmap(corr, annot=False, cmap="viridis", ax=ax2)
    else:
        ax2.text(0.5, 0.5, "Yeterli sayısal sütun yok!", ha='center', va='center')
    st.pyplot(fig2)

    st.subheader("Feature Correlations with Fraud/Not-Fraud")
    if "fraud" in numeric_data.columns and len(numeric_data.columns) > 1:
        corr2 = numeric_data.corr()["fraud"].drop("fraud").sort_values(ascending=False)
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        sns.barplot(x=corr2.values, y=corr2.index, palette='coolwarm', ax=ax3)
        st.pyplot(fig3)
    else:
        st.info("Yeterli sayısal sütun yok.")

elif page == "Model Training":
    st.title("Model Training")
    st.write(
        "Aşağıda dilediğiniz modelleri seçip hiperparametrelerini ayarlayabilir, ardından sonuçları karşılaştırabilirsiniz.")

    st.sidebar.subheader("Model ve Hiperparametre Seçimi")

    models = {}
    model_params = {}

    if st.sidebar.checkbox("Logistic Regression", value=True):
        st.sidebar.markdown("**Logistic Regression Ayarları**")
        lr_c = st.sidebar.slider("C (LR)", 0.01, 10.0, 1.0, 0.01)
        lr_max_iter = st.sidebar.number_input("Max Iter (LR)", 100, 5000, 3000, 100)
        models["Logistic Regression"] = LogisticRegression(
            C=lr_c, max_iter=lr_max_iter, solver='saga', n_jobs=-1, random_state=42
        )
        model_params["Logistic Regression"] = {"C": lr_c, "max_iter": lr_max_iter}

    if st.sidebar.checkbox("KNN", value=True):
        st.sidebar.markdown("**KNN Ayarları**")
        knn_k = st.sidebar.slider("n_neighbors", 1, 20, 5, 1)
        models["KNN"] = KNeighborsClassifier(n_neighbors=knn_k, n_jobs=-1)
        model_params["KNN"] = {"n_neighbors": knn_k}

    if st.sidebar.checkbox("Decision Tree", value=True):
        st.sidebar.markdown("**Decision Tree Ayarları**")
        dt_max_depth = st.sidebar.slider("max_depth (DT)", 1, 20, 3, 1)
        models["Decision Tree"] = DecisionTreeClassifier(max_depth=dt_max_depth, random_state=42)
        model_params["Decision Tree"] = {"max_depth": dt_max_depth}

    if st.sidebar.checkbox("Random Forest", value=True):
        st.sidebar.markdown("**Random Forest Ayarları**")
        rf_estimators = st.sidebar.slider("n_estimators (RF)", 10, 200, 20, 10)
        rf_max_depth = st.sidebar.slider("max_depth (RF)", 1, 20, 3, 1)
        models["Random Forest"] = RandomForestClassifier(
            n_estimators=rf_estimators, max_depth=rf_max_depth, random_state=42, n_jobs=-1
        )
        model_params["Random Forest"] = {"n_estimators": rf_estimators, "max_depth": rf_max_depth}

    if st.sidebar.checkbox("XGBoost", value=True):
        st.sidebar.markdown("**XGBoost Ayarları**")
        xgb_estimators = st.sidebar.slider("n_estimators (XGB)", 10, 200, 20, 10)
        xgb_max_depth = st.sidebar.slider("max_depth (XGB)", 1, 20, 3, 1)
        xgb_learning_rate = st.sidebar.slider("learning_rate (XGB)", 0.01, 0.5, 0.1, 0.01)
        models["XGBoost"] = XGBClassifier(
            n_estimators=xgb_estimators, max_depth=xgb_max_depth, learning_rate=xgb_learning_rate,
            random_state=42, verbosity=0, n_jobs=-1, use_label_encoder=False
        )
        model_params["XGBoost"] = {
            "n_estimators": xgb_estimators, "max_depth": xgb_max_depth, "learning_rate": xgb_learning_rate
        }

    if st.sidebar.checkbox("LightGBM", value=True):
        st.sidebar.markdown("**LightGBM Ayarları**")
        lgbm_estimators = st.sidebar.slider("n_estimators (LGBM)", 10, 200, 20, 10)
        lgbm_max_depth = st.sidebar.slider("max_depth (LGBM)", 1, 20, 3, 1)
        lgbm_learning_rate = st.sidebar.slider("learning_rate (LGBM)", 0.01, 0.5, 0.1, 0.01)
        models["LightGBM"] = LGBMClassifier(
            n_estimators=lgbm_estimators, max_depth=lgbm_max_depth, learning_rate=lgbm_learning_rate,
            random_state=42, n_jobs=-1
        )
        model_params["LightGBM"] = {
            "n_estimators": lgbm_estimators, "max_depth": lgbm_max_depth, "learning_rate": lgbm_learning_rate
        }

    selected_models = list(models.keys())

    st.subheader("SMOTE ile Sınıf Dengesi")
    st.write(f"Before: {Counter(y)}")
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    st.write(f"After: {Counter(y_resampled)}")

    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = []
    all_roc_curves = []
    trained_models = {}

    for name in selected_models:
        model = models[name]
        model.fit(X_train_scaled, y_train)
        y_test_pred = model.predict(X_test_scaled)
        test_acc = accuracy_score(y_test, y_test_pred)
        test_roc_auc = roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:, 1])
        report = classification_report(y_test, y_test_pred, output_dict=True)

        results.append({
            "Model": name,
            **model_params.get(name, {}),
            "Test Accuracy": test_acc,
            "Test ROC AUC": test_roc_auc,
            "Precision": report["weighted avg"]["precision"],
            "Recall": report["weighted avg"]["recall"],
            "F1-score": report["weighted avg"]["f1-score"]
        })

        fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test_scaled)[:, 1])
        all_roc_curves.append((name, fpr, tpr))
        trained_models[name] = model

    results_df = pd.DataFrame(results).sort_values(by="Test ROC AUC", ascending=False)
    st.subheader("Model Performans Sonuçları")
    st.dataframe(results_df)

    st.subheader("ROC Eğrisi Karşılaştırması")
    fig_roc, ax_roc = plt.subplots()
    for name, fpr, tpr in all_roc_curves:
        ax_roc.plot(fpr, tpr, label=name)
    ax_roc.plot([0, 1], [0, 1], linestyle='--', color='gray')
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("ROC AUC Eğrisi")
    ax_roc.legend()
    st.pyplot(fig_roc)

    st.subheader("Confusion Matrix & Classification Report")
    for name in results_df.head(2)["Model"].values:
        model = trained_models[name]
        y_pred = model.predict(X_test_scaled)
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", ax=ax)
        ax.set_title(f"Confusion Matrix: {name}")
        st.pyplot(fig)
        st.text(f"{name} Classification Report:")
        st.text(classification_report(y_test, y_pred))

    if "Random Forest" in trained_models:
        st.subheader("Özellik Önem Sıralaması (Random Forest)")
        rf = trained_models["Random Forest"]
        importances = rf.feature_importances_
        indices = np.argsort(importances)[::-1][:10]
        features = X.columns[indices]
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(x=importances[indices], y=features, palette="viridis", ax=ax)
        ax.set_title("Random Forest - En Önemli 10 Özellik")
        st.pyplot(fig)

elif page == "Fraud Detector":
    st.title("Fraud Detector")
    st.write("Aşağıdaki form ile işlem bilgilerini girip, işlemin sahte olup olmadığını kontrol edebilirsiniz.")
    input_columns = list(X.columns)
    inputs = []
    with st.form(key='fraud_detection_form'):
        for col in input_columns:
            if data[col].dtype == "float64":
                val = st.number_input(col, value=float(data[col].mean()))
            elif data[col].dtype == "int64":
                val = st.number_input(col, value=int(data[col].mean()))
            else:
                example = data[col].unique()[0] if len(data[col].unique()) > 0 else ""
                val = st.text_input(f"{col} (ör: {example})", value=example)
                if val != "" and col in label_encoders:
                    val = label_encoders[col].transform([val])[0]
                else:
                    val = 0
            inputs.append(val)
        submit_button = st.form_submit_button(label='Check your Transaction')

    if submit_button:
        try:
            input_array = np.array([float(val) for val in inputs]).reshape(1, -1)
        except ValueError:
            st.error("Tüm girdileri doldurmalı ve sayısal/kodlanmış olmalı!")
            st.stop()
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        rf = RandomForestClassifier(n_estimators=20, max_depth=3, random_state=42, n_jobs=-1)
        rf.fit(X_train_scaled, y_train)
        input_scaled = scaler.transform(input_array)
        prediction = rf.predict(input_scaled)
        color = "red" if prediction[0] == 1 else "green"
        result_text = "Fraudulent" if prediction[0] == 1 else "Not Fraudulent"
        st.markdown(f"<h3 style='color: {color};'>{result_text}</h3>", unsafe_allow_html=True)

elif page == "Müşteri Kontrolü":
    st.title("Müşteri Bazında Fraud Kontrolü")
    if "customer_id" not in data2.columns:
        st.warning("Veri setinizde 'customer_id' kolonu yok. Bu özelliği kullanmak için müşteri ID'li veri gerekir.")
    else:
        customer_ids = data2["customer_id"].unique()
        selected_id = st.selectbox("Müşteri Seçiniz", customer_ids)
        customer_rows = data2[data2["customer_id"] == selected_id]

        st.write(f"Seçilen müşteri: **{selected_id}**")
        st.write(f"Toplam işlem adedi: {len(customer_rows)}")
        st.write("İşlem detayları:")
        st.dataframe(customer_rows.head())

        # Burada örneklem değil, yine eğitim için örneklem veri (data) ile modeli fit ediyoruz!
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        rf = RandomForestClassifier(n_estimators=20, max_depth=3, random_state=42, n_jobs=-1)
        rf.fit(X_train_scaled, y_train)

        # Müşteri verisini preprocess et
        cust_rows_proc, _ = preprocess_data(customer_rows)
        model_features = list(X.columns)
        X_cust = cust_rows_proc.reindex(columns=model_features, fill_value=0)
        X_cust = X_cust[model_features]

        X_cust_scaled = scaler.transform(X_cust)
        preds = rf.predict(X_cust_scaled)

        fraud_sum = np.sum(preds)
        if fraud_sum > 0:
            st.error(
                f"Bu müşterinin işlemlerinde SAHTECİLİK şüphesi var! (Toplam {fraud_sum} işlem fraud olarak sınıflandı.)")
        else:
            st.success("Bu müşterinin işlemlerinde fraud tespit edilmedi.")

        fraud_ops = customer_rows.iloc[np.where(preds == 1)]
        if not fraud_ops.empty:
            st.write("Fraud olarak işaretlenen işlemler:")
            st.dataframe(fraud_ops)

st.sidebar.markdown("""
---
<div style="font-size: 13px;">
<b>Proje:</b> MIUUL DSMLBC17 Final Projesi - Fraud Detection<br>
<b>Geliştiriciler:</b> Onur KARASÜRMELİ, Kemal BAL, Zeynep YERLİKAYA, Emre YILDIRIM<br>
<b>Grup: 1</b>
</div>
""", unsafe_allow_html=True)