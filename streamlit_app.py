# Paketleri yüklemek için requirements.txt dosyasını çalıştır
#
# !pip install -r requirements.txt



# Gerekli kütüphaneleri yükle
#
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
import joblib
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(layout="wide")
st.title("\U0001F4CA Makine Öğrenimi ile Sahtecilik Tespiti")



# Veri Yükleme
#
data_file = st.file_uploader("Veri dosyasını yükle (CSV formatında)", type=["csv"])

if data_file:
    data = pd.read_csv(data_file)

    st.subheader("\U0001F4C4 Veri Önizlemesi")
    st.dataframe(data.head())

    if len(data) > 50000:
        data = data.sample(n=50000, random_state=42)

    numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = data.select_dtypes(include='object').columns

    imputer = SimpleImputer(strategy='median')
    data_numeric_imputed = pd.DataFrame(imputer.fit_transform(data[numeric_cols]), columns=numeric_cols)

    label_encoders = {}
    data_categorical_encoded = pd.DataFrame()
    for column in categorical_cols:
        le = LabelEncoder()
        data_categorical_encoded[column] = le.fit_transform(data[column].astype(str))
        label_encoders[column] = le

    data_imputed = pd.concat([data_numeric_imputed, data_categorical_encoded], axis=1)

    if "fraud_bool" not in data_imputed.columns:
        st.error("Veri kümesinde 'fraud_bool' adlı hedef değişken bulunamadı.")
        st.stop()

    all_features = data_imputed.drop("fraud_bool", axis=1).columns.tolist()
    selected_features = st.multiselect("Kullanılacak özellikleri seçin:", all_features, default=all_features)

    X = data_imputed[selected_features]
    y = data_imputed["fraud_bool"]

    st.subheader("⚖️ SMOTE Dengeleme")
    st.text(f"Önce: {Counter(y)}")
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    st.text(f"Sonra: {Counter(y_resampled)}")

    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model_options = {
        "Logistic Regression": LogisticRegression(max_iter=3000, solver='saga', n_jobs=-1, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        "Decision Tree": DecisionTreeClassifier(max_depth=3, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=20, max_depth=3, random_state=42, n_jobs=-1),
        "XGBoost": XGBClassifier(n_estimators=20, max_depth=3, random_state=42, verbosity=0, n_jobs=-1, use_label_encoder=False),
        "LightGBM": LGBMClassifier(n_estimators=20, max_depth=3, random_state=42, n_jobs=-1)
    }

    st.subheader("\U0001F916 Kullanılacak Modelleri Seçin")
    selected_models = st.multiselect("Model Seçin:", list(model_options.keys()), default=list(model_options.keys()))

    results = []
    all_roc_curves = []
    trained_models = {}

    for name in selected_models:
        model = model_options[name]
        model.fit(X_train_scaled, y_train)
        y_test_pred = model.predict(X_test_scaled)

        test_acc = accuracy_score(y_test, y_test_pred)
        test_roc_auc = roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:, 1])
        report = classification_report(y_test, y_test_pred, output_dict=True)

        results.append({
            "Model": name,
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
    st.subheader("\U0001F4CB Model Performans Sonuçları")
    st.dataframe(results_df)

    st.subheader("\U0001F4C9 ROC Eğrisi Karşılaştırması")
    fig_roc, ax_roc = plt.subplots()
    for name, fpr, tpr in all_roc_curves:
        ax_roc.plot(fpr, tpr, label=name)
    ax_roc.plot([0, 1], [0, 1], linestyle='--', color='gray')
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("ROC AUC Eğrisi")
    ax_roc.legend()
    st.pyplot(fig_roc)

    st.subheader("\U0001F4C8 Confusion Matrix ve Classification Report")
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
        st.subheader("\U0001F4AB Özellik Önem Sıralaması (Random Forest)")
        rf = trained_models["Random Forest"]
        importances = rf.feature_importances_
        indices = np.argsort(importances)[::-1][:10]
        features = X.columns[indices]

        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(x=importances[indices], y=features, palette="viridis", ax=ax)
        ax.set_title("Random Forest - En Önemli 10 Özellik")
        st.pyplot(fig)

    st.subheader("\U0001F4BE Modeli Kaydet")
    model_to_save = st.selectbox("Hangi modeli kaydetmek istersiniz?", list(trained_models.keys()))
    if st.button("Kaydet"):
        joblib.dump(trained_models[model_to_save], f"{model_to_save.replace(' ', '_')}_model.pkl")
        st.success(f"{model_to_save} modeli kaydedildi.")

else:
    st.warning("Lütfen bir veri dosyası yükleyin.")
