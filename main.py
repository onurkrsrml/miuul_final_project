# Paketleri yüklemek için requirements.txt dosyasını çalıştır
#
# !pip install -r requirements.txt



# Gerekli kütüphaneleri yükle
#
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
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
warnings.filterwarnings("ignore")



# Görselleştirme ayarları
#
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)



# Veri setini oku ve önizle
#
data = pd.read_csv("datasets/Base.csv")

data.head()
data.info()
data.columns
data.describe().T



# Büyük veri setleri için örneklemeyle hızlandır (Burada veri seti 1000000 satır olduğundan yalnıca 50000 satırını alıyoruz)
#
if len(data) > 50000:
    data = data.sample(n=50000, random_state=42)



# Sayısal ve kategorik sütunları ayır
#
numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = data.select_dtypes(include='object').columns



# Sayısal sütunlar için median ile doldurma
#
imputer = SimpleImputer(strategy='median')
data_numeric_imputed = pd.DataFrame(imputer.fit_transform(data[numeric_cols]), columns=numeric_cols)



# Kategorik sütunları encode et
#
label_encoders = {}
data_categorical_encoded = pd.DataFrame()
for column in categorical_cols:
    le = LabelEncoder()
    data_categorical_encoded[column] = le.fit_transform(data[column].astype(str))
    label_encoders[column] = le



# Sayısal ve kategorik verileri birleştir
#
data_imputed = pd.concat([data_numeric_imputed, data_categorical_encoded], axis=1)



# Özellikler ve hedef
#
X = data_imputed.drop("fraud_bool", axis=1)
y = data_imputed["fraud_bool"]



# SMOTE ile dengeleme
#
print("\nSMOTE Uygulanmadan Önce:", Counter(y))
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
print("SMOTE Uygulandıktan Sonra:", Counter(y_resampled))



# Eğitim ve test verisi ayır
#
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)



# Özellik ölçekleme (StandardScaler)
#
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



# Modeller
#
models = {
    "Logistic Regression": LogisticRegression(max_iter=3000, solver='saga', n_jobs=-1, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    "Decision Tree": DecisionTreeClassifier(max_depth=3, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=20, max_depth=3, random_state=42, n_jobs=-1),
    "XGBoost": XGBClassifier(n_estimators=20, max_depth=3, random_state=42, verbosity=0, n_jobs=-1, use_label_encoder=False),
    "LightGBM": LGBMClassifier(n_estimators=20, max_depth=3, random_state=42, n_jobs=-1)
}



# Model performanslarını sakla
#
results = []

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)

    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    train_roc_auc = roc_auc_score(y_train, model.predict_proba(X_train_scaled)[:, 1])
    test_roc_auc = roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:, 1])

    report = classification_report(y_test, y_test_pred, output_dict=True)

    results.append({
        "Model": name,
        "Train Accuracy": train_acc,
        "Test Accuracy": test_acc,
        "Train ROC AUC": train_roc_auc,
        "Test ROC AUC": test_roc_auc,
        "Precision": report["weighted avg"]["precision"],
        "Recall": report["weighted avg"]["recall"],
        "F1-score": report["weighted avg"]["f1-score"]
    })

    print(f"\n{name}:")
    print(f"Train Accuracy: {train_acc:.4f} | Test Accuracy: {test_acc:.4f}")
    print(f"Train ROC AUC: {train_roc_auc:.4f} | Test ROC AUC: {test_roc_auc:.4f}")



# Sonuçları DataFrame olarak yazdır
#
results_df = pd.DataFrame(results)
print("\nModel Karşılaştırmaları:")
print(results_df.sort_values(by="Test ROC AUC", ascending=False))



# Overfitting kontrolü için eğitim ve test skorlarını karşılaştır
#
print("\nOverfitting Kontrolü (Train vs Test ROC AUC):")
for i, row in results_df.iterrows():
    diff = row["Train ROC AUC"] - row["Test ROC AUC"]
    print(f"{row['Model']}: Train-Test ROC AUC Farkı = {diff:.4f}")



# En iyi 2 model için confusion matrix ve classification report
#
best_models = results_df.sort_values(by="Test ROC AUC", ascending=False).head(2)["Model"].values

for name in best_models:
    model = models[name]
    y_pred = model.predict(X_test_scaled)

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
    plt.title(f"Confusion Matrix: {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()
    plt.close()

    print(f"\n{name} Classification Report:")
    print(classification_report(y_test, y_pred))



# Random Forest için feature importance (ilk 10 özellik)
#
rf = models["Random Forest"]
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1][:10]
features = X.columns[indices]



# Özellik önem sıralamasını görselleştir
#
plt.figure(figsize=(10, 5))
sns.barplot(x=importances[indices], y=features, palette="viridis")
plt.title("Random Forest - En Önemli 10 Özellik")
plt.tight_layout()
plt.show()
plt.close()



#
#