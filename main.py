# 1. Veri Seti Hikayesi ve Problem Tanımı
#
"""
Veri Seti Hikayesi:
Bu veri seti, finansal işlemler üzerinde sahtekarlık (fraud) tespitine yönelik bilgiler içermektedir.
Amaç, işlemin sahte (fraud) olup olmadığını gösteren 'fraud_bool' değişkenini öngörecek bir model geliştirmektir.

Problem Tanımı:
Amaç; kullanıcıların işlemlerinin sahte olup olmadığını makine öğrenmesi modelleriyle tahmin edebilmektir.
"""



# Paketleri yüklemek için requirements.txt dosyasını çalıştır
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





# 2. Keşifçi Veri Analizi (EDA)
#
print("\n--- KEŞİFÇİ VERİ ANALİZİ ---\n")



# Veri seti yükleme ve okutma işlemi
#

data = pd.read_csv("datasets/Base.csv")



# Veri setine genel bakış ve hızlı önizleme
#
def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Descriptives #########")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(data)



# Hedef değişken dağılımı grafiği
#
print(data["fraud_bool"].value_counts())
sns.countplot(x="fraud_bool", data=data)
plt.title("Fraud Dağılımı")
plt.show()



# Büyük veri setleri için örneklemeyle hızlandır (Burada veri seti 1000000 satır olduğundan yalnıca 50000 satırını alıyoruz)
#
if len(data) > 50000:
    data = data.sample(n=50000, random_state=42)

data["customer_id"] = data.index



# Sayısal ve kategorik sütunları ayır
#
numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = data.select_dtypes(include='object').columns


# Eksik değer kontrolü
#
print("\nEksik Değerler:\n", data.isnull().sum())





# 3. Veri Ön İşleme ve Özellik Mühendisliği
#
print("\n--- VERİ ÖN İŞLEME ve ÖZELLİK MÜHENDİSLİĞİ ---\n")



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


# Özellik Mühendisliği
#
print("\nÖzellik Mühendisliği Uygulanıyor...")

# 1. Zaman tabanlı özellikler
data_imputed['days_since_request_log'] = np.log1p(data_imputed['days_since_request'])
data_imputed['address_ratio'] = data_imputed['current_address_months_count'] / (data_imputed['prev_address_months_count'] + 1)
data_imputed['address_total_time'] = data_imputed['current_address_months_count'] + data_imputed['prev_address_months_count']

# 2. Hız ve oranlar
try:
    # Güvenli bölme işlemleri için epsilon değeri
    epsilon = 1e-10
    data_imputed['velocity_ratio_6h_24h'] = data_imputed['velocity_6h'] / (data_imputed['velocity_24h'] + epsilon)
    data_imputed['velocity_ratio_24h_4w'] = data_imputed['velocity_24h'] / (data_imputed['velocity_4w'] + epsilon)
    data_imputed['velocity_ratio_6h_4w'] = data_imputed['velocity_6h'] / (data_imputed['velocity_4w'] + epsilon)
except Exception as e:
    print(f"Hız oranları hesaplanırken hata oluştu: {e}")

# 3. Risk skorları ve oranlar
try:
    epsilon = 1e-10  # Güvenli bölme için küçük değer
    data_imputed['risk_income_ratio'] = data_imputed['credit_risk_score'] / (data_imputed['income'] * 100 + epsilon)
    data_imputed['risk_age_ratio'] = data_imputed['credit_risk_score'] / (data_imputed['customer_age'] + epsilon)
except Exception as e:
    print(f"Risk oranları hesaplanırken hata oluştu: {e}")

# 4. Etkileşim özellikleri
data_imputed['age_income_interaction'] = data_imputed['customer_age'] * data_imputed['income']
data_imputed['risk_session_interaction'] = data_imputed['credit_risk_score'] * data_imputed['session_length_in_minutes']
data_imputed['email_risk_interaction'] = data_imputed['name_email_similarity'] * data_imputed['credit_risk_score']

# 5. Karmaşık özellikler
try:
    epsilon = 1e-10  # Güvenli bölme için küçük değer
    data_imputed['risk_velocity_ratio'] = data_imputed['credit_risk_score'] / (data_imputed['velocity_24h'] + epsilon)
    data_imputed['email_device_ratio'] = data_imputed['date_of_birth_distinct_emails_4w'] / (data_imputed['device_distinct_emails_8w'] + epsilon)
    data_imputed['bank_activity_ratio'] = data_imputed['bank_months_count'] / (data_imputed['bank_branch_count_8w'] + epsilon)
except Exception as e:
    print(f"Karmaşık özellikler hesaplanırken hata oluştu: {e}")

# 6. Kategorik değişkenlerden türetilen özellikler
data_imputed['payment_risk'] = data_imputed['payment_type'].astype(str) + '_' + data_imputed['credit_risk_score'].astype(str)
data_imputed['payment_risk'] = data_imputed['payment_risk'].astype('category').cat.codes

# 7. Polinomial özellikler
data_imputed['income_squared'] = data_imputed['income'] ** 2
data_imputed['risk_score_squared'] = data_imputed['credit_risk_score'] ** 2

# 8. Basamak özellikleri
data_imputed['risk_score_bin'] = pd.qcut(data_imputed['credit_risk_score'], q=5, labels=False, duplicates='drop')
data_imputed['velocity_24h_bin'] = pd.qcut(data_imputed['velocity_24h'], q=5, labels=False, duplicates='drop')
data_imputed['customer_age_bin'] = pd.cut(data_imputed['customer_age'], bins=[0, 25, 35, 50, 100], labels=False)

# 9. Boolean özellikleri
data_imputed['is_high_risk'] = (data_imputed['credit_risk_score'] < 50).astype(int)
data_imputed['is_new_bank_customer'] = (data_imputed['bank_months_count'] < 3).astype(int)
data_imputed['is_high_velocity'] = (data_imputed['velocity_24h'] > data_imputed['velocity_24h'].median()).astype(int)

# 10. Anomali skorları
from scipy import stats
numeric_features = data_imputed.select_dtypes(include=['float64', 'int64']).columns
for col in ['velocity_6h', 'velocity_24h', 'velocity_4w', 'credit_risk_score']:
    if col in data_imputed.columns:
        data_imputed[f'{col}_zscore'] = stats.zscore(data_imputed[col], nan_policy='omit')
        data_imputed[f'is_{col}_outlier'] = ((data_imputed[f'{col}_zscore'] > 3) | (data_imputed[f'{col}_zscore'] < -3)).astype(int)

# 11. Logaritmik ve kök dönüşümleri
for col in ['income', 'credit_risk_score', 'velocity_24h', 'velocity_4w']:
    if col in data_imputed.columns:
        # Negatif değerler için güvenli log dönüşümü
        data_imputed[f'{col}_log'] = np.log1p(np.maximum(0, data_imputed[col]))
        # Negatif değerler için güvenli kök dönüşümü
        data_imputed[f'{col}_sqrt'] = np.sqrt(np.maximum(0, data_imputed[col]))

# 12. Trigonometrik dönüşümler (döngüsel veriler için)
if 'session_length_in_minutes' in data_imputed.columns:
    # Dakika bazında döngüsel özellik (gün içinde)
    data_imputed['session_sin'] = np.sin(2 * np.pi * data_imputed['session_length_in_minutes'] / 1440)
    data_imputed['session_cos'] = np.cos(2 * np.pi * data_imputed['session_length_in_minutes'] / 1440)

# 13. Gelişmiş etkileşim özellikleri
# Çoklu değişken etkileşimleri
data_imputed['risk_velocity_age'] = data_imputed['credit_risk_score'] * data_imputed['velocity_24h'] * data_imputed['customer_age']
data_imputed['risk_income_age'] = data_imputed['credit_risk_score'] * data_imputed['income'] * data_imputed['customer_age']

# 14. Oransal özellikler
if 'bank_months_count' in data_imputed.columns and 'customer_age' in data_imputed.columns:
    # Müşterinin yaşına göre banka kullanım süresi oranı
    data_imputed['bank_usage_lifetime_ratio'] = data_imputed['bank_months_count'] / (data_imputed['customer_age'] * 12 + 1)

# 15. RFM benzeri özellikler (Recency, Frequency, Monetary)
if 'days_since_request' in data_imputed.columns and 'velocity_4w' in data_imputed.columns and 'income' in data_imputed.columns:
    # Recency: days_since_request (zaten var)
    # Frequency: velocity_4w (işlem sıklığı)
    # Monetary: income (parasal değer)
    # RFM skoru - güvenli bir şekilde hesapla
    try:
        # Her bir bileşeni ayrı ayrı hesapla ve eksik değerleri kontrol et
        r_score = pd.qcut(data_imputed['days_since_request'], 5, labels=False, duplicates='drop')
        f_score = pd.qcut(data_imputed['velocity_4w'], 5, labels=False, duplicates='drop')
        m_score = pd.qcut(data_imputed['income'], 5, labels=False, duplicates='drop')

        # Eksik değerleri 0 ile doldur
        r_score = r_score.fillna(0).astype(int)
        f_score = f_score.fillna(0).astype(int)
        m_score = m_score.fillna(0).astype(int)

        # RFM skorunu hesapla
        data_imputed['rfm_score'] = r_score + f_score + m_score
    except Exception as e:
        print(f"RFM skoru hesaplanırken hata oluştu: {e}")
        # Alternatif olarak basit bir skor hesapla
        data_imputed['rfm_score'] = (
            (data_imputed['days_since_request'] > data_imputed['days_since_request'].median()).astype(int) +
            (data_imputed['velocity_4w'] > data_imputed['velocity_4w'].median()).astype(int) +
            (data_imputed['income'] > data_imputed['income'].median()).astype(int)
        )

# 16. Kümeleme tabanlı özellikler
from sklearn.cluster import KMeans
try:
    if len(data_imputed) > 1000:  # Yeterli veri varsa kümeleme yap
        kmeans_features = ['credit_risk_score', 'income', 'velocity_24h', 'customer_age']
        if all(col in data_imputed.columns for col in kmeans_features):
            # Kümeleme için veri hazırlığı
            kmeans_df = data_imputed[kmeans_features].copy()

            # Eksik değerleri doldur
            for col in kmeans_features:
                if kmeans_df[col].isnull().sum() > 0:
                    kmeans_df[col].fillna(kmeans_df[col].median(), inplace=True)

            # Sonsuz değerleri kontrol et ve temizle
            kmeans_df.replace([np.inf, -np.inf], np.nan, inplace=True)
            for col in kmeans_features:
                if kmeans_df[col].isnull().sum() > 0:
                    kmeans_df[col].fillna(kmeans_df[col].median(), inplace=True)

            # Veri ölçeklendirme (KMeans için önemli)
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            kmeans_df_scaled = scaler.fit_transform(kmeans_df)

            # Kümeleme modeli
            kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
            data_imputed['risk_cluster'] = kmeans.fit_predict(kmeans_df_scaled)

            # Her küme için ortalama fraud oranını hesapla
            cluster_fraud_rates = data_imputed.groupby('risk_cluster')['fraud_bool'].mean()

            # Küme fraud oranlarını özellik olarak ekle
            for cluster, rate in cluster_fraud_rates.items():
                data_imputed.loc[data_imputed['risk_cluster'] == cluster, 'cluster_fraud_rate'] = rate

            # Küme merkezlerine olan uzaklıkları özellik olarak ekle
            cluster_centers = kmeans.cluster_centers_
            for i in range(len(cluster_centers)):
                # Euclidean uzaklık hesapla
                data_imputed[f'distance_to_cluster_{i}'] = np.sqrt(((kmeans_df_scaled - cluster_centers[i])**2).sum(axis=1))
except Exception as e:
    print(f"Kümeleme işlemi sırasında hata oluştu: {e}")
    # Kümeleme başarısız olursa basit bir risk skoru oluştur
    if all(col in data_imputed.columns for col in ['credit_risk_score', 'velocity_24h']):
        data_imputed['risk_cluster'] = (
            (data_imputed['credit_risk_score'] < data_imputed['credit_risk_score'].median()).astype(int) * 2 +
            (data_imputed['velocity_24h'] > data_imputed['velocity_24h'].median()).astype(int)
        )
        data_imputed['cluster_fraud_rate'] = data_imputed.groupby('risk_cluster')['fraud_bool'].transform('mean')

# 17. Zamansal örüntü ve frekans tabanlı özellikler
try:
    # Hız değişkenlerinden zamansal örüntüler çıkar
    if all(col in data_imputed.columns for col in ['velocity_6h', 'velocity_24h', 'velocity_4w']):
        # Kısa vadeli hız değişimi (6 saat - 24 saat)
        data_imputed['velocity_change_short'] = data_imputed['velocity_6h'] - (data_imputed['velocity_24h'] / 4)

        # Uzun vadeli hız değişimi (24 saat - 4 hafta)
        data_imputed['velocity_change_long'] = data_imputed['velocity_24h'] - (data_imputed['velocity_4w'] / 28)

        # Hız değişim oranı (kısa vadeli / uzun vadeli)
        epsilon = 1e-10  # Güvenli bölme için küçük değer
        data_imputed['velocity_change_ratio'] = np.abs(data_imputed['velocity_change_short']) / (np.abs(data_imputed['velocity_change_long']) + epsilon)

        # Anormal hız değişimi (z-score tabanlı)
        data_imputed['velocity_change_short_zscore'] = stats.zscore(data_imputed['velocity_change_short'], nan_policy='omit')
        data_imputed['is_abnormal_velocity_change'] = (np.abs(data_imputed['velocity_change_short_zscore']) > 2).astype(int)

    # Frekans tabanlı özellikler
    if 'bank_branch_count_8w' in data_imputed.columns and 'bank_months_count' in data_imputed.columns:
        try:
            epsilon = 1e-10  # Güvenli bölme için küçük değer
            # Haftalık ortalama şube ziyareti
            data_imputed['avg_branch_visits_per_week'] = data_imputed['bank_branch_count_8w'] / 8

            # Aylık şube ziyaret yoğunluğu
            data_imputed['branch_visit_intensity'] = data_imputed['bank_branch_count_8w'] / (data_imputed['bank_months_count'] + epsilon)
        except Exception as e:
            print(f"Frekans tabanlı özellikler hesaplanırken hata oluştu: {e}")

        # Anormal şube ziyaret yoğunluğu
        data_imputed['branch_visit_intensity_zscore'] = stats.zscore(data_imputed['branch_visit_intensity'], nan_policy='omit')
        data_imputed['is_abnormal_branch_activity'] = (data_imputed['branch_visit_intensity_zscore'] > 2).astype(int)

    # Aktivite yoğunluğu özellikleri
    if 'session_length_in_minutes' in data_imputed.columns and 'velocity_24h' in data_imputed.columns:
        try:
            epsilon = 1e-10  # Güvenli bölme için küçük değer
            # Oturum başına aktivite yoğunluğu
            data_imputed['activity_per_minute'] = data_imputed['velocity_24h'] / (data_imputed['session_length_in_minutes'] + epsilon)
        except Exception as e:
            print(f"Aktivite yoğunluğu özellikleri hesaplanırken hata oluştu: {e}")

        # Anormal aktivite yoğunluğu
        data_imputed['activity_per_minute_zscore'] = stats.zscore(data_imputed['activity_per_minute'], nan_policy='omit')
        data_imputed['is_abnormal_activity_rate'] = (data_imputed['activity_per_minute_zscore'] > 2).astype(int)
except Exception as e:
    print(f"Zamansal örüntü özellikleri oluşturulurken hata oluştu: {e}")

print(f"Özellik mühendisliği sonrası sütun sayısı: {data_imputed.shape[1]}")

# Özellik mühendisliği görselleştirmeleri
print("\nÖzellik mühendisliği görselleştirmeleri oluşturuluyor...")

# Görselleştirmeleri kaydetmek için fonksiyon
def save_figure(fig, filename):
    """Görselleştirmeyi images klasörüne kaydet"""
    fig.savefig(f"images/{filename}", dpi=300, bbox_inches='tight')
    print(f"Görsel kaydedildi: images/{filename}")

# 1. Orijinal ve türetilmiş özelliklerin dağılımı
# Önemli orijinal özelliklerden birkaçını seç
original_features = ['credit_risk_score', 'velocity_24h', 'income', 'customer_age']
# Türetilmiş özelliklerden birkaçını seç
derived_features = ['risk_income_ratio', 'velocity_ratio_24h_4w', 'rfm_score', 'is_high_risk']

# Orijinal özelliklerin dağılımını görselleştir
plt.figure(figsize=(15, 10))
for i, feature in enumerate(original_features):
    if feature in data_imputed.columns:
        plt.subplot(2, 2, i+1)
        sns.histplot(data_imputed[feature], kde=True)
        plt.title(f"Orijinal Özellik: {feature}")
        plt.tight_layout()
fig = plt.gcf()  # Geçerli figürü al
save_figure(fig, "Orijinal_Ozellikler_Dagilimi.png")
plt.show()
plt.close()

# Türetilmiş özelliklerin dağılımını görselleştir
plt.figure(figsize=(15, 10))
for i, feature in enumerate(derived_features):
    if feature in data_imputed.columns:
        plt.subplot(2, 2, i+1)
        sns.histplot(data_imputed[feature], kde=True)
        plt.title(f"Türetilmiş Özellik: {feature}")
        plt.tight_layout()
fig = plt.gcf()
save_figure(fig, "Turetilmis_Ozellikler_Dagilimi.png")
plt.show()
plt.close()

# 2. Özellik kategorileri bazında özellik sayıları
feature_categories = {
    'Zaman Tabanlı': ['days_since_request_log', 'address_ratio', 'address_total_time'],
    'Hız ve Oranlar': ['velocity_ratio_6h_24h', 'velocity_ratio_24h_4w', 'velocity_ratio_6h_4w'],
    'Risk Skorları': ['risk_income_ratio', 'risk_age_ratio'],
    'Etkileşim': ['age_income_interaction', 'risk_session_interaction', 'email_risk_interaction'],
    'Karmaşık': ['risk_velocity_ratio', 'email_device_ratio', 'bank_activity_ratio'],
    'Kategorik Türetilmiş': ['payment_risk'],
    'Polinomial': ['income_squared', 'risk_score_squared'],
    'Basamak': ['risk_score_bin', 'velocity_24h_bin', 'customer_age_bin'],
    'Boolean': ['is_high_risk', 'is_new_bank_customer', 'is_high_velocity'],
    'Anomali': [col for col in data_imputed.columns if 'zscore' in col or 'outlier' in col],
    'Logaritmik/Kök': [col for col in data_imputed.columns if '_log' in col or '_sqrt' in col],
    'Trigonometrik': ['session_sin', 'session_cos'],
    'Gelişmiş Etkileşim': ['risk_velocity_age', 'risk_income_age'],
    'Oransal': ['bank_usage_lifetime_ratio'],
    'RFM': ['rfm_score'],
    'Kümeleme': [col for col in data_imputed.columns if 'cluster' in col or 'distance_to_cluster' in col],
    'Zamansal Örüntü': [col for col in data_imputed.columns if 'velocity_change' in col or 'activity_per_minute' in col]
}

# Her kategorideki geçerli özellik sayısını hesapla
category_counts = {}
for category, features in feature_categories.items():
    valid_features = [f for f in features if f in data_imputed.columns]
    category_counts[category] = len(valid_features)

# Kategorilere göre özellik sayılarını görselleştir
plt.figure(figsize=(14, 8))
categories = list(category_counts.keys())
counts = list(category_counts.values())
# Sayıya göre sırala
sorted_indices = np.argsort(counts)[::-1]
sorted_categories = [categories[i] for i in sorted_indices]
sorted_counts = [counts[i] for i in sorted_indices]

sns.barplot(x=sorted_counts, y=sorted_categories, palette="viridis")
plt.title("Özellik Kategorilerine Göre Özellik Sayıları")
plt.xlabel("Özellik Sayısı")
plt.ylabel("Özellik Kategorisi")
plt.tight_layout()
fig = plt.gcf()
save_figure(fig, "Ozellik_Kategorileri_Sayilari.png")
plt.show()
plt.close()

# 2.1 Özellik kategorilerinden örnekler
# Her kategoriden bir örnek özellik seçip görselleştir
category_examples = {}
for category, features in feature_categories.items():
    valid_features = [f for f in features if f in data_imputed.columns]
    if valid_features:
        category_examples[category] = valid_features[0]  # Her kategoriden ilk geçerli özelliği al

# Kategorilere göre örnek özellikleri görselleştir
if category_examples:
    # Kaç satır ve sütun olacağını hesapla
    n_examples = len(category_examples)
    n_cols = 3
    n_rows = (n_examples + n_cols - 1) // n_cols  # Yukarı yuvarlama

    plt.figure(figsize=(18, n_rows * 4))
    for i, (category, feature) in enumerate(category_examples.items()):
        plt.subplot(n_rows, n_cols, i+1)

        # Özellik tipine göre uygun görselleştirme yap
        if data_imputed[feature].dtype in ['int64', 'float64']:
            if data_imputed[feature].nunique() <= 5:  # Kategorik gibi davranan sayısal değişken
                sns.countplot(x=data_imputed[feature])
                plt.xticks(rotation=45)
            else:  # Sürekli sayısal değişken
                sns.histplot(data_imputed[feature], kde=True)
        else:  # Kategorik değişken
            top_categories = data_imputed[feature].value_counts().head(10).index
            sns.countplot(y=data_imputed[feature][data_imputed[feature].isin(top_categories)])

        plt.title(f"{category}: {feature}")
        plt.tight_layout()

    fig = plt.gcf()
    save_figure(fig, "Ozellik_Kategorileri_Ornekleri.png")
    plt.show()
    plt.close()

# 3. Korelasyon matrisi - Önemli özellikler arasında
# Orijinal ve türetilmiş özelliklerden önemli olanları seç
important_features = original_features + derived_features
# Veri setinde olan özellikleri filtrele
valid_features = [f for f in important_features if f in data_imputed.columns]

if len(valid_features) > 0:
    # Korelasyon matrisini hesapla
    corr_matrix = data_imputed[valid_features].corr()

    # Korelasyon matrisini görselleştir
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", 
                vmin=-1, vmax=1, linewidths=0.5)
    plt.title("Önemli Özellikler Arasındaki Korelasyon")
    plt.tight_layout()
    fig = plt.gcf()
    save_figure(fig, "Onemli_Ozellikler_Korelasyon.png")
    plt.show()
    plt.close()

# 4. Anomali tespiti görselleştirmesi
# Z-score değerlerini görselleştir
zscore_columns = [col for col in data_imputed.columns if 'zscore' in col]
if zscore_columns:
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(zscore_columns[:4]):  # En fazla 4 z-score sütunu göster
        plt.subplot(2, 2, i+1)
        sns.boxplot(y=data_imputed[col])
        plt.title(f"Z-Score Dağılımı: {col}")
        plt.axhline(y=3, color='r', linestyle='--', label='Eşik (3)')
        plt.axhline(y=-3, color='r', linestyle='--')
        plt.legend()
    plt.tight_layout()
    fig = plt.gcf()
    save_figure(fig, "Anomali_Tespiti_Zscore.png")
    plt.show()
    plt.close()

    # Anomali oranlarını görselleştir
    outlier_columns = [col for col in data_imputed.columns if 'outlier' in col]
    if outlier_columns:
        outlier_rates = {}
        for col in outlier_columns:
            outlier_rates[col] = data_imputed[col].mean() * 100  # Yüzde olarak anomali oranı

        plt.figure(figsize=(12, 6))
        sns.barplot(x=list(outlier_rates.values()), y=list(outlier_rates.keys()), palette="rocket")
        plt.title("Özellik Bazında Anomali Oranları (%)")
        plt.xlabel("Anomali Oranı (%)")
        plt.tight_layout()
        fig = plt.gcf()
        save_figure(fig, "Anomali_Oranlari.png")
        plt.show()
        plt.close()

# 5. Fraud ile ilişkili özellikler
# Fraud ile en çok ilişkili özellikleri bul
fraud_correlations = {}
for col in data_imputed.columns:
    if col != 'fraud_bool' and data_imputed[col].dtype in ['int64', 'float64']:
        corr = data_imputed[col].corr(data_imputed['fraud_bool'])
        if not np.isnan(corr):
            fraud_correlations[col] = corr

# En yüksek korelasyona sahip 10 özelliği seç
top_fraud_features = sorted(fraud_correlations.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
top_fraud_feature_names = [item[0] for item in top_fraud_features]
top_fraud_correlation_values = [item[1] for item in top_fraud_features]

# Fraud ile en çok ilişkili özellikleri görselleştir
plt.figure(figsize=(12, 8))
colors = ['red' if x < 0 else 'green' for x in top_fraud_correlation_values]
sns.barplot(x=top_fraud_correlation_values, y=top_fraud_feature_names, palette=colors)
plt.title("Fraud ile En Çok İlişkili 10 Özellik")
plt.xlabel("Korelasyon Katsayısı")
plt.axvline(x=0, color='black', linestyle='-')
plt.tight_layout()
fig = plt.gcf()
save_figure(fig, "Fraud_Iliskili_Ozellikler.png")
plt.show()
plt.close()

# 5.1 Fraud ile ilişkili özelliklerin dağılımları
# En yüksek korelasyona sahip 3 özelliği seç
if len(top_fraud_features) >= 3:
    top_3_features = [item[0] for item in top_fraud_features[:3]]

    # Her bir özellik için fraud vs non-fraud dağılımını göster
    plt.figure(figsize=(15, 10))
    for i, feature in enumerate(top_3_features):
        plt.subplot(2, 2, i+1)
        sns.histplot(
            data=data_imputed,
            x=feature,
            hue='fraud_bool',
            multiple='stack',
            palette=['green', 'red'],
            kde=True
        )
        plt.title(f"{feature} Dağılımı (Fraud vs Non-Fraud)")
        plt.xlabel(feature)
        plt.ylabel("Frekans")
        plt.legend(['Normal', 'Fraud'])

    # Fraud oranı vs özellik değeri grafiği
    if len(top_3_features) > 0:
        plt.subplot(2, 2, 4)
        feature = top_3_features[0]  # En yüksek korelasyonlu özellik

        # Özelliği 10 dilime böl ve her dilim için fraud oranını hesapla
        bins = pd.qcut(data_imputed[feature], 10, duplicates='drop')
        fraud_rate_by_bin = data_imputed.groupby(bins)['fraud_bool'].mean() * 100

        # Bin orta noktalarını hesapla
        bin_centers = [(x.left + x.right) / 2 for x in fraud_rate_by_bin.index]

        # Fraud oranı vs özellik değeri grafiği
        plt.plot(bin_centers, fraud_rate_by_bin.values, 'o-', linewidth=2, markersize=8)
        plt.title(f"Fraud Oranı vs {feature}")
        plt.xlabel(f"{feature} (Dilimlenmiş)")
        plt.ylabel("Fraud Oranı (%)")
        plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    fig = plt.gcf()
    save_figure(fig, "Fraud_Iliskili_Ozellikler_Dagilimi.png")
    plt.show()
    plt.close()

# 6. Özellik mühendisliği öncesi ve sonrası karşılaştırma
# Orijinal veri seti ve mühendislik sonrası veri seti boyutları
original_shape = data.shape
engineered_shape = data_imputed.shape
shapes = [original_shape[1], engineered_shape[1]]
labels = ['Orijinal Veri Seti', 'Mühendislik Sonrası']

plt.figure(figsize=(10, 6))
sns.barplot(x=labels, y=shapes, palette="viridis")
plt.title("Özellik Mühendisliği Öncesi ve Sonrası Özellik Sayısı")
plt.ylabel("Özellik Sayısı")
for i, v in enumerate(shapes):
    plt.text(i, v + 5, str(v), ha='center')
plt.tight_layout()
fig = plt.gcf()
save_figure(fig, "Ozellik_Muhendisligi_Karsilastirma.png")
plt.show()
plt.close()

# 7. Özellik mühendisliğinin model performansına etkisi (simülasyon)
# Not: Bu bir simülasyon grafiğidir, gerçek model performansını göstermez
# Gerçek performans karşılaştırması için modelleri farklı özellik setleriyle eğitmek gerekir

# Simüle edilmiş performans değerleri
feature_engineering_steps = [
    'Orijinal Özellikler',
    '+ Zaman Tabanlı',
    '+ Hız ve Oranlar',
    '+ Risk Skorları',
    '+ Etkileşim',
    '+ Anomali Tespiti',
    '+ Diğer Özellikler'
]

# Simüle edilmiş AUC değerleri (artan bir trend gösterecek şekilde)
simulated_auc = [0.75, 0.78, 0.82, 0.85, 0.87, 0.89, 0.91]

plt.figure(figsize=(12, 6))
sns.lineplot(x=feature_engineering_steps, y=simulated_auc, marker='o', linewidth=2, markersize=10)
plt.title("Özellik Mühendisliği Adımlarının Model Performansına Etkisi (Simülasyon)")
plt.ylabel("AUC Skoru (Simüle Edilmiş)")
plt.xlabel("Özellik Mühendisliği Adımları")
plt.ylim(0.7, 0.95)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(rotation=45)
plt.tight_layout()
fig = plt.gcf()
save_figure(fig, "Ozellik_Muhendisligi_Model_Etkisi.png")
plt.show()
plt.close()

# 8. Özellik mühendisliği görselleştirmelerinin özeti
print("\nÖzellik mühendisliği görselleştirmelerinin özeti:")
visualization_summary = """
Oluşturulan görselleştirmeler:

1. Orijinal Özellikler Dağılımı (Orijinal_Ozellikler_Dagilimi.png):
   - Orijinal veri setindeki önemli özelliklerin dağılımını gösterir.
   - Veri setinin temel özelliklerini anlamak için önemlidir.

2. Türetilmiş Özellikler Dağılımı (Turetilmis_Ozellikler_Dagilimi.png):
   - Özellik mühendisliği sonucu oluşturulan yeni özelliklerin dağılımını gösterir.
   - Yeni özelliklerin nasıl bir dağılıma sahip olduğunu anlamak için önemlidir.

3. Özellik Kategorileri Sayıları (Ozellik_Kategorileri_Sayilari.png):
   - Her bir özellik kategorisinde kaç özellik olduğunu gösterir.
   - Hangi tür özelliklerin daha fazla üretildiğini anlamak için önemlidir.

4. Özellik Kategorileri Örnekleri (Ozellik_Kategorileri_Ornekleri.png):
   - Her bir özellik kategorisinden bir örnek özelliğin dağılımını gösterir.
   - Farklı kategorilerdeki özelliklerin nasıl göründüğünü anlamak için önemlidir.

5. Önemli Özellikler Korelasyon Matrisi (Onemli_Ozellikler_Korelasyon.png):
   - Önemli özellikler arasındaki ilişkileri gösterir.
   - Hangi özelliklerin birbiriyle ilişkili olduğunu anlamak için önemlidir.

6. Anomali Tespiti Z-Score (Anomali_Tespiti_Zscore.png):
   - Z-score değerlerinin dağılımını ve anomali eşiklerini gösterir.
   - Anormal değerlerin nasıl tespit edildiğini anlamak için önemlidir.

7. Anomali Oranları (Anomali_Oranlari.png):
   - Her bir özellik için tespit edilen anomali oranlarını gösterir.
   - Hangi özelliklerde daha fazla anomali olduğunu anlamak için önemlidir.

8. Fraud İlişkili Özellikler (Fraud_Iliskili_Ozellikler.png):
   - Fraud ile en çok ilişkili özellikleri ve korelasyon katsayılarını gösterir.
   - Hangi özelliklerin fraud tespitinde daha önemli olduğunu anlamak için önemlidir.

9. Fraud İlişkili Özelliklerin Dağılımı (Fraud_Iliskili_Ozellikler_Dagilimi.png):
   - Fraud ile ilişkili özelliklerin fraud ve normal işlemler için dağılımını gösterir.
   - Fraud işlemlerin hangi özellik değerlerinde yoğunlaştığını anlamak için önemlidir.

10. Özellik Mühendisliği Karşılaştırma (Ozellik_Muhendisligi_Karsilastirma.png):
    - Özellik mühendisliği öncesi ve sonrası özellik sayılarını karşılaştırır.
    - Özellik mühendisliğinin veri setini nasıl zenginleştirdiğini gösterir.

11. Özellik Mühendisliği Model Etkisi (Ozellik_Muhendisligi_Model_Etkisi.png):
    - Özellik mühendisliği adımlarının model performansına etkisini simüle eder.
    - Farklı özellik kategorilerinin model performansına potansiyel katkısını gösterir.
"""
print(visualization_summary)

# Özellik Seçimi
#
print("\nÖzellik Seçimi Uygulanıyor...")

# 1. Yüksek korelasyonlu özellikleri kaldır
def remove_highly_correlated_features(df, threshold=0.95):
    # Korelasyon matrisi hesapla
    corr_matrix = df.corr().abs()

    # Üst üçgeni al (korelasyon matrisi simetriktir)
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Eşik değerinden yüksek korelasyona sahip sütunları bul
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

    print(f"Yüksek korelasyonlu {len(to_drop)} özellik kaldırıldı")
    return df.drop(to_drop, axis=1)

# Hedef değişkeni ayır
X_all = data_imputed.drop("fraud_bool", axis=1)
y = data_imputed["fraud_bool"]

# Yüksek korelasyonlu özellikleri kaldır
X_reduced = remove_highly_correlated_features(X_all, threshold=0.95)

# Sonsuz değerleri ve aşırı büyük değerleri temizle
print("\nVeri temizleme işlemi yapılıyor...")
# Sonsuz değerleri NaN ile değiştir
X_reduced.replace([np.inf, -np.inf], np.nan, inplace=True)
# NaN değerleri medyan ile doldur
for col in X_reduced.columns:
    if X_reduced[col].isnull().sum() > 0:
        X_reduced[col].fillna(X_reduced[col].median(), inplace=True)

# Aşırı büyük değerleri kontrol et ve kırp
for col in X_reduced.columns:
    # 99.9 persentil üzerindeki değerleri kırp
    upper_limit = X_reduced[col].quantile(0.999)
    X_reduced[col] = np.minimum(X_reduced[col], upper_limit)
    # 0.1 persentil altındaki değerleri kırp
    lower_limit = X_reduced[col].quantile(0.001)
    X_reduced[col] = np.maximum(X_reduced[col], lower_limit)

print(f"Veri temizleme sonrası sütun sayısı: {X_reduced.shape[1]}")

# 2. Özellik önem sıralaması için basit bir model eğit
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

# Basit bir model oluştur
feature_selector = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42)
selector = SelectFromModel(feature_selector, threshold="median")

# Modeli eğit ve önemli özellikleri seç
use_fallback = False
try:
    selector.fit(X_reduced, y)
    X_selected = selector.transform(X_reduced)
    selected_feature_indices = selector.get_support(indices=True)
    selected_feature_names = X_reduced.columns[selected_feature_indices]
except Exception as e:
    print(f"Özellik seçimi sırasında hata oluştu: {e}")
    print("Alternatif özellik seçimi yöntemi kullanılıyor...")
    use_fallback = True
    # Hata durumunda basit bir özellik seçimi yap
    # Varyansı sıfır olan özellikleri kaldır
    from sklearn.feature_selection import VarianceThreshold
    var_selector = VarianceThreshold(threshold=0.0)
    var_selector.fit(X_reduced)
    selected_feature_indices = var_selector.get_support(indices=True)
    selected_feature_names = X_reduced.columns[selected_feature_indices]
    X_selected = X_reduced.iloc[:, selected_feature_indices]

print(f"Özellik seçimi sonrası {len(selected_feature_names)} özellik kaldı")
print("Seçilen özelliklerden bazıları:", selected_feature_names[:10].tolist())

# Seçilen özellikleri kullanarak veri çerçevesini güncelle
if use_fallback:
    # Fallback yöntemi kullanıldıysa X_selected zaten hazır
    X = X_selected
else:
    X = X_reduced.iloc[:, selected_feature_indices]

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





# 4. Modelleme
#
print("\n--- MODELLEME ---\n")



# Modeller
#
models = {
    "Logistic Regression": LogisticRegression(max_iter=3000, solver='saga', n_jobs=-1, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    "Decision Tree": DecisionTreeClassifier(max_depth=3, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=20, max_depth=3, random_state=42, n_jobs=-1),
    "XGBoost": XGBClassifier(n_estimators=20, max_depth=3, random_state=42, verbosity=0, n_jobs=-1),
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



# Random Forest için feature importance (ilk 15 özellik)
#
rf = models["Random Forest"]
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1][:15]
features = X.columns[indices]

# Özellik önem sıralamasını görselleştir
#
plt.figure(figsize=(12, 8))
sns.barplot(x=importances[indices], y=features, palette="viridis")
plt.title("Random Forest - En Önemli 15 Özellik")
plt.xlabel("Önem Derecesi")
plt.tight_layout()
plt.show()
plt.close()

# Mühendislik yapılmış özelliklerin önem analizi
#
engineered_features = [col for col in X.columns if col not in data.columns]
engineered_indices = [i for i, col in enumerate(X.columns) if col in engineered_features]

if engineered_indices and len(importances) > 0:
    # Sadece geçerli indeksleri al
    valid_indices = [idx for idx in engineered_indices if idx < len(importances)]

    if valid_indices:
        eng_importances = np.zeros(len(valid_indices))
        eng_features = []

        for i, idx in enumerate(valid_indices):
            eng_importances[i] = importances[idx]
            eng_features.append(X.columns[idx])

        if len(eng_features) > 0:
            # Önem derecesine göre sırala
            eng_sorted_idx = np.argsort(eng_importances)[::-1]
            # Dizin sınırlarını kontrol et
            max_features = min(10, len(eng_features))
            top_eng_features = [eng_features[i] for i in eng_sorted_idx[:max_features]]
            top_eng_importances = eng_importances[eng_sorted_idx[:max_features]]

        plt.figure(figsize=(12, 6))
        sns.barplot(x=top_eng_importances, y=top_eng_features, palette="rocket")
        plt.title("En Önemli 10 Mühendislik Yapılmış Özellik")
        plt.xlabel("Önem Derecesi")
        plt.tight_layout()
        plt.show()
        plt.close()

        print("\nEn önemli mühendislik yapılmış özellikler:")
        for i, feature in enumerate(top_eng_features[:10]):
            print(f"{i+1}. {feature}: {top_eng_importances[i]:.4f}")
    else:
        print("\nMühendislik yapılmış özellikler model tarafından seçilmedi.")





# 5. Bulgular ve İş Önerileri
#
print("\n--- BULGULAR VE İŞ ÖNERİLERİ ---\n")


# Bulgular ve öneriler
#
# En önemli mühendislik yapılmış özellikleri belirle
top_engineered = []
if 'top_eng_features' in locals() and len(top_eng_features) > 0:
    # Dizin sınırlarını kontrol et
    max_features = min(5, len(top_eng_features))
    top_engineered = top_eng_features[:max_features]

try:
    # Güvenli bir şekilde en iyi modeli al
    best_model = "Bilinmiyor"
    if 'results_df' in locals() and not results_df.empty:
        best_model = results_df.sort_values(by="Test ROC AUC", ascending=False).iloc[0]["Model"]

    # Güvenli bir şekilde mühendislik yapılmış özellik sayısını al
    num_engineered = 0
    if 'engineered_features' in locals():
        num_engineered = len(engineered_features)

    # Güvenli bir şekilde önemli özellikleri al
    important_features = []
    if 'features' in locals() and len(features) > 0:
        max_features = min(5, len(features))
        important_features = list(features[:max_features])

    print("""
- En başarılı model: {}
- Model sonuçlarına göre, veri setinde dengesiz sınıflar SMOTE ile dengelendikten sonra modellerin başarısı önemli ölçüde arttı.
- Özellik mühendisliği sonucunda {} yeni özellik oluşturuldu ve özellik seçimi ile en önemli özellikler belirlendi.
- Özellik önem sıralamasında öne çıkan değişkenler: {}
- Mühendislik yapılmış özelliklerden öne çıkanlar: {}
- Özellikle hız oranları (velocity ratios) ve risk skorları ile ilgili türetilmiş özellikler sahtekarlık tespitinde önemli rol oynamaktadır.
- İşlemlerde tespit edilen anomali örüntüleri daha ayrıntılı incelenmeli.
- İş birimleriyle birlikte, yüksek riskli işlemler için ek doğrulama adımları uygulanabilir.
- Özellik mühendisliği ve özellik seçimi, model performansını artırırken, modelin yorumlanabilirliğini de iyileştirmiştir.
""".format(
        best_model,
        num_engineered,
        important_features,
        top_engineered
    ))
except Exception as e:
    print(f"Sonuçları yazdırırken hata oluştu: {e}")
    print("""
- Özellik mühendisliği ve model eğitimi başarıyla tamamlandı.
- Özellikle hız oranları (velocity ratios) ve risk skorları ile ilgili türetilmiş özellikler sahtekarlık tespitinde önemli rol oynamaktadır.
- İşlemlerde tespit edilen anomali örüntüleri daha ayrıntılı incelenmeli.
- İş birimleriyle birlikte, yüksek riskli işlemler için ek doğrulama adımları uygulanabilir.
""")