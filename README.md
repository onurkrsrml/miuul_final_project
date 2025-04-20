# 🏦 Makine Öğrenimi ile Sahtecilik Tespiti (Fraud Detection)

Bu proje, makine öğrenimi teknikleri kullanarak finansal işlemlerde sahtecilik tespiti yapmayı amaçlar. Proje, Python ile geliştirilmiş olup veri ön işleme, modelleme ve performans değerlendirme adımlarını içermektedir. Ayrıca, interaktif bir arayüz ile kullanıcıların kendi verileri üzerinde analiz yapabilmelerine olanak tanıyan bir **Streamlit** uygulaması da sunulmaktadır.

---

## 🚀 Proje Özeti

- **Amaç:** Finansal işlemlerde sahtecilik (fraud) olup olmadığını tahmin eden, farklı makine öğrenimi modellerinin karşılaştırıldığı bir çözüm geliştirmek.
- **Temel Dosyalar:**
  - `main.py`: Veri işleme, model eğitim ve değerlendirme süreçlerini komut satırında çalıştırır.
  - `streamlit_app.py`: Kullanıcıların kendi verileriyle interaktif analiz yapmasını sağlayan web arayüzünü sunar.

---

## 📁 Proje Yapısı

```
miuul_final_project/
│
├── main.py              # Komut satırından çalıştırılan ana Python scripti
├── streamlit_app.py     # Streamlit tabanlı interaktif uygulama
├── requirements.txt     # Proje bağımlılıkları
├── Base.csv             # Örnek veri seti (opsiyonel)
└── README.md
```

---

## ⚙️ Kurulum

1. **Depoyu klonlayın:**

   ```bash
   git clone https://github.com/onurkrsrml/miuul_final_project.git
   cd miuul_final_project
   ```

2. **Gerekli Python paketlerini yükleyin:**

   ```bash
   pip install -r requirements.txt
   ```

---

## 📝 Kullanım

### 1. Komut Satırı (main.py)

Bu script, yerleşik bir veri seti ile otomatik olarak çalışır ve çeşitli makine öğrenimi modellerinin karşılaştırmalı sonuçlarını sunar.

```bash
python main.py
```

#### Özellikler:

- **Veri yükleme ve ön işleme:** Eksik değer doldurma, kategorik verilerin kodlanması, örnekleme.
- **SMOTE ile veri dengelenmesi:** Azınlık sınıfını çoğaltarak dengesiz veri sorununu çözer.
- **Çoklu model eğitimi:** Lojistik Regresyon, KNN, Karar Ağacı, Random Forest, XGBoost, LightGBM.
- **Performans raporları:** Doğruluk, ROC AUC, Precision, Recall, F1-score.
- **En iyi modeller:** confusion matrix ve classification report görselleştirmeleri.
- **Özellik önem sıralaması:** Random Forest modeline göre en önemli 10 özellik.

### 2. Web Arayüzü (streamlit_app.py)

Kendi veri setinizi yükleyerek, özellik ve model seçimi yapabilir, sonuçları anlık olarak görebilirsiniz.

```bash
streamlit run streamlit_app.py
```

#### Streamlit Arayüz Özellikleri:

- **Veri yükleme:** CSV dosyanızı yükleyin.
- **Önizleme:** Verinizin ilk satırlarını ve temel istatistikleri görün.
- **Özellik seçimi:** Hangi değişkenlerle modelleme yapılacağını seçin.
- **Model seçimi:** Birden fazla algoritmayı aynı anda deneyin.
- **Dengeleme:** SMOTE ile sınıf dengesi.
- **Sonuç tablosu:** Test doğruluğu, ROC AUC, Precision, Recall, F1-score karşılaştırmalı tablo.
- **ROC eğrisi karşılaştırması:** Seçilen tüm modellerin ROC eğrilerini aynı grafikte görün.
- **Confusion Matrix & Classification Report:** En başarılı iki model için detaylı hata analizi.
- **Özellik önem sıralaması:** Random Forest ile en önemli değişkenler.
- **Model kaydetme:** Eğitilen modelleri `.pkl` formatında kaydedin.

---

## 📊 Kullanılan Modeller

- Logistic Regression
- K-En Yakın Komşu (KNN)
- Karar Ağacı (Decision Tree)
- Rastgele Orman (Random Forest)
- XGBoost
- LightGBM

---

## 🧩 Kullanılan Kütüphaneler

- pandas, numpy, matplotlib, seaborn
- scikit-learn (modelleme ve ön işleme)
- imbalanced-learn (SMOTE)
- xgboost, lightgbm
- streamlit (web arayüzü)
- joblib (model kaydetme)

---

## 💡 Notlar

- Ana veri setinizde `fraud_bool` isimli bir hedef değişken olmalıdır.
- Büyük veri setlerinde örnekleme uygulanır (ilk 50.000 satır kullanılır).
- Özellik ve model seçimi esnektir, kullanıcıya bırakılmıştır.

---

## 📸 Ekran Görüntüleri

> ![GİRİŞ] <img src="https://github.com/onurkrsrml/miuul_final_project/blob/main/images/Giris.png">
> ![ÖN TANIM] <img src="https://github.com/onurkrsrml/miuul_final_project/blob/main/images/On%20Tanim.png">
> ![ROC AUC Eğrisi] <img src="https://github.com/onurkrsrml/miuul_final_project/blob/main/images/ROC%20AUC%20Egrisi.png">
> ![Confusion Matrix XGBoost] <img src="https://github.com/onurkrsrml/miuul_final_project/blob/main/images/Confusion%20Matrix%20Report%20XGBoost.png">
> ![Confusion Matrix LightGBM] <img src="https://github.com/onurkrsrml/miuul_final_project/blob/main/images/Confusion%20Matrix%20Report%20LightGBM.png">
> ![Random Forest] <img src="https://github.com/onurkrsrml/miuul_final_project/blob/main/images/Random%20Forest.png">


---

## ✨ Katkı ve Geri Bildirim

Lütfen önerilerinizi, hata bildirimlerinizi ve katkılarınızı GitHub Issues veya Pull Request olarak iletmekten çekinmeyin.

---

## 📄 Lisans

Bu proje [MIT](LICENSE) lisansı ile lisanslanmıştır.

---
