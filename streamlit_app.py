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
import os  # <-- EKLENDİ

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
    # Dosyanın tam yolunu belirle, hata kontrolü ekle
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets", "Base.csv")
    if not os.path.exists(csv_path):
        st.error(f"Veri dosyası bulunamadı: {csv_path}\nLütfen 'datasets/Base.csv' dosyasının mevcut olduğundan emin olun.")
        st.stop()
    return pd.read_csv(csv_path)

data = load_data()