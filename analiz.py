import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

# 1. VERİ YÜKLEME

try:
    df = pd.read_csv("veri.csv")

    # patientid analiz için gereksiz, varsa çıkar
    if "patientid" in df.columns:
        df = df.drop(columns=["patientid"])

    df = df.dropna()
    print("✅ Kalp veri seti başarıyla yüklendi.")

except FileNotFoundError:
    print("❌ veri.csv dosyası bulunamadı")


# 2. ÖN İŞLEME
X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. MODELLERİN ANALİZİ
modeller = {
    "Lojistik Regresyon": LogisticRegression(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(probability=True)
}

sonuclar = []

print("\n" + "="*60)
print(f"{'ALGORİTMA':<20} | {'BAŞARI':<10} | {'AYIRT ETME GÜCÜ (AUC)'}")
print("-" * 60)

for isim, model in modeller.items():
    model.fit(X_train_scaled, y_train)
    tahmin = model.predict(X_test_scaled)
    olasilik = model.predict_proba(X_test_scaled)[:, 1]
    
    skor = accuracy_score(y_test, tahmin)
    auc_skor = roc_auc_score(y_test, olasilik)
    
    sonuclar.append({"Model": isim, "Skor": skor})
    print(f"{isim:<20} | %{skor*100:.2f}     | {auc_skor:.3f}")

# 4. REGRESYON ANALİZİ (Özellik Önem Sırası)
# Hangi faktör kalbi daha çok yoruyor?
rf_model = modeller["Random Forest"]
onem_df = pd.DataFrame({
    'Faktör': X.columns,
    'Önem Skoru': rf_model.feature_importances_
}).sort_values(by='Önem Skoru', ascending=False)

# 5. GÖRSEL ANALİZ PANELİ
plt.figure(figsize=(15, 6))

# Grafik 1: Faktörlerin Etkisi
plt.subplot(1, 2, 1)
sns.barplot(x='Önem Skoru', y='Faktör', data=onem_df, palette='magma')
plt.title('Hangi Faktör Kalp Hastalığını Daha Çok Tetikliyor?')

# Grafik 2: Hata Matrisi (En iyi model için)
plt.subplot(1, 2, 2)
en_iyi_model_ismi = max(sonuclar, key=lambda x: x['Skor'])['Model']
cm = confusion_matrix(y_test, modeller[en_iyi_model_ismi].predict(X_test_scaled))
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds')
plt.title(f'En İyi Model ({en_iyi_model_ismi}) Tahmin Başarısı')
plt.xlabel('Tahmin Edilen')
plt.ylabel('Gerçek Durum')

plt.tight_layout()
plt.show()

print("\n" + "="*60)
print(f"🏆 SONUÇ: Bu veri setinde en yüksek doğruluğu {en_iyi_model_ismi} sağladı.")
print("="*60)
import joblib

# Basit ve hızlı: Logistic Regression kaydedelim
best_model = LogisticRegression(max_iter=2000)
best_model.fit(X_train_scaled, y_train)

joblib.dump(best_model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

# Sütun sırasını da kaydedelim (site input sırası doğru olsun diye)
joblib.dump(list(X.columns), "columns.pkl")

print("✅ model.pkl, scaler.pkl, columns.pkl kaydedildi.")
# EKLENDİ: Feature importance'ları kaydet
# ==============================
# Özellik Önemlerini Kaydet
# ==============================

feature_importance_df = pd.DataFrame({
    "feature": X.columns,
    "importance": rf_model.feature_importances_
}).sort_values(by="importance", ascending=False)

feature_importance_df.to_csv("feature_importance.csv", index=False)

print("✅ feature_importance.csv kaydedildi")
# ==============================
# VERİ SETİ ORTALAMALARI (HOCALIK KISIM)
# ==============================

group_stats = df.groupby("target").mean()

group_stats.to_csv("group_stats.csv")

print("✅ group_stats.csv (veri seti ortalamaları) kaydedildi")
