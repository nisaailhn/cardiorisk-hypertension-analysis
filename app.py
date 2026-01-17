from flask import Flask, render_template, request
import numpy as np
import joblib
import pandas as pd

# ==============================
# Flask App
# ==============================
app = Flask(__name__)

# ==============================
# Veri seti ortalamalarını oku
# ==============================
group_stats = pd.read_csv("group_stats.csv", index_col=0)

# ==============================
# Model Dosyaları
# ==============================
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")

# ==============================
# Ana Sayfa
# ==============================
@app.route("/")
def home():
    return render_template("home.html")

# ==============================
# Form Sayfası
# ==============================
@app.route("/risk")
def risk():
    return render_template("index.html")

# ==============================
# Tahmin
# ==============================
@app.route("/predict", methods=["POST"])
def predict():

    # 1️⃣ Formdan gelen veriler
    form_data = {
        "patientid": 0,
        "age": float(request.form.get("age", 0)),
        "gender": float(request.form.get("gender", 0)),
        "chestpain": float(request.form.get("chestpain", 0)),
        "restingBP": float(request.form.get("restingBP", 0)),
        "serumcholestrol": float(request.form.get("serumcholestrol", 0)),
        "fastingbloodsugar": float(request.form.get("fastingbloodsugar", 0)),
        "restingrelectro": float(request.form.get("restingrelectro", 0)),
        "maxheartrate": float(request.form.get("maxheartrate", 0)),
        "exerciseangia": float(request.form.get("exerciseangia", 0)),
        "oldpeak": float(request.form.get("oldpeak", 0)),
        "slope": float(request.form.get("slope", 0)),
        "noofmajorvessels": float(request.form.get("noofmajorvessels", 0)),
    }

    # 2️⃣ Model girdisi
    X = np.array([[form_data[c] for c in columns]])
    X_scaled = scaler.transform(X)

    # 3️⃣ Tahmin
    risk = round(model.predict_proba(X_scaled)[0][1] * 100, 2)

    # 4️⃣ Risk seviyesi
    if risk >= 70:
        level = "Yüksek"
        tips = [
            "Hipertansiyon gelişme riski yüksek görünmektedir.",
            "Tuz tüketiminin sınırlandırılması önerilir.",
            "Kardiyoloji uzmanı kontrolü önemlidir."
        ]
    elif risk >= 40:
        level = "Orta"
        tips = [
            "Bazı risk faktörleri mevcuttur.",
            "Yaşam tarzı düzenlemeleri önerilir."
        ]
    else:
        level = "Düşük"
        tips = [
            "Hipertansiyon riski düşük görünmektedir."
        ]

       # ==============================
    # 5️⃣ VERİ SETİNE DAYALI AÇIKLAMA (VERİ SETİ DİLİ)
    # ==============================
    explanations = []

    compare_cols = {
        "noofmajorvessels": "Ana damar sayısı",
        "restingBP": "Dinlenme tansiyonu",
        "age": "Yaş"
    }

    for col, label in compare_cols.items():
        user_val = form_data[col]
        mean_no = group_stats.loc[0, col]   # hipertansiyon yok
        mean_yes = group_stats.loc[1, col]  # hipertansiyon var

        if user_val > mean_yes:
            explanations.append(
                f"{label} değeri ({user_val}), veri setinde hipertansiyon görülen bireylerin ortalamasının ({mean_yes:.1f}) üzerindedir ve riski artırıcı yönde değerlendirilmiştir."
            )
        elif user_val < mean_no:
            explanations.append(
                f"{label} değeri ({user_val}), hipertansiyon görülmeyen bireylerin ortalamasının ({mean_no:.1f}) altındadır ve riski azaltıcı yönde değerlendirilmiştir."
            )
        else:
            explanations.append(
                f"{label} değeri ({user_val}), veri setindeki ortalama değerlere yakın olup belirgin bir risk artışı göstermemektedir."
            )

    # ==============================
    # 6️⃣ Sonuç
    # ==============================
    return render_template(
        "result.html",
        risk=risk,
        level=level,
        tips=tips,
        explanations=explanations
    )

# ==============================
# SSS
# ==============================
@app.route("/sss")
def sss():
    return render_template("sss.html")

# ==============================
# İletişim
# ==============================
@app.route("/iletisim", methods=["GET", "POST"])
def iletisim():
    if request.method == "POST":
        name = request.form.get("name", "")
        return render_template("iletisim_tesekkur.html", name=name)
    return render_template("iletisim.html")

# ==============================
# Çalıştır
# ==============================
if __name__ == "__main__":
    app.run(debug=True)
