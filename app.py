"""UrbanBike AI: aplikacioni lokal për parashikimin e kërkesës ditore.

Nisja nga dosja kryesore e projektit:
    streamlit run app.py
"""
from datetime import date

import joblib
import pandas as pd
import streamlit as st

from config.common import DATA, FEATURES, ROOT, TARGET

st.set_page_config(page_title="UrbanBike AI", page_icon="🚲", layout="centered")

MODELS = ROOT / "models"
DITET = ["E diel", "E hënë", "E martë", "E mërkurë", "E enjte", "E premte", "E shtunë"]


@st.cache_resource
def load_models():
    reg = joblib.load(MODELS / "regresion.joblib")
    clf = joblib.load(MODELS / "klasifikim.joblib")
    return reg, clf["model"], clf["pragu"]


@st.cache_data
def load_data():
    return pd.read_csv(DATA / "processed" / "model_data.csv", parse_dates=["date"])


def predict(row, reg, clf):
    X = pd.DataFrame([row])[FEATURES]
    sasia = max(0, round(float(reg.predict(X)[0])))
    e_larte = int(clf.predict(X)[0])
    prob = float(clf.predict_proba(X)[0][1])
    return sasia, e_larte, prob


def show_result(sasia, e_larte, prob, pragu, reale=None):
    st.divider()
    c1, c2 = st.columns(2)
    c1.metric("Parashikimi", f"{sasia:,} biçikleta".replace(",", "."))
    if e_larte:
        c2.markdown("### 🔴 Kërkesë E LARTË")
    else:
        c2.markdown("### 🟢 Kërkesë NORMALE")
    c2.caption(f"Probabiliteti 'e lartë': {prob:.0%} · pragu: {pragu:.0f}")
    if reale is not None:
        gabimi = sasia - reale
        st.info(f"Vlera reale atë ditë: **{reale:,}** · gabimi: **{gabimi:+,}**".replace(",", "."))


try:
    reg, clf, pragu = load_models()
    df = load_data()
except FileNotFoundError:
    st.error("Modelet nuk u gjetën. Nis më parë: `python src/01_pastrimi.py`, "
             "`python src/03_modelet.py` dhe `python src/04_klasifikimi.py`.")
    st.stop()

st.title("🚲 UrbanBike AI")
st.write("Parashikimi i kërkesës ditore për biçikleta me qira, "
         "nga historiku i qirave dhe moti.")

tab1, tab2 = st.tabs(["📅 Zgjidh një datë", "✏️ Vendos vlerat vetë"])

with tab1:
    st.caption("Zgjidh një ditë nga të dhënat historike. Aplikacioni përdor "
               "të dhënat e ditëve të mëparshme dhe e krahason me vlerën reale.")
    dita = st.date_input("Data", value=df.date.max().date(),
                         min_value=df.date.min().date(), max_value=df.date.max().date())
    if st.button("Parashiko", type="primary", key="b1"):
        match = df.loc[df.date.dt.date == dita]
        if match.empty:
            st.warning("Nuk ka të dhëna për këtë datë.")
        else:
            row = match.iloc[0]
            show_result(*predict(row, reg, clf), pragu, reale=int(row[TARGET]))
            pjesa = "testim" if dita > date(2012, 8, 31) else (
                "validim" if dita >= date(2012, 5, 1) else "trajnim")
            st.caption(f"Kjo ditë i përket pjesës së **{pjesa}**.")

with tab2:
    dita2 = st.date_input("Data e parashikimit", value=date(2012, 12, 1), key="d2")
    festa = st.checkbox("Ditë feste")
    st.subheader("Qiratë e ditëve të kaluara")
    a, b, c = st.columns(3)
    lag1 = a.number_input("Dje", 0, 10000, 4500, step=100)
    lag7 = b.number_input("Para 7 ditësh", 0, 10000, 4500, step=100)
    mean7 = c.number_input("Mesatarja 7 ditë", 0, 10000, 4500, step=100)
    st.subheader("Moti para 7 ditësh")
    a, b, c = st.columns(3)
    temp = a.slider("Temperatura (°C)", -10.0, 40.0, 15.0, 0.5)
    rain = b.slider("Reshjet (mm)", 0.0, 60.0, 0.0, 0.5)
    wind = c.slider("Era (m/s)", 0.0, 15.0, 3.0, 0.5)

    weekday = (dita2.weekday() + 1) % 7  # si te day.csv: 0 = e diel
    st.caption(f"Dita: {DITET[weekday]}")
    if st.button("Parashiko", type="primary", key="b2"):
        row = {
            "rentals_lag1": lag1, "rentals_lag7": lag7, "rentals_mean7": mean7,
            "temp_c_lag7": temp, "rain_mm_lag7": rain, "wind_ms_lag7": wind,
            "trend_days": (pd.Timestamp(dita2) - df.date.min()).days + 7,
            "weekday": weekday, "month": dita2.month, "holiday": int(festa),
            "workingday": int(weekday not in (0, 6) and not festa),
        }
        show_result(*predict(row, reg, clf), pragu)

st.divider()
st.caption("Të dhënat: UCI Bike Sharing (Washington DC, 2011–2012) + NASA POWER. "
           "Modelet: Scikit-learn.")