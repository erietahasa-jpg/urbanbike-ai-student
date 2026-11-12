"""Hapi 2: Grafikët për të kuptuar ndryshimet e kërkesës.

Nisja nga dosja kryesore e projektit:
    python src/02_grafiket.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # ruan PNG pa hapur dritare
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.common import DATA, FIGURES, TARGET, save_json  # noqa: E402

MUAJT = ["Jan", "Shk", "Mar", "Pri", "Maj", "Qer",
         "Kor", "Gus", "Sht", "Tet", "Nën", "Dhj"]
BLU = "#2a6fdb"


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIGURES / name, dpi=150)
    plt.close(fig)
    print("U ruajt:", FIGURES / name)


def main():
    df = pd.read_csv(DATA / "processed" / "model_data.csv", parse_dates=["date"])

    # 1. Kërkesa në kohë + mesatarja 30-ditore
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df.date, df[TARGET], color=BLU, alpha=0.35, lw=1, label="Ditore")
    ax.plot(df.date, df[TARGET].rolling(30, center=True).mean(),
            color=BLU, lw=2.5, label="Mesatarja 30-ditore")
    ax.set(title="Kërkesa ditore për biçikleta në kohë",
           xlabel="Data", ylabel="Qira në ditë")
    ax.legend(frameon=False)
    save(fig, "01_kerkesa_ne_kohe.png")

    # 2. Sipas muajit
    by_month = df.groupby("month")[TARGET].mean()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar([MUAJT[m - 1] for m in by_month.index], by_month.values, color=BLU)
    ax.set(title="Kërkesa mesatare sipas muajit", ylabel="Qira në ditë")
    save(fig, "02_sipas_muajit.png")

    # 3. Sipas ditës së javës (në day.csv: 0 = e diel)
    by_day = df.groupby("weekday")[TARGET].mean()
    names = {0: "Die", 1: "Hën", 2: "Mar", 3: "Mër", 4: "Enj", 5: "Pre", 6: "Sht"}
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([names[d] for d in by_day.index], by_day.values, color=BLU)
    ax.set(title="Kërkesa mesatare sipas ditës së javës", ylabel="Qira në ditë")
    save(fig, "03_sipas_dites.png")

    # 4. Ditë pune kundrejt pushimit
    fig, ax = plt.subplots(figsize=(6, 4))
    groups = [df.loc[df.workingday == v, TARGET] for v in (0, 1)]
    ax.boxplot(groups)
    ax.set_xticks([1, 2], ["Pushim / fundjavë", "Ditë pune"])
    ax.set(title="Ditë pune kundrejt pushimit", ylabel="Qira në ditë")
    save(fig, "04_dite_pune.png")

    # 5. Temperatura dhe reshjet
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].scatter(df.temp_c, df[TARGET], s=10, alpha=0.5, color=BLU)
    axes[0].set(title="Temperatura dhe kërkesa",
                xlabel="Temperatura (°C)", ylabel="Qira në ditë")
    rain = pd.cut(df.rain_mm, [-0.1, 0.5, 5, 20, 1000],
                  labels=["Pa shi", "Pak", "Mesatar", "Shumë"])
    by_rain = df.groupby(rain, observed=False)[TARGET].mean()
    axes[1].bar(by_rain.index.astype(str), by_rain.values, color=BLU)
    axes[1].set(title="Reshjet dhe kërkesa",
                xlabel="Reshjet ditore", ylabel="Qira mesatare")
    save(fig, "05_moti.png")

    # 6. Shpërndarja + pragu për "kërkesë e lartë" (përdoret te klasifikimi)
    limit = df.loc[df.date <= "2012-04-30", TARGET].quantile(0.75)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(df[TARGET], bins=30, color=BLU, alpha=0.8)
    ax.axvline(limit, color="#d1495b", lw=2, ls="--",
               label=f"Pragu 'e lartë' ≈ {limit:.0f}")
    ax.set(title="Shpërndarja e kërkesës ditore",
           xlabel="Qira në ditë", ylabel="Numri i ditëve")
    ax.legend(frameon=False)
    save(fig, "06_shperndarja.png")

    # 7. Korrelacioni me kërkesën
    cols = ["rentals_lag1", "rentals_lag7", "rentals_mean7",
            "temp_c", "rain_mm", "wind_ms", "trend_days"]
    corr = df[cols + [TARGET]].corr()[TARGET].drop(TARGET).sort_values()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(corr.index, corr.values,
            color=[BLU if v >= 0 else "#d1495b" for v in corr.values])
    ax.axvline(0, color="grey", lw=1)
    ax.set(title="Korrelacioni me kërkesën", xlabel="Korrelacioni (r)")
    save(fig, "07_korrelacioni.png")

    save_json("grafiket.json", {
        "mesatarja": round(float(df[TARGET].mean()), 1),
        "muaji_me_i_larte": MUAJT[int(by_month.idxmax()) - 1],
        "muaji_me_i_ulet": MUAJT[int(by_month.idxmin()) - 1],
        "pragu_e_larte": round(float(limit), 1),
        "korrelacioni": {k: round(float(v), 3) for k, v in corr.items()},
    })


if __name__ == "__main__":
    main()