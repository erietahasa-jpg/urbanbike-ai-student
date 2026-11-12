"""Hapi 1: Pastrimi i të dhënave dhe krijimi i model_data.csv.

Nisja nga dosja kryesore e projektit:
    python src/01_pastrimi.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.common import DATA, FEATURES, TARGET, save_json  # noqa: E402

RAW = DATA / "raw"
OUT = DATA / "processed"


def pick(columns, candidates, what):
    """Gjen emrin e kolonës nga një listë emrash të mundshëm."""
    for name in candidates:
        if name in columns:
            return name
    raise KeyError(
        f"Nuk u gjet kolona për '{what}'. Kolonat ekzistuese: {list(columns)}"
    )


def check(df, name):
    """Kontrolli: vlerat që mungojnë, dublimet dhe tipet."""
    return {
        "rreshta": int(len(df)),
        "vlera_qe_mungojne": {k: int(v) for k, v in df.isna().sum().items() if v},
        "rreshta_dublikate": int(df.duplicated().sum()),
        "data_dublikate": int(df["date"].duplicated().sum()),
        "tipet": {k: str(v) for k, v in df.dtypes.items()},
        "burimi": name,
    }


def load_bikes():
    df = pd.read_csv(RAW / "day.csv")
    df = df.rename(columns={"dteday": "date", "cnt": TARGET, "mnth": "month"})
    df["date"] = pd.to_datetime(df["date"])
    # casual + registered = cnt -> do të ishte "kopjim" i përgjigjes
    keep = ["date", TARGET, "weekday", "month", "workingday", "holiday"]
    return df[keep]


def load_weather():
    df = pd.read_csv(RAW / "nasa_weather.csv")
    cols = df.columns
    if any(c in cols for c in ["date", "DATE", "Date"]):
        df["date"] = pd.to_datetime(df[pick(cols, ["date", "DATE", "Date"], "data")])
    elif {"YEAR", "DOY"} <= set(cols):
        df["date"] = pd.to_datetime(
            df["YEAR"].astype(str) + df["DOY"].astype(str).str.zfill(3), format="%Y%j"
        )
    else:
        df["date"] = pd.to_datetime(
            dict(year=df["YEAR"], month=df["MO"], day=df["DY"])
        )
    out = pd.DataFrame({
        "date": df["date"],
        "temp_c": df[pick(cols, ["temp_c", "T2M"], "temperatura")],
        "rain_mm": df[pick(cols, ["rain_mm", "PRECTOTCORR", "PRECTOT"], "reshjet")],
        "wind_ms": df[pick(cols, ["wind_ms", "WS2M", "WS10M"], "era")],
    })
    # NASA POWER e shënon mungesën e vlerës me -999
    return out.replace(-999, np.nan)


def clean(df):
    df = df.drop_duplicates()
    df = df.drop_duplicates(subset="date", keep="first")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return df


def build_features(df):
    df = df.copy()
    past = df[TARGET].shift(1)  # vetëm të dhëna nga dita e kaluar
    df["rentals_lag1"] = past
    df["rentals_lag7"] = df[TARGET].shift(7)
    df["rentals_mean7"] = past.rolling(7).mean()
    for col in ["temp_c", "rain_mm", "wind_ms"]:
        df[f"{col}_lag7"] = df[col].shift(7)
    df["trend_days"] = (df["date"] - df["date"].min()).dt.days
    for col in ["weekday", "month", "workingday", "holiday"]:
        df[col] = df[col].astype(int)
    # 7 ditët e para s'kanë histori -> hiqen
    return df.dropna(subset=["rentals_lag7", "rentals_mean7"]).reset_index(drop=True)


def main():
    bikes, weather = load_bikes(), load_weather()
    report = {"para": {"biciklete": check(bikes, "day.csv"),
                       "moti": check(weather, "nasa_weather.csv")}}

    bikes, weather = clean(bikes), clean(weather)
    df = bikes.merge(weather, on="date", how="left")  # ruan çdo ditë qiraje
    df = build_features(df)

    report["pas"] = check(df, "model_data.csv")
    report["periudha"] = [str(df.date.min().date()), str(df.date.max().date())]
    save_json("pastrimi.json", report)

    df[["date", TARGET] + FEATURES + ["temp_c", "rain_mm", "wind_ms"]].to_csv(
        OUT / "model_data.csv", index=False
    )
    print(f"U ruajt: {OUT / 'model_data.csv'} ({len(df)} rreshta)")
    print("Vlera që mungojnë:", report["pas"]["vlera_qe_mungojne"] or "asnjë")


if __name__ == "__main__":
    main()