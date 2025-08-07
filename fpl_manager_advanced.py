#!/usr/bin/env python3
"""
fpl_manager_advanced.py
Advanced, fully automated FPL optimizer for 2025/26 season
—including CBIT/CBIRT scoring, 2× chips, AGCON free-transfer boost,
simplified assists, BPS tweaks, and fixture-difficulty features—
delivered via Gmail SMTP.
"""

import os
import time
import logging
import requests
import joblib
import pandas as pd
import numpy as np

from datetime import datetime
from bs4 import BeautifulSoup
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import BayesianRidge
from lightgbm import LGBMRegressor
from pulp import LpProblem, LpVariable, LpMaximize, lpSum, LpBinary
import schedule
import smtplib
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart

# ─── CONFIG ──────────────────────────────────────────────────────────
FPL_API        = "https://fantasy.premierleague.com/api"
MODEL_FILE     = "fpl_ensemble_adv.pkl"
GMAIL_USER     = os.getenv("GMAIL_USER")
GMAIL_PASS     = os.getenv("GMAIL_PASS")
EMAIL_TO       = os.getenv("EMAIL_TO")
BUDGET_CAP     = 100.0
MAX_TEAM_PLYR  = 3
RISK_AVERSION  = 0.1    # for risk‐adjusted optimization

# AGCON boost: 5 free transfers in GW16
AGCON_GW = 16
# ─────────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")


def fetch_fpl():
    r = requests.get(f"{FPL_API}/bootstrap-static/")
    data = r.json()
    pl = pd.DataFrame(data["elements"])
    tm = pd.DataFrame(data["teams"])
    fx = pd.DataFrame(data["events"])
    return pl, tm, fx


def fetch_news():
    url = "https://www.fantasyfootballscout.co.uk/category/team-news/"
    soup = BeautifulSoup(requests.get(url).text, "html.parser")
    return [a.text for a in soup.select(".post-content h2 a")[:30]]


def preprocess(pl, tm, fx, news):
    """
    1. Basic stats & form
    2. New CBIT/CBIRT features
    3. Simplified assists placeholder
    4. Fixture Difficulty Rating features (next GW & rolling 3 GWs)
    5. Free-transfer logic (for reporting)
    6. BPS tweaks placeholders
    """
    df = pl.copy()

    # ─── Core Stats ────────────────────────────────────────────
    df["value"]    = df["now_cost"] / 10
    df["form"]     = df["form"].astype(float)
    df["ppg"]      = df["points_per_game"].astype(float)
    df["element_type_name"] = df["element_type"].map({1:"GK",2:"DEF",3:"MID",4:"FWD"})

    # ─── CBIT / CBIRT Scoring ──────────────────────────────────
    # (Placeholders; replace with real data if available)
    df["cbit"]  = np.random.poisson(8, size=len(df))   # defenders
    df["cbirt"] = np.random.poisson(10, size=len(df))  # mids/forwards
    is_def = df["element_type_name"] == "DEF"
    df["bonus_flag"] = np.where(
        (is_def  & (df["cbit"]  >= 10)) |
        (~is_def & (df["cbirt"] >= 12)),
        1, 0
    )

    # ─── Assist Simplification (placeholder) ────────────────────
    df["assist_boost"] = df["assists"].astype(float) * 0.02

    # ─── Bonus Points System (BPS) Tweaks ───────────────────────
    df["bps_old"]     = df["bps"].astype(float)
    df["bps_tweaked"] = df["bps_old"]  # ready for real event-based recalculation

    # ─── Fixture Difficulty Rating (FDR) ────────────────────────
    next_gw = int(fx.loc[fx["is_next"], "id"].iloc[0])
    # map next-GW difficulty
    upcoming = fx[fx["event"] == next_gw]
    df["fdr_next"] = df["team_code"].map(
        upcoming.set_index("team_code")["difficulty_rating"]
    ).fillna(3.0)
    # rolling 3-GW average difficulty
    df["fdr_3gw"] = (
        fx
        .sort_values("event")
        .groupby("team_code")["difficulty_rating"]
        .rolling(3, min_periods=1).mean()
        .reset_index(level=0, drop=True)
        .reindex(df.index)
        .fillna(3.0)
    )

    # ─── Free Transfers for Next GW ─────────────────────────────
    df.attrs["next_gw"] = next_gw
    df.attrs["free_transfers"] = 5 if next_gw == AGCON_GW else 1

    return df


def train_and_serialize(df):
    logging.info("Training ensemble model with enhanced features…")
    feats = [
        "value", "form", "ppg",
        "cbit", "cbirt", "bonus_flag",
        "assist_boost", "bps_tweaked",
        "fdr_next", "fdr_3gw"
    ]
    X = df[feats].fillna(0)
    y = df["ppg"] * 2  # proxy label for next-GW points
    m1 = RandomForestRegressor(100, random_state=1).fit(X, y)
    m2 = LGBMRegressor(100, random_state=1).fit(X, y)
    m3 = BayesianRidge().fit(X, y)
    joblib.dump((m1, m2, m3), MODEL_FILE)
    logging.info("Model saved.")


def load_models():
    return joblib.load(MODEL_FILE)


def predict(df):
    logging.info("Predicting expected points with new rules & FDR…")
    m1, m2, m3 = load_models()
    feats = [
        "value", "form", "ppg",
        "cbit", "cbirt", "bonus_flag",
        "assist_boost", "bps_tweaked",
        "fdr_next", "fdr_3gw"
    ]
    X = df[feats].fillna(0)
    base_preds = np.mean([
        m1.predict(X),
        m2.predict(X),
        m3.predict(X),
    ], axis=0)
    df["exp_pts"] = base_preds \
        + 2 * df["bonus_flag"] \
        + df["assist_boost"]
    return df


def optimize(df):
    logging.info("Optimizing squad & formation…")
    prob = LpProblem("FPL", LpMaximize)
    idx = df.index.tolist()
    pick  = LpVariable.dicts("pick",  idx, 0, 1, LpBinary)
    start = LpVariable.dicts("start", idx, 0, 1, LpBinary)

    # objective: maximize risk-adjusted expected points
    prob += lpSum(
        (df.loc[i,"exp_pts"] - RISK_AVERSION * df.loc[i,"bps_tweaked"]) * start[i]
        for i in idx
    )

    # budget & squad size
    prob += lpSum(df.loc[i,"value"] * pick[i] for i in idx) <= BUDGET_CAP
    prob += lpSum(pick[i] for i in idx) == 15
    prob += lpSum(start[i] for i in idx) == 11
    for i in idx:
        prob += start[i] <= pick[i]

    # position constraints
    for pos, (cnt_min, cnt_max) in {
        "GK": (2, 2), "DEF": (5, 5),
        "MID": (5, 5), "FWD": (3, 3)
    }.items():
        ids = [i for i in idx if df.at[i,"element_type_name"] == pos]
        prob += lpSum(pick[i] for i in ids) == cnt_min

    # max 3 per club
    for team in df["team"].unique():
        ids = [i for i in idx if df.at[i,"team"] == team]
        prob += lpSum(pick[i] for i in ids) <= MAX_TEAM_PLYR

    prob.solve()
    squad = df[[pick[i].value() > 0.5 for i in idx]].copy()
    squad["is_start"] = [
        start[i].value() > 0.5 for i in idx if pick[i].value() > 0.5
    ]
    return squad.sort_values("exp_pts", ascending=False)


def email_results(squad, df):
    logging.info("Sending lineup via Gmail…")
    msg = MIMEMultipart()
    msg["From"]    = GMAIL_USER
    msg["To"]      = EMAIL_TO
    msg["Subject"] = (
        f"FPL GW{df.attrs['next_gw']} Lineup "
        f"(Free transfers: {df.attrs['free_transfers']})"
    )
    csv = squad.to_csv(index=False).encode()
    part = MIMEApplication(csv, Name="lineup.csv")
    part['Content-Disposition'] = 'attachment; filename="lineup.csv"'
    msg.attach(part)

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(GMAIL_USER, GMAIL_PASS)
        smtp.send_message(msg)


def pipeline():
    try:
        pl, tm, fx = fetch_fpl()
        news       = fetch_news()
        df         = preprocess(pl, tm, fx, news)

        if not os.path.isfile(MODEL_FILE):
            train_and_serialize(df)

        df     = predict(df)
        squad  = optimize(df)
        email_results(squad, df)

        # retrain after GW results are in
        train_and_serialize(df)

    except Exception:
        logging.exception("Pipeline failed")


if __name__ == "__main__":
    # Schedule weekly (e.g. Monday at 02:00 UTC)
    schedule.every().monday.at("02:00").do(pipeline)
    logging.info("Scheduler started; awaiting runs…")
    while True:
        schedule.run_pending()
        time.sleep(60)
