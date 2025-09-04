import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import json
import re
import random
import time
from datetime import datetime
from fuzzywuzzy import process
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("Premier League Relegation Predictor")

# --- Season Input ---
season = st.number_input(
    "Enter the season (e.g. for 2022/23 enter 2022)",
    min_value=2014, max_value=datetime.now().year, step=1
)

# --- User-Agent Pool for rotation ---
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/15.1 Safari/605.1.15",
    "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:89.0) Gecko/20100101 Firefox/89.0"
]

def make_request(url):
    """Make a GET request with realistic headers and delays."""
    headers = {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-GB,en-US;q=0.9,en;q=0.8",
        "Referer": "https://www.google.com/",
        "Upgrade-Insecure-Requests": "1"
    }
    time.sleep(random.uniform(2, 5))
    return requests.get(url, headers=headers, timeout=20)

@st.cache_data(show_spinner="Fetching Transfermarkt data...")
def get_transfermarkt_table(url):
    """Scrape the league table from Transfermarkt."""
    response = make_request(url)
    soup = BeautifulSoup(response.text, "html.parser")
    table = soup.find("table", class_="items")
    if not table:
        return pd.DataFrame()
    cols = [th.text.strip() for th in table.find_all("th")]
    df = pd.DataFrame(columns=cols)
    for row in table.find_all("tr")[1:]:
        cells = [td.text.strip() for td in row.find_all("td")]
        if len(cells) == len(cols):
            df.loc[len(df)] = cells
    return df.drop(0, errors='ignore')

def load_understat_data(season):
    """Scrape historical match data from Understat."""
    url = f"https://understat.com/league/EPL/{season}"
    res = make_request(url)
    soup = BeautifulSoup(res.content, "html.parser")
    match = re.search(r"var teamsData\s*=\s*JSON.parse\('(.*)'\)", str(soup))
    if not match:
        return None
    raw_json = match.group(1).encode('utf-8').decode('unicode_escape')
    return json.loads(raw_json)

if st.button("Run Prediction Pipeline"):
    with st.spinner("Collecting data..."):

        prem_url = f"https://www.transfermarkt.com/premier-league/startseite/wettbewerb/GB1/plus/?saison_id={season}"
        champ_url = f"https://www.transfermarkt.com/championship/startseite/wettbewerb/GB2/plus/?saison_id={season}"

        prem_df = get_transfermarkt_table(prem_url)
        champ_df = get_transfermarkt_table(champ_url)

        if prem_df.empty or champ_df.empty:
            st.error("❌ Could not load Premier League or Championship table from Transfermarkt. They may be blocking the request.")
            st.stop()

        union_df = pd.concat([prem_df, champ_df]).drop_duplicates().reset_index(drop=True)

        understat_data = load_understat_data(season)
        if not understat_data:
            st.error("❌ Could not load match history from Understat.")
            st.stop()

    # --- Process Understat data ---
    tdf = pd.DataFrame(understat_data).transpose().explode("history").reset_index(drop=True)
    hist = pd.json_normalize(tdf["history"])
    merged = tdf.drop(columns=["history"]).join(hist)
    merged = merged.sort_values(["title", "date"]).reset_index(drop=True)
    merged["matchday"] = merged.groupby("title").cumcount() + 1

    # --- Feature Engineering ---
    sum_cols = ["pts", "scored", "missed", "xpts"]
    mean_cols = ["xG", "xGA", "npxG", "npxGA",
                 "deep", "deep_allowed", "npxGD",
                 "ppda.att", "ppda.def",
                 "ppda_allowed.att", "ppda_allowed.def"]

    for col in sum_cols:
        merged[f"cumsum_{col}"] = merged.groupby("title")[col].cumsum()
    merged["cumcount"] = merged.groupby("title").cumcount() + 1
    for col in mean_cols:
        merged[f"cumsum_{col}"] = merged.groupby("title")[col].cumsum()
        merged[f"cummean_{col}"] = merged[f"cumsum_{col}"] / merged["cumcount"]

    merged = merged.drop(columns=[f"cumsum_{c}" for c in mean_cols])

    # --- Match names ---
    name_map = {}
    for team in union_df["name"]:
        match, score, _ = process.extractOne(team, merged["title"])
        if score > 80:
            name_map[team] = match
    union_df["name"] = union_df["name"].replace(name_map)

    df = pd.merge(union_df, merged, left_on="name", right_on="title", how="right")

    # --- Final features ---
    df["goal_difference_so_far"] = df["cumsum_scored"] - df["cumsum_missed"]
    df["points_per_game"] = df["cumsum_pts"] / df["matchday"]
    df["attack_strength"] = df["cummean_xG"] - df["cummean_xGA"]
    df["defense_solid"] = df["cummean_ppda.def"] - df["cummean_ppda_allowed.def"]
    df["tactical_balance"] = df["cummean_deep"] - df["cummean_deep_allowed"]
    df["non-penalty GD"] = df["cummean_npxG"] - df["cummean_npxGA"]
    df["avg_player_value"] = df["ø market value"].str.extract(r"(\d+\.\d+)").astype(float)

    df = df.rename(columns={'Squad': 'squad_size', 'ø age': 'avg_age'})

    # Drop unneeded columns
    drop_cols = [
        'id', 'h_a', 'xG', 'xGA', 'npxG', 'npxGA', 'result', 'date', 'wins',
        'draws', 'loses', 'pts', 'npxGD', 'ppda_allowed.att', 'ppda_allowed.def',
        'deep', 'deep_allowed', 'scored', 'missed', 'xpts', 'ppda.att',
        'ppda.def', 'cumcount', 'Total market value', 'cumsum_scored',
        'cumsum_missed', 'cummean_xG', 'cummean_xGA', 'cummean_ppda.def',
        'cummean_ppda_allowed.def', 'cummean_deep', 'cummean_deep_allowed',
        'cummean_npxG', 'cummean_npxGA', 'Club', 'title', 'ø market value'
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # --- Label relegation ---
    final_md = df["matchday"].max()
    final_game = df[df["matchday"] == final_md]
    final_game = final_game.sort_values(["cumsum_pts", "goal_difference_so_far"]).reset_index(drop=True)
    relegated = final_game.iloc[:3]["name"].tolist()
    df["relegated"] = df["name"].apply(lambda x: 1 if x in relegated else 0)

    # --- Model & Predictions ---
    prob_list = []
    for md in sorted(df["matchday"].unique()):
        train = df[df["matchday"] < md]
        test = df[df["matchday"] == md].copy()
        if train.empty or test.empty:
            continue
        X_train = train.drop(columns=["name", "relegated"])
        y_train = train["relegated"]
        X_test = test.drop(columns=["name", "relegated"])
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        test["relegation_proba"] = model.predict_proba(X_test)[:, 1]
        test["matchday"] = md
        prob_list.append(test[["name", "matchday", "relegation_proba"]])

    if not prob_list:
        st.warning("Not enough data for predictions.")
        st.stop()

    results = pd.concat(prob_list)

    st.success("✅ Prediction complete!")

    # --- Visualize ---
    st.subheader("Relegation Probability Over Matchdays")
    fig, ax = plt.subplots(figsize=(10, 6))
    for team in results["name"].unique():
        td = results[results["name"] == team]
        ax.plot(td["matchday"], td["relegation_proba"], label=team)
    ax.set_xlabel("Matchday")
    ax.set_ylabel("Relegation Probability")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True)
    st.pyplot(fig)

    st.subheader("Relegation Probabilities at Matchday 5")
    md5 = results[results["matchday"] == 5].drop_duplicates("name")
    st.dataframe(md5.sort_values("relegation_proba", ascending=False).reset_index(drop=True))
