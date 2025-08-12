<<<<<<< HEAD
import streamlit
=======
import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
from fuzzywuzzy import process
import json
import re
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import numpy as np

st.set_page_config(layout="wide")
st.title("Premier League Relegation Predictor")

# --- User Input ---
season = st.number_input("Enter the starting season year (2014 onwards)", min_value=2014, max_value=datetime.now().year, step=1)

if st.button("Run Prediction Pipeline"):

    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept-Language": "en-US,en;q=0.9"
    }

    @st.cache_data(show_spinner="Fetching Transfermarkt data...")
    def get_transfermarkt_table(url, headers):
        r = requests.get(url, headers=headers)
        soup = BeautifulSoup(r.text, "html.parser")
        league_table = soup.find("table", class_="items")
        if league_table is None:
            return pd.DataFrame()
        titles = [th.text.strip() for th in league_table.find_all("th")]
        df = pd.DataFrame(columns=titles)
        for row in league_table.find_all("tr")[1:]:
            row_data = [td.text.strip() for td in row.find_all("td")]
            if len(row_data) == len(titles):
                df.loc[len(df)] = row_data
        return df.drop(0, errors='ignore')

    # --- Scrape Transfermarkt ---
    prem_url = f"https://www.transfermarkt.com/premier-league/startseite/wettbewerb/GB1/plus/?saison_id={season}"
    champ_url = f"https://www.transfermarkt.com/championship/startseite/wettbewerb/GB2/plus/?saison_id={season}"

    prem_df = get_transfermarkt_table(prem_url, headers)
    champ_df = get_transfermarkt_table(champ_url, headers)

    if prem_df.empty or champ_df.empty:
        st.error("Failed to retrieve Premier League or Championship data.")
        st.stop()

    union_df = pd.concat([prem_df, champ_df]).drop_duplicates().reset_index(drop=True)

    # --- Scrape Understat Data ---
    understat_url = f"https://understat.com/league/EPL/{season}"
    res = requests.get(understat_url)
    soup = BeautifulSoup(res.content, "html.parser")
    ugly_soup = str(soup)

    team_data = re.search(r"var teamsData\s*=\s*JSON.parse\('(.*)'\)", ugly_soup)
    if not team_data:
        st.error("Failed to retrieve Understat data.")
        st.stop()

    json_str = team_data.group(1)
    json_data = json_str.encode("utf8").decode("unicode_escape")
    data = json.loads(json_data)
    t_df = pd.DataFrame(data).transpose()
    exploded = t_df.explode("history").reset_index(drop=True)
    history = pd.json_normalize(exploded["history"])
    result = exploded.drop(columns=["history"]).join(history)
    result = result.sort_values(['title', 'date']).reset_index(drop=True)
    result["matchday"] = result.groupby("title").cumcount() + 1

    # --- Feature Engineering ---
    sum_cols = ["pts", "scored", "missed", "xpts"]
    mean_cols = ["xG", "xGA", "npxG", "npxGA", "deep", "deep_allowed", 
                 "npxGD", "ppda.att", "ppda.def", "ppda_allowed.att", "ppda_allowed.def"]

    for col in sum_cols:
        result[f"cumsum_{col}"] = result.groupby("title")[col].cumsum()

    result["cumcount"] = result.groupby("title").cumcount() + 1
    for col in mean_cols:
        result[f"cumsum_{col}"] = result.groupby("title")[col].cumsum()
        result[f"cummean_{col}"] = result[f"cumsum_{col}"] / result["cumcount"]

    result = result.drop(columns=[f"cumsum_{col}" for col in mean_cols])

    # --- Fuzzy Match Team Names ---
    mapped = {}
    left_names = union_df["name"]
    right_names = result["title"]
    for name in left_names:
        match, score, _ = process.extractOne(name, right_names)
        if score > 80:
            mapped[name] = match
    union_df["name"] = union_df["name"].replace(mapped)

    df = pd.merge(union_df, result, left_on="name", right_on="title", how="right")

    # --- More Feature Engineering ---
    df["goal_difference_so_far"] = df["cumsum_scored"] - df["cumsum_missed"]
    df["points_per_game"] = df["cumsum_pts"] / df["matchday"]
    df["attack_strength"] = df["cummean_xG"] - df["cummean_xGA"]
    df["defense_solid"] = df["cummean_ppda.def"] - df["cummean_ppda_allowed.def"]
    df["tactical_balance"] = df["cummean_deep"] - df["cummean_deep_allowed"]
    df["non-penality GD"] = df["cummean_npxG"] - df["cummean_npxGA"]

    df["avg_player_value"] = df["ø market value"].str.extract(r"(\d+\.\d+)").astype(float)
    df = df.rename(columns={'cummean_npxGD': 'non-penalty xGD', 'cummean_ppda.att': 'pressing intensity',
                            'cummean_ppda_allowed.att': 'defending pressing intensity', 'ø age': 'avg_age',
                            'Squad': 'squad_size'})

    drop_cols = ['id', 'h_a', 'xG', 'xGA', 'npxG', 'npxGA', 'result', 'date', 'wins', 'draws', 'loses', 'pts',
                 'npxGD', 'ppda_allowed.att', 'ppda_allowed.def', 'deep', 'deep_allowed', 'scored', 'missed',
                 'xpts', 'ppda.att', 'ppda.def', 'cumcount', 'Total market value', 'cumsum_scored',
                 'cumsum_missed', 'cummean_xG', 'cummean_xGA', 'cummean_ppda.def', 'cummean_ppda_allowed.def',
                 'cummean_deep', 'cummean_deep_allowed', 'cummean_npxG', 'cummean_npxGA', 'Club', 'title',
                 'ø market value']
    df = df.drop(columns=[col for col in drop_cols if col in df.columns])

    # --- Relegation Labels ---
    final_game = df[df["matchday"] == df["matchday"].max()]
    final_game = final_game.sort_values(["cumsum_pts", "goal_difference_so_far"]).reset_index(drop=True)
    relegated_teams = final_game.iloc[:3]["name"].tolist()
    df["relegated"] = df["name"].apply(lambda x: 1 if x in relegated_teams else 0)

    # --- ML Model ---
    prob_list = []
    for md in sorted(df["matchday"].unique()):
        train_df = df[df["matchday"] < md]
        test_df = df[df["matchday"] == md].copy()
        if train_df.empty or test_df.empty:
            continue
        X_train = train_df.drop(columns=["name", "relegated"])
        y_train = train_df["relegated"]
        X_test = test_df.drop(columns=["name", "relegated"])
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        test_df["relegation_proba"] = model.predict_proba(X_test)[:, 1]
        test_df["matchday"] = md
        prob_list.append(test_df[["name", "matchday", "relegation_proba"]])

    if prob_list:
        relegation_probs = pd.concat(prob_list)
        st.success("Prediction complete!")

        st.subheader("Relegation Probability Over Matchdays")
        fig, ax = plt.subplots(figsize=(10, 6))
        for team in relegation_probs["name"].unique():
            team_data = relegation_probs[relegation_probs["name"] == team]
            ax.plot(team_data["matchday"], team_data["relegation_proba"], label=team)
        ax.set_xlabel("Matchday")
        ax.set_ylabel("Relegation Probability")
        ax.set_title(f"Relegation Probability per Team Over the {season} Season")
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax.grid(True)
        st.pyplot(fig)

        st.subheader("Latest Relegation Probabilities")
        latest_probs = relegation_probs[relegation_probs["matchday"] == 5]
        latest_probs = latest_probs.drop_duplicates(subset="name", keep="first")
        st.dataframe(latest_probs.sort_values("relegation_proba", ascending=False).reset_index(drop=True))

    else:
        st.warning("Not enough data to make predictions.")
>>>>>>> 681714be74fa493a4d3290bfcf9e8714037dbc68
