import requests
from bs4 import BeautifulSoup
import pandas as pd
from fuzzywuzzy import process
import json
import re
from datetime import datetime

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/115.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9"
}

def get_transfermarkt_table(url, headers):
    """
    Scrapes a Transfermarkt league table from the given URL and returns it as a pandas DataFrame.

    Args:
        url (str): The URL of the Transfermarkt league page.
        headers (Dict[str, str]): HTTP headers to use (e.g user-agent) for the request to circumvent bot protection (mimics a normal web browser)

    Returns:
        pd.DataFrame: A DataFrame containing the cleaned table data.
    """
    r = requests.get(url, headers=headers)
    soup = BeautifulSoup(r.text, "html.parser")

    league_table = soup.find("table", class_="items")
    if league_table is None:
        raise ValueError("Could not find table with class 'items' on the page.")

    titles = league_table.find_all("th")
    clean_titles = [title.text.strip() for title in titles]

    df = pd.DataFrame(columns=clean_titles)

    for row in league_table.find_all("tr")[1:]:
        row_data = row.find_all("td")
        individual_row_data = [data.text.strip() for data in row_data]

        if len(individual_row_data) == len(clean_titles):
            df.loc[len(df)] = individual_row_data

    return df.drop(0, errors='ignore')

prem_base_url = "https://www.transfermarkt.com/premier-league/startseite/wettbewerb/GB1/plus/?saison_id="
champ_base_url = "https://www.transfermarkt.com/premier-league/startseite/wettbewerb/GB1/plus/?saison_id="

def run_data_collection():
    try:
        prem_season_input = input("Please enter the starting season year: ")
        prem_season = int(prem_season_input)

        if 2014 <= prem_season <= datetime.now().year:
            prem_url = prem_base_url + str(prem_season)
        else:
            raise ValueError("Premier League season must be a whole number from 2014 onwards.")

        champ_season_input = input("Please enter the starting season year (same as above): ")
        champ_season = int(champ_season_input)

        if 2014 <= champ_season <= datetime.now().year and champ_season == prem_season:
            champ_url = champ_base_url + str(champ_season)
        else:
            raise ValueError("Championship season must be the same as Premier League and valid.")

        prem_mkt_val_df = get_transfermarkt_table(prem_url, headers)
        champ_mkt_val_df = get_transfermarkt_table(champ_url, headers)

        return prem_mkt_val_df, champ_mkt_val_df, prem_season

    except ValueError as ve:
        print(f"Input error: {ve}")
    except Exception as e:
        print(f"An error occurred: {e}")

# run run_data_collection only if file is executed directly - not when imported
if __name__ == "__main__":
    run_data_collection()

prem_mkt_val_df, champ_mkt_val_df, prem_season = run_data_collection()

union_df = pd.concat([prem_mkt_val_df, champ_mkt_val_df]).drop_duplicates().reset_index(drop=True)

understat_base_url = "https://understat.com/league/EPL/"

try:
    understat_season = int(input('Please enter the starting season year (same as before): '))
    if 2014 <= understat_season <= datetime.now().year and understat_season == prem_season:
        understat_url = understat_base_url + str(understat_season)
    else:
        raise ValueError("Season must be same as previous season and valid")
    
except ValueError as ve:
    print(f"Input error: {ve}")
except Exception as e:
    print(f"An error occurred: {e}")

res = requests.get(understat_url)
soup = BeautifulSoup(res.content, "html.parser")
ugly_soup = str(soup)

team_data = re.search(r"var teamsData\s*=\s*JSON.parse\('(.*)'\)", ugly_soup)

json_str = team_data.group(1)

json_data = json_str.encode("utf8").decode("unicode_escape")
data = json.loads(json_data)

json_df = pd.DataFrame(data)

t_df = json_df.transpose()

df_exploded = t_df.explode("history").reset_index(drop=True)

history_expanded = pd.json_normalize(df_exploded["history"])

result = df_exploded.drop(columns=["history"]).join(history_expanded)

result = result.sort_values(['title', 'date']).reset_index(drop=True)
result['matchday'] = result.groupby('title').cumcount() + 1

result = result.sort_values(["title", "matchday"]).reset_index(drop=True)

result["matchday"] = result.groupby("title").cumcount() + 1

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

left_names = union_df["name"]
right_names = result["title"]

mapped = {}
for name in left_names:
    match, score, _ = process.extractOne(name, right_names)
    if score > 80:
        mapped[name] = match

union_df["name"] = union_df["name"].replace(mapped)

df = pd.merge(union_df, result, left_on="name", right_on="title", how="right")

df["goal_difference_so_far"] = df["cumsum_scored"] - df["cumsum_missed"]
df["points_per_game"] = df["cumsum_pts"] / df["matchday"]
df["attack_strength"] = df["cummean_xG"] - df["cummean_xGA"]
df["defense_solid"] = df["cummean_ppda.def"] - df["cummean_ppda_allowed.def"]
df["tactical_balance"] = df["cummean_deep"] - df["cummean_deep_allowed"]
df["non-penality GD"] = df["cummean_npxG"] - df["cummean_npxGA"]

df = df.drop(columns=['id', 'h_a', 'xG', 'xGA', 'npxG',
    'npxGA', 'result', 'date', 'wins', 'draws', 'loses', 'pts', 'npxGD',
    'ppda_allowed.att', 'ppda_allowed.def', 'deep', 'deep_allowed', 'scored', 'missed',
    'xpts', 'ppda.att', 'ppda.def', 'cumcount', 'Total market value', 'cumsum_scored', 'cumsum_missed',
    'cummean_xG', 'cummean_xGA', 'cummean_ppda.def', 
    'cummean_ppda_allowed.def', 'cummean_deep', 'cummean_deep_allowed', 'cummean_npxG', 'cummean_npxGA', 'Club',
    'title'
    ])

df = df.rename(columns={'cummean_npxGD': 'non-penalty xGD', 'cummean_ppda.att': 'pressing intensity',
                    'cummean_ppda_allowed.att': 'defending pressing intensity', 'ø age': 'avg_age',
                    'Squad': 'squad_size'
                    })

final_game = df[df["matchday"] == df["matchday"].max()]
final_game = final_game.sort_values(["cumsum_pts", "goal_difference_so_far"]).reset_index(drop=True)
relegated_teams = final_game.iloc[:3, 0].tolist()

df["relegated"] = df["name"].apply(lambda x: 1 if x in relegated_teams else 0)

df["avg_player_value"] = df["ø market value"].str.extract(r"(\d+\.\d+)")

df = df.drop(columns=["ø market value"])

print(df)