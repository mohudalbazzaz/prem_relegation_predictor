import data.prem_data_collection
import importlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GroupKFold, cross_val_score

importlib.reload(data.prem_data_collection)
df = data.prem_data_collection.df

train_df = df[df["matchday"] <= 4]
test_df = df[df["matchday"] == 5]

X_train = train_df.drop(columns=["name", "relegated"])
y_train = train_df["relegated"]

X_test = test_df.drop(columns=["name", "relegated"])
y_test = test_df["relegated"]

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# from RandomForest module 
test_df["relegation_proba"] = model.predict_proba(X_test)[:, 1]

print(test_df[["name", "matchday", "relegation_proba"]].sort_values("relegation_proba", ascending=False))

prob_list = []

for md in sorted(df["matchday"].unique()):

    train_df = df[df["matchday"] < md]
    test_df = df[df["matchday"] == md]

    # when md = 1 there exists not training data
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

# combines all probabilities into one df so easier to visualise 
relegation_probs = pd.concat(prob_list)

plt.figure(figsize=(10, 6))

for team in relegation_probs["name"].unique():
    team_data = relegation_probs[relegation_probs["name"] == team]
    plt.plot(team_data["matchday"], team_data["relegation_proba"], label=team)

plt.xlabel("Matchday")
plt.ylabel("Relegation Probability")
season = data.prem_data_collection.prem_season
plt.title(f"Relegation Probability per Team Over the Season {(season)}")
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)
plt.grid(True)
plt.tight_layout()
plt.show()

# from RandomForest
importances = model.feature_importances_

# match them to column names
feature_names = X_train.columns
feature_importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

print(feature_importance_df)