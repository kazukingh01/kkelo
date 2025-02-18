import tqdm
import pandas as pd
import numpy as np
from kkelo import Elo, TrueSkill, TrueSkillOriginal
import zipfile

with zipfile.ZipFile('./boatrace_rating.zip', 'r') as z:
    with z.open('boatrace_rating.csv') as f:
        df = pd.read_csv(f)
list_players = df["id_racecourse_sk"].unique().tolist() + df["id_player_sk"].unique().tolist()
df      = df.sort_values(["race_id", "number"]).reset_index(drop=True)
df_test = df.loc[df["race_id"] >= 202009000000].copy()
df      = df.loc[df["race_id"] <  202009000000].copy()

dict_rating = {}
for _class in [TrueSkill, TrueSkillOriginal]:
    rating = _class(mu=25.0, monitors=["pk4320", "jk24_1"]) # p_draw=1e-3
    rating.add_players(list_players)
    for race_id, dfwk in tqdm.tqdm(df.groupby("race_id")):
        rating.update(*dfwk[["id_racecourse_sk", "id_player_sk"]].to_numpy(dtype=object).tolist(), ranks=dfwk["rank"].tolist())
    dict_rating[_class] = rating

rating = Elo(diff=10, k=5, monitors=["pk4320", "jk24_1"])
rating.add_players(list_players)
for race_id, dfwk in tqdm.tqdm(df.groupby("race_id")):
    rating.update(*dfwk[["id_racecourse_sk", "id_player_sk"]].to_numpy(dtype=object).tolist(), ranks=dfwk["rank"].tolist())
dict_rating[Elo] = rating

# eval
ndf_id   = df_test[["id_racecourse_sk", "id_player_sk"]].to_numpy(dtype=object).reshape(-1, 2*6)
ndf_rank = df_test["rank"].to_numpy(dtype=int).reshape(-1, 6)
for _class in [TrueSkill, TrueSkillOriginal, Elo]:
    ndf_eval = dict_rating[_class].evaluate(ndf_id, ranks=ndf_rank, structure=[2, 4, 6, 8, 10])
    print(ndf_eval.mean())
