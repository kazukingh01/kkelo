import trueskill, tqdm
import pandas as pd
import numpy as np
from kkelo import Elo, TrueSkill, evaluate_ndcg
import zipfile

with zipfile.ZipFile('./boatrace_rating.zip', 'r') as z:
    with z.open('boatrace_rating.csv') as f:
        df = pd.read_csv(f)
list_players = df["id_racecourse_sk"].unique().tolist() + df["id_player_sk"].unique().tolist()

# My class
rating = TrueSkill()
rating.add_players(list_players)
for race_id, dfwk in tqdm.tqdm(df.groupby("race_id")):
    rating.update(*dfwk[["id_racecourse_sk", "id_player_sk"]].to_numpy(dtype=object).tolist(), ranks=dfwk["rank"].tolist())

# public class
env = trueskill.TrueSkill(draw_probability=0.0)
dict_rate = {x: env.Rating() for x in list_players}
for race_id, dfwk in tqdm.tqdm(df.groupby("race_id")):
    list_grps    = dfwk[["id_racecourse_sk", "id_player_sk"]].to_numpy(dtype=object).tolist()
    list_ratings = trueskill.rate([[dict_rate[y] for y in x] for x in list_grps], ranks=dfwk["rank"].tolist(), min_delta=1e-10)
    for x, a in zip(list_grps, list_ratings):
        for y, b in zip(x, a):
            dict_rate[y] = b

# eval
list_eval1 = []
list_eval2 = []
ndf_id   = df[["id_racecourse_sk", "id_player_sk"]].to_numpy(dtype=object).reshape(-1, 2*6)
ndf_rank = df["rank"].to_numpy(dtype=int).reshape(-1, 6)
# rating.evaluate(ndf_id, ranks=ndf_rank, structure=[2, 4, 6, 8, 10])
for i in range(ndf_id.shape[0]):
    ndf1 = rating.rating[ndf_id[i]][:, 0].reshape(-1, 2).sum(axis=-1)
    ndf2 = np.array([dict_rate[x].mu for x in ndf_id[i]]).reshape(-1, 2).sum(axis=1)
    list_eval1.append(evaluate_ndcg(np.argsort(np.argsort(-ndf1)) + 1, ndf_rank[i]))
    list_eval2.append(evaluate_ndcg(np.argsort(np.argsort(-ndf2)) + 1, ndf_rank[i]))
list_eval1 = np.array(list_eval1)
list_eval2 = np.array(list_eval2)