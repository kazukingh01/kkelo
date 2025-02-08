import numpy as np
from .util import check_type_list, evaluate_ndcg


__all__ = [
    "Elo",
    "evaluate_ndcg",
]


class Elo:
    def __init__(self, init_rating: int | float=500, diff: int=400, k: int=10, n_round: int=3, is_check: bool=True):
        """
        Ref: https://arxiv.org/pdf/2105.14069.pdf
        experience::
            >>> import time
            >>> import numpy as np
            >>> import pandas as pd
            >>> N = 1000
            >>> indexes = np.arange(1000000, 1000000 + N).astype(str)
            >>> ndf = np.random.rand(indexes.shape[0])
            >>> dict_map = {x: i for i, x in enumerate(indexes)}
            >>> se_map   = pd.Series(np.arange(indexes.shape[0]), index=indexes)
            >>> x, y, z = 0, 0, 0
            >>> for i in range(1, 1000): a = time.time(); b = ndf[[dict_map[x] for x in indexes[np.random.permutation(indexes.shape[0])]      ]]; x += time.time() - a; x / i
            >>> for i in range(1, 1000): a = time.time(); b =     [dict_map[x] for x in indexes[np.random.permutation(indexes.shape[0])]      ];  y += time.time() - a; y / i
            >>> for i in range(1, 1000): a = time.time(); b =                se_map.loc[indexes[np.random.permutation(indexes.shape[0])]      ];  z += time.time() - a; z / i
            >>> x, y, z = 0, 0, 0
            >>> for i in range(1, 1000): a = time.time(); b = ndf[[dict_map[x] for x in indexes[np.random.permutation(indexes.shape[0])][:100]]]; x += time.time() - a; x / i
            >>> for i in range(1, 1000): a = time.time(); b =     [dict_map[x] for x in indexes[np.random.permutation(indexes.shape[0])][:100]];  y += time.time() - a; y / i
            >>> for i in range(1, 1000): a = time.time(); b =                se_map.loc[indexes[np.random.permutation(indexes.shape[0])][:100]];  z += time.time() - a; z / i
            >>> x, y, z = 0, 0, 0
            >>> for i in range(1, 1000): a = time.time(); b = ndf[[dict_map[x] for x in indexes[np.random.permutation(indexes.shape[0])][:10 ]]]; x += time.time() - a; x / i
            >>> for i in range(1, 1000): a = time.time(); b =     [dict_map[x] for x in indexes[np.random.permutation(indexes.shape[0])][:10 ]];  y += time.time() - a; y / i
            >>> for i in range(1, 1000): a = time.time(); b =                se_map.loc[indexes[np.random.permutation(indexes.shape[0])][:10 ]];  z += time.time() - a; z / i
            # N = 1000  & select 1000  -> dict, np, se
            # N = 1000  & select 10    -> dict, np, se
            # N = 10000 & select 10000 -> se, np, dict
            # N = 10000 & select 100   -> dict, np, se
            # N = 10000 & select 10    -> dict, np, se
        """
        assert isinstance(init_rating, (int, float))
        assert isinstance(diff,    int) and diff    > 0
        assert isinstance(k,       int) and k       > 0
        assert isinstance(n_round, int) and n_round >= 0
        self.rating      = {}
        self.init_rating = float(init_rating)
        self.diff        = diff
        self.k           = k
        self.n_round     = n_round
        self.is_check    = is_check
    
    def add_players(self, name: str | list[str], rating: float | list[float]=None):
        if isinstance(name, str): name = [name,]
        assert check_type_list(name, str)
        if rating is None:
            rating = [self.init_rating] * len(name)
        else:
            if isinstance(rating, (list, tuple)):
                assert len(rating) == len(name)
            else:
                rating = [rating] * len(name)
            assert check_type_list(rating, [int, float])
        for x, y in zip(name, rating):
            if x not in self.rating:
                self.rating[x] = float(y)

    def ratings(self, *teams: str | list[str] | np.ndarray) -> tuple[None | list[list[str]], np.ndarray | list[list[float]]]:
        is_np = (len(teams) == 1 and isinstance(teams[0], np.ndarray))
        if is_np: teams = teams[0]
        if self.is_check:
            if is_np:
                assert len(teams.shape) in [1, 2]
                assert np.issubdtype(teams.dtype, np.str_) or teams.dtype in [np.object_]
            else:
                assert check_type_list(teams, [list, str], str)
        if is_np:
            ratings = np.array([self.rating[x] for x in teams.reshape(-1)]).reshape(teams.shape)
            return None, ratings
        else:
            teams = [x if isinstance(x, (tuple, list)) else [x, ] for x in teams]
            return teams, [[self.rating[y] for y in x] for x in teams]

    def ratings_team(self, *teams: str | list[str], ratings: list[list[float]]=None) -> list[float]:
        if self.is_check:
            assert check_type_list(teams, [list, str], str)
        if ratings is None:
            _, ratings = self.ratings(*teams)
        if self.is_check:
            assert check_type_list(ratings, list, float)
        return [sum(x) for x in ratings]

    def weights_team(self, *teams: str | list[str], ratings: list[list[float]]=None) -> list[list[float]]:
        if self.is_check:
            assert check_type_list(teams, list, str)
        if ratings is None:
            _, ratings = self.ratings(*teams)
        if self.is_check:
            assert check_type_list(ratings, list, float)
        weights = [sum(x) for x in ratings]
        return [[z / y for z in x] for x, y in zip(ratings, weights)]
    
    def probability(self, *teams: str | list[str], ratings_team: list[float]=None, diff: int=None) -> np.ndarray:
        if self.is_check:
            assert check_type_list(teams, list, str)
        if ratings_team is None:
            ratings_team = self.ratings_team(*teams)
        if diff is None:
            diff = self.diff
        if self.is_check:
            assert check_type_list(ratings_team, float)
            assert isinstance(diff, int) and diff > 0
        n_teams      = len(ratings_team)
        ratings_team = np.array(ratings_team)
        n_pairwise   = (n_teams * (n_teams - 1) / 2)
        return ((1 / (np.exp((np.tile(ratings_team, (n_teams, 1)) - ratings_team.reshape(-1, 1)) / diff) + 1)).sum(axis=-1) - 0.5) / n_pairwise # The 0.5 is introduced to cancel out (or offset) the contribution from i = j.
    
    def update(self, *teams: str | list[str], ranks: list[int]=None, diff: int=None, k: int=None, weights: np.ndarray=None, mask: list[str]=None):
        """
        params::
            ranks: over 1. 1 means top rank.
        """
        if diff is None: diff = self.diff
        if k    is None: k    = self.k
        if mask is None: mask = []
        if self.is_check:
            assert check_type_list(teams, [list, str], str)
            assert check_type_list(ranks, int)
            for x in ranks: assert x >= 1
            assert len(teams) == len(ranks)
            assert isinstance(diff, int)
            assert isinstance(k,    int)
            assert check_type_list(mask, str)
        ranks            = np.array(ranks)
        n_teams          = len(teams)
        ranks_norm       = (n_teams - ranks) / (n_teams * (n_teams - 1) / 2)
        indexes, ratings = self.ratings(*teams)
        ratings_team     = self.ratings_team(ratings=ratings)
        probs            = self.probability(ratings_team=ratings_team, diff=diff)
        ratings_team     = np.array(ratings_team)
        ratings_team_new = ratings_team + k * (ranks_norm - probs)
        if weights is None: weights = self.weights_team(ratings=ratings)
        else:               weights = [[weights], ] * n_teams
        ratings_new      = [[y + z * x for y, z in zip(ratings[i], weights[i])] for i, x in enumerate(ratings_team_new - ratings_team)]
        for x, y in zip(indexes, ratings_new):
            for a, b in zip(x, y):
                if a in mask: continue
                self.rating[a] = round(b, self.n_round)

    def evaluate(self, *teams: str | list[str] | np.ndarray, ranks: list[int] | np.ndarray=None, structure: list[int]=None):
        """
        If you want to evalate many data, you have to use np.ndarray form.
        """
        is_np = (len(teams) == 1 and isinstance(teams[0], np.ndarray))
        if is_np: teams = teams[0]
        if self.is_check:
            assert ranks is not None
            if is_np:
                assert isinstance(ranks, np.ndarray)
                assert len(teams.shape) == len(ranks.shape) == 2
                assert teams.shape[0] == ranks.shape[0]
                assert ranks.dtype in [int, np.int32, np.int64]
                assert np.issubdtype(teams.dtype, np.str_) or teams.dtype in [np.object_]
                if structure is None:
                    assert teams.shape == ranks.shape
                else:
                    assert (len(structure) + 1) == ranks.shape[-1]
                    assert check_type_list(structure, int)
            else:
                assert check_type_list(teams, [list, str], str)
                assert isinstance(ranks, (tuple, list))
                assert check_type_list(ranks, int)
                for x in ranks: assert x >= 1
                assert len(teams) == len(ranks)
        if is_np:
            _, ratings = self.ratings(teams)
            if structure is not None:
                structure = [0, ] + structure + [teams.shape[-1], ]
                ratings   = [ratings[:, structure[i]:structure[i+1]].sum(axis=-1) for i in range(len(structure) - 1)]
                ratings   = np.stack(ratings).T
        else:
            ratings = self.ratings_team(*teams)
            ratings = np.array(ratings)
        ranks_pred = np.argsort(np.argsort(-ratings, axis=-1), axis=-1) + 1
        return evaluate_ndcg(ranks_pred, np.array(ranks))
