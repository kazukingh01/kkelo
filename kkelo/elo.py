import numpy as np
import pandas as pd
from .util import check_type_list, evaluate_ndcg


__all__ = [
    "Elo",
    "evaluate_ndcg",
]


class NumpyDict:
    def __init__(self, is_check: bool=True, dtype=float, default_size: int=1000, is_series: bool=False):
        assert isinstance(is_check, bool)
        assert isinstance(default_size, int) and default_size > 0
        self.is_check     = is_check
        self.is_series    = is_series
        self.keys         = pd.Series(dtype=int) if is_series else {}
        self.sets         = set()
        self.values       = np.zeros(default_size, dtype=dtype)
        self.dtype        = dtype
        self.default_size = default_size
    def __str__(self) -> str:
        return f"{__class__.__name__}({self.values.__str__()})"
    def __repr__(self) -> str:
        return f"{__class__.__name__}({self.values.__repr__()})"
    def __getitem__(self, idx: object | list[object] | np.ndarray[object]):
        if isinstance(idx, list):
            if self.is_series:
                return self.values[self.keys[idx]]
            else:
                return self.values[[self.keys[x] for x in idx]]
        elif isinstance(idx, np.ndarray):
            if self.is_check:
                assert len(idx.shape) in [1, 2]
            if self.is_series:
                return self.values[self.keys[idx.reshape(-1)]].reshape(idx.shape)
            else:
                if len(idx.shape) == 1:
                    return self.values[[self.keys[x] for x in idx]]
                else:
                    return self.values[np.array([self.keys[x] for x in idx.reshape(-1)]).reshape(idx.shape)]
        else:
            return self.values[self.keys[idx]]
    def __setitem__(self, keys: object | list[object] | np.ndarray, values: object | list | np.ndarray):
        if isinstance(keys, (list, np.ndarray)):
            if self.is_check:
                assert isinstance(values, (list, np.ndarray))
                assert len(keys) == len(values)
                assert not isinstance(values[0], (list, np.ndarray))
            if self.is_series:
                self.values[self.keys[keys]] = values
            else:
                self.values[[self.keys[x] for x in keys]] = values
        else:
            if self.is_check:
                assert isinstance(values, self.dtype)
            self.values[self.keys[keys]] = values
    def __len__(self):
        return len(self.keys)
    def update(self, keys: object | list[object] | np.ndarray, values: object | list | np.ndarray):
        """
        If you want to add new key, use this.
        """
        if isinstance(keys, (list, np.ndarray)):
            if self.is_check:
                assert isinstance(values, (list, np.ndarray))
                assert len(keys) == len(values)
            if len(self.keys) == 0:
                if self.is_series:
                    self.keys = pd.Series({x: i for i, x in enumerate(keys)})
                else:
                    self.keys = {x: i for i, x in enumerate(keys)}
                self.values[np.arange(len(keys), dtype=int)] = values
            else:
                keys   = np.array(keys, dtype=object)
                values = np.array(values, dtype=self.dtype)
                boolwk = [x in self.sets for x in keys]
                if np.any(boolwk):
                    if self.is_series:
                        self.values[self.keys[keys[boolwk]]] = values[boolwk]
                    else:
                        self.values[[self.keys[x] for x in keys[boolwk]]] = values[boolwk]
                    boolwk = [not x for x in boolwk]
                    keys   = keys[  boolwk]
                    values = values[boolwk]
                if keys.shape[0] > 0:
                    new_idxs  = np.arange(len(self.keys), len(self.keys) + keys.shape[0], dtype=int)
                    if self.is_series:
                        self.keys = pd.concat([self.keys, pd.Series({x: int(y) for x, y in zip(keys, new_idxs)})])
                    else:
                        self.keys = self.keys | {x: int(y) for x, y in zip(keys, new_idxs)}
                    self.values[new_idxs] = values
                    self.sets.update(keys.tolist())
        else:
            if self.is_check:
                assert isinstance(values, self.dtype)
            if keys in self.sets:
                self.values[self.keys[keys]] = values
            else:
                n_keys = len(self.keys)
                self.keys[keys] = n_keys
                if n_keys >= self.values.shape[0]:
                    self.values = np.concatenate([self.values, np.zeros(self.default_size, dtype=self.dtype)])
                self.values[n_keys] = values
                self.sets.add(keys)
    def to_dict(self) -> dict:
        return {x: self.values[y] for x, y in self.keys.items()}
    def to_pandas(self) -> pd.DataFrame:
        df = pd.DataFrame(list(self.keys.items()), columns=["keys", "values"])
        df["values"] = self.values[df["values"].to_numpy(dtype=int)]
        return df


class Elo:
    def __init__(self, init_rating: int | float=500, diff: int=400, k: int=10, n_round: int=3, is_check: bool=True, dtype=np.float32):
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
        self.rating      = NumpyDict(is_check=False, dtype=dtype, default_size=1000000, is_series=False)
        self.init_rating = dtype(init_rating)
        self.diff        = diff
        self.k           = k
        self.n_round     = n_round
        self.is_check    = is_check
        self.dtype       = dtype
    
    def add_players(self, name: str | list[str] | np.ndarray[str], rating: float | list[float] | np.ndarray=None):
        if isinstance(name, str): name = [name,]
        if self.is_check:
            if isinstance(name, np.ndarray):
                assert len(name.shape) == 1
                assert name.dtype == object
            else:
                assert check_type_list(name, str)
        if rating is None:
            rating = np.ones(len(name), dtype=self.dtype) * self.init_rating
        else:
            if isinstance(rating, (list, tuple, np.ndarray)):
                rating = np.array(rating, dtype=self.dtype)
            else:
                rating = np.ones(len(name), dtype=self.dtype) * rating
        if self.is_check:
            assert len(rating) == len(name)
            if isinstance(rating, np.ndarray):
                assert rating.dtype == self.dtype
            else:
                assert check_type_list(rating, [int, float, self.dtype])
        self.rating.update(name, rating)

    def ratings(self, *teams: str | list[str] | np.ndarray) -> tuple[None | list[list[str]], np.ndarray | list[list[float]]]:
        is_np = (len(teams) == 1 and isinstance(teams[0], np.ndarray))
        if is_np: teams = teams[0]
        if self.is_check:
            if is_np:
                assert len(teams.shape) in [1, 2]
                assert teams.dtype in [np.object_] # np.str_ access is slower than pure str. remove condition of np.issubdtype(teams.dtype, )
            else:
                assert check_type_list(teams, [list, str], str)
        if is_np:
            ratings = self.rating[teams]
            return None, ratings
        else:
            teams = [x if isinstance(x, (list, tuple)) else [x, ] for x in teams]
            return teams, [self.rating[x] for x in teams]

    def ratings_team(self, *teams: str | list[str], ratings: list[np.ndarray]=None) -> np.ndarray:
        if self.is_check:
            assert check_type_list(teams, [list, str], str)
        if ratings is None:
            _, ratings = self.ratings(*teams)
        if self.is_check:
            assert check_type_list(ratings, np.ndarray)
        return np.array([x.sum() for x in ratings], dtype=self.dtype)

    def weights_team(self, *teams: str | list[str], ratings: list[np.ndarray]=None) -> list[np.ndarray]:
        if self.is_check:
            assert check_type_list(teams, list, str)
        if ratings is None:
            _, ratings = self.ratings(*teams)
        if self.is_check:
            assert check_type_list(ratings, np.ndarray)
        return [x / x.sum() for x in ratings]
    
    def probability(self, *teams: str | list[str], ratings_team: np.ndarray=None, diff: int=None) -> np.ndarray:
        if self.is_check:
            assert check_type_list(teams, list, str)
        if ratings_team is None:
            ratings_team = self.ratings_team(*teams)
        if diff is None:
            diff = self.diff
        if self.is_check:
            assert isinstance(ratings_team, np.ndarray)
            assert isinstance(diff, int) and diff > 0
        n_teams    = len(ratings_team)
        n_pairwise = (n_teams * (n_teams - 1) / 2)
        return ((1 / (np.exp((np.tile(ratings_team, (n_teams, 1)) - ratings_team.reshape(-1, 1)) / diff) + 1)).sum(axis=-1) - 0.5) / n_pairwise # The 0.5 is introduced to cancel out (or offset) the contribution from i = j.
    
    def update(self, *teams: str | list[str], ranks: list[int]=None, diff: int=None, k: int=None, weights: np.ndarray=None, mask: list[bool]=None):
        """
        params::
            ranks: over 1. 1 means top rank.
        """
        if diff is None: diff = self.diff
        if k    is None: k    = self.k
        if self.is_check:
            assert check_type_list(teams, [list, str], str)
            assert check_type_list(ranks, int)
            for x in ranks: assert x >= 1
            assert len(teams) == len(ranks)
            assert isinstance(diff, int)
            assert isinstance(k,    int)
            assert mask is None or check_type_list(mask, bool)
            if mask is not None:
                assert check_type_list(teams, list, str)
                _n = len(teams[0])
                for x in teams: assert len(x) == _n == len(mask)
        ranks            = np.array(ranks, dtype=int)
        n_teams          = len(teams)
        ranks_norm       = (n_teams - ranks) / (n_teams * (n_teams - 1) / 2)
        indexes, ratings = self.ratings(*teams)
        ratings_team     = self.ratings_team(ratings=ratings)
        probs            = self.probability(ratings_team=ratings_team, diff=diff)
        ratings_team_new = ratings_team + k * (ranks_norm - probs)
        if weights is None: weights = self.weights_team(ratings=ratings)
        else:               weights = [weights, ] * n_teams
        ratings_new      = [ratings[i] + weights[i] * x for i, x in enumerate(ratings_team_new - ratings_team)]
        indexes, ratings_new = np.concatenate(indexes, dtype=object), np.concatenate(ratings_new, dtype=self.dtype)
        if mask is not None:
            mask = np.tile(mask, n_teams)
            indexes, ratings_new = indexes[mask], ratings_new[mask]
        self.rating[indexes] = np.round(ratings_new, decimals=self.n_round)

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
        ranks_pred = np.argsort(np.argsort(-ratings, axis=-1), axis=-1) + 1
        return evaluate_ndcg(ranks_pred, np.array(ranks, dtype=int))
