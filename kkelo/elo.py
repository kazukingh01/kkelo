from math import sqrt
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
import trueskill
# local package
from .com import NumpyDict
from .util import check_type_list, evaluate_ndcg, normal_cdf, normal_pdf, ppf


__all__ = [
    "Elo",
    "TrueSkill",
    "TrueSkillOriginal",
    "evaluate_ndcg",
]


class BaseRating:
    def __init__(self, dtype=np.float32, default_size: int=1000000, n_round: int=3, is_check: bool=True, monitors: list[str]=None, init_values: float | list[float]=0.):
        assert isinstance(is_check, bool)
        assert monitors is None or (isinstance(monitors, list) and check_type_list(monitors, str))
        assert isinstance(n_round, int) and n_round >= 0
        self.i_step   = 0
        self.is_check = is_check
        self.monitors = monitors
        self.rating   = NumpyDict(is_check=False, dtype=dtype, default_size=default_size, is_series=False, init_values=init_values)
        self.dtype    = dtype
        self.n_round  = n_round
        self.idx_mtrs:  list[int]  = []
        self.list_mtrs: list[dict] = []    
    def add_players(self):
        raise NotImplementedError()
    def ratings(self, *teams: str | list[str] | np.ndarray, idx_ret: int=None) -> tuple[None | list[list[str]], np.ndarray | list[list[float]]]:
        is_np = (len(teams) == 1 and isinstance(teams[0], np.ndarray))
        if is_np: teams = teams[0]
        if self.is_check:
            assert idx_ret is None or (isinstance(idx_ret, int) and idx_ret in [-2, -1, 0, 1])
            if is_np:
                assert len(teams.shape) in [1, 2]
                assert teams.dtype in [np.object_] # np.str_ access is slower than pure str. remove condition of np.issubdtype(teams.dtype, )
            else:
                assert check_type_list(teams, [list, str], str)
        if is_np:
            ratings = self.rating[teams]
            if idx_ret is None:
                return None, ratings
            else:
                return None, ratings[..., idx_ret]
        else:
            teams = [x if isinstance(x, (list, tuple)) else [x, ] for x in teams]
            if idx_ret is None:
                return teams, [self.rating[x] for x in teams]
            else:
                return teams, [self.rating[x][..., idx_ret] for x in teams]
    def __repr__(self):
        return self.__str__()
    def ratings_team(self):
        raise NotImplementedError()
    def update(self):
        raise NotImplementedError()
    def update_common(self):
        self.i_step += 1
        if self.monitors is not None and self.i_step % 100 == 0:
            self.list_mtrs.append({x: self.rating.values[y].copy() if self.rating.is_multi_dim else self.rating.values[y] for x, y in zip(self.monitors, self.idx_mtrs)})
    def evaluate(self, *teams: str | list[str] | np.ndarray, ranks: list[int] | np.ndarray=None, structure: list[int]=None, idx_ret: int=None):
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
            _, ratings = self.ratings(teams, idx_ret=idx_ret)
            if structure is not None:
                structure = [0, ] + structure + [teams.shape[-1], ]
                ratings   = [ratings[:, structure[i]:structure[i+1]].sum(axis=-1) for i in range(len(structure) - 1)]
                ratings   = np.stack(ratings).T
        else:
            ratings = self.ratings_team(*teams)
        ranks_pred = np.argsort(np.argsort(-ratings, axis=-1), axis=-1) + 1
        return evaluate_ndcg(ranks_pred, np.array(ranks, dtype=int))
    def monitors_to_pandas(self):
        if self.monitors is not None:
            return pd.DataFrame(self.list_mtrs)
        else:
            raise AttributeError(f"monitors is not set.")
    def monitors_to_plot(self, figsize: tuple[int, int]=(10, 6)):
        if self.monitors is not None:
            dforg = pd.DataFrame(self.list_mtrs)
            if isinstance(dforg.iloc[0, 0], (list, np.ndarray)):
                n_len = len(dforg.iloc[0, 0])
                df = pd.DataFrame(index=dforg.index)
                for x in dforg.columns:
                    for y in range(n_len):
                        df[f"{x}_{y}"] = dforg[x].str[y]
            else:
                df = dforg.copy()
            df.plot(figsize=figsize)
            plt.xlabel('n steps ( x 100 )')
            plt.ylabel('Rating')
            plt.title(self.__str__())
            plt.legend()
            plt.show()
        else:
            raise AttributeError(f"monitors is not set.")
    

class Elo(BaseRating):
    def __init__(
        self, init_rating: int | float=500, diff: int=400, k: int=10, n_round: int=3,
        dtype=np.float32, default_size: int=1000000, is_check: bool=True, monitors: list[str] = None
    ):
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
        super().__init__(dtype=dtype, default_size=default_size, n_round=n_round, is_check=is_check, monitors=monitors, init_values=init_rating)
        self.init_rating = dtype(init_rating)
        self.diff        = diff
        self.k           = k
    
    def __str__(self):
        return f"{__class__.__name__}(init: {self.init_rating}, diff: {self.diff}, k: {self.k})"

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
            assert isinstance(rating, np.ndarray)
            assert rating.dtype == self.dtype
        self.rating.update(name, rating)
        if self.monitors is not None:
            self.idx_mtrs = [self.rating.keys.get(x) for x in self.monitors]

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
        super().update_common()


class TrueSkill(BaseRating):
    def __init__(
        self, mu: int | float=25.0, sigma: int | float=None, beta: int | float=None, d_factor: float=None, p_draw: float=None,
        n_round: int=3, dtype=np.float64, default_size: int=1000000, is_check: bool=True, monitors: list[str] = None
    ):
        """
        https://herbrich.me/papers/trueskill.pdf
        https://uwaterloo.ca/computational-mathematics/sites/default/files/uploads/documents/justin_dastous_research_paper.pdf
        https://www.diva-portal.org/smash/get/diva2:1322103/FULLTEXT01.pdf
        """
        assert isinstance(mu, (float, int)) and mu > 0
        assert sigma    is None or (isinstance(sigma, (float, int)) and sigma > 0)
        assert beta     is None or (isinstance(beta,  (float, int)) and beta > 0)
        assert d_factor is None or (isinstance(d_factor,     float) and d_factor > 0)
        assert p_draw   is None or (isinstance(p_draw,       float) and p_draw   > 0)
        super().__init__(dtype=dtype, default_size=(default_size, 2), n_round=n_round, is_check=is_check, monitors=monitors, init_values=[mu, sigma])
        self.mu        = dtype(mu)
        self.sigma     = dtype(sigma) if sigma is not None else self.mu / 3
        self.var       = self.sigma ** 2
        self.beta      = dtype(beta) if beta is not None else self.sigma / 2
        self.beta2     = self.beta ** 2
        self.d_factor  = dtype(d_factor) if d_factor is not None else self.sigma / 100
        self.d_factor2 = self.d_factor ** 2
        self.p_draw    = dtype(p_draw) if p_draw is not None else p_draw
        if self.p_draw is not None:
            self._p_draw   = norm.ppf((self.p_draw + 1) / 2.) * self.beta

    def __str__(self):
        return f"{__class__.__name__}(mu: {self.mu}, sigma: {self.sigma}, beta: {self.beta}, dynamics factor: {self.d_factor})"

    def add_players(self, name: str | list[str] | np.ndarray[str], mu: float | list[float] | np.ndarray=None, sigma: float | list[float] | np.ndarray=None):
        if isinstance(name, str): name = [name,]
        if self.is_check:
            if isinstance(name, np.ndarray):
                assert len(name.shape) == 1
                assert name.dtype == object
            else:
                assert check_type_list(name, str)
        if mu is None:
            assert sigma is None
            rating = np.ones((len(name), 2), dtype=self.dtype)
            rating[:, 0] = self.mu
            rating[:, 1] = self.var
        else:
            if isinstance(mu, (list, tuple, np.ndarray)):
                if self.is_check:
                    assert type(mu) == type(sigma)
                    assert len(mu)  == len(sigma)
                rating = np.stack([
                    np.array(mu, dtype=self.dtype),
                    np.array(sigma ** 2, dtype=self.dtype)
                ], dtype=self.dtype).T
            else:
                rating = np.ones((len(name), 2), dtype=self.dtype)
                rating[:, 0] = self.dtype(mu)
                rating[:, 1] = self.dtype(sigma ** 2)
        if self.is_check:
            assert len(rating) == len(name)
            assert isinstance(rating, np.ndarray)
            assert rating.dtype == self.dtype
        self.rating.update(name, rating)
        if self.monitors is not None:
            self.idx_mtrs = [self.rating.keys[x] for x in self.monitors]

    def ratings_team(self, *teams: str | list[str], ratings: list[np.ndarray]=None) -> np.ndarray:
        if self.is_check:
            assert check_type_list(teams, [list, str], str)
        if ratings is None:
            _, ratings = self.ratings(*teams)
        if self.is_check:
            assert check_type_list(ratings, np.ndarray)
        return np.array([x[:, 0].sum() for x in ratings], dtype=self.dtype)

    def update(self, *teams: str | list[str], ranks: list[int]=None, mask: list[bool]=None):
        """
        params::
            ranks: over 1. 1 means top rank.
        """
        if self.is_check:
            assert check_type_list(teams, [list, str], str)
            assert check_type_list(ranks, int)
            for x in ranks: assert x >= 1
            assert len(teams) == len(ranks)
            assert mask is None or check_type_list(mask, bool)
            if mask is not None:
                assert check_type_list(teams, list, str)
                _n = len(teams[0])
                for x in teams: assert len(x) == _n == len(mask)
        indexes, ratings = self.ratings(*teams)
        ranks     = np.array(ranks, dtype=int)
        idx_rank  = np.argsort(ranks)
        # idx_invrs = np.argsort(idx_rank)
        idx_pairs = np.stack([idx_rank[:-1], idx_rank[1:]])
        i_center  = idx_pairs.shape[-1] // 2
        list_idxs = [idx_pairs[:, [i_center]], ] + [idx_pairs[:, [i_center - i, i_center + i]] for i in np.arange(1, i_center + 1, dtype=int)]
        skill_mu  = [x[:, 0]                  for x in ratings]
        skill_var = [x[:, 1] + self.d_factor2 for x in ratings]
        # update
        for _ in range(1):
            for idx_pair in list_idxs: # win or lose process
                prfrm_mu  = skill_mu
                prfrm_var = [x + self.beta2 for x in skill_var]
                teams_mu  = np.array([x.sum() for x in prfrm_mu ], dtype=self.dtype)
                teams_var = np.array([x.sum() for x in prfrm_var], dtype=self.dtype)
                delta     = teams_mu[ idx_pair[0]] - teams_mu[ idx_pair[1]] # 0 means win, 1 means lose
                if self.p_draw is not None:
                    n_teams     = np.array([len(x) for x in indexes], dtype=int)
                    draw_margin = self._p_draw * np.sqrt(n_teams[idx_pair[0]] + n_teams[idx_pair[1]])
                    draw_margin[delta < 0] *= -1
                    delta       = delta - draw_margin
                c2        = teams_var[idx_pair[0]] + teams_var[idx_pair[1]]
                c         = np.sqrt(c2)
                t         = delta / c
                v         = normal_pdf(t) / normal_cdf(t)
                w         = v * (v + t)
                v_c       = v / c
                # w_c2      = w / c2
                weight    = [x / x.sum() for x in skill_var]
                # teams_mu[idx_pair[0]]  = teams_mu[idx_pair[0]] + (v_c * teams_var[idx_pair[0]])
                # teams_mu[idx_pair[1]]  = teams_mu[idx_pair[1]] - (v_c * teams_var[idx_pair[1]])
                # teams_var[idx_pair[0]] = teams_var[idx_pair[0]] * (1 - (w_c2 * teams_var[idx_pair[0]]))
                # teams_var[idx_pair[1]] = teams_var[idx_pair[1]] * (1 - (w_c2 * teams_var[idx_pair[1]]))
                for i, j in enumerate(idx_pair[0]):
                    skill_mu[j]  = skill_mu[j] + (v_c[i] * teams_var[j] * weight[j])
                    skill_var[j] = skill_var[j] * (1 - (skill_var[j] * w[i] / c2[i]))
                for i, j in enumerate(idx_pair[1]):
                    skill_mu[j]  = skill_mu[j] - (v_c[i] * teams_var[j] * skill_var[j] / skill_var[j].sum())
                    skill_var[j] = skill_var[j] * (1 - (skill_var[j] * w[i] / c2[i]))
        for i, x, y in zip(indexes, skill_mu, skill_var):
            self.rating[i] = np.stack([x, y]).T
        super().update_common()

    def evaluate(self, *teams: str | list[str] | np.ndarray, ranks: list[int] | np.ndarray=None, structure: list[int]=None):
        return super().evaluate(*teams, ranks=ranks, structure=structure, idx_ret=0)


class TrueSkillOriginal:
    def __init__(
        self, mu: int | float=25.0, sigma: int | float=None, beta: int | float=None, d_factor: float=None, p_draw: float=None,
        min_delta: float=10, n_round: int=None, is_check: bool=True, monitors: list[str] = None
    ):
        assert isinstance(mu, (float, int)) and mu > 0
        assert sigma    is None or (isinstance(sigma, (float, int)) and sigma > 0)
        assert beta     is None or (isinstance(beta,  (float, int)) and beta > 0)
        assert d_factor is None or (isinstance(d_factor,     float) and d_factor > 0)
        assert p_draw   is None or (isinstance(p_draw,       float) and p_draw   > 0)
        assert isinstance(min_delta, (int, float)) and min_delta > 0
        assert n_round  is None or (isinstance(n_round, int) and n_round > 0)
        assert isinstance(is_check, bool)
        assert monitors is None or (isinstance(monitors, list) and check_type_list(monitors, str))
        sigma          = sigma    if sigma    is not None else mu / 3.
        beta           = beta     if beta     is not None else sigma / 2.
        d_factor       = d_factor if d_factor is not None else sigma / 100.
        p_draw         = p_draw   if p_draw   is not None else 0.0
        self.env       = trueskill.TrueSkill(mu=mu, sigma=sigma, beta=beta, tau=d_factor, draw_probability=p_draw, backend=None)
        self.rating    = defaultdict(lambda: self.env.create_rating())
        self.min_delta = min_delta
        self.n_round   = n_round
        self.is_check  = is_check
        self.monitors  = monitors
        self.list_mtrs = []
        self.i_step    = 0

    def __str__(self):
        return self.env.__str__()

    def add_players(self, name: str | list[str] | np.ndarray[str], mu: float | list[float] | np.ndarray=None, sigma: float | list[float] | np.ndarray=None, is_override: bool=True):
        if isinstance(name, str): name = [name,]
        if mu is None:
            mu = [None] * len(name)
        elif not isinstance(mu, (list, np.ndarray)):
            mu = [round(mu, self.n_round)] * len(name) if self.n_round is not None else [mu] * len(name)
        else:
            mu = [round(x, self.n_round) for x in mu] if self.n_round is not None else mu
        if sigma is None:
            sigma = [None] * len(name)
        elif not isinstance(sigma, (list, np.ndarray)):
            sigma = [round(sigma, self.n_round)] * len(name) if self.n_round is not None else [sigma] * len(name)
        else:
            sigma = [round(x, self.n_round) for x in sigma] if self.n_round is not None else sigma
        assert len(name) == len(mu) == len(sigma)
        if is_override:
            self.rating = self.rating | {x: self.env.create_rating(mu=y, sigma=z) for x, y, z in zip(name, mu, sigma)}
        else:
            self.rating = self.rating | {x: self.env.create_rating(mu=y, sigma=z) for x, y, z in zip(name, mu, sigma) if x not in self.rating}

    def update(self, *teams: str | list[str], ranks: list[int]=None, mask: list[bool]=None):
        """
        params::
            ranks: over 1. 1 means top rank.
        """
        if self.is_check:
            assert check_type_list(teams, [list, str], str)
            assert check_type_list(ranks, int)
            for x in ranks: assert x >= 1
            assert len(teams) == len(ranks)
            assert mask is None or check_type_list(mask, bool)
            if mask is not None:
                assert check_type_list(teams, list, str)
                _n = len(teams[0])
                for x in teams: assert len(x) == _n == len(mask)
        teams     = [x if isinstance(x, (list, tuple)) else [x, ] for x in teams]
        list_vals = self.env.rate([[self.rating[y] for y in x] for x in teams], ranks=ranks, min_delta=self.min_delta)
        for x, a in zip(teams, list_vals):
            if mask is None:
                for y, b in zip(x, a):
                    if self.n_round is None:
                        self.rating[y] = b
                    else:
                        self.rating[y] = self.env.create_rating(mu=round(b.mu, self.n_round), sigma=round(b.sigma, self.n_round))
            else:
                for y, b, is_mask in zip(x, a, mask):
                    if is_mask:
                        if self.n_round is None:
                            self.rating[y] = b
                        else:
                            self.rating[y] = self.env.create_rating(mu=round(b.mu, self.n_round), sigma=round(b.sigma, self.n_round))
        self.i_step += 1
        if self.monitors is not None and self.i_step % 100 == 0:
            self.list_mtrs.append({x: self.rating[x].mu for x in self.monitors})

    def evaluate(self, teams: np.ndarray, ranks: np.ndarray=None, structure: list[int]=None):
        if self.is_check:
            assert isinstance(teams, np.ndarray) and len(teams.shape) == 2
            assert isinstance(ranks, np.ndarray) and len(ranks.shape) == 2
            assert teams.shape[0] == ranks.shape[0]
            assert teams.dtype in [np.object_]
            assert (ranks < 1).sum() == 0
            if structure is None:
                assert teams.shape == ranks.shape
            else:
                assert (len(structure) + 1) == ranks.shape[-1]
                assert check_type_list(structure, int)
        ratings = np.array([self.rating[x].mu for x in teams.reshape(-1)], dtype=float).reshape(teams.shape)
        if structure is not None:
            structure = [0, ] + structure + [teams.shape[-1], ]
            ratings   = [ratings[:, structure[i]:structure[i+1]].sum(axis=-1) for i in range(len(structure) - 1)]
            ratings   = np.stack(ratings).T
        return evaluate_ndcg(np.argsort(np.argsort(-ratings)) + 1, ranks)

    def monitors_to_pandas(self):
        if self.monitors is not None:
            return pd.DataFrame(self.list_mtrs)
        else:
            raise AttributeError(f"monitors is not set.")

    def monitors_to_plot(self, figsize: tuple[int, int]=(10, 6)):
        if self.monitors is not None:
            df = pd.DataFrame(self.list_mtrs)
            df.plot(figsize=figsize)
            plt.xlabel('n steps ( x 100 )')
            plt.ylabel('Rating')
            plt.title(self.__str__())
            plt.legend()
            plt.show()
        else:
            raise AttributeError(f"monitors is not set.")
    
