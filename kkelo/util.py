import math
import numpy as np
from scipy.special import erf


__all__ = [
    "evaluate_ndcg",
    "check_type",
    "check_type_list",
    "normal_pdf",
    "normal_cdf",
]


def evaluate_ndcg(
    ranks_pred: np.ndarray[float | int], ranks_answer: np.ndarray[int], points: dict[int, int]=None,
    k: int=None, idx_groups: list[int | str] | np.ndarray=None, is_point_to_rank: bool=False
):
    """
    ranks_pred, ranks_answer:
        interger, 1 ~. 1 menas top rank
    idx | rank | point  ||  idx | pred  ||  point sort | rank idx | pred point
     0      1      6         0      1            6          0            6
     1      2      5         1      2            5          1            5
     2      6      1         2      3            4          3            1
     3      3      4         3      6            3          5            3
     4      5      2         4      5            2          4            2
     5      4      3         5      4            1          2            4
    >>> import numpy as np
    >>> ranks_pred    = np.array([[1, 2, 3, 6, 5, 4], [1, 3, 4, 5, 6, 2]])
    >>> ranks_answer  = np.array([[1, 2, 6, 3, 5, 4], [1, 2, 4, 5, 6,]])
    >>> points_answer = np.argsort(np.argsort(-ranks_answer, axis=-1), axis=-1) + 1
    >>> idx_answer    = np.argsort(-points_answer, axis=-1)
    >>> evaluate_ndcg(ranks_pred, ranks_answer)
    >>> evaluate_ndcg(ranks_pred, ranks_answer, k=3)
    """
    assert isinstance(ranks_pred,   np.ndarray)
    assert isinstance(ranks_answer, np.ndarray)
    assert len(ranks_pred.shape) in [1, 2]
    assert ranks_pred.shape == ranks_answer.shape
    assert (ranks_answer < 1).sum() == 0
    assert isinstance(is_point_to_rank, bool)
    if is_point_to_rank == False:
        assert (ranks_pred < 1).sum() == 0
    assert k is None or (isinstance(k, int) and k > 0)
    if idx_groups is not None:
        assert isinstance(idx_groups, (list, np.ndarray))
        if isinstance(idx_groups, list):
            idx_groups = np.array(idx_groups)
        assert len(idx_groups.shape) == 1
        assert idx_groups.shape == ranks_pred.shape
        assert k is not None
        df = pl.DataFrame([
            pl.Series(ranks_pred  ).alias("pred"),
            pl.Series(ranks_answer).alias("answer"),
            pl.Series(idx_groups  ).alias("group"),
        ])
        df = df.group_by("group").agg(pl.all())
        ranks_pred   = df.select([pl.col("pred"  ).list.get(i, null_on_oob=True).alias(f"{i}") for i in range(k)]).to_numpy()
        ranks_answer = df.select([pl.col("answer").list.get(i, null_on_oob=True).alias(f"{i}") for i in range(k)]).to_numpy()
    if is_point_to_rank:
        ndf_nan    = np.isnan(ranks_pred)
        ranks_pred = (np.argsort(np.argsort(-ranks_pred, axis=-1)) + 1).astype(float)
        ranks_pred[ndf_nan] = float("nan")
    if points is not None:
        assert isinstance(points_answer, dict)
        points_answer = np.vectorize(lambda x: points.get(x) if x in points else float("nan"), otypes=[float])(ranks_answer)
    else:
        points_answer = (np.argsort(np.argsort(-ranks_answer, axis=-1), axis=-1) + 1).astype(float)
        points_answer[np.isnan(ranks_answer)] = float("nan")
    if k is None:
        if len(ranks_pred.shape) == 1:
            k = ranks_pred.shape[0]
        else:
            k = ranks_pred.shape[-1]
    ndf      = np.log2(np.arange(k) + 1)
    ndf[0]   = 1
    p_pred   = np.take_along_axis(points_answer, np.argsort(ranks_pred, axis=-1), axis=-1)
    p_answer = -np.sort(-points_answer, axis=-1)
    if len(ranks_pred.shape) > 1:
        ndf   = np.tile(ndf, (ranks_pred.shape[0], 1))
    dcg_i = np.nansum(p_answer[..., :k] / ndf[..., :k], axis=-1)
    dcg   = np.nansum(p_pred[  ..., :k] / ndf[..., :k], axis=-1)
    return dcg / dcg_i

def check_type(instance: object, _type: object | list[object]):
    _type = [_type] if not (isinstance(_type, list) or isinstance(_type, tuple)) else _type
    is_check = [isinstance(instance, __type) for __type in _type]
    if sum(is_check) > 0:
        return True
    else:
        return False

def check_type_list(instances: list[object], _type: object | list[object], *args: object | list[object]):
    """
    Usage::
        >>> check_type_list([1,2,3,4], int)
        True
        >>> check_type_list([1,2,3,[4,5]], int, int)
        True
        >>> check_type_list([1,2,3,[4,5,6.0]], int, int)
        False
        >>> check_type_list([1,2,3,[4,5,6.0]], int, [int,float])
        True
    """
    if isinstance(instances, (list, tuple)):
        for instance in instances:
            if len(args) > 0 and isinstance(instance, list):
                is_check = check_type_list(instance, *args)
            else:
                is_check = check_type(instance, _type)
            if is_check == False: return False
        return True
    else:
        return check_type(instances, _type)

def normal_pdf(x):
    return 1 / np.sqrt(2 * np.pi) * np.exp(-x**2 / 2)

def normal_cdf(x):
    return 0.5 * (1 + erf(x / np.sqrt(2)))

def _gen_erfcinv(erfc, math=math):
    """Generates the inverse function of erfc by the given erfc function and
    math module.
    """
    def erfcinv(y):
        """The inverse function of erfc."""
        if y >= 2:
            return -100.
        elif y <= 0:
            return 100.
        zero_point = y < 1
        if not zero_point:
            y = 2 - y
        t = math.sqrt(-2 * math.log(y / 2.))
        x = -0.70711 * \
            ((2.30753 + t * 0.27061) / (1. + t * (0.99229 + t * 0.04481)) - t)
        for i in range(2):
            err = erfc(x) - y
            x += err / (1.12837916709551257 * math.exp(-(x ** 2)) - x * err)
        return x if zero_point else -x
    return erfcinv

def _gen_ppf(erfc, math=math):
    """ppf is the inverse function of cdf.  This function generates cdf by the
    given erfc and math module.
    """
    erfcinv = _gen_erfcinv(erfc, math)
    def ppf(x, mu=0, sigma=1):
        """The inverse function of cdf."""
        return mu - sigma * math.sqrt(2) * erfcinv(2 * x)
    return ppf

def erfc(x):
    """Complementary error function (via `http://bit.ly/zOLqbc`_)"""
    z = abs(x)
    t = 1. / (1. + z / 2.)
    r = t * math.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (
        0.37409196 + t * (0.09678418 + t * (-0.18628806 + t * (
            0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (
                -0.82215223 + t * 0.17087277
            )))
        )))
    )))
    return 2. - r if x < 0 else r

ppf = _gen_ppf(erfc)

