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


def evaluate_ndcg(ranks_pred: np.ndarray, ranks_answer: np.ndarray, points_answer: np.ndarray=None, k: int=None):
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
    assert (ranks_pred   < 1).sum() == 0
    assert (ranks_answer < 1).sum() == 0
    if points_answer is not None:
        assert isinstance(points_answer, np.ndarray)
        assert ranks_pred.shape == points_answer.shape
    else:
        points_answer = np.argsort(np.argsort(-ranks_answer, axis=-1), axis=-1) + 1
    assert k is None or (isinstance(k, int) and k > 0)
    if k is None:
        if len(ranks_pred.shape) == 1:
            k = ranks_pred.shape[0]
        else:
            k = ranks_pred.shape[-1]
    ndf    = np.log2(np.arange(k) + 1)
    p_pred = np.take_along_axis(points_answer, np.argsort(ranks_pred, axis=-1), axis=-1)
    if len(ranks_pred.shape) == 1:
        p_answer = np.sort(points_answer, axis=-1)[::-1]
    else:
        p_answer = np.sort(points_answer, axis=-1)[:, ::-1]
    if len(ranks_pred.shape) == 1:
        dcg_i = p_answer[0] + (p_answer[1:k] / ndf[1:k]).sum()
        dcg   = p_pred[  0] + (p_pred[  1:k] / ndf[1:k]).sum()
        return dcg / dcg_i
    else:
        ndf   = np.tile(ndf,      (ranks_pred.shape[0], 1))
        dcg_i = p_answer[:, 0] + (p_answer[:, 1:k] / ndf[:, 1:k]).sum(axis=-1)
        dcg   = p_pred[  :, 0] + (p_pred[  :, 1:k] / ndf[:, 1:k]).sum(axis=-1)
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

