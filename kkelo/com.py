import numpy as np
import pandas as pd


__all__ = [
    "NumpyDict",
]


class NumpyDict:
    def __init__(self, is_check: bool=True, dtype=float, default_size: int | tuple[int, int]=1000, is_series: bool=False):
        assert isinstance(is_check, bool)
        if isinstance(default_size, int):
            assert default_size > 0
        else:
            assert isinstance(default_size, tuple)
            assert len(default_size) == 2
            assert default_size[0] > 0 and default_size[1] > 0
        self.is_check     = is_check
        self.is_series    = is_series
        self.keys         = pd.Series(dtype=int) if is_series else {}
        self.sets         = set()
        self.values       = np.zeros(default_size, dtype=dtype)
        self.dtype        = dtype
        self.default_size = default_size
        self.is_multi_dim = not isinstance(self.default_size, int)
        if self.is_series: assert self.is_multi_dim == False
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
                if self.is_multi_dim:
                    assert self.is_series == False
                    if isinstance(values, list):
                        assert len(values) == self.default_size[1]
                    else:
                        assert isinstance(values, np.ndarray)
                        assert len(values.shape) == 2
                        assert values.shape[-1] == self.default_size[1]
                        assert len(keys) == len(values)
                else:
                    assert isinstance(values, (list, np.ndarray)) # Not allowed just float or int value
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
                        # This is only for single dimention
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
                if self.is_multi_dim:
                    assert isinstance(values, list, np.ndarray)
                else:
                    assert isinstance(values, self.dtype)
            if keys in self.sets:
                self.values[self.keys[keys]] = values
            else:
                n_keys = len(self.keys)
                self.keys[keys] = n_keys
                if n_keys >= self.values.shape[0]:
                    self.values = np.concatenate([self.values, np.zeros(self.default_size, dtype=self.dtype)], axis=0)
                self.values[n_keys] = values
                self.sets.add(keys)
    def to_dict(self) -> dict:
        return {x: self.values[y] for x, y in self.keys.items()}
    def to_pandas(self) -> pd.DataFrame:
        df = pd.DataFrame(list(self.keys.items()), columns=["keys", "values"])
        df["values"] = self.values[df["values"].to_numpy(dtype=int)]
        return df
