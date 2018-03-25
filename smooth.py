import numpy as np

class Smoother(object):

    def __init__(self, runner, M, Minv, token_names, diff_thresholds):
        self._runner = runner
        self._token_names = token_names
        self._M = M
        self._Minv = Minv

        self._last = {tk: None for tk in token_names}

        self._dt = diff_thresholds

    def __call__(self, im):

        res = {}

        self._runner.run(image=im, M=self._M, Minv=self._Minv)

        for tk in self._token_names:

            val = runner[tk]

            if self._last[tk] is None: # the first point
                self._last[tk] = val
                res[tk] = val
                continue

            diff = val - self._last[tk]

            if np.any( np.abs(diff) > self._dt[tk] ):
                new_val = self._last[tk]
            else:
                new_val = 0.5 * (val + self._last[tk])

            self._last[tk] = new_val
            res[tk] = new_val

        return res


class Memory(object):

    def __init__(self, size=5):
        self._data = []
        self._n = 0
        self._size = size

    def insert(self, val):

        self._data.append(val)
        self._n += 1

        if self._n > self._size:
            self._data = self._data[1:]

    def last(self):
        return self._data[-1]

    def mean(self):
        return np.mean(self._data, axis=0)

    def std(self):
        return np.std(self._data, axis=0)

    def is_empty(self):
        return self._data == []

    def is_full(self):
        return len(self._data) == self._size
