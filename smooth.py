import numpy as np


def compute_diffs(arr):

    n_records = arr.shape[0]
    new_shape = (n_records-1,) + arr.shape[1:]

    diffs = []

    for i in range(1, n_records):
        diffs.append( arr[i] - arr[i - 1] )

    return np.array(diffs)


class GenericSmoother(object):

    def __init__(self, proc_func, diff_threshold):

        self._proc_func = proc_func
        self._diff_threshold = diff_threshold
        self._last = None

    def __call__(self, input_object):

        val = self._proc_func(input_object)

        if self._first_run():
            self._remember_value(val)
            return val

        diff = val - self._get_last()

        if np.any( np.abs(diff) > self._diff_threshold ):
            new_val = self._get_last()
        else:
            new_val = self._average(val)

        self._remember_value(new_val)

        return new_val

    def _first_run(self):
        return self._last is None

    def _average(self, val):
        return 0.5 * (val + self._last)

    def _get_last(self):
        return self._last

    def _remember_value(self, val):
        self._last = val


class GenericSmootherWithMemory(GenericSmoother):

    def __init__(self, proc_func, diff_threshold, memory_size=4):

        self._proc_func = proc_func
        self._diff_threshold = diff_threshold
        self._mem = Memory(memory_size)

    def _first_run(self):
        return self._mem.is_empty()

    def _average(self, val):
        return 0.5 * (val + self._mem.mean())

    def _get_last(self):
        return self._mem.last()

    def _remember_value(self, val):
        self._mem.insert(val)


class Smoother(object):
    '''
    Performs processing of a stream of images with build-in smooting
    and outlier rejection
    '''

    def __init__(self, runner, M, token_names, diff_thresholds):
        '''
        runner -- an instance of CompGraphRunner
        M -- perspective transform matrix mapping from the original image
             to the corresponding bird's-eye view of the road
        token_names -- names of tokens to be smoothed / monitored for outliers
        diff_thresholds -- a dictionary of threshold values on the absolute
                           differences between two subsequent measurements
                           (for each token in token_names)
        '''

        self._runner = runner
        self._token_names = token_names
        self._M = M

        self._last = {tk: None for tk in token_names}

        self._dt = diff_thresholds

    def __call__(self, im):

        res = {}

        self._runner.run(image=im, M=self._M)

        for tk in self._token_names:

            val = self._runner[tk]

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

    @property
    def runner(self):
        return self._runner


class Memory(object):
    '''
    A container for storing n last values of a parameter of interest
    (scalars or 1D NumPy arrays)
    '''

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

    def size(self):
        return len(self._data)

    def is_empty(self):
        return self._data == []

    def is_full(self):
        return len(self._data) == self._size
