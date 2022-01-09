from typing import Dict, List, Optional, Set, Tuple

from allennlp.common import Registrable
from allennlp.training.metrics.metric import Metric

import math

class MomentsMetric(Metric, Registrable):
    def __init__(self):
        self.reset()

    def reset(self):
        self._num_values = 0.0
        self._mean = 0.0
        self._m2 = 0.0

    def __call__(self, xs):
        xs, = Metric.unwrap_to_tensors(xs)
        for i in range(xs.size(0)):
            x = xs[i].item()
            self._num_values += 1
            delta1 = x - self._mean
            self._mean += delta1 / self._num_values
            delta2 = x - self._mean
            self._m2 += delta1 * delta2

    def get_metric(self, reset = False):
        stdev = math.sqrt(self._m2 / self._num_values) if self._num_values > 0 else 0.0
        return {
            "n": self._num_values,
            "mean": self._mean,
            "stdev": stdev
        }
