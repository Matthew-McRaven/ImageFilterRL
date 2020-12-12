import math

import torch, torch.nn, torch.nn.functional, torch.distributions

from ..ptree import ProbabilisticLeaf
from ..actions import SwapFilters as _SwapFilters
class SwapFilters(ProbabilisticLeaf):
    def _dists(self, weight_dict):
        return torch.distributions.Uniform(
            weight_dict['w_swap_mindex'], weight_dict['w_swap_maxdex']
        )
    def _sample(self, weight_dict, device):
        dist = self._dists(weight_dict)
        n0, n1 = dist.rsample(), dist.rsample()
        return _SwapFilters(n0, n1, self, device)

    def _log_prob(self, action, weight_dict, device):
        dist = self._dists(weight_dict)
        return dist.log_prob(action.n0) + dist.log_prob(action.n1)

