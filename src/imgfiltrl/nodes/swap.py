import math

import torch, torch.nn, torch.nn.functional, torch.distributions

from ..ptree import ProbabilisticLeaf

class SwapFilters(ProbabilisticLeaf):
    def _sample(self, weight_dict, device):
        raise NotImplementedError()

    def _log_prob(self, action, weight_dict, device):
        raise NotImplementedError()

