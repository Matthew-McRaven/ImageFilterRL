import torch
from scipy.special import logit as _logit
import torch, torch.nn, torch.nn.functional, torch.distributions

from ..ptree import ProbabilisticLeaf
from ..actions import AddContrastFilter as _AddContrastFilter
from ..actions import AddClipFilter as _AddClipFilter
from ..actions import AddGlobalHistogramEq as _AddGlobalHistogramEq
from ..actions import AddLocalHistogramEq as _AddLocalHistogramEq

class AddFilter(ProbabilisticLeaf):
    def __init__(self, where, *args):
        super(AddFilter, self).__init__(*args)
        self.where = where

class AddConstrastStretch(AddFilter):
    def _sample(self, weight_dict, device):
        return _AddContrastFilter(self.where, self, device)
    def _log_prob(self, action, weight_dict, device):
        return torch.tensor(0., requires_grad = True) 

class AddGlobalHistogramEq(AddFilter):
    def _sample(self, weight_dict, device):
        return _AddGlobalHistogramEq(self.where, self, device)
    def _log_prob(self, action, weight_dict, device):
        return torch.tensor(0., requires_grad = True) 

class AddLocalHistEq(AddFilter):
    def _dists(self, weight_dict):
        return weight_dict['p_00']
    def _sample(self, weight_dict, device):
        radius_dist = self._dists(weight_dict)
        radius = radius_dist.rsample()
        return _AddLocalHistogramEq(self.where, radius, self, device)
    def _log_prob(self, action, weight_dict, device):
        lp = self._dists(weight_dict).log_prob(action.radius)
        assert not torch.isnan(lp).any()
        return lp

class AddClip(AddFilter):
    def _sample(self, weight_dict, device):
        lb, ub = weight_dict['p_01'], weight_dict['p_02']
        lb, ub = lb.rsample(), ub.rsample()
        return _AddClipFilter(self.where, lb.item(), ub.item(), self, device)

    def _log_prob(self, action, weight_dict, device):
        
        lb, ub = weight_dict['p_01'], weight_dict['p_02']
        n0, n1 = lb.log_prob(torch.tensor(action.min_i)), ub.log_prob(torch.tensor(action.max_i))
        lp = n0 + n1
        assert not torch.isnan(lp).any()
        return lp