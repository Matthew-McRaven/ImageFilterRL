import torch
from scipy.special import logit as _logit
import torch, torch.nn, torch.nn.functional, torch.distributions

from ..ptree import ProbabilisticLeaf
from ..actions import AddContrastFilter as _AddContrastFilter
from ..actions import AddClipFilter as _AddClipFilter

class AddFilter(ProbabilisticLeaf):
    def __init__(self, where, *args):
        super(AddFilter, self).__init__(*args)
        self.where = where

class AddConstrastStretch(AddFilter):
    def _sample(self, weight_dict, device):
        return _AddContrastFilter(self.where, self, device)
    def _log_prob(self, action, weight_dict, device):
        return torch.tensor(0., requires_grad = True) 

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