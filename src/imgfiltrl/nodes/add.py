import torch

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
        return _AddClipFilter(self.where, 0., 255., self, device)
    def _log_prob(self, action, weight_dict, device):
        return torch.tensor(0., requires_grad = True) 