import torch
from scipy.special import logit as _logit
import torch, torch.nn, torch.nn.functional, torch.distributions

from ..ptree import ProbabilisticLeaf
from .. import actions as _actions 

class AddFilter(ProbabilisticLeaf):
    def __init__(self, where, *args):
        super(AddFilter, self).__init__(*args)
        self.where = where

class AddConstrastStretch(AddFilter):
    def _sample(self, weight_dict, device):
        return _actions.AddContrastFilter(self.where, self, device)
    def _log_prob(self, action, weight_dict, device):
        return torch.tensor(0., requires_grad = True) 

class AddGlobalHistogramEq(AddFilter):
    def _sample(self, weight_dict, device):
        return _actions.AddGlobalHistogramEq(self.where, self, device)
    def _log_prob(self, action, weight_dict, device):
        return torch.tensor(0., requires_grad = True) 

class AddLocalHistEq(AddFilter):
    def _dists(self, weight_dict):
        return weight_dict['p_00']
    def _sample(self, weight_dict, device):
        radius_dist = self._dists(weight_dict)
        radius = radius_dist.rsample()
        return _actions.AddLocalHistogramEq(self.where, radius, self, device)
    def _log_prob(self, action, weight_dict, device):
        lp = self._dists(weight_dict).log_prob(action.radius)
        assert not torch.isnan(lp).any()
        return lp

class AddClip(AddFilter):
    def _sample(self, weight_dict, device):
        lb, ub = weight_dict['p_01'], weight_dict['p_02']
        lb, ub = lb.rsample(), ub.rsample()
        return _actions.AddClipFilter(self.where, lb.item(), ub.item(), self, device)

    def _log_prob(self, action, weight_dict, device):
        
        lb, ub = weight_dict['p_01'], weight_dict['p_02']
        n0, n1 = lb.log_prob(torch.tensor(action.min_i)), ub.log_prob(torch.tensor(action.max_i))
        lp = n0 + n1
        assert not torch.isnan(lp).any()
        return lp

# Blurs!
class AddBoxBlur(AddFilter):
    def _dists(self, weight_dict):
        return weight_dict['p_00']
    def _sample(self, weight_dict, device):
        radius_dist = self._dists(weight_dict)
        radius = radius_dist.rsample()
        return _actions.AddBoxBlur(self.where, radius.item(), self, device)
    def _log_prob(self, action, weight_dict, device):
        lp = self._dists(weight_dict).log_prob(torch.tensor(action.radius, requires_grad=True))
        assert not torch.isnan(lp).any()
        return lp

class AddGaussBlur(AddFilter):
    def _dists(self, weight_dict):
        return weight_dict['p_01']
    def _sample(self, weight_dict, device):
        sigma_dist = self._dists(weight_dict)
        sigma = sigma_dist.rsample()
        return _actions.AddGaussianBlur(self.where, sigma.item(), self, device)
    def _log_prob(self, action, weight_dict, device):
        lp = self._dists(weight_dict).log_prob(torch.tensor(action.sigma, requires_grad=True))
        assert not torch.isnan(lp).any()
        return lp

class AddMedianBlur(AddFilter):
    def _dists(self, weight_dict):
        return weight_dict['p_00']
    def _sample(self, weight_dict, device):
        radius_dist = self._dists(weight_dict)
        radius = radius_dist.rsample()
        return _actions.AddMedianBlur(self.where, radius.item(), self, device)
    def _log_prob(self, action, weight_dict, device):
        lp = self._dists(weight_dict).log_prob(torch.tensor(action.radius, requires_grad=True))
        assert not torch.isnan(lp).any()
        return lp

# Medial Axis skeletonization:
# Threshold + skeletonize the image.
class AddMedialAxisSkeltonization(AddFilter):
    def _dists(self, weight_dict):
        return weight_dict['p_01']
    def _sample(self, weight_dict, device):
        threshold_dist = self._dists(weight_dict)
        threshold = threshold_dist.rsample()
        return _actions.AddSkeletonize(self.where, threshold.item(), self, device)
    def _log_prob(self, action, weight_dict, device):
        lp = self._dists(weight_dict).log_prob(torch.tensor(action.threshold, requires_grad=True))
        assert not torch.isnan(lp).any()
        return lp

# Edge detection
# Threshold + skeletonize the image.
class AddEdgeDetector(AddFilter):
    def __init__(self, where, *args, kind="sobel"):
        super(AddEdgeDetector, self).__init__(where, *args)
        self.kind = kind
    def _sample(self, weight_dict, device):
        return _actions.AddEdgeDetection(self.where, self, device, kind=self.kind)
    def _log_prob(self, action, weight_dict, device):
        return torch.tensor(0., requires_grad = True) 