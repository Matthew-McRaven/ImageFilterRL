
import torch
from scipy.special import logit as _logit
import torch, torch.nn, torch.nn.functional, torch.distributions

from ..ptree import ProbabilisticLeaf
from ..actions import ModifyFilter as _ModifyFilter

class ModifyFilter(ProbabilisticLeaf):
    def __init__(self, *args):
        super(ModifyFilter, self).__init__(*args)
    def _dists(self, weight_dict):
        layer_dist = torch.distributions.categorical.Categorical(probs=weight_dict['w_modify_layer'])
        param_dist = torch.distributions.uniform.Uniform(-.5, 2.5)
        list_shifts = [weight_dict['shift_00'],weight_dict['shift_01'],weight_dict['shift_02']]
        return layer_dist, param_dist, list_shifts

    def _sample(self, weight_dict, device):
        layer_dist, param_dist, list_shifts = self._dists(weight_dict)
        layer_num = layer_dist.sample()
        param_idx = param_dist.sample()
        param_shift = list_shifts[round(param_idx.item())].sample().item()
        assert isinstance(param_shift, float)
        
        return _ModifyFilter(layer_num, param_idx, param_shift, self, device)
         
    def _log_prob(self, action, weight_dict, device):
        layer_dist, param_dist, list_shifts = self._dists(weight_dict)
        layer_prob = layer_dist.log_prob(action.layer_num)
        param_prob = param_dist.log_prob(action.param_idx)
        shift = torch.tensor(action.param_shift, requires_grad=True)
        shift_prob = list_shifts[round(action.param_idx.item())].log_prob(shift)
        print(layer_prob, param_prob, shift_prob)
        lp = layer_prob + param_prob + shift_prob
        assert not torch.isnan(lp).any()
        return lp