import torch

from ..ptree import ProbabilisticLeaf
from ..actions import DeleteFilter as _DeleteFilter


# Either pop the first or last filter.
class DeleteFilter(ProbabilisticLeaf):
    def __init__(self, where, *args):
        super(DeleteFilter, self).__init__(*args)
        self.where = where
    def _sample(self, weight_dict, device):
        return _DeleteFilter(self.where, self, device)

    def _log_prob(self, action, weight_dict, device):
        return torch.tensor(0., requires_grad=True)
