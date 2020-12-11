
from ..ptree import ProbabilisticLeaf

class DeleteFilter(ProbabilisticLeaf):
    def __init__(self, where, *args):
        super(DeleteFilter, self).__init__(*args)
    def _sample(self, weight_dict, device):
        raise NotImplementedError()

    def _log_prob(self, action, weight_dict, device):
        raise NotImplementedError()
