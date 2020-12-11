from .ptree import *
from .nodes import *
from imgfiltrl import nodes

def DecisionTree():
    # Add Nodes #
    # Prepend
    node_prepend_01 = nodes.AddConstrastStretch(where.front)
    node_prepend_02 = nodes.AddClip(where.front)
    node_prepend = ProbabalisticBranch([node_prepend_01, node_prepend_02], 
        ["w_pre_01", "w_pre_02"]
    )
    # Modify an existing node
    # TODO: Magic
    # Everything else
    node_swap = nodes.SwapFilters()
    node_popf = nodes.DeleteFilter(where.front)
    node_popb = nodes.DeleteFilter(where.back)
    return ProbabalisticBranch([node_prepend, node_swap, node_popf, node_popb],
        ["w_pre", "w_swap", "w_popf", "w_popb"]
    )

# Represent a policy that uses a probabilistic decision tree to generate actions.
class TreePolicy:
    def __init__(self, decision_tree, weights, device):
        #print(decision_tree)
        self.decision_tree = decision_tree
        self.weights = weights
        self.device = device

    def sample(self, count):
        return self.decision_tree.sample(count, self.weights, self.device)

    def log_prob(self, actions):
        # librl expects a tensor of logprobs matching the shape of actions.
        return torch.stack([action.log_prob(self.weights) for action in actions])