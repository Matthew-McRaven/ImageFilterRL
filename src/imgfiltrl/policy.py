from .ptree import *
from .nodes import *
from imgfiltrl import nodes
# Create a probabilisitic decision tree corresponding to my possible actions.
# Slides provide an example.
def DecisionTree():
    # Add Nodes #
    # Prepend
    prepend_list = [
        nodes.AddConstrastStretch(where.front), #0
        nodes.AddClip(where.front), #1
        nodes.AddGlobalHistogramEq(where.front), #2
        nodes.AddLocalHistEq(where.front), #3
        nodes.AddBoxBlur(where.front), #4
        nodes.AddGaussBlur(where.front), #5
        #nodes.AddMedianBlur(where.front), #6
        nodes.AddMedialAxisSkeltonization(where.front), #7
        nodes.AddEdgeDetector(where.front, kind="sobel"), #8
        nodes.AddEdgeDetector(where.front, kind="prewitt"), #9
        nodes.AddEdgeDetector(where.front, kind="roberts"), #10
        nodes.AddEdgeDetector(where.front, kind="scharr"), #11
    ]
    prepend_weights = [f"w_pre_{i:02d}" for i in range(len(prepend_list))]
    masks =  range(len(prepend_weights))
    prepend_weights = [(v if i in masks else "ninf") for i,v in enumerate(prepend_weights)]
    node_prepend = ProbabalisticBranch(prepend_list, 
        prepend_weights
    )
    # Modify an existing node
    node_modify = nodes.ModifyFilter()
    # Everything else
    node_swap = nodes.SwapFilters()
    node_popf = nodes.DeleteFilter(where.front)
    node_popb = nodes.DeleteFilter(where.back)
    return ProbabalisticBranch([node_prepend, node_swap, node_popf, node_popb, node_modify],
        ["w_pre", "w_swap", "w_popf", "w_popb", "w_modify"]
    )

# Represent a policy that uses a probabilistic decision tree to generate actions.
class TreePolicy:
    def __init__(self, decision_tree, weights, device):
        self.decision_tree = decision_tree
        self.weights = weights
        self.device = device

    def sample(self, count):
        return self.decision_tree.sample(count, self.weights, self.device)

    def log_prob(self, actions):
        # librl expects a tensor of logprobs matching the shape of actions.
        return torch.stack([action.log_prob(self.weights) for action in actions])