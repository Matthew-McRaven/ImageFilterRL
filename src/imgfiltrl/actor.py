import functools

import more_itertools
import torch
import torch.nn as nn
import torch.distributions, torch.nn.init
import torch.optim

from .policy import DecisionTree, TreePolicy

# Actor that generates weightings for imgfiltrl.DecisionTree
class FilterTreeActor(nn.Module):
    def __init__(self, neural_module,  observation_space, output_dimension=(10,)):
        super(FilterTreeActor, self).__init__()
        self.observation_space = observation_space
        self.decision_tree = DecisionTree()

        self.input_dimension = list(more_itertools.always_iterable(neural_module.output_dimension))
        self._input_size = functools.reduce(lambda x,y: x*y, self.input_dimension, 1)
        self.neural_module = neural_module
        self.output_dimension = output_dimension
        self._output_size = functools.reduce(lambda x,y: x*y, self.output_dimension, 1)
        self._output_size = torch.tensor(self._output_size, requires_grad=True, dtype=torch.float)

        # Our output layers are used as the seed for some set of random number generators.
        # These random number generators are used to generate edge pairs.
        v = {}
        v['w_pre'] = nn.Linear(self._input_size, 1)
        v['w_swap'] = nn.Linear(self._input_size, 1)
        v['w_popf'] = nn.Linear(self._input_size, 1)
        v['w_popb'] = nn.Linear(self._input_size, 1)
        for i in range(3):
            key = f'w_pre_{i:02d}'
            v[key] = nn.Linear(self._input_size, 1)
        for i in range(1, 3):
            v[f'param_{i:02d}_mu'] = nn.Linear(self._input_size, 1)
            v[f'param_{i:02d}_std'] = nn.Linear(self._input_size, 1)
            
        self.weight_layers = torch.nn.ModuleDict(v)

        # Initialize NN
        for x in self.parameters():
            if x.dim() > 1:
                nn.init.kaiming_normal_(x)

    def recurrent(self):
        return self.neural_module.recurrent()
        
    def save_hidden(self):
        assert self.recurrent()
        return self.neural_module.save_hidden()

    def restore_hidden(self, state=None):
        assert self.recurrent()
        self.neural_module.restore_hidden(state)
        
    def get_policy_weights(self, input):
        # Detect NAN poisoning.
        try:
            output = self.neural_module(input).view(-1, self._input_size)
        except AssertionError as e:
            print(input, self)
            raise e

        weight_dict = {}
        for key in self.weight_layers:
            weight_dict[key] = self.weight_layers[key](output)
        # If values 1..n are all equally, they're probably all 0.
        # In that case, we shouldn't allow a delete.
        if torch.eq(input[1], input[2:]).all():
            weight_dict['w_popf'] = torch.tensor(float("-inf")).to(input.device)
            weight_dict['w_popb'] = torch.tensor(float("-inf")).to(input.device)
        weight_dict['w_swap'] = torch.tensor(float("-inf")).to(input.device)
        for i in range(1, 3):
            mu = weight_dict[f'param_{i:02d}_mu']
            std = weight_dict[f'param_{i:02d}_std']**2
            weight_dict[f'p_{i:02d}'] = torch.distributions.Normal(mu, std**2)

        return weight_dict

    def forward(self, input):
        weight_dict = self.get_policy_weights(input)
        # Encapsulate our poliy in an object so downstream classes don't
        # need to know what kind of distribution to re-create.
        policy = TreePolicy(self.decision_tree, weight_dict, input.device)

        actions = policy.sample(1)
        # Each actions is drawn independtly of others, so joint prob
        # is all of them multiplied together. However, since we have logprobs,
        # we need to sum instead.
        log_prob = sum(policy.log_prob(actions)) # type: ignore

        return actions, log_prob, policy