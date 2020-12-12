import collections
import copy

import gym, gym.spaces
import more_itertools
import numpy as np
import torch
import torchvision.datasets, torchvision.transforms
import torch.nn as nn

import librl.nn.core, librl.nn.classifier
import librl.task, librl.task.classification
import librl.train.classification.label

import imgfiltrl.reward
import imgfiltrl.actions as _actions
from imgfiltrl.nodes import where as _where

class ImageClassifictionEnv(gym.Env):
    def _shapes(self):
        i, n = float("inf"), float("-inf")
        return [0, n, n, n], [4, i, i, i]
    def __init__(self, baseline, augmented, inner_loss=None, train_dataset=None, 
        validation_dataset=None, normalize_fn = None, classes=10, reward_fn=imgfiltrl.reward.Linear, adapt_steps=3,
        device='cpu'
    ):
        super(ImageClassifictionEnv, self).__init__()

        assert classes > 1

        self.layer_params = 4
        self.layer_depth = 3
        mins, maxs  = self._shapes()
        low = np.tile(np.fromiter(mins, dtype=np.float64, count=self.layer_params), (self.layer_depth, 1))
        high = np.tile(np.fromiter(maxs, dtype=np.float64, count=self.layer_params), (self.layer_depth, 1))

        self.observation_space = gym.spaces.Box(low, high, [self.layer_depth, self.layer_params], dtype=np.int16)
        self.classes = classes
        self.reward_fn = reward_fn
        self.adapt_steps = adapt_steps
        self.device = device

        self.baseline = baseline
        self.augmented = augmented
        self.inner_loss = inner_loss
        self.train_percent = .05
        self.validation_percent = .1
        
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.normalize_fn = normalize_fn

    def reset(self):
        self.state = collections.deque([], self.layer_depth)

        # At the start of a new training epoch, erase all learning of inner networks.
        for x in self.baseline.parameters():
            if x.dim() > 1:
                nn.init.kaiming_normal_(x)
        # For fairness, init both networks to same values.
        params = copy.deepcopy(self.baseline.state_dict())
        self.augmented.load_state_dict(params)

        return self._convert_state(self.state)

    def step(self, actions):
        for action in actions: self._apply_action(action)
        total = []
        baseline_correct, augmented_correct = [], []
        for _ in range(self.adapt_steps + 1):
            t, bc, ac = self._train_step()
            total.append(t)
            baseline_correct.append(bc)
            augmented_correct.append(ac)
        reward_baseline = self.reward_fn(baseline_correct[-1], total[-1], self.classes, classifier=self.baseline)
        reward_augmented = self.reward_fn(augmented_correct[-1], total[-1], self.classes, classifier=self.augmented)
        return self._convert_state(self.state), (reward_augmented-reward_baseline), False, {}
        
    def _apply_action(self, action):
        # Add
        if isinstance(action, _actions.AddFilter):
            if action.where == _where.front: self.state.appendleft(action)
            elif action.where == _where.back: self.state.append(action)
        # Swap
        elif isinstance(action, _actions.SwapFilters):
            n0, n1 = round(action.n0.item()), round(action.n1.item())
            self.state[n0], self.state[n1] = self.state[n1], self.state[n0]
        # Delete
        elif isinstance(action, _actions.DeleteFilter):
            if action.where == _where.front: self.state.popleft()
            elif action.where == _where.back: self.state.pop()
        else: raise NotImplementedError(f"Don't know this filter type: {type(action)}")


    def _transform_baseline(self):
        return self.normalize_fn

    def _transform_augmented(self):
        return self.normalize_fn      

    def _train_step(self):
        total, correct_baseline, correct_augmented = 0,0,0
        self.baseline.train(), self.augmented.train()
        observed_train_data = 0
        dataloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=100, shuffle=True)
        for _, (data, target) in enumerate(dataloader):
            data_baseline = self._transform_baseline()(data)
            data_augmented = self._transform_augmented()(data)
            data_baseline, data_augmented = data_baseline.to(self.device), data_augmented.to(self.device)
            target = target.to(self.device)
            # Transform data for baseline.
            librl.task.classification.train_one_batch(self.baseline, self.inner_loss, data_baseline, target)
            # Transform data for augmented.
            librl.task.classification.train_one_batch(self.augmented, self.inner_loss, data_augmented, target)
            observed_train_data += len(data)
            if observed_train_data >= self.train_percent * len(dataloader):
                break

        
        self.baseline.eval(), self.augmented.eval()
        observed_valid_data = 0
        dataloader = torch.utils.data.DataLoader(self.validation_dataset, batch_size=100, shuffle=True)
        for _, (data, target) in enumerate(dataloader):
            data_baseline = self._transform_baseline()(data)
            data_augmented = self._transform_augmented()(data)
            data_baseline, data_augmented = data_baseline.to(self.device), data_augmented.to(self.device)
            target = target.to(self.device)
                # Transform data for baseline.
            _, selected_baseline = librl.task.classification.test_one_batch(self.baseline, self.inner_loss, data, target)
            # Transform data for augmented.
            _, selected_augmented = librl.task.classification.test_one_batch(self.augmented, self.inner_loss, data, target)
            correct_baseline += torch.eq(selected_baseline, target).sum() 
            correct_augmented += torch.eq(selected_augmented, target).sum() 
            total += float(target.shape[0])
            observed_valid_data += len(data)
            if observed_valid_data >= self.validation_percent * len(dataloader):
                break
        return total, correct_baseline, correct_augmented

    def _convert_state(self, state):
        np_state = np.zeros((self.layer_depth, self.layer_params))
        if state:
            for idx, filter in enumerate(state): np_state[idx] = filter.array()
        return np_state