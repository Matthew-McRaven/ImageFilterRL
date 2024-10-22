import math

import numpy as np
import torch 
def Linear(correct, total, *args, **kwargs):
    return 100 * correct / total
class PolyAsymptotic:
    def __init__(self, neg_scale=-1, pos_scale=1, neg_clip=-5, pos_clip=10, penalty_fn = lambda *_,**__: 0):
        assert callable(penalty_fn)
        self.penalty_fn = penalty_fn

        self.neg_scale = neg_scale
        self.pos_scale = pos_scale
        self.neg_clip = neg_clip
        self.pos_clip = pos_clip
    def __call__(self, correct, total, classes, **kwargs):
        p = correct/total
        # Exponetially decay reward when classifiation is worse than random chance.
        # Set r=0 @ p == 1/classes
        if p < classes: r = 10**self.neg_scale*((-1/(p*classes)) + 1)
        # Exponentially increase reward when classification is bettern than random chance.
        # Set r=0 @ p == 1/classes
        elif p<100:r = -10**self.pos_scale/(p-1) + (classes*10**self.pos_scale)/(classes-1)
        r = np.clip(r, self.neg_clip, self.pos_clip)
        return r + self.penalty_fn(**kwargs)