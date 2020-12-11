class Action:
    def __init__(self, parent, device):
        self.parent = parent
        self.device = device
    def log_prob(self, weight_dict):
        return self.parent.log_prob(self, weight_dict, self.device)

class SwapFilters:
    def __init__(self, n0, n1, *args):
        super(SwapFilters, self).__init__(*args)
        self.n0 = n0
        self.n1 = n1

class DeleteFilter(Action):
    def __init__(self, where, *args):
        super(DeleteFilter, self).__init__(*args)

class AddContrastFilter(Action):
    def __init__(self, where, filter, *args):
        super(AddContrastFilter, self).__init__(*args)
        self.filter = filter

class AddClipFilter(Action):
    def __init__(self, where, filter, *args):
        super(AddClipFilter, self).__init__(*args)
        self.filter = filter