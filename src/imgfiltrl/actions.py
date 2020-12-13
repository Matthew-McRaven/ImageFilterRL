import functools
import numpy as np
import torch
import libimg
import libimg.equalize, libimg.interpolate
import skimage.exposure
import libimg.image

from imgfiltrl.filters import Filters as _Filters
class Action:
    def __init__(self, parent, device):
        self.parent = parent
        self.device = device
    def log_prob(self, weight_dict):
        return self.parent.log_prob(self, weight_dict, self.device)

class SwapFilters(Action):
    def __init__(self, n0, n1, *args):
        super(SwapFilters, self).__init__(*args)
        self.n0 = n0
        self.n1 = n1
class ModifyFilter(Action):
    def __init__(self, layer_num, param_idx, param_shift, *args):
        super(ModifyFilter, self).__init__(*args)
        self.layer_num = layer_num
        self.param_idx = param_idx
        self.param_shift = param_shift
class DeleteFilter(Action):
    def __init__(self, where, *args):
        super(DeleteFilter, self).__init__(*args)
        self.where = where
class AddFilter(Action):
    def __init__(self, where, filter, *args):
        super(AddFilter, self).__init__(*args)
        self.where = where
        self.filter = filter
    def modify(self, param_idx, param_shift):
        pass
class AddContrastFilter(AddFilter):
    def __init__(self, where, *args):
        filter = functools.partial(libimg.equalize.stretch_contrast)
        super(AddContrastFilter, self).__init__(where, filter, *args)

    def array(self):
        return np.asarray([_Filters.ContrastStretch, 0, 0, 0], dtype=np.int)
class AddGlobalHistogramEq(AddFilter):
    
    def __init__(self, where, *args):
        def wrap(image):
            rv = skimage.exposure.equalize_hist(image.data)
            return libimg.image.Image(rv)
        super(AddGlobalHistogramEq, self).__init__(where, wrap, *args)

    def array(self):
        return np.asarray([_Filters.GlobalHistEq, 0, 0, 0], dtype=np.int)
class AddLocalHistogramEq(AddFilter):
    def _create_filter(self, radius):
        self.radius = radius
        import skimage.filters
        import skimage.morphology
        import skimage.util
        def wrap(image):
            selem = skimage.morphology.disk(5*radius.item())
            rv = skimage.exposure.rescale_intensity(image.data)
            rv = skimage.util.img_as_ubyte(rv).reshape(28, 28)
            rv = skimage.filters.rank.equalize(rv, selem=selem)
            rv = skimage.util.img_as_float(rv).reshape(1, 28, 28)
            rv = skimage.exposure.rescale_intensity(image.data, out_range=(0., 255.))
            return libimg.image.Image(rv)

        return wrap

    def __init__(self, where, radius, *args):
        filter = self._create_filter(radius)
        super(AddLocalHistogramEq, self).__init__(where, filter, *args)

    def array(self):
        return np.asarray([_Filters.LocalHistEq, 0, 0, 0], dtype=np.int)

    def modify(self, param_idx, param_shift):
        if param_idx == 0:
            radius = torch.clamp(self.radius+param_shift, min=0, max=1)
            self.filter = self._create_filter(radius)
        
class AddClipFilter(AddFilter):
    def _create_filter(self, min_i, max_i):
        self.min_i, self.max_i = min_i, max_i
        l, u = self.min_i, self.max_i
        l, u = sorted([255*l,255*u])
        return functools.partial(libimg.interpolate.intensity_clip, new_min=l, new_max=u)
        
    def __init__(self, where, min_i, max_i, *args):
        print(f"{min_i},{max_i}")
        filter = self._create_filter(min_i, max_i)
        super(AddClipFilter, self).__init__(where, filter, *args)

    def array(self):
        return np.asarray([_Filters.Clip, self.radius, 0, 0], dtype=np.int)

    def modify(self, param_idx, param_shift):
        min_i, max_i = self.min_i, self.max_i
        if param_idx == 1:
            min_i = torch.clamp(self.min_i+param_shift, min=0, max=1)
        if param_idx == 2:
            max_i = torch.clamp(self.max_i+param_shift, min=0, max=1)
        self.filter = self._create_filter(min_i, max_i)
