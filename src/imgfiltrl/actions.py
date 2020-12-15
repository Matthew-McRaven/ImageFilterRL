import functools
import numpy as np
import torch
import libimg
import libimg.equalize, libimg.interpolate
import skimage.exposure
import skimage.filters
import skimage.morphology
import skimage.util
import scipy.ndimage
import libimg.image
import libimg.functional.convolve

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
        assert isinstance(param_shift, float)
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
        return np.asarray([_Filters.ContrastStretch, 0, 0, 0], dtype=np.float64)
class AddGlobalHistogramEq(AddFilter):
    
    def __init__(self, where, *args):
        def wrap(image):
            rv = skimage.exposure.equalize_hist(image.data)
            return libimg.image.Image(rv)
        super(AddGlobalHistogramEq, self).__init__(where, wrap, *args)

    def array(self):
        return np.asarray([_Filters.GlobalHistEq, 0, 0, 0], dtype=np.float64)
class AddLocalHistogramEq(AddFilter):
    def _create_filter(self, radius):
        self.radius = radius

        def wrap(image):
            # Implemented as per:
            #   https://scikit-image.org/docs/dev/auto_examples/color_exposure/plot_equalize.html#sphx-glr-auto-examples-color-exposure-plot-equalize-py
            selem = skimage.morphology.disk(5*radius)
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
        return np.asarray([_Filters.LocalHistEq, self.radius, 0, 0], dtype=np.float64)

    def modify(self, param_idx, param_shift):
        if param_idx == 0:
            radius = np.clip(self.radius+param_shift, a_min=0, a_max=1)
            assert isinstance(radius, float)
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
        return np.asarray([_Filters.Clip, 0, self.min_i, self.max_i], dtype=np.float64)

    def modify(self, param_idx, param_shift):
        min_i, max_i = self.min_i, self.max_i
        if param_idx == 1:
            min_i = np.clip(self.min_i+param_shift, a_min=0, a_max=1)
        if param_idx == 2:
            max_i = np.clip(self.max_i+param_shift, a_min=0, a_max=1)
        assert isinstance(min_i, float)
        assert isinstance(max_i, float)
        self.filter = self._create_filter(min_i, max_i)

# Blurs

class AddBoxBlur(AddFilter):
    def _create_filter(self, radius):
        self.radius = radius
        radius = round(radius*5)
        radius = np.clip(radius, 1, 14)
        filter = libimg.functional.convolve.BoxFilter(radius) 
        return filter.apply
    def __init__(self, where, radius, *args):
        filter = self._create_filter(radius)
        super(AddBoxBlur, self).__init__(where, filter, *args)

    def array(self):
        return np.asarray([_Filters.BoxBlur, self.radius, 0, 0], dtype=np.float64)

    def modify(self, param_idx, param_shift):
        if param_idx == 0:
            radius = np.clip(self.radius+param_shift, a_min=0, a_max=1)
            assert isinstance(radius, float)
            self.filter = self._create_filter(radius) 

class AddGaussianBlur(AddFilter):
    def _create_filter(self, sigma):
        self.sigma = sigma
        def wrap(image):
            rv = skimage.filters.gaussian(image.data, sigma=2*sigma)
            return libimg.image.Image(rv)
        return wrap

    def __init__(self, where, sigma, *args):
        filter = self._create_filter(sigma)
        super(AddGaussianBlur, self).__init__(where, filter, *args)

    def array(self):
        print(self.sigma)
        return np.asarray([_Filters.GaussianBlur, 0, self.sigma, 0], dtype=np.float64)
    
    def modify(self, param_idx, param_shift):
        if param_idx == 1:
            sigma = np.clip(self.sigma+param_shift, a_min=0, a_max=1)
            assert isinstance(sigma, float)
            self.filter = self._create_filter(sigma) 

class AddMedianBlur(AddFilter):
    def _create_filter(self, radius):
        self.radius = radius
        def wrap(image):
            r = np.clip(round(3*radius), 0, 14)
            assert r < 14
            # Flatten image to 2d, since median filter doesn't understand.
            rv = scipy.ndimage.median_filter(image.data.reshape(28,28), size=r)
            return libimg.image.Image(rv.reshape(1,28,28))
        return wrap

    def __init__(self, where, radius, *args):
        filter = self._create_filter(radius)
        super(AddMedianBlur, self).__init__(where, filter, *args)

    def array(self):
        return np.asarray([_Filters.MedianBlur, self.radius, 0, 0], dtype=np.float64)

    def modify(self, param_idx, param_shift):
        if param_idx == 0:
            radius = np.clip(self.radius+param_shift, a_min=0, a_max=1)
            assert isinstance(radius, float)
            self.filter = self._create_filter(radius)

# Skeletonization
class AddSkeletonize(AddFilter):
    def _create_filter(self, threshold):
        self.threshold = threshold
        def wrap(image):
            # Flatten image to 2d, since filter doesn't understand.
            
            bin = image.data.reshape(28, 28) > 255*threshold
            
            if self.alg == "skeletonize":
                rv = skimage.morphology.skeletonize(bin)
            elif self.alg == "medial_axis":
                rv = skimage.morphology.medial_axis(bin)
            else: raise NotImplementedError("This skeletonization not offered.")
            # Remap to [0,255], otherwise NN will be confused.
            rv = skimage.exposure.rescale_intensity(rv, out_range=(0., 255.))
            return libimg.image.Image(rv.reshape(1,28,28))
        return wrap

    def __init__(self, where, threshold, *args, alg="skeletonize"):
        self.alg = alg
        filter = self._create_filter(threshold)
        super(AddSkeletonize, self).__init__(where, filter, *args)

    def array(self):
        return np.asarray([_Filters.MedialAxisSkeleton, 0, self.threshold, 0], dtype=np.float64)

    def modify(self, param_idx, param_shift):
        if param_idx == 1:
            threshold = np.clip(self.threshold+param_shift, a_min=0, a_max=1)
            assert isinstance(threshold, float)
            self.filter = self._create_filter(threshold)

# Edge detection
class AddEdgeDetection(AddFilter):
    
    def __init__(self, where, *args, kind=""):
        def wrap(image):
            rv = image.data.reshape(28,28)
            if kind == "sobel": rv = skimage.filters.sobel(rv)
            elif kind == "scharr": rv = skimage.filters.scharr(rv)
            elif kind == "prewitt": rv = skimage.filters.prewitt(rv)
            elif kind == "roberts": rv = skimage.filters.roberts(rv)
            else: raise NotImplementedError(f"Haven't implemented filter type '{kind}'")
            rv = skimage.exposure.equalize_hist(image.data)
            rv = skimage.exposure.rescale_intensity(image.data, out_range=(0., 255.))
            return libimg.image.Image(rv)
        super(AddEdgeDetection, self).__init__(where, wrap, *args)

    def array(self):
        return np.asarray([_Filters.EdgeDetection, 0, 0, 0], dtype=np.float64)