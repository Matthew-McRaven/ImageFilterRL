import enum

# Record what kinds of filters I allow
class Filters(enum.Enum):
    NA = 0
    ContrastStretch = 1
    GlobalHistEq = 2
    LocalHistEq = 3
    Clip = 4
    BoxBlur = 5
    GaussianBlur = 6
    MedianBlur = 7
    MedialAxisSkeleton = 8
    EdgeDetection = 9

    # enum implictly assigns .value
    # Use these methods to convert enumerated type to float/int.
    
    def __int__(self):
        return self.value

    def __float__(self):
        return float(self.value)