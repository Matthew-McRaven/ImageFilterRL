import enum

class Filters(enum.Enum):
    NA = 0
    ContrastStretch = 1
    GlobalHistEq = 2
    LocalHistEq = 3
    Clip = 4
    BoxBlur = 5
    GaussianBlur = 6
    MedianBlur = 7

    def __new__(cls, value):
        member = object.__new__(cls)
        member._value_ = value
        return member

    def __int__(self):
        return self.value

    def __float__(self):
        return float(self.value)