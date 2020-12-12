import enum

class Filters(enum.Enum):
    NA = 0
    ContrastStretch = 1
    Clip = 2
    def __new__(cls, value):
        member = object.__new__(cls)
        member._value_ = value
        return member

    def __int__(self):
        return self.value