import enum

from .swap import *
from .delete import *
from .add import *

class where(enum.Enum):
    front = enum.auto()
    back = enum.auto()