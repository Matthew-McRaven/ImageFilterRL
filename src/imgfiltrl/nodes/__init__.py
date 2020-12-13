import enum

from .swap import *
from .delete import *
from .add import *
from .modify import *

class where(enum.Enum):
    front = enum.auto()
    back = enum.auto()