import enum

from .swap import *
from .delete import *
from .add import *
from .modify import *

# Refernece the head / tail of filter list.
class where(enum.Enum):
    front = enum.auto()
    back = enum.auto()