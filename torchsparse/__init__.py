import torchsparse.backends as backends

from .operators import *
from .tensor import *
from .utils.tune import tune
from .version import __version__

try:
    backends.init()
except:
    print("No CUDA device available.")
