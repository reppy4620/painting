from .layers import (
    Concat,
    Branch, BranchAdd,
    Identity
)

from .loss import hinge_loss

from .activations import (
    mish, Mish,
    tanhexp, TanhExp
)
from .utils import make_channels
