import random
import torch.nn as nn


class OneOf(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.transforms = args

    def forward(self, x):
        return random.choice(self.transforms)(x)
