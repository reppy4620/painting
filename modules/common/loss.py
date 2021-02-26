import torch.nn.functional as F


def hinge_loss(fake, real):
    return (F.relu(1 + fake) + F.relu(1 - real)).mean()
