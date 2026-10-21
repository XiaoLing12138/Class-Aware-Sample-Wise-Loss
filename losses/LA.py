import torch
import numpy as np


class LogitAdjustment:
    # Logit Adjustment

    def __init__(self, num_per_cls):
        nps = np.array(num_per_cls)
        class_freq = nps / nps.sum()
        adjustments = np.log(class_freq ** 1 + 1e-12)
        self.adjustment = torch.from_numpy(adjustments)
