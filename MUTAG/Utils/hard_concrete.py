import torch
from torch.nn.parameter import Parameter
import numpy as np
from torch import sigmoid

class HardConcrete(torch.nn.Module):

    def __init__(self, beta=4 / 7, gamma=-0.2, zeta=1.0, fix_temp=True, loc_bias=5):
        super(HardConcrete, self).__init__()

        self.temp = beta if fix_temp else Parameter(torch.zeros(1).fill_(beta))
        self.gamma = gamma
        self.zeta = zeta
        self.gamma_zeta_ratio = np.math.log(-gamma / zeta)

        self.loc_bias = loc_bias

    def forward(self, input_element, summarize_penalty=True):
        input_element = input_element + self.loc_bias

        if self.training:
            u = torch.empty_like(input_element).uniform_(0+1e-6, 1.0-1e-6)
            s = sigmoid((torch.log(u) - torch.log(1 - u) + input_element) / self.temp)

            penalty = sigmoid(input_element - self.temp * self.gamma_zeta_ratio)
            penalty = penalty
        else:
            s = sigmoid(input_element)
            penalty = torch.zeros_like(input_element)

        if summarize_penalty:
            penalty = penalty.mean()

        s = s * (self.zeta - self.gamma) + self.gamma

        clipped_s = s.clamp(0, 1)
        hard_concrete = (clipped_s > 0.5).float()
        clipped_s = clipped_s + (hard_concrete - clipped_s).detach()

        return clipped_s, penalty

