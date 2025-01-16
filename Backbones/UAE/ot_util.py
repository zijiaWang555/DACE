import torch
import numpy as np
import random
import time
import scipy
import torch.nn as nn
from torch.autograd import grad     

class OT(nn.Module):
    def __init__(self, device):
        super(OT, self).__init__()
        self.device = device
        self.eps = 0.05
        self.max_iter = 10

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i)
                          for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def cost_matrix(self, x, y, p=2):
        x_col = x.unsqueeze(1)
        y_lin = y.unsqueeze(0)
        c = torch.sum((x_col - y_lin) ** p, dim=2)
        return c

    def M(self, C, u, v):
        return (-C + u.unsqueeze(1) + v.unsqueeze(0)) / self.eps

    def sinkhorn(self, x, y):
        C = self.cost_matrix(x, y)
        x_norm = (x ** 2).sum(dim=1, keepdims=True) ** 0.5
        y_norm = (y ** 2).sum(dim=1, keepdims=True) ** 0.5
        mu = (x_norm[:, 0] / x_norm.sum()).detach().to(self.device)
        nu = (y_norm[:, 0] / y_norm.sum()).detach().to(self.device)
        u = torch.zeros_like(mu)
        v = torch.zeros_like(nu)

        actual_nits = 0
        thresh = 0.1

        for i in range(self.max_iter):
            u1 = u
            u = self.eps * (torch.log(mu + 1e-8) - torch.logsumexp(self.M(C, u, v), dim=-1)) + u
            v = self.eps * (torch.log(nu + 1e-8) - torch.logsumexp(self.M(C, u, v).t(), dim=-1)) + v
            err = (u - u1).abs().sum()

            actual_nits += 1
            if err.item() < thresh:
                break

        U, V = u, v
        pi = torch.exp(self.M(C, U, V))
        cost = torch.sum(pi * C)
        return cost, pi, C

    def forward(self, source, target):
        Cs, Ct = (source.unsqueeze(1) - source.unsqueeze(0)) ** 2, (target.unsqueeze(1) - target.unsqueeze(0)) ** 2
        loss = torch.norm(Cs.sum(2) - Ct.sum(2), p=2) / (source.shape[0] * source.shape[0])

        return loss
