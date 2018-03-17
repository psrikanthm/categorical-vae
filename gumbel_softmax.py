from __future__ import division

import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import torch.optim as optim
from torch import nn
import matplotlib.pyplot as plt


def gumbel_sample(pi, tou, K):
    eps = 1e-20
    log_pi = torch.log(pi)
    softmax = torch.nn.Softmax()

    U = torch.from_numpy(np.random.uniform(0, 1, size=log_pi.size())).float()

    # for doing operation between Variable and Tensor, a tensor has to be wrapped 
    # insider Variable. However, set requires_grad as False so that back propagation doesn't 
    # pass through it
    # gumbel sample is -log(-log(U))

    g = Variable(-torch.log(-torch.log(U + eps) + eps), requires_grad=False)
    y = (log_pi + g) / tou
    y = softmax((log_pi + g)/tou)
    y = y.view(-1, K)

    return y
