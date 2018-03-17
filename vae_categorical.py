from __future__ import division

import torch
import argparse
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import torch.optim as optim
from torch import nn

class Encoder(torch.nn.Module):
    def __init__(self, D_in, D_out, D_layers=[]):
        """
        Currently it is assumed that number of hidden layers = 2
        """
        super(Encoder, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, D_layers[0])
        self.linear2 = torch.nn.Linear(D_layers[0], D_layers[1])
        self.linear3 = torch.nn.Linear(D_layers[1], D_out)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        return self.linear3(x)

class Decoder(torch.nn.Module):
    def __init__(self, D_in, D_out, D_layers=[]):
        """
        Currently it is assumed that number of hidden layers = 2
        """
        super(Decoder, self).__init__()
        self.linear1 = torch.nn.Linear(D_out, D_layers[1])
        self.linear2 = torch.nn.Linear(D_layers[1], D_layers[0])
        self.linear3 = torch.nn.Linear(D_layers[0], D_in)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        return self.linear3(x)

class VAE(torch.nn.Module):

    def __init__(self, K, N, temperature, input_dim, hidden_layers, iscuda):
        """
        Categorical Variational Autoencoder
        K: Number of Cateories or Classes
        N: Number of Categorical distributions 
        N x K: Dimension of latent variable
        hidden_layers: A list containing number of nodes in each hidden layers
                        of both encoder and decoder
        """
        super(VAE, self).__init__()
        self.encoder = Encoder(input_dim, K*N, hidden_layers)
        self.decoder = Decoder(input_dim, K*N, hidden_layers[::-1])
        self.K=K
        self.N=N
        self.temperature = temperature
        self.iscuda = iscuda

    def _sample_latent(self, tou):
        """
        Return the latent normal sample y ~ gumbel_softmax(x)
        tou = temperature Variable to be learnt
        """
        eps = 1e-20
        
        # generates a h_enc.size() shaped batch of reparameterized 
        # Gumbel samples with location = 0, sclae = 1
        U = torch.from_numpy(np.random.uniform(0, 1, size=self.hidden.size())).float()

        # for doing operation between Variable and Tensor, a tensor has to be wrapped
        # insider Variable. However, set requires_grad as False so that back propagation doesn't
        # pass through it
        # gumbel sample is -log(-log(U))
        g = Variable(-torch.log(-torch.log(U + eps) + eps), requires_grad=False)
        if self.iscuda:
            g = g.cuda()

        # Gumbel-Softmax samples are - softmax((probs + gumbel(0,1).sample)/temperature)
        y = self.hidden + g
        softmax = torch.nn.Softmax(dim=-1) # -1 indicates the last dimension

        return softmax(y/1.0) #keep the temperature fixed at 1.0

    def forward(self, x):
        """
        Forward computation Graph
        x = inputs
        """
        
        # dynamic binarization of input
        t = Variable(torch.rand(x.size()), requires_grad=False)
        if self.iscuda:
            t = t.cuda()

        net = t < x
    
        h_enc = self.encoder(net.float())
        tou = Variable(torch.from_numpy(np.array([self.temperature])), requires_grad=False)
        if self.iscuda:
            tou = tou.cuda()

        self.hidden = h_enc.view(-1, self.N, self.K)
        bsize = self.hidden.size()[0]
        self.latent = self._sample_latent(tou)
        x_hat = self.decoder(self.latent.view(bsize,-1))
        return x_hat

    def loss_fn(self, x, x_hat):
        """
        Total Loss = Reconstruction Loss + KL Divergence
        x = input to forward()
        x_hat = output of forward()
        Reconstruction Loss = binary cross entropy between inputs and outputs
        KL Divergence = KL Divergence between gumbel softmax distributions with 
                        self.hidden and uniform log-odds
        """
        eps = 1e-20 # to avoid log of 0

        # Reconstruction Loss
        softmax = torch.nn.Softmax(dim=-1)
        x_prob = softmax(x_hat)
        recons_loss = torch.sum(x * torch.log(x_prob + eps), dim=1)

        # KL Divergence = entropy (self.latent) - cross_entropy(self.latent, uniform log-odds)
        q_y = softmax(self.hidden) # convert hidden layer values to probabilities
        kl1 = q_y * torch.log(q_y + eps) # entropy (self.latent)
        kl2 = q_y * np.log((1.0/self.K) + eps)

        KL_divergence = torch.sum(torch.sum(kl1 - kl2, 2),1)
        
        # total loss = reconstruction loss + KL Divergence
        loss = -torch.mean(recons_loss - KL_divergence)
        self.recons_loss = -torch.mean(recons_loss).data[0] # for visualization purposes
        self.kl_loss = -torch.mean(-KL_divergence).data[0]  # for visualization purposes
        return loss 
