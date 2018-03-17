# categorical-vae
Implement Categorical Variational autoencoder using Pytorch

Reproducing the results of https://arxiv.org/pdf/1611.01144.pdf in Pytorch framework. 
The original implementation by authors use Tensorflow. In Pytorch there is no readily available 
Gumbel Softmax distribution to sample from, so have to implement the Relaxed Categorical representation to sample the latent representation.
