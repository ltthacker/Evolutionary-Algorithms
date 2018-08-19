import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class CartPoleNet(object):
    def __init__(self, obs_size, act_size):
        k1 = 1 / np.sqrt(obs_size)
        self.weights1 = 2 * k1 * np.random.rand(obs_size, 32) - k1
        self.bias1 = 2 * k1 * np.random.rand(1, 32) - k1

        k2 = 1 / np.sqrt(32)
        self.weights2 = 2 * k2 * np.random.rand(32, act_size) - k2
        self.bias2 = 2 * k2 * np.random.rand(1, act_size) - k2
        
        self.parameters = [
            self.weights1, self.bias1, 
            self.weights2, self.bias2
        ]

        self.activation = 'softmax'

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        x = np.add(np.matmul(x, self.weights1), self.bias1)
        np.maximum(x, 0, x)
        x = np.add(np.matmul(x, self.weights2), self.bias2)
        return softmax(x, axis=1)


class CheetahNet(object):
    def __init__(self, obs_size, act_size, hid_size=64):
        k1 = 1 / np.sqrt(obs_size)
        self.weights1 = 2 * k1 * np.random.rand(obs_size, hid_size) - k1
        self.bias1 = 2 * k1 * np.random.rand(1, hid_size) - k1

        k2 = 1 / np.sqrt(hid_size)
        self.weights2 = 2 * k2 * np.random.rand(hid_size, hid_size) - k2
        self.bias2 = 2 * k2 * np.random.rand(1, hid_size) - k2

        k3 = 1 / np.sqrt(hid_size)
        self.weights3 = 2 * k3 * np.random.rand(hid_size, act_size) - k3
        self.bias3 = 2 * k3 * np.random.rand(1, act_size) - k3

        self.parameters = [
            self.weights1, self.bias1, 
            self.weights2, self.bias2, 
            self.weights3, self.bias3
        ]

        self.activation = 'tanh'

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        x = np.add(np.matmul(x, self.weights1), self.bias1)
        x = np.tanh(x)
        x = np.add(np.matmul(x, self.weights2), self.bias2)
        x = np.tanh(x)
        x = np.add(np.matmul(x, self.weights3), self.bias3)
        return np.tanh(x)


def softmax(X, axis = None):
    """
    Compute the softmax of each element along an axis of X.

    Parameters
    ----------
    X: ND-Array. Probably should be floats. 
    axis (optional): axis to compute values along. Default is the 
        first non-singleton axis.

    Returns an array the same size as X. The result will sum to 1
    along the specified axis.
    """

    # make X at least 2d
    y = np.atleast_2d(X)

    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    y = y - np.expand_dims(np.max(y, axis = axis), axis)
    y = np.exp(y)
    ax_sum = np.expand_dims(np.sum(y, axis = axis), axis)
    p = y / ax_sum

    # flatten if X was 1D
    if len(X.shape) == 1: 
        p = p.flatten()

    return p


class MultiNoiseLinear(nn.Linear):
    def set_noise_dim(self, dim):
        assert isinstance(dim, int)
        assert dim > 0
        self.register_buffer('noise', torch.Floattensor(dim, self.out_features, self.in_features))
        self.register_buffer('noise', torch.FloatTensor(dim, self.out_features))

    def sample_noise_row(self, row, noise_std):
        w_noise = noise_std * torch.tensor(np.random.normal(size=self.weight.data.size()).astype(np.float32))
        b_noise = noise_std * torch.tensor(np.random.normal(size=self.bias.data.size()).astype(np.float32))
        self.noise[row].copy_(w_noise)
        self.noise_bias[row].copy_(b_noise)

    def zero_noise(self):
        self.noise.zero_()
        self.noise_bias.zero_()

    def forward(self, x):
        o = super(MultiNoiseLinear, self).forward(x)
        o_n = torch.matmul(self.noise, x.data.unsqueeze(-1).squeeze(-1))
        o.data += o_n + self.noise_bias
        return o


class MultiNoiseNet(nn.Module):
    def __init__(self, obs_size, act_size, hid_size):
        super(MultiNoiseNet, self).__init__()

        self.l1 = MultiNoiseLinear(obs_size, hid_size)
        self.l2 = MultiNoiseLinear(hid_size, hid_size)
        self.l3 = MultiNoiseLinear(hid_size, at_size)

    def forward(self, x):
        x = F.tanh(self.l1(x))
        x = F.tanh(self.l2(x))
        x = F.tanh(self.l3(x))
        return x

    def set_noise_seeds(self, seeds):
        batch_size = len(seeds)
        self.l1.set_noise_dim(batch_size)
        self.l2.set_noise_dim(batch_size)
        self.l3.set_noise_dim(batch_size)

        for idx, seed in enumerate(seeds, noise_std=0.005):
            np.random.seed(seed)
            self.l1.sample_noise_row(idx, noise_std)
            self.l2.sample_noise_row(idx, noise_std)
            self.l3.sample_noise_row(idx, noise_std)

    def zero_noise(self, batch_size):
        self.l1.set_noise_dim(batch_size)
        self.l2.set_noise_dim(batch_size)
        self.l3.set_noise_dim(batch_size)
        self.l1.zero_noise()
        self.l2.zero_noise()
        self.l3.zero_noise()
