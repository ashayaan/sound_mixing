import torch
import torch.nn as  nn
import numpy


def attention_across_track(H, h, B_1, b_t, dimensions):
    '''The multiplication of the parameter matrices H and the
       hidden representation of the ith channel'''

    X = torch.matmul(H.view(dimensions['c'], -1, dimensions['r'], dimensions['k']),
                        h.view(dimensions['c'], dimensions['T'], dimensions['k'],1))

    ''' The multiplication of the Parameter matrices B and the
        the mixing vector that is outputted form the uni-RNN'''

    Y = torch.matmul(B_1, b_t)
    Y = Y.repeat(dimensions['T'], 1).view(dimensions['T'], 1, dimensions['r'])

    '''Calculating the context vector for the attention 
       over the channels'''

    Lambda = torch.matmul(Y.view(-1, dimensions['T'], 1, dimensions['r']), X)

    soft_max = nn.Softmax(dim=1)
    alpha = soft_max(Lambda.view(dimensions['c'], dimensions['T'], 1))

    return alpha


def attention_across_channels(B_2, b_t, F, alpha, dimensions):
    X = torch.matmul(F, alpha)

    Y = torch.matmul(B_2, b_t)

    Lambda = torch.matmul(Y.view(-1, 1, dimensions['r']), X).view(dimensions['c'])

    soft_max = nn.Softmax(dim=0)

    print Lambda
    beta = soft_max(Lambda)

    return beta


def sample_dirchlet():
    # TODO: Shayaan
    pass


def sample_scaling_factors_from_distribution():
    # TODO: Shayaan
    pass


def apply_scaling_factors():
    # TODO: Shayaan
    pass


def get_mixed_mfcc_at_t():
    # TODO: Shayaan
    pass


if __name__ == '__main__':
    '''Setting the parameters value'''
    dimensions = {'c':50,'T':10,'r':100,'k':200,'l':300}

    H = torch.randn(dimensions['c'], dimensions['r'], dimensions['k'])
    B_1 = torch.randn(dimensions['r'], dimensions['l'])
    b_t = torch.randn(dimensions['l'], 1)
    h = torch.randn(dimensions['c'], dimensions['T'], 1, dimensions['k'])
    F = torch.randn(dimensions['c'], dimensions['r'], dimensions['T'])
    B_2 = torch.randn(dimensions['r'], dimensions['l'])

    #The attention over the channels
    alpha = attention_across_track(H, B_1, b_t, h, dimensions)

    #The attentions over the tracks
    beta = attention_across_channels(B_2, b_t, alpha, F, dimensions)
    print beta