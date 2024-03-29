import re
import torch
import torch.nn as nn
import numpy as np

from torch.distributions.dirichlet import Dirichlet
from python_speech_features import mfcc
from torch.autograd import Variable

from graphviz import Digraph

from model_params import hidden_dim_bidlstm                 # k value in our notes
from model_params import num_channels                       # C value in our notes
from model_params import num_chunks                         # T value in our notes
from model_params import chunk_size                         # lambda/T value in our notes

from model_params import hidden_dim_unilstm                 # l value in our notes
from model_params import mfcc_chunk_size
from model_params import parameter_matrix_dim                           # r value in our notes

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def attention_across_track(H, h, B_1, b_t):
    '''The multiplication of the parameter matrices H and the
       hidden representation of the ith channel'''

    X = torch.matmul(H.view(num_channels, -1, parameter_matrix_dim, hidden_dim_bidlstm),
                     h.view(num_channels, num_chunks, hidden_dim_bidlstm,1))

    # The multiplication of the Parameter matrices B and the mixing vector that is outputted form the uni-RNN
    Y = torch.matmul(B_1, b_t)
    Y = Y.repeat(num_chunks, 1).view(num_chunks, 1, parameter_matrix_dim)

    # Calculate attention weights
    Lambda = torch.matmul(Y.view(-1, num_chunks, 1, parameter_matrix_dim), X)
    soft_max = nn.Softmax(dim=1)
    alpha = soft_max(Lambda.view(num_channels, num_chunks, 1))

    # Calculating the context vector for the attention over the channels
    h_alpha = h*alpha.view(num_channels, num_chunks, 1, 1).repeat(1, 1, 1, hidden_dim_bidlstm)
    h_alpha_context_vec = torch.sum(h_alpha, dim=1)
    return h_alpha_context_vec, alpha


def attention_across_channels(B_2, b_t, F, h_alpha):
    X = torch.matmul(F, h_alpha.reshape(num_channels, hidden_dim_bidlstm, 1))
    Y = torch.matmul(B_2, b_t)

    Lambda = torch.matmul(Y.view(-1, 1, parameter_matrix_dim), X).view(num_channels)
    soft_max = nn.Softmax(dim=0)
    beta_t1 = soft_max(Lambda)
    return beta_t1.to(device)


def sample_dirchlet(beta_t1):
    dirichlet_distribution = Dirichlet(beta_t1)
    sample = dirichlet_distribution.rsample(sample_shape=(chunk_size, ))
    sample = sample.view(num_channels, chunk_size)
    return torch.tensor(sample, dtype=torch.float).to(device)

def apply_scaling_factors(scaling_factor,raw_tracks,time_step):
    raw_tracks[:, time_step, :] *= scaling_factor
    raw_tracks_at_t = raw_tracks[:, time_step, :]
    mixed_raw_tracks = raw_tracks
    return mixed_raw_tracks, raw_tracks_at_t


def get_mixed_mfcc_at_t(raw_tracks, time_step_value):
    with torch.no_grad():
        blended_track_at_t = torch.sum(raw_tracks[:, time_step_value, :], dim=0)

    mfcc_features_blended_song = mfcc(blended_track_at_t.cpu().numpy())
    mfcc_features_blended_song = np.sum(mfcc_features_blended_song, axis=0)
    mfcc_features_blended_song = torch.tensor([mfcc_features_blended_song], dtype=torch.float).to(device)

    return mfcc_features_blended_song


def get_original_mfcc_at_t(original_track, time_step_value):
    with torch.no_grad():
        mfcc_features_original_song = mfcc(original_track[time_step_value])
        mfcc_features_original_song = np.sum(mfcc_features_original_song, axis=0)
        mfcc_features_original_song = torch.tensor([mfcc_features_original_song], requires_grad=False, dtype=torch.float).to(device)

    return mfcc_features_original_song


if __name__ == '__main__':
    '''Setting the parameters value'''
    # dimensions = {'c':50,'T':10,'r':100,'k':200,'l':300}

    # H = torch.randn(num_channels, parameter_matrix_dim, hidden_dim_bidlstm)
    # B_1 = torch.randn(parameter_matrix_dim, hidden_dim_unilstm)
    # b_t = torch.randn(hidden_dim_unilstm, 1)
    # h = torch.randn(num_channels, num_chunks, 1, hidden_dim_bidlstm)
    # # print h.view(num_channels,num_chunks,hidden_dim_bidlstm,1)
    # F = torch.randn(num_channels, parameter_matrix_dim, num_chunks)
    # B_2 = torch.randn(parameter_matrix_dim, hidden_dim_unilstm)
    #
    # #The attention over the channels
    # alpha = attention_across_track(H,h,B_1,b_t)
    #
    # #The attentions over the tracks
    # beta = attention_across_channels(B_2, b_t,F,alpha)
    # print (beta)

    raw_tracks = torch.tensor(np.random.randn(num_channels, num_chunks, chunk_size)).to(device)
    original_track = torch.rand(num_chunks, chunk_size).to(device)
    print(get_mixed_mfcc_at_t(raw_tracks, time_step_value=1).shape)
    print(get_mixed_mfcc_at_t(raw_tracks, time_step_value=1).view(1, 1, -1).shape)
    print(get_original_mfcc_at_t(original_track, time_step_value=1).shape)
