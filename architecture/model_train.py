import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from model import (MFCCUniLSTM, RawTrackBiLSTM)
from model_tools import (attention_across_track, attention_across_channels,
                         sample_dirchlet, apply_scaling_factors,
                         get_original_mfcc_at_t, get_mixed_mfcc_at_t)

from model_params import hidden_dim_bidlstm                             # k value in our notes
from model_params import num_channels                                   # C value in our notes
from model_params import num_chunks                                     # T value in our notes
from model_params import chunk_size                                     # lambda/T value in our notes

from model_params import hidden_dim_unilstm                             # l value in our notes
from model_params import mfcc_chunk_size
from model_params import parameter_matrix_dim                           # r value in our notes

from model_params import delta                                          #parameter for the loss funciton

torch.manual_seed(1)


class PolicyNetwork(object):
    def __init__(self, num_channels, num_chunks, chunk_size, hidden_dim_bidlstm, mfcc_chunk_size, hidden_dim_unilstm, parameter_matrix_dim):
        """
        :param num_channels:
        :param num_chunks:
        :param chunk_size:
        :param hidden_dim_bidlstm:
        :param mfcc_chunk_size:
        :param hidden_dim_unilstm:
        :param parameter_matrix_dim:
        """
        # network parameters
        self.num_channels = num_channels
        self.num_chunks = num_chunks
        self.chunk_size = chunk_size
        self.hidden_dim_bidlstm = hidden_dim_bidlstm
        self.mfcc_chunk_size = mfcc_chunk_size
        self.hidden_dim_unilstm = hidden_dim_unilstm
        self.parameter_matrix_dim = parameter_matrix_dim

        # B1                        shape: (r x l)
        self.parameter_matrix_individual_tracks = torch.randn(parameter_matrix_dim, hidden_dim_unilstm, dtype=torch.float, requires_grad=True)
        # H_1 ... H_c               shape: (C x r x k)
        self.parameter_matrix_for_each_channel_1 = torch.randn(num_channels, parameter_matrix_dim, hidden_dim_bidlstm, dtype=torch.float, requires_grad=True)

        # B2                        shape: (r x l)
        self.parameter_matrix_across_channels = torch.randn(parameter_matrix_dim, hidden_dim_unilstm, dtype=torch.float, requires_grad=True)
        # F_1 ... F_c               shape: (C x r x T)
        self.parameter_matrix_for_each_channel_2 = torch.randn(num_channels, parameter_matrix_dim, num_chunks, dtype=torch.float, requires_grad=True)

        # the models
        self.raw_track_bilstm_model = RawTrackBiLSTM(self.chunk_size, self.hidden_dim_bidlstm, self.num_channels)
        self.mfcc_unilstm_model = MFCCUniLSTM(self.chunk_size, self.hidden_dim_bidlstm)

        self.beta_t = None

        # TODO
        self.optimizer = optim.SGD()

    def initialize_dirchlet_parameters(self):
        self.beta_t = torch.rand(num_channels, 1)


    def forward(self, raw_tracks, original_mfcc_at_t, time_step_value):
        """
        :param raw_tracks:
        :param original_mfcc_at_t:
        :param time_step_value:
        :return:
        """
        num_channels_bidlstm_hidden = self.raw_track_bilstm_model(raw_tracks)

        mixed_mfcc_at_t = get_mixed_mfcc_at_t(raw_tracks, time_step_value)
        blended_at_t = self.mfcc_unilstm_model(mixed_mfcc_at_t)

        h_alpha, alpha = attention_across_track(self.parameter_matrix_for_each_channel_1, num_channels_bidlstm_hidden, self.parameter_matrix_individual_tracks, blended_at_t)
        beta_t1 = attention_across_channels(self.parameter_matrix_across_channels, blended_at_t, self.parameter_matrix_for_each_channel_2, alpha)

        scaling_factor_distibution = sample_dirchlet(beta_t1)
        mixed_raw_tracks, next_step_mixed_raw = apply_scaling_factors(scaling_factor_distibution, raw_tracks, time_step_value+1)

        loss = self.calculate_loss(original_mfcc_at_t, mixed_mfcc_at_t, self.beta_t, beta_t1)

        # setting beta_t1 of previous step (for next forward pass) as beta_t
        self.beta_t = beta_t1

        return mixed_raw_tracks, loss


    def calculate_loss(self, original_mfcc_at_t, mixed_mfcc_at_t, beta_t, beta_t1):
        def kl_divergence(beta_t, beta_t1):
            return np.sum(np.where(beta_t != 0, beta_t * np.log(beta_t / beta_t1), 0))

        difference_mfcc_features = original_mfcc_at_t - mixed_mfcc_at_t
        difference_mfcc_features_norm = difference_mfcc_features.dot(difference_mfcc_features)
        kl_divergence_value = kl_divergence(beta_t,beta_t1)

        return np.exp((difference_mfcc_features_norm + delta*kl_divergence_value))


def train_network(network, optimizer, songs):
    for (raw_tracks, original_song) in songs:

        network.initialize_dirchlet_parameters()
        network.optimizer.zero_grad()

        for time_step_value in range(num_chunks):
            original_mfcc_at_t = get_original_mfcc_at_t(original_song, time_step_value)
            raw_tracks, loss = network.forward(original_song, original_mfcc_at_t, time_step_value)
            loss.backward()
            optimizer.step()


if __name__ == "__main__":
    network = PolicyNetwork(num_channels, num_chunks, chunk_size, hidden_dim_bidlstm, mfcc_chunk_size, hidden_dim_unilstm, parameter_matrix_dim)
    songs = []
    for i in range(10):
        raw_tracks = torch.tensor(np.random.randn(num_channels, num_chunks, chunk_size))
        original_song = torch.tensor(np.zeros(num_chunks, chunk_size))
        songs.append((raw_tracks, original_song))
    train_network(network, songs)