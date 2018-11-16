import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from model import (MFCCUniLSTM, RawTrackBiLSTM)
from model_tools import (attention_across_track, attention_across_channels,
                         sample_dirchlet, sample_scaling_factors_from_distribution, apply_scaling_factors,
                         get_mixed_mfcc_at_t)

from model_params import hidden_dim_bidlstm                             # k value in our notes
from model_params import num_channels                                   # C value in our notes
from model_params import num_chunks                                     # T value in our notes
from model_params import chunk_size                                     # lambda/T value in our notes

from model_params import hidden_dim_unilstm                             # l value in our notes
from model_params import mfcc_chunk_size
from model_params import parameter_matrix_dim                           # r value in our notes

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
        self.parameter_matrix_individual_tracks = torch.tensor(np.random.randn(parameter_matrix_dim, hidden_dim_unilstm), dtype=torch.float, requires_grad=True)
        # H_1 ... H_c               shape: (C x r x k)
        self.parameter_matrix_for_each_channel_1 = torch.tensor(np.random.randn(num_channels, parameter_matrix_dim, hidden_dim_bidlstm), dtype=torch.float, requires_grad=True)

        # B2                        shape: (r x l)
        self.parameter_matrix_across_channels = torch.tensor(np.random.randn(parameter_matrix_dim, hidden_dim_unilstm), dtype=torch.float, requires_grad=True)
        # F_1 ... F_c               shape: (C x r x T)
        self.parameter_matrix_for_each_channel_2 = torch.tensor(np.random.randn(num_channels, parameter_matrix_dim, num_chunks), dtype=torch.float, requires_grad=True)

        # the models
        self.raw_track_bilstm_model = RawTrackBiLSTM(self.chunk_size, self.hidden_dim_bidlstm, self.num_channels)
        self.mfcc_unilstm_model = MFCCUniLSTM(self.chunk_size, self.hidden_dim_bidlstm)

        self.beta_t = None

    def initialize_dirchlet_parameters(self):
        #TODO Shayaan
        # to be done every time a new song is loaded into the network for training
        pass

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

        alpha = attention_across_track(self.parameter_matrix_for_each_channel_1, num_channels_bidlstm_hidden, self.parameter_matrix_individual_tracks, blended_at_t)
        beta_t1 = attention_across_channels(self.parameter_matrix_across_channels, blended_at_t, self.parameter_matrix_for_each_channel_2, alpha)

        scaling_factor_distibution = sample_dirchlet(beta_t1)
        scaling_factors_for_all_channels = sample_scaling_factors_from_distribution(scaling_factor_distibution)

        mixed_raw_tracks, next_step_mixed_raw = apply_scaling_factors(scaling_factors_for_all_channels, raw_tracks, time_step_value + 1)

        self.calculate_loss(original_mfcc_at_t, mixed_mfcc_at_t, self.beta_t, beta_t1)
        return mixed_raw_tracks

    def calculate_loss(self, original_mfcc_at_t, mixed_mfcc_at_t, beta_t, beta_t1):
        #TODO Shayaan
        pass


if __name__ == "__main__":
    network = PolicyNetwork(num_channels, num_chunks, chunk_size, hidden_dim_bidlstm, mfcc_chunk_size, hidden_dim_unilstm, parameter_matrix_dim)