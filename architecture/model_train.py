import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from torch.distributions.kl import kl_divergence
import numpy as np

import itertools
import argparse

from model import (MFCCUniLSTM, RawTrackBiLSTM, instantiate_all_channels_bilstms, forward_pass_all_channel_bilstms)
from model_tools import (attention_across_track, attention_across_channels,
                         sample_dirchlet, apply_scaling_factors,
                         get_original_mfcc_at_t, get_mixed_mfcc_at_t)

from pytorch_viz import make_dot
from tqdm import tqdm

from model_params import hidden_dim_bidlstm                             # k value in our notes
from model_params import num_channels                                   # C value in our notes
from model_params import num_chunks                                     # T value in our notes
from model_params import chunk_size                                     # lambda/T value in our notes

from model_params import hidden_dim_unilstm                             # l value in our notes
from model_params import mfcc_chunk_size
from model_params import parameter_matrix_dim                           # r value in our notes
from model_params import delta                                          # parameter for the loss function
from model_params import epoch


import sys
sys.path.insert(0,'../tools/')
from db_handle import get_chunked_songs

torch.manual_seed(1)

# Checking if the GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PolicyNetwork(nn.Module):
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
        super(PolicyNetwork, self).__init__()

        # network parameters
        self.num_channels = num_channels
        self.num_chunks = num_chunks
        self.chunk_size = chunk_size
        self.hidden_dim_bidlstm = hidden_dim_bidlstm
        self.mfcc_chunk_size = mfcc_chunk_size
        self.hidden_dim_unilstm = hidden_dim_unilstm
        self.parameter_matrix_dim = parameter_matrix_dim

        # attention matrices across tracks
        # B1                        shape: (r x l)
        self.parameter_matrix_individual_tracks = nn.Parameter(torch.randn(parameter_matrix_dim, hidden_dim_unilstm, dtype=torch.float, requires_grad=True,device=device))
        # H_1 ... H_c               shape: (C x r x k)
        self.parameter_matrix_for_each_channel_1 = nn.Parameter(torch.randn(num_channels, parameter_matrix_dim, hidden_dim_bidlstm, dtype=torch.float, requires_grad=True,device=device))

        # attention matrices acriss channels
        # B2                        shape: (r x l)
        self.parameter_matrix_across_channels = nn.Parameter(torch.randn(parameter_matrix_dim, hidden_dim_unilstm, dtype=torch.float, requires_grad=True,device=device))
        # F_1 ... F_c               shape: (C x r x k)
        self.parameter_matrix_for_each_channel_2 = nn.Parameter(torch.randn(num_channels, parameter_matrix_dim, hidden_dim_bidlstm, dtype=torch.float, requires_grad=True,device=device))

        # the models
        self.raw_track_channels_bilstms = instantiate_all_channels_bilstms(self.chunk_size, self.hidden_dim_bidlstm, self.num_channels)
        self.mfcc_unilstm_model = MFCCUniLSTM(self.mfcc_chunk_size, self.hidden_dim_unilstm)

        # initialize beta_t - dirichlet parameters at time t=0 for every input song
        self.beta_list = []
        self.mixed_mfcc_till_t = []

        model_parameters = []
        for model in self.raw_track_channels_bilstms:
            model_parameters.extend(list(model.parameters()))

        model_parameters.extend(list(self.mfcc_unilstm_model.parameters()))

        parameter_matrices = [self.parameter_matrix_individual_tracks,self.parameter_matrix_for_each_channel_1,self.parameter_matrix_across_channels,self.parameter_matrix_for_each_channel_2]
        model_parameters.extend(parameter_matrices)

        self.optimizer = optim.Adam(model_parameters)
        # self.optimizer.add_param_group({"params": [self.parameter_matrix_individual_tracks,
        #                                                           self.parameter_matrix_for_each_channel_1,
        #                                                           self.parameter_matrix_across_channels,
        #                                                           self.parameter_matrix_for_each_channel_2]})

    def initialize_dirchlet_parameters(self):
        self.beta_list.append(torch.rand(num_channels,).to(device))

    def forward(self, raw_tracks, original_mfcc_at_t, time_step_value):
        """
        :param raw_tracks:
        :param original_mfcc_at_t:
        :param time_step_value:
        :return:
        """

        all_channels_bidlstm_hidden = forward_pass_all_channel_bilstms(raw_tracks, self.raw_track_channels_bilstms)

        mixed_mfcc_at_t = get_mixed_mfcc_at_t(raw_tracks, time_step_value)
        blended_at_t = self.mfcc_unilstm_model(mixed_mfcc_at_t)

        h_alpha, alpha = attention_across_track(self.parameter_matrix_for_each_channel_1, all_channels_bidlstm_hidden, self.parameter_matrix_individual_tracks, blended_at_t)
        beta_t1 = attention_across_channels(self.parameter_matrix_across_channels, blended_at_t, self.parameter_matrix_for_each_channel_2, h_alpha)

        scaling_factor_distibution = sample_dirchlet(beta_t1)
        if time_step_value + 1 != self.num_chunks:
            mixed_raw_tracks, next_step_mixed_raw = apply_scaling_factors(scaling_factor_distibution, raw_tracks, time_step_value+1)
        else:
            mixed_raw_tracks = raw_tracks

        beta_t = self.beta_list[-1]
        loss = self.calculate_loss(original_mfcc_at_t, mixed_mfcc_at_t, beta_t, beta_t1)

        # setting beta_t1 of previous step (for next forward pass) as beta_t
        with torch.no_grad():
            self.beta_list.append(beta_t1.clone())

        return mixed_raw_tracks, loss

    def calculate_loss(self, original_mfcc_at_t, mixed_mfcc_at_t, beta_t, beta_t1):
        difference_mfcc_features = original_mfcc_at_t - mixed_mfcc_at_t
        difference_mfcc_features_norm = torch.matmul(difference_mfcc_features, difference_mfcc_features.view(mfcc_chunk_size, 1))

        kl_divergence_value = F.kl_div(beta_t, beta_t1)
        loss = torch.clamp(torch.exp(difference_mfcc_features_norm + delta*kl_divergence_value), max=1)
        return loss


def repackage_hidden(h):
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def process_one_song(network, raw_tracks, original_song):
    total_loss = 0
    for time_step_value in tqdm(range(num_chunks)):

        # zero the gradients and detach hidden states
        for channel_bilstm in network.raw_track_channels_bilstms:
            channel_bilstm.zero_grad()
            channel_bilstm.hidden = repackage_hidden(channel_bilstm.hidden)
            channel_bilstm.hidden = channel_bilstm.init_hidden()

        network.mfcc_unilstm_model.zero_grad()
        network.mfcc_unilstm_model.hidden = repackage_hidden(network.mfcc_unilstm_model.hidden)
        network.mfcc_unilstm_model.hidden = network.mfcc_unilstm_model.init_hidden()

        # get the ground truth original song mixing mfcc
        original_mfcc_at_t = get_original_mfcc_at_t(original_song, time_step_value)

        # forward pass across the network
        raw_tracks, loss = network.forward(raw_tracks, original_mfcc_at_t, time_step_value)

        # view the computation graph
        # graph = make_dot(loss)
        # graph.view()

        network.optimizer.zero_grad()

        total_loss += loss.item()

        # https://discuss.pytorch.org/t/solved-training-a-simple-rnn/9055/8
        loss.backward(retain_graph=True)

        network.optimizer.step()

    print('LOSS: {}'.format(total_loss))

    return network, raw_tracks, original_song


def train_network(network, path_to_songs):
    song_counter = 1
    for (raw_tracks, original_song) in get_chunked_songs(path_to_songs):
        network.initialize_dirchlet_parameters()
        raw_tracks = raw_tracks.to(device)
        print("SONG NUMBER: {}".format(str(song_counter)))
        network, raw_tracks_processed, original_song_processed = process_one_song(network, raw_tracks, original_song)
        song_counter += 1
        print('-------------------------------------------------------------------------------------------------------')

    return network


if __name__ == "__main__":
    print  "Device in use: " + str(device)
    print  "\n"
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", type=str, default="./", help="path to the dataset")
    args = parser.parse_args()

    network = PolicyNetwork(num_channels, num_chunks, chunk_size, hidden_dim_bidlstm, mfcc_chunk_size, hidden_dim_unilstm, parameter_matrix_dim)
    songs = []

    # DUMMY DATA
    # print('LOADING DATASET')
    # for i in tqdm(range(10)):
    #     raw_tracks = torch.tensor(np.random.rand(num_channels, num_chunks, chunk_size), dtype=torch.float, requires_grad=False)
    #     original_song = torch.tensor(np.random.rand(num_chunks, chunk_size), dtype=torch.float, requires_grad=False)
    #     songs.append((raw_tracks, original_song))

    for epoch in range(epoch):
        print('=======================================================================================================')
        print('EPOCH : {}'.format(epoch + 1))
        print('=======================================================================================================')

        network = train_network(network, path_to_songs=args.datapath)