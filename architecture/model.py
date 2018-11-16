import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from model_params import hidden_dim_bidlstm                 # k value in our notes
from model_params import num_channels                       # C value in our notes
from model_params import num_chunks                         # T value in our notes
from model_params import chunk_size                         # lambda/T value in our notes

from model_params import hidden_dim_unilstm                 # l value in our notes
from model_params import mfcc_chunk_size

torch.manual_seed(1)


class MFCCUniLSTM(nn.Module):
    def __init__(self, mfcc_chunk_size, hidden_dim_unilstm):
        """
        :param mfcc_chunk_size:
        :param hidden_dim_unilstm:
        """
        super(MFCCUniLSTM, self).__init__()

        self.hidden_dim = hidden_dim_unilstm
        self.lstm = nn.LSTM(mfcc_chunk_size, hidden_dim_unilstm)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.zeros(1, 1, self.hidden_dim),
                torch.zeros(1, 1, self.hidden_dim))

    def forward(self, mixed_mfcc_at_t):
        #TODO: Check features, update dimensions of lstm pass (view)
        lstm_out, self.hidden = self.lstm(mixed_mfcc_at_t.view(len(mixed_mfcc_at_t), 1, -1), self.hidden)
        blended_till_t = lstm_out[-1]
        return blended_till_t.view(-1, 1)


class RawTrackBiLSTM(nn.Module):
    def __init__(self, chunk_size, hidden_dim, num_channels):
        """
        :param chunk_size:
        :param hidden_dim:
        :param num_channels:
        """
        super(RawTrackBiLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_channels = num_channels

        # instantiating the bi-directional lstm
        # creating a list for the num_channels x bi-directional lstm
        # initializing hiddenstates for the num_channels x bi-directional lstm
        self.channels_bilstms = []
        self.channels_bilstms_hidden = []

        for i in range(num_channels):
            self.channels_bilstms.append(nn.LSTM(chunk_size, hidden_dim // 2, num_layers=1, bidirectional=True))
            self.channels_bilstms_hidden.append(self.init_hidden())

    def init_hidden(self):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(2, 1, self.hidden_dim // 2),
                torch.zeros(2, 1, self.hidden_dim // 2))

    def forward(self, raw_tracks):
        """
        :param raw_tracks:
        :return:
        """
        all_channels_bidlstm_hidden = []

        for i in range(self.num_channels):
            current_single_channel_info = torch.tensor(raw_tracks[i], dtype=torch.float).view(num_chunks, 1, -1)
            current_single_channel_hidden = self.channels_bilstms_hidden[i]

            lstm_out, _ = self.channels_bilstms[i](current_single_channel_info, current_single_channel_hidden)
            all_channels_bidlstm_hidden.append(lstm_out)

        all_channels_bidlstm_hidden = torch.stack(all_channels_bidlstm_hidden)
        return all_channels_bidlstm_hidden


if __name__ == "__main__":
    import numpy as np
    bilstms_model = RawTrackBiLSTM(chunk_size, hidden_dim_bidlstm, num_channels)

    # each song data input - raw_tracks will have info for all channels and split inro num_chunks
    raw_tracks = np.random.randn(num_channels, num_chunks, chunk_size)

    bilstms_model.zero_grad()

    num_channels_bidlstm_hidden = bilstms_model(raw_tracks)
    print(num_channels_bidlstm_hidden.shape)

    unilstms_model = MFCCUniLSTM(mfcc_chunk_size, hidden_dim_unilstm)
