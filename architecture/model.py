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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        lstm_out, self.hidden = self.lstm(mixed_mfcc_at_t.view(len(mixed_mfcc_at_t), 1, -1), self.hidden)
        blended_at_t = lstm_out[-1]
        return blended_at_t.view(-1, 1)


class RawTrackBiLSTM(nn.Module):
    def __init__(self, chunk_size, hidden_dim, num_channels):
        """
        :param chunk_size:
        :param hidden_dim:
        :param num_channels:
        """
        super(RawTrackBiLSTM, self).__init__()

        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(chunk_size, hidden_dim // 2, num_layers=1, bidirectional=True)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(2, 1, self.hidden_dim // 2),
                torch.zeros(2, 1, self.hidden_dim // 2))

    def forward(self, current_channel_raw_track):
        """
        :param raw_tracks:
        :return:
        """

        lstm_out, self.hidden = self.lstm(current_channel_raw_track.view(num_chunks, 1, -1), self.hidden)
        return lstm_out


def instantiate_all_channels_bilstms(chunk_size, hidden_dim_bidlstm, num_channels):
    """
    instantiating the bi-directional lstm
    creating a list for the num_channels x bi-directional lstm
    :param chunk_size:
    :param hidden_dim_bidlstm:
    :param num_channels:
    :return:
    """

    channels_bilstms = []
    for i in range(num_channels):
        channels_bilstms.append(RawTrackBiLSTM(chunk_size, hidden_dim_bidlstm, num_channels))

    return channels_bilstms


def forward_pass_all_channel_bilstms(raw_tracks, channels_bilstms):
    """
    :param raw_tracks:
    :param channels_bilstms:
    :return:
    """
    all_channels_bidlstm_hidden = []
    for i in range(num_channels):
        current_single_channel_info = torch.tensor(raw_tracks[i], dtype=torch.float, requires_grad=False).view(num_chunks, 1, -1)

        channels_bilstms[i].init_hidden()
        channels_bilstms[i].hidden = (channels_bilstms[i].hidden[0].detach(), channels_bilstms[i].hidden[1].detach())
        lstm_out = channels_bilstms[i](current_single_channel_info)
        all_channels_bidlstm_hidden.append(lstm_out)

    all_channels_bidlstm_hidden = torch.stack(all_channels_bidlstm_hidden)
    return all_channels_bidlstm_hidden


if __name__ == "__main__":
    import numpy as np
    from model_tools import get_mixed_mfcc_at_t

    raw_tracks = torch.tensor(np.random.randn(num_channels, num_chunks, chunk_size))
    channels_bilstms = instantiate_all_channels_bilstms()
    all_channels_bidlstm_hidden = forward_pass_all_channel_bilstms(raw_tracks, channels_bilstms)

    mixed_mfcc_at_t = get_mixed_mfcc_at_t(raw_tracks, time_step_value=1)
    unilstms_model = MFCCUniLSTM(mfcc_chunk_size, hidden_dim_unilstm)
    print(unilstms_model(mixed_mfcc_at_t).shape)

