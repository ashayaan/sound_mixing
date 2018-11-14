import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from model_params import embedding_dim_bidlstm
from model_params import hidden_dim_bidlstm                 # k value in our notes
from model_params import num_channels                       # C value in our notes
from model_params import num_chunks                         # T value in our notes
from model_params import chunk_size                         # lambda/T value in our notes

torch.manual_seed(1)


class BiLSTM(nn.Module):
    def __init__(self, chunk_size, hidden_dim, num_channels):
        super(BiLSTM, self).__init__()
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
        num_channels_bidlstm_hidden = []

        for i in range(self.num_channels):
            current_single_channel_info = torch.tensor(raw_tracks[i], dtype=torch.float).view(num_chunks, 1, -1)
            current_single_channel_hidden = self.channels_bilstms_hidden[i]

            lstm_out, _ = self.channels_bilstms[i](current_single_channel_info, current_single_channel_hidden)
            num_channels_bidlstm_hidden.append(lstm_out)

        num_channels_bidlstm_hidden = torch.stack(num_channels_bidlstm_hidden)
        return num_channels_bidlstm_hidden


if __name__ == "__main__":
    import numpy as np
    model = BiLSTM(embedding_dim_bidlstm, hidden_dim_bidlstm, num_channels)

    # each song data input - raw_tracks will have info for all channels and split inro num_chunks
    raw_tracks = np.random.randn(num_channels, num_chunks, chunk_size)

    model.zero_grad()

    num_channels_bidlstm_hidden = model(raw_tracks)
    print(num_channels_bidlstm_hidden.shape)