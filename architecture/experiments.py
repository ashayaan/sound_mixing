import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

# EXPERIMENTS
c = 10
embedding_dim = 3
hidden_dim = 4
c_bilstms = []
for i in range(c):
    c_bilstms.append(nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=1, bidirectional=True))
print(c_bilstms)

lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=1, bidirectional=True)  # Input dim is 3, output dim is 3
uni_lstm = nn.LSTM(embedding_dim, hidden_dim)  # Input dim is 3, output dim is 3

inputs = [torch.randn(1, embedding_dim) for _ in range(5)]  # make a sequence of length 5

# initialize the hidden state.
hidden = (torch.randn(2, 1, hidden_dim // 2),
          torch.randn(2, 1, hidden_dim // 2))

# initialize the hidden state.
uni_hidden = (torch.randn(1, 1, hidden_dim),
          torch.randn(1, 1, hidden_dim))

for i in inputs:
    # Step through the sequence one element at a time.
    # after each step, hidden contains the hidden state.
    out, hidden = lstm(i.view(1, 1, -1), hidden)


for i in inputs:
    # Step through the sequence one element at a time.
    # after each step, hidden contains the hidden state.
    uni_out, uni_hidden = uni_lstm(i.view(1, 1, -1), uni_hidden)

# alternatively, we can do the entire sequence all at once.
# the first value returned by LSTM is all of the hidden states throughout
# the sequence. the second is just the most recent hidden state
# (compare the last slice of "out" with "hidden" below, they are the same)
# The reason for this is that:
# "out" will give you access to all hidden states in the sequence
# "hidden" will allow you to continue the sequence and backpropagate,
# by passing it as an argument  to the lstm at a later time
# Add the extra 2nd dimension
inputs = torch.cat(inputs).view(len(inputs), 1, -1)
hidden = (torch.randn(2, 1, hidden_dim // 2), torch.randn(2, 1, hidden_dim // 2))  # clean out hidden state
uni_hidden = (torch.randn(1, 1, hidden_dim), torch.randn(1, 1, hidden_dim))  # clean out hidden state
out, hidden = lstm(inputs, hidden)
uni_out, uni_hidden = uni_lstm(inputs, uni_hidden)

print(out.shape)
print(out.view(len(inputs), -1).shape)
print(hidden[0].shape)
print('---------------------------------------------')
print(uni_out.shape)
print(uni_out.view(len(inputs), -1).shape)
print(uni_hidden[0].shape)

print(uni_out[-1].view(-1,1).shape)
print(uni_hidden[0].shape)