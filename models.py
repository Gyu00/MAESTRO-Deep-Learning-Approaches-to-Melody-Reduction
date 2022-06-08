import torch
import numpy as np
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class LSTM(nn.Module):
	def __init__(self, hidden_size, n_layers = 2, input_size=9):
		super(LSTM, self).__init__()
		self.hidden_size = hidden_size
		self.n_layers = n_layers

		self.rnn = nn.LSTM( input_size, hidden_size, n_layers)
		self.decoder = nn.Linear(in_features = hidden_size, out_features = 2)
		self.log_softmax = nn.Softmax(dim=-1)

	def forward(self, x, hidden):

		output, hidden = self.rnn(x, hidden)

		output = self.decoder(output)

		pred = self.log_softmax(output)

		return pred, hidden
	def init_hidden(self, batch_size, random_init=False):
		if random_init:
			return torch.randn(self.n_layers, batch_size, self.hidden_size), \
					torch.randn(self.n_layers, batch_size, self.hidden_size)
		else:
			return torch.zeros(self.n_layers, batch_size, self.hidden_size),\
					torch.zeros(self.n_layers, batch_size, self.hidden_size)
