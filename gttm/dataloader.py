import numpy as np
import torch
from torch.utils.data import Dataset
import data_load as load_d
from torch.nn.utils.rnn import pack_padded_sequence

class GTTMDataset(Dataset):
	def __init__(self,batch_size, train=True):
		self.batch_size = batch_size
		self.batch_idx = 0
		if train:
			s = np.arange(80)
		else:
			s = np.arange(80,100)


		t=0
		self.seq_lengths = torch.zeros(len(s))
		for i in s:
			x, y = load_d.load_gttm(i)
			self.seq_lengths[t] = len(y)
			t = t+1

		q=0
		seq_data = torch.zeros((len(s),int(self.seq_lengths.max().item()),9))
		seq_label = torch.zeros((len(s),int(self.seq_lengths.max().item()))).long()
		for i in s:
			seq_data[q,:int(self.seq_lengths[q].item()),:],seq_label[q,:int(self.seq_lengths[q].item())] = load_d.load_gttm(i)
			q=q+1

		self.data = seq_data
		self.label = seq_label



	def __len__(self):
		return len(self.seq_lengths)

	def __getitem__(self, index):
		return self.data[index,:,:], self.label[index,:], self.seq_lengths[index]
	def next_batch(self):
		x = self.data[self.batch_idx*self.batch_size:(self.batch_idx+1)*self.batch_size,:,:]
		y = self.label[self.batch_idx*self.batch_size:(self.batch_idx+1)*self.batch_size,:]

		self.batch_idx = self.batch_idx +1
		if self.batch_idx == self.total_batch:
			self.batch_idx = 0
		return x, y, self.seq_lengths[self.batch_idx*self.batch_size:(self.batch_idx+1)*self.batch_size]

if __name__ == '__main__':
	train_dataset = GTTMDataset()
	data_generator = iter(train_dataset)

	data_example = next(data_generator)
	print(data_example)


