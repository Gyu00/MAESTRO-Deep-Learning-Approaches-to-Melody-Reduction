import os
import sys

sys.path.append('/mnt/GTTM/model')
sys.path.append('/mnt/GTTM/model/gttm')
import torch
import models as models
import gttm.dataloader as dataloader
import gttm.generate_sheet as gs

from torch.utils.data import DataLoader

def load_model(checkpoint_path):
	checkpoint = torch.load(checkpoint_path)
	hidden_size = checkpoint['hidden_size']
	model = models.LSTM(hidden_size)

	model.load_state_dict(checkpoint['state_dict'])

	return model

def generate_label(model, loader):
	device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
	with torch.no_grad():
		model.eval()
		model = model.to(device)

		for batch_idx, (inputs, targets, seq_lengths) in enumerate(loader):
			inputs = inputs.to(device)
			c_0, h_0 = model.init_hidden(batch_size=1, random_init = False)
			c_0 = c_0.to(device)
			h_0 = h_0.to(device)
			init_hidden = (c_0, h_0)

			hidden = init_hidden
			inputs = inputs.transpose(0,1)

			answer = torch.zeros(int(seq_lengths[0]))
			answer= answer.to(device)

			for steps in range(int(seq_lengths[0])):
				step_inputs = inputs[steps,:,:].unsqueeze(0)
				pred, hidden = model(x=step_inputs , hidden=hidden)

				ans = int(pred[0,:,0] < pred[0,:,1])
				answer[steps] = ans
			print (answer)
			print (batch_idx)
			#cheol
			gs.generate_new_sheet(answer,batch_idx)
			

		return answer

def main():
	dataset = dataloader.GTTMDataset( 1, train = False)
	generateloader = DataLoader(dataset, 1, shuffle=False)
	model = load_model("./checkpoint/our_model_model_best.pth.tar")
	answer = generate_label(model, generateloader)




if __name__ == '__main__':
	main()



