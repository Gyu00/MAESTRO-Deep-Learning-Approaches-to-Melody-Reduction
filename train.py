import os
import sys
import shutil
sys.path.append('/mnt/GTTM/model')
sys.path.append('/mnt/GTTM/model/gttm')

import torch
from torch.optim.lr_scheduler import StepLR
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

import models as models

from gttm import dataloader as dataloader
import logger as lg

import torch.nn as nn


#hyper parameters
epochs = 50

train_batch_size = 1
test_batch_size = 1

hidden_size = 10

prefix = 'our_model_1'

learning_rate= 1e-2
learning_rate_decay_steps=500
learning_rate_decay_rate=0.98

train_sr=80
valid_sr=20

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

def accuracy(pred, target):
    cor = (pred==target)
    return sum(cor)/cor.shape[0]
    
def main():
	trainset = dataloader.GTTMDataset(train_batch_size, train=True)
	trainloader = DataLoader(trainset, train_batch_size, shuffle=True)
	
	validset = dataloader.GTTMDataset(test_batch_size, train=False)
	validloader = DataLoader(validset, test_batch_size, shuffle=False)
    #define model
	model = models.LSTM(hidden_size)

    #training
	optimizer = torch.optim.Adam(model.parameters(), learning_rate)
	criterion = nn.CrossEntropyLoss()
	scheduler = StepLR( optimizer, step_size = learning_rate_decay_steps, gamma = learning_rate_decay_rate)

	#for tensorboard
	logger = lg.Logger('./logs_1')

	model = model.to(device)

	cudnn.benchmark = True
	print(' Total params : %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
	start_epoch = 0
	best_acc =0
	for epoch in range(start_epoch, epochs):

		print('\nEpoch: [%d | %d]' % (epoch+1, epochs))
		train_loss, train_acc = train(trainloader, model, criterion, optimizer,scheduler, epoch,logger )
		test_loss, test_acc = valid(validloader, model, criterion,scheduler,  epoch, logger)
		print('  train loss : %.4f | train acc : %.4f | test loss : %.4f | testt acc : %.4f'%(train_loss, train_acc, test_loss, test_acc))
		is_best = test_acc > best_acc
		best_acc = max(test_acc, best_acc)
		save_checkpoint({
				'state_dict' : model.state_dict(),
				'optimizer' : optimizer.state_dict(),
				'hidden_size' : hidden_size,
				}, is_best, prefix)


def train(trainloader, model, criterion, optimizer,scheduler, epoch, logger):
  
	avg_loss = 0
	correct = 0
	i=0

	for batch_idx, (inputs, targets, seq_lengths) in enumerate(trainloader):
		scheduler.step()
		optimizer.zero_grad()
		inputs = inputs.to(device)
		targets = targets.to(device)

		c0, h0 = model.init_hidden(inputs.shape[0])
		c0 = c0.to(device)
		h0 = h0.to(device)
		init_hidden = (c0, h0)

		hidden = init_hidden
		loss = 0.0

		inputs = inputs.transpose(0,1)
		targets = targets.transpose(0,1)

		for steps in range(int(seq_lengths[0])):
			step_inputs = inputs[steps,:,:].unsqueeze(0)
			pred, hidden = model(x=step_inputs , hidden = hidden)

			loss += criterion(pred[0,:,:], targets[steps])
			i +=1
			ans = int(pred[0,:,0] < pred[0,:,1])
			correct += int(ans ==targets[steps])
		loss.backward()
		optimizer.step()

		avg_loss += loss

	avg_loss=avg_loss/batch_idx
	acc=correct/i

	info = { 'train_loss' : avg_loss, 'train_accuarcy' : acc}
	for tag, value in info.items():
		logger.scalar_summary(tag, value, epoch)

	return avg_loss, acc

def valid(validloader, model, criterion,scheduler, epoch, logger ):
	with torch.no_grad():
		correct=0
		i=0
		avg_loss=0
		for batch_idx, (inputs, targets, seq_lengths)  in enumerate(validloader):
			scheduler.step()
            
			inputs=inputs.to(device)
			targets=targets.to(device)

			c0, h0 = model.init_hidden(inputs.shape[0])
			c0 = c0.to(device)
			h0 = h0.to(device)
			init_hidden = (c0, h0)
			hidden = init_hidden

			inputs=inputs.transpose(0,1)
			targets = targets.transpose(0,1)
			loss = 0.0
			for steps in range(int(seq_lengths[0])):
				step_inputs = inputs[steps,:,:].unsqueeze(0)
				pred, hidden = model(x=step_inputs, hidden = hidden)
				loss += criterion(pred[0,:,:], targets[steps])
				i += 1
				ans = int(pred[0,:,0]<pred[0,:,1])
				correct += int(ans ==targets[steps])
			avg_loss += loss
	avg_loss= avg_loss/batch_idx
	acc = correct/i

	info = { 'test_loss' : avg_loss, 'test_accuracy' : acc}
	for tag, value in info.items():
		logger.scalar_summary(tag,value,epoch)

	return avg_loss, acc


def save_checkpoint(state, is_best, prefix):
	filename = './checkpoint/%s_checkpoint.pth.tar'%prefix
	torch.save(state, filename)
	if is_best:
		shutil.copyfile(filename, './checkpoint/%s_model_best.pth.tar'%prefix)



if __name__ == '__main__':
	main()



