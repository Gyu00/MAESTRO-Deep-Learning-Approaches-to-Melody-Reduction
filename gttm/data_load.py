import xml.etree.ElementTree as elemTree
import os
import torch
import numpy as np

#torch.set_default_tensor_type('torch.cuda.FloatTensor')
def Convert2Data(data_list):
	
	converted_list=[]
	converted_list.append(int(data_list[0]))
	converted_list.append(int(data_list[1]))
	# major 0 / minor 1
	if data_list[2]=='major':
		converted_list.append(0)
	elif data_list[2]=='minor':
		converted_list.append(1)
	else:
		print("Error!!!")
	
	#beats 
	converted_list.append(data_list[3])

	#clef_sign
	if data_list[4]=='G':
		converted_list.append(1)
	else:
		converted_list.append(0)

	#clef_line
	if data_list[5]=='2':
		converted_list.append(2)
	else:
		converted_list.append(int(data_list[5]))

	#Step
	StepVector = ['C','D','E','F','G','A','B','#']
	
	for i in range(0,8):
		if StepVector[i] == data_list[6]:
			converted_list.append(i)
			break
	
	#octave
	if data_list[7]!='#':
		converted_list.append(int(data_list[7]))
	else:
		# rest symbol  => -1 .. ? 
		converted_list.append(-1)

	#duration
	converted_list.append(float(data_list[8]))
	
	return converted_list


def data_processing(data_path):
	tree = elemTree.parse(data_path)
	
	root = tree.getroot()

	division=''
	fifths=''
	mode=''
	time=''
	clef_sign=''
	clef_line=''
		
	
	counter=0
	for j in root.iter('measure'):
		for k in j.iter('note'):
			counter=counter+1
	
	data_list = torch.zeros(counter,9)

	counter=0

	for m in root.iter('measure'):
		att = m.find('attributes')
		
		if att != None:
			div = att.find('divisions')
			key = att.find('key')
			time2 = att.find('time')
			clef = att.find('clef')

			if div != None:
				division=div.text
			if key != None:
				fifths = key.findtext('fifths')
				mode = key.findtext('mode')
			if time2 != None:
				x =  float(time2.findtext('beats'))/float(time2.findtext('beat-type'))
				if x != None:
					time=x
					
			if clef != None:
				clef_sign = clef.findtext('sign')
				clef_line = clef.findtext('line')
		
		
		for single_note in m.iter('note'):
			pitch = single_note.find('pitch')
			if single_note.find('duration') != None:
				duration = single_note.findtext('duration')
			else:
				duration = -9999


			step=''
			octave=''
			if pitch != None:
				if pitch.find('step') != None:
					step=pitch.findtext('step')
					octave=pitch.findtext('octave')
			else:
				step='#'
				octave='#'
			
			# *** IMPORTANT ***
			#feature sequence 
			temp_list = [division, fifths,mode,time,clef_sign,clef_line,step,octave,duration]
			
			# Coverting
			temp_data = Convert2Data(temp_list)
			#print(temp_data)
			for i in range(0,9):
				data_list[counter][i]=temp_data[i]
				
			#print(data_list[counter])
			counter=counter+1
	
#	data_list = data_list.t()

	return data_list

def make_label(data_path, origin_data_path):

	ori_tree = elemTree.parse(origin_data_path)
	ori_root = ori_tree.getroot()
	rest_checker=[]
	measure_checker=[]
	for a in ori_root.iter('measure'):
		for b in a.iter('note'):
			pitch = b.find('pitch')
			if pitch != None:
				rest_checker.append(0)
				measure_checker.append(a)
			else:
				rest_checker.append(1)
				measure_checker.append(a)


	tree = elemTree.parse(data_path)
	root = tree.getroot()

	label = []

	for j in root.iter('measure'):
		for k in j.iter('note'):
			pitch = k.find('pitch')
			if pitch != None:
				step = pitch.findtext('step')
				if step == '':
					label.append(0)
				else:
					label.append(1)
				
			else:
				#rest symbol.. 
				label.append(1)

	rest_index=[]
	for i in range(1,len(rest_checker)):
		if (rest_checker[i-1] ==1 and rest_checker[i]==1) and (measure_checker[i-1] == measure_checker[i]):
			rest_index.append(i)
	
	real_label=[]
	counter=0

	rest_counter=0
	start=len(rest_checker)-1
	while (True):
		if rest_checker[start]==1:
			rest_counter=rest_counter+1

		elif rest_checker[start]!=1:
			break

		start=start-1

	for i in range(len(rest_checker)-rest_counter):
		if rest_index.count(i) != 0:
			real_label.append(0)
		else:
			real_label.append(label[counter])
			counter=counter+1
	
	for i in range(0,rest_counter):
		real_label.append(0)


	tensor_label = torch.zeros(len(real_label))
	for i in range(0,len(real_label)):
		tensor_label[i] = real_label[i]

	return tensor_label

def load_gttm(index):
	
	path_dir = '/mnt/GTTM/model/gttm/test_set/'
	path_dir2 = '/mnt/GTTM/model/gttm/easy_set/'
	file_list = os.listdir(path_dir) 
	
	item=file_list[index]

	data_x = data_processing('/mnt/GTTM/model/gttm/test_set/'+item)
	data_y = make_label('/mnt/GTTM/model/gttm/easy_set/easy'+item[0]+item[1]+'.xml','/mnt/GTTM/model/gttm/test_set/'+item)
#	print(data_x.size(), data_y.size())
#	print(data_x, data_y)
#	print(data_x.type(), data_y.type())
	return data_x, data_y

	'''
	data_x = []
	data_y = []
#	i=0
	item = file_list
	train=True
	if train:
		for item in file_list[0:100]:
			input_tensor = data_processing('/mnt/GTTM/model/gttm/test_set/'+item)
			label = make_label('/mnt/GTTM/model/gttm/easy_set/easy'+item[0]+item[1]+'.xml','/mnt/GTTM/model/gttm/test_set/'+item)
#			i=i+1
			data_x.append(input_tensor)
			data_y.append(label)
			t_x=input_tensor.size()
			t_y=label.size()
			if t_x[1] != t_y[0]:
				print(item)
				print('wrong'+str(t_x[1])+' '+str(t_y[0]))
	else:
		for item in file_list[80:100]:
			input_tensor = data_processing('/mnt/GTTM/model/gttm/test_set/'+item)
			label = make_label('/mnt/GTTM/model/gttm/easy_set/easy'+item[0]+item[1]+'.xml','/mnt/GTTM/model/gttm/test_set/'+item)
#           i=i+1
			data_x.append(input_tensor)
			data_y.append(label)
			print(input_tensor.size())
			print(label.size())


#	data_x = data_processing('./test_set/' +item)
#	data_y = make_label('./easy_set/easy'+item[0]+item[1]+'.xml')

	return data_x, data_y
	'''
if __name__ == '__main__':
	load_gttm(1)


