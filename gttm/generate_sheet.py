import xml.etree.ElementTree as ET
import os

def generate_new_sheet(output,data_path):

	path_dir = '/mnt/GTTM/model/gttm/test_set'

	file_list = os.listdir(path_dir)

	item = file_list[80+data_path]
	

	tree = ET.parse('/mnt/GTTM/model/gttm/test_set/'+item)
	root = tree.getroot()

	counter=0
	for i in root.iter('measure'):
		for j in i.iter('note'):
			counter=counter+1

	if counter != output.size():
	
	counter=0
	for i in root.iter('measure'):
		for j in i.iter('note'):
			if output[counter].item() == 0:
				#remove the note
				new_step = ''
				new_octave = '0'
				
				pitch = j.find('pitch')
				
				if pitch != None:
					step = pitch.find('step')
					octave = pitch.find('octave')

					step.text = new_step
					octave.text = new_octave
				else:
					#rest
					i.remove(j)
			
			counter = counter + 1
	

	
	tree.write('./output/'+item+str(data_path)+'_output.xml', encoding='utf-8', xml_declaration=True)


if __name__ == '__main__':
	generate_new_sheet()
