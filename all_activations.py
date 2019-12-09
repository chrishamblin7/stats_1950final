#script for pulling the activations from all of a networks units. 

from __future__ import print_function
import os
from PIL import Image
import torch
torch.set_printoptions(threshold=5000)
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models, transforms
import pandas as pd
import pdb
#paths
input_data_path = '../../stimuli/enumeration_unit_selection/wob/unit_select_2/'



''' CSV OUTPUT
output_file = open('pretrained_alexnet.csv','w+')

#headers for output file
output_file.write('neuron,activation,numerosity,image,channel,dim1,dim2,areacontrol\n')
output_file.flush()
'''

#Pandas df
columns = ['neuron','activation','numerosity','image','channel','dim1','dim2','areacontrol']
rows = []

#model
#model = models.alexnet(pretrained=True)

model = torch.load('../../outputs/enumeration_wob_big_alexnet_classes20_pretrained_12-08-2019:22_47/model_50.pt')
model.to('cpu')

#hook function
def get_activation(name):
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook


#image preprocessing 
default_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])



# cycle through images

image_list = sorted(os.listdir(input_data_path))
i = 0
for image_name in image_list:
	i+=1
	print(i)
	activations = {}
	#pull info from image name
	image_name_split = image_name.split('_')
	numerosity = image_name_split[0]
	if numerosity == '0':
		continue
	areacontrol = image_name_split[2]
	#get actual image data
	image = Image.open(os.path.join(input_data_path,image_name))
	image = default_transform(image)
	image = image.unsqueeze(0)
	model.features[12].register_forward_hook(get_activation('final_features'))
	model_output = model(image)
	for chan in range(len(activations['final_features'][0])):
		for dim1 in range(len(activations['final_features'][0][chan])):
			for dim2 in range(len(activations['final_features'][0][chan][dim1])):
				activation = round(float(activations['final_features'][0][chan][dim1][dim2]),5)
				#output_file.write(','.join(['maxpool3_%s_%s_%s'%(str(chan),str(dim1),str(dim2)),activation,numerosity,image_name,str(chan),str(dim1),str(dim2),areacontrol])+'\n')
				rows.append(['maxpool3_%s_%s_%s'%(str(chan),str(dim1),str(dim2)),activation,numerosity,image_name,chan,dim1,dim2,areacontrol])



df = pd.DataFrame(rows,columns=columns)
df.to_feather('alexnet_trained_wob_sel2_final_layer_responses.feather')


'''
store = pd.HDFStore("alexnet_random.hdf5", "w", complib=str("zlib"), complevel=5)
store.put("dataframe", df, data_columns=df.columns)
store.close()
#df.to_hdf('alexnet_random.h5', key='df', mode='w')
'''
