import torch.nn as nn
import pickle

baseDataPath = './data'

from torchvision import transforms
from models import generator,discriminator

size = (64,64)

transform = transforms.Compose([transforms.Resize(size), 
                                transforms.ToTensor()])

data_path = f"{baseDataPath}/file_names_jules.csv"


args = {
	'file_path_csv' : data_path,

	'epochs' : 500,

	'batch_size' : 128,

	'num_workers' : 16,
    
	'nz' : 100,
    
	'ngf' : 64,
    
	'nc' : 3,
    
	'ngpu' : 1,

	'ndf' : 64,

	'model_path' : './results/gan_dc_hyperparameters_jules',

	'Discriminator' : discriminator,

	'Generator' : generator,

	# 'num_layers' : 1,

	'loss_criterion' : nn.MSELoss(),

	'learning_rate_gen': 2e-4,
    
	'learning_rate_dis': 2e-4,
    
	'noise_len': 100,
    
	'beta' : 0.5,

	'transforms' : transform,

}
