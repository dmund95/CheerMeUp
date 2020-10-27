#!/usr/bin/env python
# coding: utf-8


import nntools as nt
from models import generator,discriminator
import torch
import torch.nn as nn
from config import args
import os
from dataloader import get_loader
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F
import sys
import numpy as np


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
        

generator_ = generator.Generator(args["nz"], args["ngf"], args["nc"], args["ngpu"]).to(device)
discriminator_ = discriminator.Discriminator(args["nc"],args["ndf"]).to(device)

generator_.apply(weights_init)
discriminator_.apply(weights_init)

criterion = args['loss_criterion']

params_gen = list(generator_.parameters())
params_dis = list(discriminator_.parameters())

optimizer_gen = torch.optim.Adam(params_gen, lr=args['learning_rate_gen'], betas=(args['beta'], 0.999))
optimizer_dis = torch.optim.Adam(params_dis, lr=args['learning_rate_dis'], betas=(args['beta'], 0.999))


d_stats_manager, g_stats_manager = nt.StatsManager(),nt.StatsManager()


exp1 = nt.Experiment(generator_, discriminator_, device, criterion, optimizer_gen, optimizer_dis,
                     d_stats_manager,g_stats_manager,
                     output_dir=args['model_path'])


#exp1.run(num_epochs=args['epochs'])


def generateZtoY(sampleSize, path):

	classifierCheckpoint = torch.load('./results/classifier')

	from models import classifier
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	extractor = classifier.Extractor().to(device)
	emotion_classifier = classifier.Classifier(7).to(device)
	gender_classifier = classifier.Classifier(2).to(device)

	extractor.load_state_dict(classifierCheckpoint['extractor'])
	emotion_classifier.load_state_dict(classifierCheckpoint['emotion_classifier'])
	gender_classifier.load_state_dict(classifierCheckpoint['gender_classifier'])

	extractor.eval()
	emotion_classifier.eval()
	gender_classifier.eval()
	
	numToGenerate = sampleSize
	batch = 128
	currentNum = 0

	allData = []
	curData = False

	while(currentNum < numToGenerate):

		images, z = exp1.eval(batch)

		feature = extractor.forward(images).squeeze()

		gender_output = gender_classifier.forward(feature)
		emotion_output = emotion_classifier.forward(feature)

		gender_probs = F.softmax(gender_output, dim=1)
		emotion_probs = F.softmax(emotion_output, dim=1)

		category = torch.cat((emotion_probs, gender_probs), 1)

		data = torch.cat((z, category), 1)

		if(curData == False):
			allData = data
			curData = True
		else:
			allData = torch.cat((allData, data), 0)

		currentNum += batch

	all_data_numpy = allData.cpu().detach().numpy()

	np.save('./data/'+path + '.npz', all_data_numpy)

def calculateCoefficients(paths):

	from models import feature_axis
	curData = False
	allData = []

	for p in paths:
		data = np.load('./data/'+p+'.npz.npy')
		if(not curData):
			allData = data
			curData = False
		else:
			allData = torch.cat((allData, data), 0)

	Z = allData[:,:100]
	Y = allData[:,100:]
	coefficients, score = feature_axis.feature_axis(Z,Y,method='tanh')
	# coenormalize_feature_axis(feature_slope)
	print(score)
	np.save('./data/coefficients_tanh_jules', coefficients)
	#normalizedCoefficients = feature_axis.normalize_feature_axis(coefficients)
	#print(normalizedCoefficients.shape)

sampleSize = 800
generateZtoY(sampleSize, 'samples1_jules')
generateZtoY(sampleSize, 'samples2_jules')
generateZtoY(sampleSize, 'samples3_jules')
generateZtoY(sampleSize, 'samples4_jules')
calculateCoefficients(['samples1_jules','samples2_jules','samples3_jules','samples4_jules'])


