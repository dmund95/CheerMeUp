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


if __name__ == "__main__":
	exp1.run(num_epochs=args['epochs'])
