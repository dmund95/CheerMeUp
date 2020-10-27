import os
import torch
import torchvision
from torch import nn
import torch.utils.data as td
from dataloader import get_loader
from config import args
from matplotlib import pyplot as plt
import sys
import json
from time import time
from pprint import pprint

class StatsManager(object):
    """
    A class meant to track the loss during a neural network learning experiment.

    Though not abstract, this class is meant to be overloaded to compute and
    track statistics relevant for a given task. For instance, you may want to
    overload its methods to keep track of the accuracy, top-5 accuracy,
    intersection over union, PSNR, etc, when training a classifier, an object
    detector, a denoiser, etc.
    """

    def __init__(self):
        self.init()

    def __repr__(self):
        """Pretty printer showing the class name of the stats manager. This is
        what is displayed when doing ``print(stats_manager)``.
        """
        return self.__class__.__name__

    def init(self):
        """Initialize/Reset all the statistics"""
        self.running_loss = 0
        self.number_update = 0

    def accumulate(self, loss, num, x=None, y=None, d=None):
        """Accumulate statistics

        Though the arguments x, y, d are not used in this implementation, they
        are meant to be used by any subclasses. For instance they can be used
        to compute and track top-5 accuracy when training a classifier.

        Arguments:
            loss (float): the loss obtained during the last update.
            x (Tensor): the input of the network during the last update.
            y (Tensor): the prediction of by the network during the last update.
            d (Tensor): the desired output for the last update.
        """
        self.running_loss += loss
        self.number_update += num

    def summarize(self):
        """Compute statistics based on accumulated ones"""
        return self.running_loss / self.number_update



class Experiment(object):
    """
    A class meant to run a neural network learning experiment.

    After being instantiated, the experiment can be run using the method
    ``run``. At each epoch, a checkpoint file will be created in the directory
    ``output_dir``. Two files will be present: ``checkpoint.pth.tar`` a binary
    file containing the state of the experiment, and ``config.txt`` an ASCII
    file describing the setting of the experiment. If ``output_dir`` does not
    exist, it will be created. Otherwise, the last checkpoint will be loaded,
    except if the setting does not match (in that case an exception will be
    raised). The loaded experiment will be continued from where it stopped when
    calling the method ``run``. The experiment can be evaluated using the method
    ``evaluate``.

    Attributes/Properties:
        epoch (integer): the number of performed epochs.
        history (list): a list of statistics for each epoch.
            If ``perform_validation_during_training``=False, each element of the
            list is a statistic returned by the stats manager on training data.
            If ``perform_validation_during_training``=True, each element of the
            list is a pair. The first element of the pair is a statistic
            returned by the stats manager evaluated on the training set. The
            second element of the pair is a statistic returned by the stats
            manager evaluated on the validation set.

    Arguments:
        net (NeuralNetork): a neural network.
        train_set (Dataset): a training data set.
        val_set (Dataset): a validation data set.
        stats_manager (StatsManager): a stats manager.
        output_dir (string, optional): path where to load/save checkpoints. If
            None, ``output_dir`` is set to "experiment_TIMESTAMP" where
            TIMESTAMP is the current time stamp as returned by ``time.time()``.
            (default: None)
        batch_size (integer, optional): the size of the mini batches.
            (default: 16)
        perform_validation_during_training (boolean, optional): if False,
            statistics at each epoch are computed on the training set only.
            If True, statistics at each epoch are computed on both the training
            set and the validation set. (default: False)
    """
    

    def __init__(self, generator, discriminator, device, criterion,
                 optimizer_gen, optimizer_dis, d_stats_manager, g_stats_manager,output_dir=None):

        # Define data loaders
        dataloader = get_loader(args['file_path_csv'], args['transforms'], args['batch_size'], args['num_workers'],shuffle=True)

        # Initialize history
        history = {
            'losses' : [],
            'best_g_loss' : 10000.0,
            'best_epoch' : -1
        }

        # Define checkpoint paths
        if output_dir is None:
            output_dir = 'experiment_{}'.format(time.time())
        
        os.makedirs(output_dir, exist_ok=True)
        checkpoint_path = os.path.join(output_dir, "checkpoint.pth.tar")
        config_path = os.path.join(output_dir, "config.txt")
        bestmodel_path = os.path.join(output_dir, "bestmodel.pth.tar")
        bestmodel_config_path = os.path.join(output_dir, "bestmodel_config.txt")
        plot_path = os.path.join(output_dir, "loss_plot.png")
        generated_imgs_path = os.path.join(output_dir, "generated")
        os.makedirs(generated_imgs_path, exist_ok=True)


        # Transfer all local arguments/variables into attributes
        locs = {k: v for k, v in locals().items() if k is not 'self'}
        self.__dict__.update(locs)

        # Load checkpoint and check compatibility
        if os.path.isfile(config_path):
            with open(config_path, 'r') as f:
                if f.read()[:-1] != repr(self):
                    raise ValueError(
                        "Cannot create this experiment: "
                        "I found a checkpoint conflicting with the current setting.")
            self.load()
        else:
            self.save()

    @property
    def epoch(self):
        """Returns the number of epochs already performed."""
        return len(self.history['losses'])

    def setting(self):
        """Returns the setting of the experiment."""
        return {'Generator': self.generator,
                'Discriminator': self.discriminator,
                'DOptimizer': self.optimizer_dis,
                'GOptimizer': self.optimizer_gen,
                'DStatsManager': self.d_stats_manager,
                'GStatsManager': self.g_stats_manager}

    def __repr__(self):
        """Pretty printer showing the setting of the experiment. This is what
        is displayed when doing ``print(experiment)``. This is also what is
        saved in the ``config.txt`` file.
        """
        string = ''
        for key, val in self.setting().items():
            string += '{}({})\n'.format(key, val)
        return string

    def state_dict(self):
        """Returns the current state of the experiment."""
        return {'Generator': self.generator.state_dict(),
                'Discriminator' : self.discriminator.state_dict(),
                'DOptimizer': self.optimizer_dis.state_dict(),
                'GOptimizer': self.optimizer_gen.state_dict(),
                'History': self.history}

    def load_state_dict(self, checkpoint):
        """Loads the experiment from the input checkpoint."""
        self.generator.load_state_dict(checkpoint['Generator'])
        self.discriminator.load_state_dict(checkpoint['Discriminator'])
        self.optimizer_dis.load_state_dict(checkpoint['DOptimizer'])
        self.optimizer_gen.load_state_dict(checkpoint['GOptimizer'])
        self.history = checkpoint['History']

        # The following loops are used to fix a bug that was
        # discussed here: https://github.com/pytorch/pytorch/issues/2830
        # (it is supposed to be fixed in recent PyTorch version)
        for state in self.optimizer_gen.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)
        
        for state in self.optimizer_dis.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)

    def save(self):
        """Saves the experiment on disk, i.e, create/update the last checkpoint."""
        torch.save(self.state_dict(), self.checkpoint_path)
        with open(self.config_path, 'w') as f:
            print(self, file=f)
    
    def save_bestmodel(self):
        """Saves the best experiment on disk"""
        torch.save(self.state_dict(), self.bestmodel_path)
        with open(self.bestmodel_config_path, 'w') as f:
            print(self, file=f)        

    def load(self):
        """Loads the experiment from the last checkpoint saved on disk."""
        checkpoint = torch.load(self.checkpoint_path,
                                map_location=self.device)
        self.load_state_dict(checkpoint)
        del checkpoint

    def load_bestmodel(self):
        bestmodel = torch.load(self.bestmodel_path,
                                map_location=self.device)
        self.load_state_dict(bestmodel)
        del bestmodel

    def plot(self):
        #plots the
        trainLosses, valLosses = zip(*self.history['losses'])
        base = [i+1 for i in list(range(len(trainLosses)))]
        plt.figure()
        plt.plot(base, trainLosses)
        plt.plot(base, valLosses)
        plt.gca().legend(('generator','discriminator'))
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Losses vs Epochs')
        plt.savefig(self.plot_path)


    def run(self, num_epochs, plot=None):
        """Runs the experiment, i.e., trains the network using backpropagation
        based on the optimizer and the training set. Also performs statistics at
        each epoch using the stats manager.

        Arguments:
            num_epoch (integer): the number of epoch to perform.
            plot (func, optional): if not None, should be a function taking a
                single argument being an experiment (meant to be ``self``).
                Similar to a visitor pattern, this function is meant to inspect
                the current state of the experiment and display/plot/save
                statistics. For example, if the experiment is run from a
                Jupyter notebook, ``plot`` can be used to display the evolution
                of the loss with ``matplotlib``. If the experiment is run on a
                server without display, ``plot`` can be used to show statistics
                on ``stdout`` or save statistics in a log file. (default: None)
        """
        self.generator.train()
        self.discriminator.train()
        self.d_stats_manager.init()
        self.g_stats_manager.init()
        start_epoch = self.epoch
        device = self.device
        min_g_eloss = self.history['best_g_loss']
        fixed_noise = torch.randn(args["batch_size"], args["nz"], 1, 1, device=device) 
        
        print(f"Start/Continue training from epoch {start_epoch}")
        if plot is not None:
            plot(self)
        for epoch in range(start_epoch, num_epochs):
            s = time()
            self.d_stats_manager.init()
            self.g_stats_manager.init()
            for idx, (images) in enumerate(self.dataloader):

                if(list(images.size())[0] == 1):
                    continue
                images = images.to(device)
                
                ones = torch.ones(list(images.size())[0],1,1,1).to(device)
                zeros = torch.zeros(list(images.size())[0],1,1,1).to(device)
                ones_d = ones*0.8
                zeros_d = ones*0.2

                self.discriminator.zero_grad()
                self.generator.zero_grad()
                
                x_real = images[:,:3, :, :].to(device)
                #print(x_real.size())
                z = torch.randn(list(images.size())[0], args["nz"], 1, 1, device=device) 

                dr_out = self.discriminator(x_real)
                d_loss_real = self.criterion(dr_out, ones_d)
                d_loss_real.backward()

                x_gen = self.generator(z)
                dg_out = self.discriminator(x_gen.detach())
                d_loss_fake = self.criterion(dg_out, zeros_d)
                d_loss_fake.backward()

                d_loss = d_loss_real + d_loss_fake
                self.optimizer_dis.step()

                # g_loss = loss(dg_out, ones)
                # d_loss = loss(dr_out, ones) + loss(dg_out, zeros)

                dg_out = self.discriminator(x_gen)
                g_loss = self.criterion(dg_out, ones)
                g_loss.backward()
                self.optimizer_gen.step()

                
                with torch.no_grad():
                    self.d_stats_manager.accumulate(d_loss.item(), list(images.size())[0])
                    self.g_stats_manager.accumulate(g_loss.item(), list(images.size())[0])

            print(f'Time taken for train : {time()-s}')  
                       
            d_eloss = self.d_stats_manager.summarize()
            g_eloss = self.g_stats_manager.summarize()
            
            
            print(f'epoch : {epoch}, generator loss : {g_eloss : 0.6f}, discriminator loss : {d_eloss: 0.6f}' )
            self.history['losses'].append((g_eloss, d_eloss))

            if(g_eloss < min_g_eloss):
                min_g_eloss = g_eloss
                self.save_bestmodel()
                self.history['best_g_loss'] = min_g_eloss
                self.history['best_epoch'] = epoch
                print('Best model saved with generator loss', min_g_eloss)
                

            with open(os.path.join(self.output_dir, 'history.json'), 'w') as f:
                json.dump(self.history, f)
                
            if (epoch %3) == 0 :
                fake = self.generator(fixed_noise)
                torchvision.utils.save_image(fake.detach(),
                                             f'{self.generated_imgs_path}/fake_samples_epoch_{epoch}.png',
                                             normalize=True)
        
            
            self.save()
            self.plot()
            
            
        print(f"Finished training for {num_epochs} epochs")

    def eval(self, num):

    	self.generator.eval()
    	device = self.device
    	z = torch.randn(num, args["nz"], 1, 1, device=device)
    	images = self.generator(z)

    	return images, z.squeeze()

