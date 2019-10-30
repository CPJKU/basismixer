from abc import ABC, abstractmethod
import logging
import os

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, Sampler
import torch.nn.functional as functional
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)


class NNTrainer(ABC):
    """
    Class for training neural networks
    """

    def __init__(self, model, train_loss, optimizer,
                 train_dataloader,
                 valid_loss=None,
                 valid_dataloader=None,
                 best_comparison='smaller',
                 n_gpu=1,
                 epochs=100,
                 save_freq=10,
                 early_stopping=100,
                 out_dir='.',
                 resume_from_saved_model=None):
        self.n_gpu = n_gpu
        self.device, self.dtype = self.prepare_device()
        self.model = model.to(self.device)
        self.out_dir = out_dir

        if self.n_gpu > 1:
            self.model = torch.nn.DataParallel(model,
                                               device_ids=self.device_ids)
        self.train_loss = loss
        self.valid_loss = valid_loss
        self.optimizer = optimizer

        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader

        self.start_epoch = 0
        self.epochs = epochs
        self.save_freq = save_freq
        self.early_stopping = early_stopping
        self.best_loss = np.inf
        self.best_comparison = best_comparison

        if not os.path.exists(self.out_dir):
            os.mkdir(self.out_dir)

        self.loss_progress = {'train_loss': [], 'val_loss': []}

        if resume_from_saved_model is not None:
            self.resume_checkpoint(resume_from_saved_model)

    @abstractmethod
    def train_step(self, *args, **kwargs):
        pass

    @abstractmethod
    def valid_step(self, *args, **kwargs):
        pass

    def train(self):

        train_loss_name = getattr(self.train_loss, 'name', 'Train Loss')
        # Initialize TrainProgressMonitors
        train_losses = TrainProgressMonitor(train_loss_name,
                                            fn=train_fn)
        valid_loss_name = None
        valid_losses = None
        if self.valid_loss is not None:
            if isinstance(self.valid_loss, (list, tuple)):
                valid_loss_name = [getattr(crit, 'name', 'Valid Loss {0}'.format(i))
                                   for i, crit in enumerate(self.valid_loss)]
            else:
                valid_loss_name = [getattr(self.valid_loss, 'name', 'Valid Loss')]

            valid_losses = TrainProgressMonitor(valid_loss_name,
                                                fn=valid_fn,
                                                show_epoch=False)

        validations_wo_improvement = 0

        # save before training
        self.save_checkpoint(-1, False, True)
        try:
            for epoch in range(self.start_epoch, self.epochs):
                tl = self.train_step()

                train_loader.update(epoch, tl)

                do_checkpoint = np.mod(epoch + 1, self.save_freq) == 0

                if do_checkpoint:
                    if self.valid_dataloader is not None:
                        vl = self.valid_step()
                        valid_losses.update(epoch, vl)
                        LOGGER.info(train_losses.last_loss + '\t' + valid_losses.last_loss)
                    else:
                        vl = [tl]
                        LOGGER.info(train_losses.last_loss)
                    if self.best_comparison == 'smaller':
                        is_best = vl[0] < best_loss
                        self.best_loss = min(vl[0], best_loss)
                    elif self.best_comparison == 'larger':
                        is_best = vl[0] > best_loss
                        self.best_loss = max(vl[0], best_loss)

                    self.save_checkpoint(epoch,
                                         validate=do_checkpoint,
                                         is_best=is_best)

                    if is_best:
                        validations_wo_improvement = 0
                    else:
                        validations_wo_improvement += 1

                    if validations_wo_improvement > self.early_stopping:
                        break
        except KeyboardInterrupt:
            LOGGER.info('Training interrupted')
            pass

        # Load best model
        best_backup = torch.load(os.path.join(self.out_dir, 'best_model.pth'))
        LOGGER.info('Loading best model (epoch {0}: {1:.4f})'.format(best_backup['epoch'],
                                                                     best_backup['best_loss']))
        self.model.load_state_dict(best_backup['state_dict'])

    def save_checkpoint(self, epoch, validate, is_best):
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_loss': self.best_loss
        }
        if validate:
            filename = os.path.join(self.out_dir, 'checkpoint-epoch-{}.pth'.format(epoch))
            torch.save(state, filename)
            LOGGER.info("Saving checkpoint: {0} ...".format(filename))
        if is_best:
            filename = os.path.join(self.out_dir, 'best_model.pth')
            torch.save(state, filename)
            LOGGER.info("Saving current best: {0} ...".format(filename))

    def prepare_device(self):
        n_gpu = torch.cuda.device_count()
        if self.n_gpu > 0 and n_gpu == 0:
            LOGGER.warning('No GPU available! Training will be performed on CPU.')
            self.n_gpu = 0
        if self.n_gpu > n_gpu:
            LOGGER.warning('Only {0} GPUs availabe. '
                           '(`n_gpu` is {1})'.format(n_gpu, self.n_gpu))
            self.n_gpu = n_gpu

        device = torch.device('cuda:0' if self.n_gpu > 0 else 'cpu')
        if device.type == 'cuda':
            dtype = 'torch.cuda.FloatTensor'
        else:
            dtype = 'torch.FloatTensor'

        return device, dtype

    def resume_checkpoint(self, checkpoint_fn):
        LOGGER.info("Loading checkpoint: {0} ...".format(checkpoint_fn))
        checkpoint = torch.load(checkpoint_fn)
        self.start_epoch = checkpoint['epoch']
        self.best_loss = checkpoint['best_loss']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        LOGGER.info("Checkpoint loaded. "
                    "Resume training from epoch {0}".format(self.start_epoch))


class TrainProgressMonitor(object):
    """
    Monitor the progress of training a model.
    """

    def __init__(self, name='', fn='/tmp/train_progres.txt',
                 show_fmt='{0} {1:.3f}', write_fmt='{0:.8f}', show_epoch=True):
        self.name = name
        self.losses = []
        self.epochs = []
        self.fn = fn
        self.show_fmt = show_fmt
        self.write_fmt = write_fmt
        self.show_epoch = show_epoch

        try:
            os.unlink(self.fn)
        except OSError:
            pass

        with open(self.fn, 'w') as f:
            if isinstance(self.name, (list, tuple)):
                header = '# Epoch' + '\t' + '\t'.join(self.name) + '\n'
            else:
                header = '# Epoch' + '\t' + self.name + '\n'

            f.write(header)

    def update(self, epoch, loss):
        """
        Append new loss(es) and update the log file
        """
        self.losses.append(loss)

        self.epochs.append(epoch)

        self.update_log()

    @property
    def last_loss(self):

        fmt = self.show_fmt

        if isinstance(self.losses[-1], (list, tuple, np.ndarray)):
            out_str = [fmt.format(n, l) for n, l in zip(self.name, self.losses[-1])]
        else:
            out_str = [fmt.format(self.name, self.losses[-1])]

        if self.show_epoch:
            return 'Epoch:{0}\t{1}'.format(self.epochs[-1], '\t'.join(out_str))
        else:
            return '\t'.join(out_str)

    def update_log(self):

        if isinstance(self.losses[-1], (list, tuple, np.ndarray)):
            out_str = '\t'.join([self.write_fmt.format(l) for l in self.losses[-1]])
        else:
            out_str = self.write_fmt.format(float(self.losses[-1]))

        with open(self.fn, 'a') as f:
            f.write('{0}\t{1}\n'.format(self.epochs[-1], out_str))


class FeedForwardTrainer(NNTrainer):
    def __init__(self, model, loss, optimizer,
                 train_data_loader,
                 valid_data_loader=None,
                 lr_scheduler=None,
                 n_gpu=1,
                 epochs=100,
                 save_freq=10,
                 early_stop=100,
                 out_dir='.',
                 resume_from_saved_model=None):
        super().__init__(model=model,
                         loss=loss,
                         optimizer=optimizer,
                         n_gpu=n_gpu,
                         epochs=epochs,
                         save_freq=save_freq,
                         early_stop=early_stop,
                         out_dir=out_dir,
                         resume_from_saved_model=resume_from_saved_model)

        self.train_data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader

    def train_step(self, epoch):
        self.model.train()

        bar = tqdm(enumerate(self.train_data_loader), total=len(self.train_data_loader))

        epoch_loss = 0
        for b_idx, (x, y) in bar:
            x, y = x.type(self.dtype), y.type(self.dtype)

            self.optimizer.zero_grad()

            y_h = self.model(y_h)

            loss = self.loss(y, y_h)

            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()

            bar.set_description("Epoch: {0}/{1}".format(epoch + 1, self.epochs + 1))

        if self.validate:

        return
