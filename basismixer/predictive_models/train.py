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

    def __init__(self, model, loss, optimizer,
                 n_gpu=1,
                 epochs=100,
                 save_period=10,
                 early_stop=100,
                 out_dir='.',
                 resume_from_saved_model=None):
        self.n_gpu = n_gpu
        self.device, self.dtype = self.prepare_device()
        self.model = model.to(self.device)
        self.out_dir = out_dir

        if self.n_gpu > 1:
            self.model = torch.nn.DataParallel(model,
                                               device_ids=self.device_ids)
        self.loss = loss
        self.optimizer = optimizer

        self.start_epoch = 0
        self.epochs = epochs
        self.save_period = save_period
        self.early_stop = early_stop
        self.best_loss = np.inf
        self.not_improved_count = 0

        if not os.path.exists(self.out_dir):
            os.mkdir(self.out_dir)

        self.loss_progress = {'train_loss': [], 'val_loss': []}

        if resume_from_saved_model is not None:
            self.resume_checkpoint(resume_from_saved_model)

    @abstractmethod
    def train_step(self):
        pass

    @abstractmethod
    def valid_step(self):
        pass

    def train(self):
        for epoch in range(self.start_epoch, self.epochs):
            train_loss = self.train_step()

            self.loss_progress['train_loss'].append(result['train_loss'])
            try:
                self.loss_progress['val_loss'].append(result['val_loss'])
            except:
                pass

            for k, v in result.items():
                LOGGER.info("  {}: {}".format(str(k), v))

            is_best = False

            validate = (epoch + 1) % self.save_period == 0
            if validate:
                val_loss = self.valid_step(epoch)

                try:
                    improved = result['val_loss'] <= self.best_loss

                    if improved:
                        self.best_loss = result['val_loss']
                        self.not_improved_count = 0
                        is_best = True
                    else:
                        self.not_improved_count += 1

                    if self.not_improved_count == self.early_stop:
                        LOGGER.info('No improvement for {0} epochs. '
                                    'Stoping training...'.format(self.early_stop))
                        break
                except Exception:
                    pass

            if validate or is_best:
                self.save_checkpoint(epoch,
                                     validate=validate,
                                     is_best=is_best)

        LOGGER.info("The best loss: {}".format(self.best_loss))
        LOGGER.info("Loss of training progress saved.")
        np.save(os.path.join(self.out_dir, 'loss_progress.npy'), self.loss_progress)

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
            filename = str(self.checkpoint_dir / 'model_best.pth')
            torch.save(state, filename)
            LOGGER.info("Saving current best: %s ..." % filename)

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
                 save_period=10,
                 early_stop=100,
                 out_dir='.',
                 resume_from_saved_model=None):
        super().__init__(model=model,
                         loss=loss,
                         optimizer=optimizer,
                         n_gpu=n_gpu,
                         epochs=epochs,
                         save_period=save_period,
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
