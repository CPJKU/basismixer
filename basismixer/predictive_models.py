import numpy as np
import torch


from torch import nn
from torch.utils.data import Dataset, Sampler
import torch.nn.functional as functional


class PredictiveModel(object):
    """
    Predictive Model
    """

    def __init__(self, input_names=None,
                 output_names=None,
                 is_rnn=False,
                 input_type=None):
        # name of input features
        self.input_names = input_names
        # name of output features
        self.output_names = output_names
        # if the model is
        self.is_rnn = is_rnn
        # if the input is onsetwise or notewise
        self.input_type = input_type

    def predict(self, x):
        """
        Predict
        """
        # reshape for recurrent models
        do_reshape = self.is_rnn and len(x.shape) < 3
        if do_reshape:
            x = x[np.newaxis]

        # model predictions
        predictions = self(x)

        # predictions as numpy array
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.data.detach().cpu().numpy()

        if do_reshape:
            predictions = predictions[0]

        # Output as structured array
        if predictions.ndim < 3:
            output = np.zeros(len(predictions),
                              dtype=[(pn, 'f4') for pn in self.output_names])
            for i, pn in enumerate(self.output_names):
                output[pn] = predictions[:, i]
        else:
            output = []
            for p in predictions:
                out = np.zeros(len(p),
                               dtype=[(pn, 'f4') for pn in self.output_names])
                for i, pn in enumerate(self.output_names):
                    out[pn] = p[:, i]
                output.append(out)
            output = np.array(output)

        return output


class RecurrentModel(nn.Module, PredictiveModel):
    def __init__(self,
                 input_size, output_size,
                 recurrent_size, hidden_size,
                 n_layers, dropout=0.0,
                 batch_first=True,
                 input_names=None,
                 output_names=None,
                 input_type=None):
        nn.Module.__init__(self)
        PredictiveModel.__init__(self,
                                 input_names=input_names,
                                 output_names=output_names,
                                 is_rnn=True,
                                 input_type=input_names)

        self.input_size = input_size
        self.recurrent_size = recurrent_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.batch_first = batch_first
        self.rnn = nn.GRU(input_size, self.recurrent_size,
                          self.n_layers,
                          batch_first=batch_first, dropout=dropout)
        self.dense = nn.Linear(in_features=self.recurrent_size,
                               out_features=self.hidden_size)
        self.output = nn.Linear(in_features=self.hidden_size,
                                out_features=output_size)

        if self.output_names is None:
            self.output_names = [str(i) for i in range(self.output_size)]

    def init_hidden(self, batch_size):
        return torch.zeros(self.n_layers, batch_size, self.recurrent_size)

    def forward(self, x, origin_len=None):
        batch_size = x.size(0)
        seq_len = x.size(1)
        h0 = self.init_hidden(batch_size).type(x.type())

        # pack_padded_sequence() prevents rnns to process padded data
        if origin_len is not None:
            origin_len = torch.as_tensor(
                origin_len, dtype=torch.int64, device='cpu')
            x = nn.utils.rnn.pack_padded_sequence(
                x, origin_len, batch_first=True, enforce_sorted=False)

        output, h = self.rnn(x, h0)

        if isinstance(output, nn.utils.rnn.PackedSequence):
            output, _ = nn.utils.rnn.pad_packed_sequence(
                output, batch_first=True)

        dense = self.dense(output.contiguous().view(-1, self.recurrent_size))
        y = self.output(dense)
        y = y.view(batch_size, seq_len, self.output_size)

        return y


def recurrent_mse_loss(output, target, mask, loss_fun=functional.mse_loss):
    """
    Mean cross entropy

    Parameters
    ----------
    output : torch.Tensor
        Predictions of the neural network (output of the model)
    target : torch.Tensor
        Target values for supervised training.
    mask : torch.Tensor
        Valid elements of the loss for computing the normalization

    Returns
    -------
    loss : torch.Tensor
       Aggregated value of the loss across the batch dimension
    """
    batch_size, time_step, out_size = output.size()
    # flatten the variables
    output_f = output.view(-1, out_size)
    target_f = target.view(-1, out_size)

    loss = loss_fun(input=output_f,
                    target=target_f,
                    reduction='none')

    mask_f = mask.view(-1, 1).type(loss.type())
    valid_timestep = mask_f.sum().item()
    loss *= mask_f

    return loss.mean(dim=-1).sum(dim=0).div(valid_timestep)


class SubsequenceSampler(Sampler):

    def __init__(self, seq_length, data_source, epoch_size='auto',
                 seq_indices=None, shuffle=True,
                 random_state=np.random.RandomState(1984),
                 valid_sequences=None):
        super().__init__(data_source)
        self.data_source = data_source
        self.seq_length = seq_length
        self.epoch_size = epoch_size

        self.seq_indices = seq_indices

        if self.seq_indices is None:

            if hasattr(self.data_source, 'seq_indices'):
                self.seq_indices = self.data_source.seq_indices
            else:
                raise ValueError('Dataset must have a list of indices for each'
                                 'unique sequence')

        self.shuffle = shuffle

        self.rng = random_state

        self.seq_lengths = [len(l) for l in self.seq_indices]

        if isinstance(self.seq_length, int):
            if any([sl < self.seq_length for sl in self.seq_lengths]):
                self.seq_length = [min(self.seq_lengths)] * len(self.seq_lengths)

            else:
                self.seq_length = [self.seq_length] * len(self.seq_lengths)

        elif self.seq_length == 'full':
            self.seq_length = self.seq_lengths

        self.valid_sequences = valid_sequences

        if self.valid_sequences is None:
            self.valid_sequences = np.arange(len(self.seq_indices))

    def __len__(self):

        if self.epoch_size == 'auto':

            return sum([sl for i, sl in enumerate(self.seq_lengths)
                        if i in self.valid_sequences]) // min(self.seq_length)
        elif isinstance(self.epoch_size, int):
            return self.epoch_size

    def __iter__(self):

        # select sequence
        if self.shuffle:
            sequences = self.rng.randint(0, len(self.valid_sequences), size=len(self))
        else:
            sequences = np.arange(0, len(self.valid_sequences))

        indices = []

        for pi in sequences:
            # select subsequence
            si = self.valid_sequences[pi]
            sl = self.seq_lengths[si]
            max_sl = self.seq_length[si]
            if self.shuffle and self.seq_length:
                start_idx = int(self.rng.randint(0, sl - max_sl))
            else:
                start_idx = 0
            indices.append(self.seq_indices[si][start_idx] + torch.arange(max_sl))

        return iter(indices)


class SimpleBaseDataset(Dataset):
    """
    Base class to create vanilla datasets
    """

    def __init__(self, dtype=torch.float, device=None):

        self.device = device if device is not None else torch.device('cpu')
        self.dtype = dtype

    def __len__(self):
        raise NotImplementedError

    def to(self, device):
        raise NotImplementedError

    def cuda(self):
        """
        Set device to CUDA
        """
        self.device = torch.device('cuda')
        self.to(self.device)

    def cpu(self):
        """
        Set device to CPU
        """
        self.device = torch.device('cpu')
        self.to(self.device)

    def __len__(self):
        return self.inputs.shape[0]

    def __iter__(self):
        self.iter_idx = 0
        return self

    def __next__(self):
        if self.iter_idx == len(self):
            raise StopIteration
        res = self[self.iter_idx]
        self.iter_idx += 1
        return res

    def __getitem__(self, idx):
        return NotImplementedError


class BasisMixerDataset(SimpleBaseDataset):
    """
    Base class for building datasets
    """

    def __init__(self, X, Y,
                 is_piecewise=True,
                 dtype=torch.float, device=None):

        super().__init__(dtype=dtype, device=device)
        # Name of all pieces in the dataset
        self.pieces = list(all_performances.keys())
        # Boolean indicating if the Dataset iterates onsetwise or piece-wise
        self.is_piecewise = is_piecewise

        # Initialize list for containing beat periods
        # this attribute will be a concatenate all
        # beat periods of all pieces into a single 1D array
        self.beat_periods = []
        # List of score IOIs (will be concatenated into a 1D array)
        self.s_iois = []
        # List of the score features (depending on the type of score features
        # if any), will either be None, a single 1D or 2D array or a list of
        # arrays (all with the same length as self.beat_periods)
        self.s_features = []
        # Performed onsets (will be concatenated into a 1D array)
        self.p_onsets = []
        # Index of the pieces to which each row in self.beat_periods,
        # self.s_iois, self.s_features and self.p_onsets belong
        # to.
        self.piece_indices = []
        # A list of lists containing the indices of the onsets for each piece
        # (which indices in self.beat_periods et al. belong to which piece)
        self.onset_indices = []

        # Append information to the arrays
        k = 0
        for pi, pn in enumerate(all_performances):
            # unpack performance information
            po, sioi, bp, sf = all_performances[pn]

            # Check that the sizes are correct
            assert len(po) == len(bp)
            assert len(sioi) == len(po) - 1
            if sf is not None:
                if isinstance(sf, (list, tuple)):
                    for s in sf:
                        assert len(s) == len(po)
                else:
                    assert len(sf) == len(po)

            # Extend lists
            self.p_onsets.append(po)
            self.beat_periods.append(bp)
            # the score IOIs are padded with a 0 at the begining so that they
            # have the same size as beat_periods and p_onsets (so that it is
            # easy to have acces to the correct elements using the indices)
            self.s_iois.append(np.r_[0, sioi])
            self.s_features.append(sf)
            self.piece_indices.append(np.ones_like(bp) * pi)
            self.onset_indices.append(np.arange(k, k + len(bp)).astype(np.int))
            k += len(bp)

        # Get main dtypes
        if isinstance(self.dtype, (list, tuple)):
            # dtype for beat period, s_iois and p_onsets
            dt = self.dtype[0]
            # dtypes for the score features
            sf_dt = self.dtype[1:]
        else:
            # dtype for beat period, s_iois and p_onsets
            dt = self.dtype
            # Use the same dtype for all score features
            sf_dt = self.dtype

        # Concatenate arrays of indices
        self.piece_indices = np.hstack(self.piece_indices).astype(np.int)
        self.all_indices = np.hstack(self.onset_indices).astype(np.int)
        # For using with SubsequenceSampler
        self.seq_indices = self.onset_indices

        # Concatenate performance and score information and cast it as torch.Tensors
        self.p_onsets = torch.from_numpy(np.hstack(self.p_onsets)).type(dt)
        self.beat_periods = torch.from_numpy(np.hstack(self.beat_periods)).type(dt)
        self.s_iois = torch.from_numpy(np.hstack(self.s_iois)).type(dt)
        if all([sf is not None for sf in self.s_features]):
            if isinstance(sf, (list, tuple)):
                cat_fun = []
                for sf in self.s_features[0]:
                    if sf.ndim == 1:
                        cat_fun.append(np.hstack)
                    elif sf.ndim > 1:
                        cat_fun.append(np.vstack)

                if isinstance(sf_dt, (list, tuple)):
                    assert len(sf_dt) == len(cat_fun)
                else:
                    sf_dt = [sf_dt] * len(cat_fun)

                s_features = []
                for ci, cf in enumerate(cat_fun):
                    s_features.append(torch.from_numpy(
                        cf([f[ci] for f in self.s_features])).type(sf_dt[ci]))

            else:
                if self.s_features[0].ndim == 1:
                    self.s_features = torch.from_numpy(np.hstack(self.s_features)).type(sf_dt)
                else:
                    self.s_features = torch.from_numpy(np.vstack(self.s_features)).type(st_dt)

        else:
            self.s_features = None

    def __len__(self):
        if self.is_piecewise:
            return len(self.pieces)
        else:
            return len(self.all_indices)

    def to(self, device):
        self.device = device
        self.p_onsets = self.p_onsets.to(device)
        self.beat_periods = self.beat_periods.to(device)
        self.s_iois = self.s_iois.to(device)

        if self.s_features is not None:
            if isinstance(self.s_features, list):
                self.s_features = [sf.to(device) for sf in self.s_features]
            else:
                self.s_features = self.s_features.to(device)

    def __getitem__(self, idx):
        if self.is_piecewise:
            idxs = self.onset_indices[idx]
        else:
            idxs = idx

        output = (self.p_onsets[idxs],
                  self.s_iois[idxs],
                  self.beat_periods[idxs])
        if self.s_features is not None:
            if isinstance(self.s_features, (list, tuple)):
                output += (s[idxs] for s in self.s_features)
            else:
                output += (self.s_features[idxs],)

        return output


class BaseTrainer:
    def __init__(self, model, loss, optimizer, args):
        self.args = args
        self.n_gpu_use = args.n_gpu_use
        self.device, self.data_type = self._prepare_device()
        self.model = model.to(self.device)

        if self.n_gpu_use > 1:
            self.model = torch.nn.DataParallel(model, device_ids=self.device_ids)

        self.loss = loss
        self.optimizer = optimizer

        self.start_epoch = 1
        self.epochs = args.epochs
        self.save_period = args.save_period
        self.early_stop = args.early_stop
        self.mnt_best = inf
        self.not_improved_count = 0

        ensure_dir(self.args.save_dir)
        self.checkpoint_dir = Path(args.save_dir)
        self.loss_progress = {'loss': [], 'val_loss': []}

        if args.resume is not None:
            self.resume_path = Path(args.resume).resolve()
            self._resume_checkpoint()
        else:
            self._save_config()

    def _train_epoch(self, epoch):
        raise NotImplementedError

    def train(self):
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)

            self.loss_progress['loss'].append(result['loss'])
            try:
                self.loss_progress['val_loss'].append(result['val_loss'])
            except Exception:
                pass

            for k, v in result.items():
                print("  {}: {}".format(str(k), v))

            is_best = False
            try:
                improved = result['val_loss'] <= self.mnt_best

                if improved:
                    self.mnt_best = result['val_loss']
                    self.not_improved_count = 0
                    is_best = True
                else:
                    self.not_improved_count += 1

                if self.not_improved_count == self.early_stop:
                    print("Validation loss has not improved for {} epochs. Stop training.".format(self.early_stop))
                    break
            except Exception:
                pass

            is_period = epoch % self.save_period == 0
            if (is_period) or (is_best):
                self._save_checkpoint(epoch, is_period=is_period, is_best=is_best)

        print("The best loss: {}".format(self.mnt_best))
        print("Loss of training progress saved.")
        np.save(str(self.checkpoint_dir / 'loss_progress.npy'), self.loss_progress)

    def _save_checkpoint(self, epoch, is_period, is_best):
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best,
            'args': self.args
        }
        if is_period:
            filename = str(self.checkpoint_dir / 'checkpoint-epoch-{}.pth'.format(epoch))
            torch.save(state, filename)
            print("Saving checkpoint: %s ..." % filename)
        if is_best:
            filename = str(self.checkpoint_dir / 'model_best.pth')
            torch.save(state, filename)
            print("Saving current best: %s ..." % filename)

    def _save_config(self):
        with open(str(self.checkpoint_dir / 'cmd.txt'), 'w') as f:
            f.write(' '.join(sys.argv))

        orig_stdout = sys.stdout
        with open(str(self.checkpoint_dir / 'arch_summary.txt'), 'w') as f:
            sys.stdout = f
            print(self.model)
        sys.stdout = orig_stdout

        with open(str(self.checkpoint_dir / 'args.txt'), 'w') as f:
            f.write(str(self.args))

    def _prepare_device(self):
        n_gpu = torch.cuda.device_count()
        if self.n_gpu_use > 0 and n_gpu == 0:
            warnings.warn("Warning: There is no GPU available, training will be performed on CPU.")
            self.n_gpu_use = 0
        if self.n_gpu_use > n_gpu:
            warnings.warn("Warning: N. of GPU specified to use: %d, only %d are availabe." % (self.n_gpu_use, n_gpu))
            self.n_gpu_use = n_gpu

        device = torch.device('cuda:0' if self.n_gpu_use > 0 else 'cpu')
        if device.type == 'cuda':
            default_dtype = 'torch.cuda.FloatTensor'
        else:
            default_dtype = 'torch.FloatTensor'
        # torch.set_default_tensor_type(default_dtype)

        return device, default_dtype

    def _resume_checkpoint(self):
        resume_path = str(self.resume_path)
        print("Loading checkpoint: %s ..." % resume_path)
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        print("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))


class RecurrentTrainer(BaseTrainer):
    def __init__(self, model, loss, optimizer, args, data_loader,
                 valid_data_loader=None, lr_scheduler=None, fold=None,
                 vae_model=None, loss_type='weighted_cross_entropy'):
        # configure the folder name to save models
        if args.subfolder:
            dataset = str(Path(args.path_data).stem)
            hidden_size = str(args.hidden_size)
            n_layer = str(args.n_layers)
            n_dropout = str(args.dropout)
            loss_mode = 'weighted' if args.loss == 'weighted_cross_entropy'\
                else 'mean'
            try:
                vae = str(Path(args.vae).stem)
            except:
                vae = ''
            note = str(args.note) if args.note != 'None' else ''
            subfolder = '-'.join([dataset, '_'.join(['loss', loss_mode]),
                                  '_'.join(['hs', hidden_size]),
                                  '_'.join(['nl', n_layer]),
                                  '_'.join(['do', n_dropout])])
            if len(vae) != 0:
                subfolder = '-'.join([subfolder, vae])
            if len(note) != 0:
                subfolder = '-'.join([subfolder, note])

            args.save_dir = Path(args.save_dir) / subfolder

        super(TonalRnnTrainer, self).__init__(model, loss, optimizer, args)

        if args.cross_valid > 1:
            dir_fold = 'fold-%d' % fold
            result_subfolder = str(Path(self.args.save_dir) / dir_fold)
            ensure_dir(result_subfolder)
            self.checkpoint_dir = Path(result_subfolder)

        self.args = args
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.loss_type = loss_type
        self.vae = vae_model.to(self.device) if vae_model\
            is not None else None

        if self.do_validation:
            np.save(str(self.checkpoint_dir / 'valid_index'),
                    valid_data_loader.sampler.indices)

    def _compute_loss(self, x, y, output, mask):
        loss_args = [output, y, mask]

        if self.loss_type == 'weighted_cross_entropy':
            loss_weights = w_io(x, y)  # add and play with extra arguments
            loss_args.append(loss_weights)

        loss = self.loss(*loss_args)

        return loss

    def _vae_encode(self, x):
        if self.vae is not None:
            x_reshape = x.view(-1, x.shape[-1])
            _, _, z = self.vae.encode(x_reshape)
            z = z.view(self.args.batch_size, -1, self.args.input_size)
        else:
            z = x
        return z

    def _vae_decode(self, z):
        if self.vae is not None:
            z = z.view(-1, z.shape[-1])
            x = self.vae.decode(z)
            x = x.view(self.args.batch_size, -1, x.shape[-1])
        else:
            x = z
        return x

    def _train_epoch(self, epoch):
        self.model.train()

        total_loss = 0
        bar = tqdm(enumerate(self.data_loader), total=len(self.data_loader))
        # for batch_idx, (x, y) in enumerate(self.data_loader):
        for batch_idx, (x, y, mask, origin_len) in bar:
            x, y = x.type(self.data_type), y.type(self.data_type)

            self.optimizer.zero_grad()
            z = self._vae_encode(x)
            output, h = self.model(z, origin_len)
            output = self._vae_decode(output)

            loss = self._compute_loss(x, y, output, mask)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            bar.set_description("epoch: {}/{}".format(epoch, self.epochs))

        log = {
            'loss': total_loss / len(self.data_loader)
        }

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log = {**log, **val_log}

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return log

    def _valid_epoch(self, epoch):
        self.model.eval()

        total_val_loss = 0
        with torch.no_grad():
            for batch_idx, (x, y, mask, origin_len) in enumerate(self.valid_data_loader):
                x, y = x.type(self.data_type), y.type(self.data_type)

                z = self._vae_encode(x)
                output, h = self.model(z, origin_len)
                output = self._vae_decode(output)

                loss = self._compute_loss(x, y, output, mask)

                total_val_loss += loss.item()

        return {
            'val_loss': total_val_loss / len(self.valid_data_loader)
        }
