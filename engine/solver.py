import os
import sys
import json
import time
import torch
import numpy as np
import torch.nn.functional as F

from pathlib import Path
from tqdm.auto import tqdm
from ema_pytorch import EMA
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from Utils.io_utils import instantiate_from_config, get_model_parameters_info


sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

def cycle(dl):
    while True:
        for data in dl:
            yield data


class Trainer(object):
    def __init__(self, config, args, model, dataloader, logger=None, val_dataloader=None):
        super().__init__()
        self.model = model
        self.device = self.model.betas.device
        self.train_num_steps = config['solver']['max_epochs']
        self.gradient_accumulate_every = config['solver']['gradient_accumulate_every']
        self.save_cycle = config['solver']['save_cycle']
        self.dl = cycle(dataloader['dataloader'])
        self.dataloader = dataloader['dataloader']
        self.step = 0
        self.milestone = 0
        self.args, self.config = args, config
        self.logger = logger

        # ---- validation-loss tracking (used by HPO) ---- #
        self.val_dataloader = val_dataloader
        self.val_seed = int(config['solver'].get('val_seed', 2024))
        self.val_num_repeats = int(config['solver'].get('val_num_repeats', 3))
        self.best_val_loss = float('inf')
        self.best_val_milestone = -1
        self.train_loss_history = []  # list of (step, train_loss)
        self.val_loss_history = []    # list of (step, val_loss)

        self.results_folder = Path(config['solver']['results_folder'] + f'_{model.seq_length}')
        os.makedirs(self.results_folder, exist_ok=True)

        start_lr = config['solver'].get('base_lr', 1.0e-4)
        ema_decay = config['solver']['ema']['decay']
        ema_update_every = config['solver']['ema']['update_interval']

        self.opt = Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=start_lr, betas=[0.9, 0.96])
        self.ema = EMA(self.model, beta=ema_decay, update_every=ema_update_every).to(self.device)

        sc_cfg = config['solver']['scheduler']
        sc_cfg['params']['optimizer'] = self.opt
        self.sch = instantiate_from_config(sc_cfg)

        if self.logger is not None:
            self.logger.log_info(str(get_model_parameters_info(self.model)))
        self.log_frequency = 100

    def save(self, milestone, verbose=False):
        if self.logger is not None and verbose:
            self.logger.log_info('Save current model to {}'.format(str(self.results_folder / f'checkpoint-{milestone}.pt')))
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema.state_dict(),
            'opt': self.opt.state_dict(),
        }
        torch.save(data, str(self.results_folder / f'checkpoint-{milestone}.pt'))
    
    def save_classifier(self, milestone, verbose=False):
        if self.logger is not None and verbose:
            self.logger.log_info('Save current classifer to {}'.format(str(self.results_folder / f'ckpt_classfier-{milestone}.pt')))
        data = {
            'step': self.step_classifier,
            'classifier': self.classifier.state_dict()
        }
        torch.save(data, str(self.results_folder / f'ckpt_classfier-{milestone}.pt'))

    @torch.no_grad()
    def evaluate_val_loss(self):
        """Average diffusion (val) loss of the EMA model over the val set.

        The diffusion loss samples a random timestep / noise per call, so we fix
        the seed and average over a few passes to make the value comparable
        across checkpoints and across HPO configs. No histogram metric is used.
        """
        if self.val_dataloader is None:
            return None
        was_training = self.ema.ema_model.training
        self.ema.ema_model.eval()
        rng_state = torch.get_rng_state()
        cuda_rng_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        torch.manual_seed(self.val_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.val_seed)

        total, count = 0.0, 0
        for _ in range(self.val_num_repeats):
            for data in self.val_dataloader:
                data = data.to(self.device)
                loss = self.ema.ema_model(data, target=data)
                total += loss.item() * data.shape[0]
                count += data.shape[0]

        torch.set_rng_state(rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state_all(cuda_rng_state)
        if was_training:
            self.ema.ema_model.train()
        return total / max(count, 1)

    def _write_val_metrics(self, current_val_loss):
        metrics = {
            'best_val_loss': self.best_val_loss,
            'best_val_milestone': self.best_val_milestone,
            'last_val_loss': current_val_loss,
            'last_milestone': self.milestone,
            'last_step': self.step,
        }
        with open(str(self.results_folder / 'val_metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)

    def load(self, milestone, verbose=False):
        if self.logger is not None and verbose:
            self.logger.log_info('Resume from {}'.format(str(self.results_folder / f'checkpoint-{milestone}.pt')))
        device = self.device
        name = 'checkpoint-best.pt' if milestone == 'best' else f'checkpoint-{milestone}.pt'
        data = torch.load(str(self.results_folder / name), map_location=device)
        self.model.load_state_dict(data['model'])
        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        self.ema.load_state_dict(data['ema'])
        self.milestone = milestone

    def load_classifier(self, milestone, verbose=False):
        if self.logger is not None and verbose:
            self.logger.log_info('Resume from {}'.format(str(self.results_folder / f'ckpt_classfier-{milestone}.pt')))
        device = self.device
        data = torch.load(str(self.results_folder / f'ckpt_classfier-{milestone}.pt'), map_location=device)
        self.classifier.load_state_dict(data['classifier'])
        self.step_classifier = data['step']
        self.milestone_classifier = milestone

    def train(self):
        device = self.device
        step = 0
        if self.logger is not None:
            tic = time.time()
            self.logger.log_info('{}: start training...'.format(self.args.name), check_primary=False)

        with tqdm(initial=step, total=self.train_num_steps) as pbar:
            while step < self.train_num_steps:
                total_loss = 0.
                for _ in range(self.gradient_accumulate_every):
                    data = next(self.dl).to(device)
                    loss = self.model(data, target=data)
                    loss = loss / self.gradient_accumulate_every
                    loss.backward()
                    total_loss += loss.item()

                pbar.set_description(f'loss: {total_loss:.6f}')

                clip_grad_norm_(self.model.parameters(), 1.0)
                self.opt.step()
                self.sch.step(total_loss)
                self.opt.zero_grad()
                self.step += 1
                step += 1
                self.ema.update()

                with torch.no_grad():
                    if self.step != 0 and self.step % self.save_cycle == 0:
                        self.milestone += 1
                        self.save(self.milestone)
                        # self.logger.log_info('saved in {}'.format(str(self.results_folder / f'checkpoint-{self.milestone}.pt')))

                        val_loss = self.evaluate_val_loss()
                        if val_loss is not None:
                            self.val_loss_history.append((self.step, val_loss))
                            if val_loss < self.best_val_loss:
                                self.best_val_loss = val_loss
                                self.best_val_milestone = self.milestone
                                self.save('best')
                            self._write_val_metrics(val_loss)
                            if self.logger is not None:
                                self.logger.add_scalar(tag='val/loss', scalar_value=val_loss, global_step=self.step)
                                self.logger.log_info(
                                    'step {}: val_loss={:.6f} (best={:.6f} @milestone {})'.format(
                                        self.step, val_loss, self.best_val_loss, self.best_val_milestone))

                    if self.step % self.log_frequency == 0:
                        self.train_loss_history.append((self.step, total_loss))

                    if self.logger is not None and self.step % self.log_frequency == 0:
                        # info = '{}: train'.format(self.args.name)
                        # info = info + ': Epoch {}/{}'.format(self.step, self.train_num_steps)
                        # info += ' ||'
                        # info += '' if loss_f == 'none' else ' Fourier Loss: {:.4f}'.format(loss_f.item())
                        # info += '' if loss_r == 'none' else ' Reglarization: {:.4f}'.format(loss_r.item())
                        # info += ' | Total Loss: {:.6f}'.format(total_loss)
                        # self.logger.log_info(info)
                        self.logger.add_scalar(tag='train/loss', scalar_value=total_loss, global_step=self.step)

                pbar.update(1)

        print('training complete')
        self._save_loss_curve()
        if self.logger is not None:
            self.logger.log_info('Training done, time: {:.2f}'.format(time.time() - tic))

    def _save_loss_curve(self):
        """Save a per-run train/val loss curve as PDF into the results folder."""
        if not self.train_loss_history and not self.val_loss_history:
            return
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except Exception as e:  # matplotlib missing / headless issue
            if self.logger is not None:
                self.logger.log_info(f'Skip loss curve (matplotlib unavailable): {e}')
            return

        fig, ax1 = plt.subplots(figsize=(8, 5))
        if self.train_loss_history:
            ts, tl = zip(*self.train_loss_history)
            ax1.plot(ts, tl, color='tab:blue', label='train loss', linewidth=1.0)
        ax1.set_xlabel('step')
        ax1.set_ylabel('train loss', color='tab:blue')
        ax1.tick_params(axis='y', labelcolor='tab:blue')

        if self.val_loss_history:
            ax2 = ax1.twinx()
            vs, vl = zip(*self.val_loss_history)
            ax2.plot(vs, vl, color='tab:red', marker='o', markersize=3,
                     label='val loss', linewidth=1.0)
            ax2.set_ylabel('val loss', color='tab:red')
            ax2.tick_params(axis='y', labelcolor='tab:red')
            if self.best_val_milestone >= 0:
                ax2.axhline(self.best_val_loss, color='tab:red', linestyle='--',
                            linewidth=0.8, alpha=0.6)

        name = getattr(self.args, 'name', 'run')
        plt.title(f'{name}: best val={self.best_val_loss:.5f} @milestone {self.best_val_milestone}')
        fig.tight_layout()
        out = str(self.results_folder / 'loss_curve.pdf')
        fig.savefig(out)
        plt.close(fig)
        if self.logger is not None:
            self.logger.log_info(f'Saved loss curve to {out}')
        print(f'saved loss curve to {out}')

    def sample(self, num, size_every, shape=None, model_kwargs=None, cond_fn=None):
        if self.logger is not None:
            tic = time.time()
            self.logger.log_info('Begin to sample...')
        samples = np.empty([0, shape[0], shape[1]])
        num_cycle = int(num // size_every) + 1

        for _ in range(num_cycle):
            sample = self.ema.ema_model.generate_mts(batch_size=size_every, model_kwargs=model_kwargs, cond_fn=cond_fn)
            samples = np.row_stack([samples, sample.detach().cpu().numpy()])
            torch.cuda.empty_cache()

        if self.logger is not None:
            self.logger.log_info('Sampling done, time: {:.2f}'.format(time.time() - tic))
        return samples

    def restore(self, raw_dataloader, shape=None, coef=1e-1, stepsize=1e-1, sampling_steps=50):
        if self.logger is not None:
            tic = time.time()
            self.logger.log_info('Begin to restore...')
        model_kwargs = {}
        model_kwargs['coef'] = coef
        model_kwargs['learning_rate'] = stepsize
        samples = np.empty([0, shape[0], shape[1]])
        reals = np.empty([0, shape[0], shape[1]])
        masks = np.empty([0, shape[0], shape[1]])

        for idx, (x, t_m) in enumerate(raw_dataloader):
            x, t_m = x.to(self.device), t_m.to(self.device)
            if sampling_steps == self.model.num_timesteps:
                sample = self.ema.ema_model.sample_infill(shape=x.shape, target=x*t_m, partial_mask=t_m,
                                                          model_kwargs=model_kwargs)
            else:
                sample = self.ema.ema_model.fast_sample_infill(shape=x.shape, target=x*t_m, partial_mask=t_m, model_kwargs=model_kwargs,
                                                               sampling_timesteps=sampling_steps)

            samples = np.row_stack([samples, sample.detach().cpu().numpy()])
            reals = np.row_stack([reals, x.detach().cpu().numpy()])
            masks = np.row_stack([masks, t_m.detach().cpu().numpy()])
        
        if self.logger is not None:
            self.logger.log_info('Imputation done, time: {:.2f}'.format(time.time() - tic))
        return samples, reals, masks
        # return samples

    def forward_sample(self, x_start):
       b, c, h = x_start.shape
       noise = torch.randn_like(x_start, device=self.device)
       t = torch.randint(0, self.model.num_timesteps, (b,), device=self.device).long()
       x_t = self.model.q_sample(x_start=x_start, t=t, noise=noise).detach()
       return x_t, t

    def train_classfier(self, classifier):
        device = self.device
        step = 0
        self.milestone_classifier = 0
        self.step_classifier = 0
        dataloader = self.dataloader
        dataloader.dataset.shift_period('test')
        dataloader = cycle(dataloader)

        self.classifier = classifier
        self.opt_classifier = Adam(filter(lambda p: p.requires_grad, self.classifier.parameters()), lr=5.0e-4)
        
        if self.logger is not None:
            tic = time.time()
            self.logger.log_info('{}: start training classifier...'.format(self.args.name), check_primary=False)
        
        with tqdm(initial=step, total=self.train_num_steps) as pbar:
            while step < self.train_num_steps:
                total_loss = 0.
                for _ in range(self.gradient_accumulate_every):
                    x, y = next(dataloader)
                    x, y = x.to(device), y.to(device)
                    x_t, t = self.forward_sample(x)
                    logits = classifier(x_t, t)
                    loss = F.cross_entropy(logits, y)
                    loss = loss / self.gradient_accumulate_every
                    loss.backward()
                    total_loss += loss.item()

                pbar.set_description(f'loss: {total_loss:.6f}')

                self.opt_classifier.step()
                self.opt_classifier.zero_grad()
                self.step_classifier += 1
                step += 1

                with torch.no_grad():
                    if self.step_classifier != 0 and self.step_classifier % self.save_cycle == 0:
                        self.milestone_classifier += 1
                        self.save(self.milestone_classifier)
                                            
                    if self.logger is not None and self.step_classifier % self.log_frequency == 0:
                        self.logger.add_scalar(tag='train/loss', scalar_value=total_loss, global_step=self.step)

                pbar.update(1)

        print('training complete')
        if self.logger is not None:
            self.logger.log_info('Training done, time: {:.2f}'.format(time.time() - tic))

        # return classifier

