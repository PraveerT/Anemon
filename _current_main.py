"""Minimal trainer for PGCNet on NVGesture.

Loads a yaml config, instantiates the model + dataloader + optimizer, runs
the train/eval loop with optional auxiliary loss support, saves checkpoints.

No telegram, no oracle/fusion telemetry, no shuffle-mix, no qcc scheduling,
no branch-specific losses, no sample weighting. Just training.
"""
import argparse
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import sys
import yaml
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import get_parser, import_class, GpuDataParallel, Optimizer, Recorder, Stat, RandomState
from utils.gpu_augment import GpuAugmentor


def dynamic_pts_size(epoch, arg):
    """Linear ramp from 48 to pts_ramp_target over ep [0, pts_ramp_epochs), constant
    after. Defaults (target 172, epochs 50) reproduce the original schedule."""
    target = int(getattr(arg, 'pts_ramp_target', None) or 172)
    ramp_ep = int(getattr(arg, 'pts_ramp_epochs', None) or 50)
    if epoch < ramp_ep:
        return int(48 + (target - 48) * (epoch / ramp_ep))
    return target


class GPUBatcher:
    """Yields batches by indexing a GPU-resident tensor -- no DataLoader workers, no
    IPC, no per-batch host->device copy. The whole set is preloaded + pinned in RAM
    (NvidiaLoader._preload), so moving it once to the GPU removes the multiprocessing
    overhead that dominates wall-clock when the model is small. Drop-in for the train
    DataLoader: yields (image, label), both already on the GPU."""

    def __init__(self, x, y, batch_size, shuffle=True):
        self.x = x
        self.y = y
        self.bs = int(batch_size)
        self.shuffle = shuffle
        self.n = x.shape[0]

    def __len__(self):
        return (self.n + self.bs - 1) // self.bs

    def __iter__(self):
        if self.shuffle:
            idx = torch.randperm(self.n, device=self.x.device)
        else:
            idx = torch.arange(self.n, device=self.x.device)
        for i in range(0, self.n, self.bs):
            j = idx[i:i + self.bs]
            yield self.x[j], self.y[j]      # advanced indexing copies -> safe to augment in place


class Processor:
    def __init__(self, arg):
        self.arg = arg
        self.save_arg()
        if self.arg.random_fix:
            self.rng = RandomState(seed=self.arg.random_seed)
        self.device = GpuDataParallel()
        self.device.set_device(self.arg.device)
        self.recoder = Recorder(self.arg.work_dir, self.arg.print_log)
        self.data_loader = {}
        self.topk = (1, 5)
        self.stat = Stat(self.arg.model_args['num_classes'], self.topk)
        self.model, self.optimizer = self.Loading()
        self.loss = self.criterion()
        self.augmentor = GpuAugmentor()
        self.device.model_to_device(self.augmentor)
        self.best_accuracy = float(getattr(self.arg, "min_best_acc", 0.0))
        self.use_static_pts = ('--pts-size' in sys.argv) or (
            not getattr(self.arg, 'dynamic_pts_size', True)
        )

    # ---------------------------------------------------------------- loss
    def criterion(self):
        loss = nn.CrossEntropyLoss(label_smoothing=0.1, reduction='none')
        return self.device.criterion_to_device(loss)

    # ---------------------------------------------------------------- model setup
    def Loading(self):
        self.recoder.print_log('Loading model')
        model_class = import_class(self.arg.model)
        model = model_class(**self.arg.model_args)
        if self.arg.weights:
            self._load_weights(model, self.arg.weights)
        model = self.device.model_to_device(model)
        optimizer = Optimizer(model, self.arg.optimizer_args)
        if self.arg.resume:
            self._resume_optimizer_state(optimizer)
        self.recoder.print_log('Loading model finished.')
        self.load_data()
        return model, optimizer

    def _load_weights(self, model, weights_path):
        self.recoder.print_log(f'Initializing model weights from {weights_path}.')
        payload = torch.load(weights_path, map_location='cpu')
        state = payload['model_state_dict'] if (
            isinstance(payload, dict) and 'model_state_dict' in payload
        ) else payload
        # Normalize 'module.' prefix from DataParallel checkpoints
        state = {(k[7:] if k.startswith('module.') else k): v for k, v in state.items()}
        res = model.load_state_dict(state, strict=self.arg.strict_load)
        if res.missing_keys:
            self.recoder.print_log(f'  missing keys: {len(res.missing_keys)}')
        if res.unexpected_keys:
            self.recoder.print_log(f'  unexpected keys: {len(res.unexpected_keys)}')

    def _resume_optimizer_state(self, optimizer):
        ckpt = torch.load(self.arg.weights, map_location='cpu')
        if not isinstance(ckpt, dict):
            return
        if 'optimizer_state_dict' in ckpt:
            try:
                optimizer.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            except (ValueError, RuntimeError) as e:
                self.recoder.print_log(f'optimizer state restore skipped: {e}')
        # Skip loading scheduler_state_dict: ckpts trained with a different
        # scheduler structure (e.g. 2-phase -> 3-phase) silently fail to load
        # and the new scheduler then starts at step 0, producing base_lr instead
        # of the intended phase. Build it fresh, then advance to start_epoch.
        if 'epoch' in ckpt:
            self.arg.optimizer_args['start_epoch'] = ckpt['epoch'] + 1
            self.recoder.print_log(
                f'Resuming from checkpoint: epoch {self.arg.optimizer_args["start_epoch"]}'
            )
            for _ in range(self.arg.optimizer_args['start_epoch']):
                optimizer.scheduler.step()
            cur_lr = optimizer.optimizer.param_groups[0]['lr']
            self.recoder.print_log(
                f'Scheduler advanced to epoch {self.arg.optimizer_args["start_epoch"]}: lr={cur_lr:.2e}'
            )

    # ---------------------------------------------------------------- data
    def load_data(self):
        self.recoder.print_log('Loading data')
        dataset_class = import_class(self.arg.dataloader)
        if self.arg.phase == 'train':
            if os.environ.get('GPU_RESIDENT', '0') == '1':
                ds = dataset_class(**self.arg.train_loader_args)
                x = self.device.data_to_device(ds.tensor)          # whole train set -> GPU once
                y = self.device.data_to_device(ds.labels_tensor)
                self.data_loader['train'] = GPUBatcher(x, y, self.arg.batch_size, shuffle=True)
                self.recoder.print_log(f'GPU-resident train set: {tuple(x.shape)} on {x.device}')
            else:
                self.data_loader['train'] = torch.utils.data.DataLoader(
                    dataset_class(**self.arg.train_loader_args),
                    batch_size=self.arg.batch_size, shuffle=True,
                    num_workers=4, pin_memory=False, persistent_workers=True,
                )
        self.data_loader['test'] = torch.utils.data.DataLoader(
            dataset_class(**self.arg.test_loader_args),
            batch_size=self.arg.test_batch_size, shuffle=False,
            num_workers=4, pin_memory=False, persistent_workers=True,
        )
        self.recoder.print_log('Loading data finished.')

    # ---------------------------------------------------------------- training
    def train(self, epoch):
        self.model.train()
        self.augmentor.train()
        model_ref = self.model.module if hasattr(self.model, 'module') else self.model

        # pts_size scheduling
        alt = os.environ.get('PTS_ALT', '')
        if alt:
            sizes = [int(s) for s in alt.split(',')]
            pts_size = sizes[epoch % len(sizes)]
            self.recoder.print_log(
                f'Training epoch: {epoch + 1} | pts_size: {pts_size} (alt {sizes})'
            )
        elif self.use_static_pts:
            pts_size = self.arg.pts_size
            tag = '--pts-size' if '--pts-size' in sys.argv else 'config'
            self.recoder.print_log(
                f'Training epoch: {epoch + 1} | pts_size: {pts_size} (static from {tag})'
            )
        else:
            pts_size = dynamic_pts_size(epoch, self.arg)
            self.recoder.print_log(
                f'Training epoch: {epoch + 1} | pts_size: {pts_size} (dynamic)'
            )
        model_ref.pts_size = pts_size
        self.arg.model_args['pts_size'] = pts_size

        loader = self.data_loader['train']
        loss_value = []
        correct, total = 0, 0
        aux_loss_value, aux_correct = [], 0   # supervised aux-CE (skew auxce head)
        model_aux_loss_value = []             # generic model.aux_loss (CTE, SSL, etc.)
        self.recoder.timer_reset()
        cur_lr = [g['lr'] for g in self.optimizer.optimizer.param_groups]

        bar = tqdm(enumerate(loader), total=len(loader), desc=f'Epoch {epoch + 1}', leave=False)
        for batch_idx, data in bar:
            self.recoder.record_timer('dataloader')
            image = self.device.data_to_device(data[0])
            label = self.device.data_to_device(data[1])
            self.recoder.record_timer('device')
            image = self.augmentor(image)

            # --- Mixup (input-level, soft labels); env MIXUP_ALPHA (0=off) ---
            _mix_a = float(__import__('os').environ.get('MIXUP_ALPHA', '0.0'))
            _lam, _label_b = 1.0, label
            if _mix_a and _mix_a > 0:
                _lam = float(np.random.beta(_mix_a, _mix_a))
                _perm = torch.randperm(image.shape[0], device=image.device)
                image = _lam * image + (1.0 - _lam) * image[_perm]
                _label_b = label[_perm]

            if hasattr(model_ref, 'current_labels'):
                model_ref.current_labels = label if _lam == 1.0 else None
            output = self.model(image)
            self.recoder.record_timer('forward')

            loss = (_lam * torch.mean(self.loss(output, label))
                    + (1.0 - _lam) * torch.mean(self.loss(output, _label_b)))
            aux = getattr(model_ref, 'aux_loss', None)
            if aux is not None:
                loss = loss + aux
                model_aux_loss_value.append(float(aux.detach().item()))
            # supervised aux-CE: deep-supervise a decoupled aux head exposed via
            # model.latest_aux_logits (keeps it alive vs the self-destruct under decay)
            aux_logits = getattr(model_ref, 'latest_aux_logits', None)
            if aux_logits is not None:
                _w = getattr(model_ref, 'aux_ce_weight', 0.0)
                aux_ce = torch.mean(self.loss(aux_logits, label)) * _w
                loss = loss + aux_ce
                aux_loss_value.append(aux_ce.item())
                aux_correct += int(aux_logits.argmax(1).eq(label).sum().item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.recoder.record_timer('backward')

            loss_value.append(loss.item())
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(label.view_as(pred)).sum().item()
            total += label.size(0)

            bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100. * correct / total:.2f}%',
                'LR': f'{cur_lr[0]:.6f}',
                'PTS': pts_size,
            })

            if batch_idx % self.arg.log_interval == 0:
                self.recoder.print_log(
                    f'\tEpoch: {epoch}, Batch({batch_idx}/{len(loader)}) done. '
                    f'Loss: {loss.item():.8f}  lr:{cur_lr[0]:f}'
                )
                self.recoder.print_time_statistics()

        train_acc = 100. * correct / max(1, total)
        train_loss = float(np.mean(loss_value)) if loss_value else 0.0
        self.recoder.print_log(f'\tMean training acc:  {train_acc:.4f}%.')
        self.recoder.print_log(f'\tMean training loss: {train_loss:.10f}.')
        if model_aux_loss_value:
            self.recoder.print_log(f'\tMean model aux loss: {float(np.mean(model_aux_loss_value)):.6f}')
        if aux_loss_value:
            self.recoder.print_log(f'\tMean auxiliary loss: {float(np.mean(aux_loss_value)):.6f}')
            self.recoder.print_log(f'\tMean auxiliary acc: {100. * aux_correct / max(1, total):.4f}%')
        if hasattr(model_ref, 'log_hooks'):
            self.recoder.print_log('\t' + model_ref.log_hooks())

        self.optimizer.scheduler.step()
        return train_acc, train_loss

    # ---------------------------------------------------------------- evaluation
    def eval(self, loader_name=('test',)):
        self.model.eval()
        self.stat.reset_statistic()
        eval_loss_values = []
        n_samples = 0
        test_logits, test_labels, test_sigs = [], [], []
        eval_ref = self.model.module if hasattr(self.model, 'module') else self.model
        aux_te_correct, aux_te_total = 0, 0
        with torch.no_grad():
            for name in loader_name:
                loader = self.data_loader[name]
                self.stat.test_size = len(loader.dataset)
                for data in loader:
                    image = self.device.data_to_device(data[0])
                    label = self.device.data_to_device(data[1])
                    output = self.model(image)
                    _al = getattr(eval_ref, 'latest_aux_logits', None)
                    if _al is not None and name == 'test':
                        aux_te_correct += int(_al.argmax(1).eq(label).sum().item())
                        aux_te_total += label.size(0)
                    loss = torch.mean(self.loss(output, label))
                    eval_loss_values.append(loss.item() * label.size(0))
                    n_samples += label.size(0)
                    self.stat.update_accuracy(output.data.cpu(), label.cpu(), topk=self.topk)
                    if name == 'test':
                        test_logits.append(output.detach().cpu().numpy())
                        test_labels.append(label.detach().cpu().numpy())
                        if len(data) >= 3:
                            test_sigs.extend([str(s) for s in data[2]])
        mean_loss = sum(eval_loss_values) / max(1, n_samples)
        self.recoder.print_log(f'mean loss: {mean_loss}')
        if aux_te_total:
            self.recoder.print_log(f'\tAux test acc: {100. * aux_te_correct / aux_te_total:.4f}%')
        if test_logits:
            try:
                import re as _re
                L = np.concatenate(test_logits)
                Y = np.concatenate(test_labels)

                def _sig(s):
                    m = _re.search(r'class_(\d+)/subject(\d+)_r(\d+)', s)
                    return (f'class_{m.group(1)}/subject{m.group(2)}_r{m.group(3)}'
                            if m else s)

                S = np.array([_sig(s) for s in test_sigs]) if test_sigs                     else np.array([str(i) for i in range(len(Y))])
                out_path = os.path.join(self.arg.work_dir, 'test_logits')
                ep_marker = np.array([self._eval_epoch_marker], dtype=np.int64)                     if hasattr(self, '_eval_epoch_marker') else np.array([-1], dtype=np.int64)
                np.savez(out_path, logits=L, labels=Y, sigs=S, epoch=ep_marker)
            except Exception as e:
                self.recoder.print_log(f'test_logits dump skipped: {e}')

    # ---------------------------------------------------------------- main loop
    def start(self):
        if self.arg.phase == 'train':
            for epoch in range(self.arg.optimizer_args['start_epoch'], self.arg.num_epoch):
                eval_interval = int(getattr(self.arg, 'eval_interval', 0) or (
                    10 if (epoch + 1) < 75 else 1
                ))
                save_interval = self.arg.save_interval if (epoch + 1) < 100 else 1
                save_now = (epoch + 1) % save_interval == 0 or (epoch + 1) == self.arg.num_epoch
                eval_now = (epoch + 1) % eval_interval == 0 or (epoch + 1) == self.arg.num_epoch

                train_acc, train_loss = self.train(epoch)
                if save_now:
                    self.save_model(epoch, self.model, self.optimizer,
                                    f'{self.arg.work_dir}/epoch{epoch + 1}_model.pt')
                if eval_now:
                    self._eval_epoch_marker = epoch + 1
                    self.eval(loader_name=['test'])
                    self.print_inf_log(epoch + 1, 'Test', train_acc, train_loss)
        elif self.arg.phase == 'test':
            if not self.arg.weights:
                raise ValueError('phase=test requires --weights')
            self.recoder.print_log(f'Evaluating: {self.arg.weights}')
            self.eval(loader_name=['test'])
            self.print_inf_log(0, 'Test')

    # ---------------------------------------------------------------- logging
    def print_inf_log(self, epoch, mode, train_acc=None, train_loss=None):
        static = self.stat.show_accuracy(f'{self.arg.work_dir}/{mode}_confusion_mat')
        prec1 = static[str(self.topk[0])] / self.stat.test_size * 100
        prec5 = static[str(self.topk[1])] / self.stat.test_size * 100
        self.recoder.print_log(
            f'Epoch {epoch}, {mode}, Evaluation: prec1 {prec1:.4f}, prec5 {prec5:.4f}'
        )
        self.recoder.print_log(f'Confusion Matrix (epoch {epoch}, {mode}):')
        cm = self.stat.confusion_matrix
        n_correct = int(cm.diagonal().sum())
        n_total = int(cm.sum())
        self.recoder.print_log(f'  Total Correct: {n_correct}.0/{n_total}.0')
        overall = 100. * n_correct / max(1, n_total)
        self.recoder.print_log(f'  Overall Accuracy: {overall:.2f}%')
        if prec1 > self.best_accuracy:
            self.best_accuracy = float(prec1)
            best_path = f'{self.arg.work_dir}/best_model.pt'
            self.save_model(epoch, self.model, self.optimizer, best_path)
            self.recoder.print_log(
                f'  Saved new best to {best_path} at prec1={prec1:.2f}% (prec1={prec1:.2f}%)'
            )

    # ---------------------------------------------------------------- checkpoint
    def save_model(self, epoch, model, optimizer, save_path):
        model_state = (model.module if hasattr(model, 'module') else model).state_dict()
        torch.save({
            'epoch': epoch,
            'model_state_dict': model_state,
            'optimizer_state_dict': optimizer.optimizer.state_dict(),
            'scheduler_state_dict': optimizer.scheduler.state_dict(),
        }, save_path)

    def save_arg(self):
        os.makedirs(self.arg.work_dir, exist_ok=True)
        with open(f'{self.arg.work_dir}/config.yaml', 'w') as f:
            yaml.dump(vars(self.arg), f, default_flow_style=False)
        self.recoder = self.recoder if hasattr(self, 'recoder') else None
        if hasattr(self, 'recoder') and self.recoder is not None:
            self.recoder.print_log(f'Parameters:\n{vars(self.arg)}')


if __name__ == '__main__':
    sparser = get_parser()
    p = sparser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
            default_arg = yaml.load(f, Loader=yaml.FullLoader)
        keys = vars(p).keys()
        for k in default_arg.keys():
            if k not in keys:
                raise ValueError(f'unrecognized config key: {k}')
        sparser.set_defaults(**default_arg)
    args = sparser.parse_args()
    Processor(args).start()
