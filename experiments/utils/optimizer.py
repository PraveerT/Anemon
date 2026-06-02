import pdb
import torch
import numpy as np
import torch.optim as optim


class Optimizer(object):
    def __init__(self, model, optim_dict):
        self.optim_dict = optim_dict
        self.spatial_prefix = self.optim_dict.get('spatial_prefix', 'spatial_branch').strip('.')
        self.temporal_prefix = self.optim_dict.get('temporal_prefix', 'temporal_branch').strip('.')

        # Separate parameter groups for different learning rates
        spatial_params = []
        temporal_params = []
        fusion_params = []
        no_decay_params = []
        low_lr_params = []

        fusion_lr_mult = self.optim_dict.get('fusion_lr_mult', 1.0)
        low_lr_mult = self.optim_dict.get('low_lr_mult', 0.05)

        # Optional: model may expose .no_decay_param_names() to exempt some
        # parameters from weight decay, and .low_lr_param_names() to put some
        # parameters in a small-LR group (soft-freeze).
        no_decay_names = set()
        low_lr_names = set()
        m = model.module if hasattr(model, 'module') else model
        if hasattr(m, 'no_decay_param_names'):
            try:
                no_decay_names = set(m.no_decay_param_names())
            except Exception:
                no_decay_names = set()
        if hasattr(m, 'low_lr_param_names'):
            try:
                low_lr_names = set(m.low_lr_param_names())
            except Exception:
                low_lr_names = set()

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            normalized_name = name[7:] if name.startswith('module.') else name
            if normalized_name in no_decay_names:
                no_decay_params.append(param)
                continue
            if normalized_name in low_lr_names:
                low_lr_params.append(param)
                continue
            if self.spatial_prefix and (
                normalized_name == self.spatial_prefix
                or normalized_name.startswith(self.spatial_prefix + '.')
            ):
                spatial_params.append(param)
            elif self.temporal_prefix and (
                normalized_name == self.temporal_prefix
                or normalized_name.startswith(self.temporal_prefix + '.')
            ):
                temporal_params.append(param)
            else:
                fusion_params.append(param)

        base_lr = self.optim_dict['base_lr']
        global_wd = self.optim_dict['weight_decay']

        param_groups = []
        if temporal_params:
            param_groups.append({'params': temporal_params, 'lr': base_lr, 'name': 'temporal'})
        if spatial_params:
            param_groups.append({'params': spatial_params, 'lr': base_lr, 'name': 'spatial'})
        if fusion_params:
            param_groups.append({'params': fusion_params, 'lr': base_lr * fusion_lr_mult, 'name': 'fusion'})
        if low_lr_params:
            param_groups.append({
                'params': low_lr_params,
                'lr': base_lr * low_lr_mult,
                'name': 'low_lr',
            })
            print(f'[optimizer] {sum(p.numel() for p in low_lr_params)} '
                  f'params in low_lr group (lr_mult={low_lr_mult})')
        if no_decay_params:
            param_groups.append({
                'params': no_decay_params,
                'lr': base_lr,
                'name': 'no_decay',
                'weight_decay': 0.0,
            })
            print(f'[optimizer] {sum(p.numel() for p in no_decay_params)} '
                  f'params in no_decay group')
        if not param_groups:
            raise ValueError("No trainable parameters were found for optimizer setup.")

        if self.optim_dict["optimizer"] == 'SGD':
            self.optimizer = optim.SGD(
                param_groups,
                momentum=0.9,
                nesterov=self.optim_dict['nesterov'],
                weight_decay=global_wd,
            )
        elif self.optim_dict["optimizer"] == 'Adam':
            self.optimizer = optim.Adam(
                param_groups,
                weight_decay=global_wd,
            )
        else:
            raise ValueError()
        self.scheduler = self.define_lr_scheduler(self.optimizer, self.optim_dict['step'])

    def define_lr_scheduler(self, optimizer, milestones):
        scheduler_type = self.optim_dict.get('scheduler', 'multistep')
        if scheduler_type == 'cosine':
            num_epochs = self.optim_dict.get('num_epochs', milestones[-1] if milestones else 200)
            start_epoch = self.optim_dict.get('start_epoch', 0)
            return optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=num_epochs - start_epoch,
                eta_min=self.optim_dict['base_lr'] * 0.01,
            )
        if scheduler_type == 'constant_then_cosine':
            # Constant base_lr until cosine_start, then cosine decay to num_epochs.
            num_epochs = self.optim_dict['num_epochs']
            cosine_start = self.optim_dict['cosine_start']
            eta_min = self.optim_dict['base_lr'] * self.optim_dict.get('eta_min_ratio', 0.01)
            constant = optim.lr_scheduler.ConstantLR(
                optimizer, factor=1.0, total_iters=cosine_start,
            )
            cosine = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=num_epochs - cosine_start, eta_min=eta_min,
            )
            return optim.lr_scheduler.SequentialLR(
                optimizer, [constant, cosine], milestones=[cosine_start],
            )
        if scheduler_type == 'constant_then_cosine_then_lock':
            # Phase 1: constant base_lr until cosine_start.
            # Phase 2: cosine decay to eta_min between cosine_start and lock_start.
            # Phase 3: lock at eta_min from lock_start to num_epochs.
            num_epochs = self.optim_dict['num_epochs']
            cosine_start = self.optim_dict['cosine_start']
            lock_start = self.optim_dict.get('lock_start', num_epochs)
            base_lr = self.optim_dict['base_lr']
            eta_min = base_lr * self.optim_dict.get('eta_min_ratio', 0.01)
            constant1 = optim.lr_scheduler.ConstantLR(
                optimizer, factor=1.0, total_iters=cosine_start,
            )
            cosine = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, lock_start - cosine_start), eta_min=eta_min,
            )
            lock = optim.lr_scheduler.ConstantLR(
                optimizer, factor=eta_min / max(base_lr, 1e-12),
                total_iters=max(1, num_epochs - lock_start),
            )
            return optim.lr_scheduler.SequentialLR(
                optimizer, [constant1, cosine, lock],
                milestones=[cosine_start, lock_start],
            )
        if self.optim_dict["optimizer"] in ['SGD', 'Adam']:
            return optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
        raise ValueError()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        self.optimizer.step()

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)
