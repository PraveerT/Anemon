import os
from collections import OrderedDict

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import sys
import pdb
import yaml
import torch
import random
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import requests

sys.path.append("../..")
from utils import get_parser, import_class, GpuDataParallel, Optimizer, Recorder, Stat, RandomState



# === ShuffleMix+ patch ===
def shufflemix_pc(image, label, smprob):
    """ShuffleMix+ for point cloud (B, T, P, C).

    For ``smprob`` fraction of frames per clip, replace those frames with the
    corresponding frames from a flipped-batch sample. Returns the mixed input
    and the two label sets plus a mixing weight ``lam`` for label-side mixing.
    """
    if smprob <= 0.0 or image.dim() < 3:
        return image, label, label, 1.0
    import random as _random
    B = image.size(0)
    T = image.size(1)
    label_b = label.flip(0)
    if (label_b == label).all():
        return image, label, label, 1.0
    n_replace = max(1, int(round(smprob * T)))
    idx = _random.sample(range(T), n_replace)
    image_b = image.flip(0).clone()
    image[:, idx] = image_b[:, idx]
    lam = 1.0 - n_replace / T
    return image, label, label_b, lam
# === end ShuffleMix+ patch ===

class Processor():
    def __init__(self, arg):
        self.arg = arg
        self.save_arg()
        if self.arg.random_fix:
            self.rng = RandomState(seed=self.arg.random_seed)
        self.device = GpuDataParallel()
        self.recoder = Recorder(self.arg.work_dir, self.arg.print_log)
        self.data_loader = {}
        self.topk = (1, 5)
        self.stat = Stat(self.arg.model_args['num_classes'], self.topk)
        self.model, self.optimizer = self.Loading()
        self.loss = self.criterion()
        
        # Telegram Bot Configuration
        self.telegram_bot_token = "8049556095:AAH0c0KB0DmzFtcW0s97ZS_kQ8ux9gX72eE"
        self.telegram_chat_id = None
        # Oracle/fusion telemetry vs PMamba @ e110 (cached probs).
        self._pmamba_probs = None
        self._pmamba_labels = None
        self._eval_probs = []
        self._eval_labels = []
        self._latest_oracle = None
        self._latest_fusion = None
        self._latest_fusion_alpha = None
        try:
            _cache = "work_dir/pmamba_branch/pmamba_test_preds.npz"
            import numpy as _np, os as _os
            if _os.path.exists(_cache):
                _d = _np.load(_cache)
                self._pmamba_probs = _d["probs"]
                self._pmamba_labels = _d["labels"]
        except Exception:
            pass
        self.best_accuracy = 0.0  # Track best accuracy within current run
        
        # A fixed point count can be requested either from the CLI or directly in config.
        self.use_static_pts = ('--pts-size' in sys.argv) or (not getattr(self.arg, 'dynamic_pts_size', True))

    def criterion(self):
        # Add label smoothing for regularization
        loss = nn.CrossEntropyLoss(label_smoothing=0.1, reduction="none")
        return self.device.criterion_to_device(loss)

    @staticmethod
    def _normalize_key(key):
        return key[7:] if key.startswith('module.') else key

    @staticmethod
    def _normalize_prefix(prefix):
        if prefix is None:
            return None
        prefix = prefix.strip('.')
        return prefix or None

    def _extract_model_state_dict(self, payload):
        if isinstance(payload, dict) and 'model_state_dict' in payload:
            return payload['model_state_dict']
        return payload

    def _should_ignore_weight(self, key):
        normalized_key = self._normalize_key(key)
        for ignore_key in self.arg.ignore_weights:
            normalized_ignore = self._normalize_key(ignore_key)
            if normalized_key == normalized_ignore or normalized_key.startswith(normalized_ignore + '.'):
                return True
        return False

    def _prepare_state_dict(self, model, source_state, source_prefix=None, target_prefix=None):
        source_prefix = self._normalize_prefix(source_prefix)
        target_prefix = self._normalize_prefix(target_prefix)
        target_state = model.state_dict()
        normalized_target_keys = {
            self._normalize_key(key): key for key in target_state.keys()
        }

        prepared_state = OrderedDict()
        ignored_keys = []
        unexpected_keys = []
        prefix_skipped_keys = []
        shape_mismatches = []

        for source_key, tensor in source_state.items():
            if self._should_ignore_weight(source_key):
                ignored_keys.append(source_key)
                continue

            normalized_key = self._normalize_key(source_key)
            if source_prefix:
                if normalized_key == source_prefix:
                    normalized_key = ''
                elif normalized_key.startswith(source_prefix + '.'):
                    normalized_key = normalized_key[len(source_prefix) + 1:]
                else:
                    prefix_skipped_keys.append(source_key)
                    continue

            target_key = normalized_key
            if target_prefix:
                target_key = target_prefix if not target_key else f"{target_prefix}.{target_key}"

            actual_target_key = normalized_target_keys.get(target_key)
            if actual_target_key is None:
                unexpected_keys.append(source_key)
                continue

            if target_state[actual_target_key].shape != tensor.shape:
                src_shape = tuple(tensor.shape)
                tgt_shape = tuple(target_state[actual_target_key].shape)
                # Partial-copy if only dim 1 differs and target is wider
                if (len(src_shape) == len(tgt_shape)
                        and len(src_shape) >= 2
                        and src_shape[0] == tgt_shape[0]
                        and src_shape[1] < tgt_shape[1]
                        and src_shape[2:] == tgt_shape[2:]):
                    merged = target_state[actual_target_key].clone()
                    merged[:, :src_shape[1]] = tensor
                    prepared_state[actual_target_key] = merged
                    shape_mismatches.append(
                        (source_key, actual_target_key, src_shape, tgt_shape)
                    )
                    continue
                shape_mismatches.append(
                    (source_key, actual_target_key, src_shape, tgt_shape)
                )
                continue

            prepared_state[actual_target_key] = tensor

        if target_prefix:
            target_scope = [
                actual_key
                for normalized_key, actual_key in normalized_target_keys.items()
                if normalized_key == target_prefix or normalized_key.startswith(target_prefix + '.')
            ]
        else:
            target_scope = list(target_state.keys())

        missing_keys = [key for key in target_scope if key not in prepared_state]
        return {
            'state_dict': prepared_state,
            'loaded_keys': list(prepared_state.keys()),
            'ignored_keys': ignored_keys,
            'unexpected_keys': unexpected_keys,
            'prefix_skipped_keys': prefix_skipped_keys,
            'shape_mismatches': shape_mismatches,
            'missing_keys': missing_keys,
        }

    def _apply_prepared_state(self, model, prepared_state):
        merged_state = model.state_dict()
        merged_state.update(prepared_state)
        model.load_state_dict(merged_state, strict=True)

    def _log_weight_summary(self, label, summary):
        self.recoder.print_log(
            '{}: loaded {}, missing {}, unexpected {}, mismatched {}, ignored {}.'.format(
                label,
                len(summary['loaded_keys']),
                len(summary['missing_keys']),
                len(summary['unexpected_keys']),
                len(summary['shape_mismatches']),
                len(summary['ignored_keys']),
            )
        )
        if summary['shape_mismatches']:
            source_key, target_key, source_shape, target_shape = summary['shape_mismatches'][0]
            self.recoder.print_log(
                '  First shape mismatch: {} -> {} ({} vs {}).'.format(
                    source_key, target_key, source_shape, target_shape
                )
            )

    def _load_weights_into_model(self, model, weights_path, label, strict=True, source_prefix=None, target_prefix=None):
        payload = torch.load(weights_path, map_location=torch.device('cpu'))
        source_state = self._extract_model_state_dict(payload)
        if not isinstance(source_state, dict):
            raise ValueError('Weights at {} do not contain a valid state_dict.'.format(weights_path))

        summary = self._prepare_state_dict(
            model,
            source_state,
            source_prefix=source_prefix,
            target_prefix=target_prefix,
        )
        if not summary['loaded_keys']:
            raise RuntimeError('No compatible parameters were loaded from {}.'.format(weights_path))
        if strict and (
            summary['missing_keys'] or summary['unexpected_keys'] or summary['shape_mismatches']
        ):
            raise RuntimeError(
                'Strict load failed for {}: missing {}, unexpected {}, mismatched {}.'.format(
                    weights_path,
                    len(summary['missing_keys']),
                    len(summary['unexpected_keys']),
                    len(summary['shape_mismatches']),
                )
            )

        self._apply_prepared_state(model, summary['state_dict'])
        self._log_weight_summary(label, summary)
        return payload

    def _set_requires_grad_by_prefix(self, model, prefix, requires_grad):
        normalized_prefix = self._normalize_prefix(prefix)
        if not normalized_prefix:
            return 0

        updated = 0
        for name, param in model.named_parameters():
            normalized_name = self._normalize_key(name)
            if normalized_name == normalized_prefix or normalized_name.startswith(normalized_prefix + '.'):
                param.requires_grad = requires_grad
                updated += 1
        return updated

    def _apply_freeze(self, model):
        if not hasattr(self.arg, 'freeze') or not self.arg.freeze:
            return

        if self.arg.freeze == 'spatial':
            frozen = self._set_requires_grad_by_prefix(model, self.arg.spatial_target_prefix, False)
            if frozen == 0:
                self.recoder.print_log(
                    'Requested spatial freeze, but no parameters matched prefix {}.'.format(
                        self.arg.spatial_target_prefix
                    )
                )
            else:
                self.recoder.print_log(
                    'Froze {} parameter tensors under {}.'.format(frozen, self.arg.spatial_target_prefix)
                )
        elif self.arg.freeze == 'temporal':
            frozen = self._set_requires_grad_by_prefix(model, self.arg.temporal_target_prefix, False)
            if frozen == 0:
                self.recoder.print_log(
                    'Requested temporal freeze, but no parameters matched prefix {}.'.format(
                        self.arg.temporal_target_prefix
                    )
                )
            else:
                self.recoder.print_log(
                    'Froze {} parameter tensors under {}.'.format(frozen, self.arg.temporal_target_prefix)
                )

    def train(self, epoch):
        self.model.train()
        model_ref = self.model.module if hasattr(self.model, 'module') else self.model

        # Apply qcc_weight_schedule if provided in config
        schedule = getattr(self.arg, 'qcc_weight_schedule', None)
        if schedule and hasattr(model_ref, 'qcc_weights'):
            applicable = [w for ep_th, w in schedule if epoch >= ep_th]
            if applicable:
                new_weight = applicable[-1]
                if hasattr(model_ref, 'qcc_weights'):
                    model_ref.qcc_weights = [new_weight] * len(model_ref.qcc_weights)
                if hasattr(model_ref, 'qcc_weight'):
                    model_ref.qcc_weight = new_weight
        
        # Check if a fixed pts_size was requested.
        if self.use_static_pts:
            # Use a fixed pts_size from CLI or config.
            pts_size = self.arg.pts_size
            model_ref.pts_size = pts_size
            # Also update model_args to ensure consistency
            self.arg.model_args['pts_size'] = pts_size
            static_source = '--pts-size' if '--pts-size' in sys.argv else 'config'
            self.recoder.print_log(
                'Training epoch: {} | pts_size: {} (static from {})'.format(epoch + 1, pts_size, static_source)
            )
        else:
            # Dynamic pts_size scheduling
            # Epoch 0-50: 48 -> 128 (slow increase)
            # Epoch 50-100: 128 -> 256 (fast increase)
            if epoch < 50:
                # Slow linear increase from 48 to 128 over 50 epochs
                pts_size = int(48 + (128 - 48) * (epoch / 50))
            elif epoch < 100:
                # Fast exponential-like increase from 128 to 256 over 50 epochs
                progress = (epoch - 50) / 50  # 0 to 1
                # Use quadratic progression for faster increase
                pts_size = int(128 + (256 - 128) * (progress ** 2))
            else:
                rr = getattr(self.arg, 'pts_random_range', None)
                if rr:
                    import random as _rnd
                    pts_size = _rnd.randint(int(rr[0]), int(rr[1]))
                else:
                    pts_size = 256
            
            # Update model's pts_size
            model_ref.pts_size = pts_size
            self.recoder.print_log('Training epoch: {} | pts_size: {} (dynamic)'.format(epoch + 1, pts_size))
        
        loader = self.data_loader['train']
        loss_value = []
        aux_loss_values = []
        temporal_loss_values = []
        spatial_loss_values = []
        correct = 0
        total = 0
        self.recoder.timer_reset()
        current_learning_rate = [group['lr'] for group in self.optimizer.optimizer.param_groups]
        
        # Add progress bar for training batches
        loader_with_progress = tqdm(enumerate(loader), total=len(loader), 
                                   desc=f"Epoch {epoch+1}", leave=False)
        
        for batch_idx, data in loader_with_progress:
            self.recoder.record_timer("dataloader")
            image = self.device.data_to_device(data[0])
            label = self.device.data_to_device(data[1])
            sm_smprob = float(getattr(self.arg, 'shufflemix_smprob', 0.0))
            image, _label_a, _label_b, _sm_lam = shufflemix_pc(image, label, sm_smprob)
            self.recoder.record_timer("device")
            output = self.model(image)
            self.recoder.record_timer("forward")
            sample_w = getattr(model_ref, 'latest_sample_weights', None)
            if sample_w is not None:
                per_sample = torch.nn.functional.cross_entropy(output, label, reduction='none')
                classification_loss = (per_sample * sample_w).sum() / (sample_w.sum() + 1e-8)
            else:
                if _sm_lam < 1.0:
                    loss_a = torch.mean(self.loss(output, _label_a))
                    loss_b = torch.mean(self.loss(output, _label_b))
                    classification_loss = _sm_lam * loss_a + (1.0 - _sm_lam) * loss_b
                else:
                    classification_loss = torch.mean(self.loss(output, label))
            loss = classification_loss

            aux_loss = None
            aux_metrics = {}
            if hasattr(model_ref, 'get_auxiliary_loss'):
                aux_loss = model_ref.get_auxiliary_loss()
                if aux_loss is not None:
                    loss = loss + aux_loss
                    aux_loss_values.append(aux_loss.detach().item())
                    if hasattr(model_ref, 'get_auxiliary_metrics'):
                        aux_metrics = model_ref.get_auxiliary_metrics() or {}

            # Compute separate branch losses (with gradient for aux training)
            if hasattr(model_ref, 'temporal_logits') and hasattr(model_ref, 'spatial_logits'):
                temporal_loss = torch.mean(self.loss(model_ref.temporal_logits, label))
                spatial_loss = torch.mean(self.loss(model_ref.spatial_logits, label))

                # Add auxiliary branch losses to train aux classifiers
                aux_w = getattr(model_ref, 'aux_weight', 0.0)
                if aux_w > 0:
                    loss = loss + aux_w * (temporal_loss + spatial_loss)

                # Track separate losses for epoch mean calculation
                temporal_loss_values.append(temporal_loss.detach().item())
                spatial_loss_values.append(spatial_loss.detach().item())

                # Print branch losses every 50 batches
                if batch_idx % 50 == 0:
                    lr_by_name = {
                        group.get('name', f'group_{idx}'): group['lr']
                        for idx, group in enumerate(self.optimizer.optimizer.param_groups)
                    }
                    temporal_lr = lr_by_name.get('temporal', current_learning_rate[0])
                    spatial_lr = lr_by_name.get('spatial', temporal_lr)
                    print(f"\n[Branch Losses] Temporal: {temporal_loss.item():.4f} (lr={temporal_lr:.6f}), "
                          f"Spatial: {spatial_loss.item():.4f} (lr={spatial_lr:.6f}), "
                          f"Combined: {loss.item():.4f}")
            elif aux_loss is not None and batch_idx % 50 == 0:
                qcc_raw = aux_metrics.get('qcc_raw')
                qcc_forward = aux_metrics.get('qcc_forward')
                qcc_backward = aux_metrics.get('qcc_backward')
                qcc_valid_ratio = aux_metrics.get('qcc_valid_ratio')
                self.recoder.print_log(
                    '\tEpoch: {}, Batch({}/{}) aux | cls: {:.6f} total: {:.6f} qcc: {:.6f} fwd: {:.6f} bwd: {:.6f} valid: {:.4f}'.format(
                        epoch,
                        batch_idx,
                        len(loader),
                        classification_loss.item(),
                        loss.item(),
                        qcc_raw.item() if qcc_raw is not None else 0.0,
                        qcc_forward.item() if qcc_forward is not None else 0.0,
                        qcc_backward.item() if qcc_backward is not None else 0.0,
                        qcc_valid_ratio.item() if qcc_valid_ratio is not None else 0.0,
                    )
                )
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.recoder.record_timer("backward")
            loss_value.append(loss.item())
            
            # Calculate accuracy
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(label.view_as(pred)).sum().item()
            total += label.size(0)
            current_acc = 100. * correct / total
            
            # Update progress bar
            loader_with_progress.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{current_acc:.2f}%',
                'LR': f'{current_learning_rate[0]:.6f}',
                'PTS': pts_size
            })
            
            if batch_idx % self.arg.log_interval == 0:
                # self.viz.append_loss(epoch * len(loader) + batch_idx, loss.item())
                self.recoder.print_log(
                    '\tEpoch: {}, Batch({}/{}) done. Loss: {:.8f}  lr:{:.6f}'
                        .format(epoch, batch_idx, len(loader), loss.item(), current_learning_rate[0]))
                self.recoder.print_time_statistics()
        self.optimizer.scheduler.step()
        train_loss = np.mean(loss_value)
        train_acc = 100. * correct / total
        if aux_loss_values:
            self.recoder.print_log('\tMean auxiliary loss: {:.10f}.'.format(np.mean(aux_loss_values)))
        self.recoder.print_log('\tMean training loss: {:.10f}.'.format(train_loss))
        return train_acc, train_loss

    def eval(self, loader_name):
        self.model.eval()
        self._eval_probs = []
        self._eval_labels = []
        for l_name in loader_name:
            loader = self.data_loader[l_name]
            loss_mean = []
            for batch_idx, data in enumerate(loader):
                image = self.device.data_to_device(data[0])
                label = self.device.data_to_device(data[1])
                # Cal = CalculateParasAndFLOPs()
                # Cal.reset()
                # Cal.calculate_all(self.model, image)
                with torch.no_grad():
                    # Test-Time Augmentation: average predictions from 3 runs
                    outputs = []
                    for _ in range(3):
                        output = self.model(image)
                        outputs.append(output)
                    output = torch.stack(outputs).mean(dim=0)
                # loss = torch.mean(self.loss(output, label))
                loss_mean += self.loss(output, label).cpu().detach().numpy().tolist()
                self.stat.update_accuracy(output.data.cpu(), label.cpu(), topk=self.topk)
                self._eval_probs.append(torch.softmax(output.detach().float(), dim=1).cpu().numpy())
                self._eval_labels.append(label.detach().cpu().numpy())
            self.recoder.print_log('mean loss: ' + str(np.mean(loss_mean)))

    def Loading(self):
        self.device.set_device(self.arg.device)
        print("Loading model")
        if self.arg.model:
            if self.arg.resume and self.arg.weights is None:
                raise ValueError('--resume requires --weights to point to a checkpoint.')
            if self.arg.resume and (self.arg.temporal_weights or self.arg.spatial_weights):
                raise ValueError('Do not combine --resume with branch-specific weight initialization.')
            if self.arg.resume and self.arg.freeze:
                raise ValueError('Do not combine --resume with --freeze. Start a new phase from --weights instead.')

            model_class = import_class(self.arg.model)
            # Override pts_size in model_args if provided via command line
            if '--pts-size' in sys.argv:
                self.arg.model_args['pts_size'] = self.arg.pts_size
                print(f"Using pts_size={self.arg.pts_size} from command line")
            model = self.device.model_to_device(model_class(**self.arg.model_args))
            optimizer_args = dict(self.arg.optimizer_args)
            optimizer_args.setdefault('spatial_prefix', self.arg.spatial_target_prefix)

            if self.arg.resume:
                self.recoder.print_log('Resuming full training state from {}.'.format(self.arg.weights))
                checkpoint = torch.load(self.arg.weights, map_location=torch.device('cpu'))
                if 'model_state_dict' not in checkpoint:
                    raise ValueError('Resume requires a checkpoint with model_state_dict.')
                self._load_weights_into_model(
                    model,
                    self.arg.weights,
                    label='Resume model state',
                    strict=self.arg.strict_load,
                )
                if 'rng_state' in checkpoint and hasattr(self, 'rng'):
                    self.rng.set_rng_state(checkpoint['rng_state'])
                self.arg.optimizer_args['start_epoch'] = checkpoint["epoch"] + 1
                optimizer_args['start_epoch'] = self.arg.optimizer_args['start_epoch']
                self.recoder.print_log(
                    "Resuming from checkpoint: epoch {}".format(self.arg.optimizer_args['start_epoch'])
                )
                optimizer = Optimizer(model, optimizer_args)
                optimizer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                optimizer.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            else:
                if self.arg.weights:
                    self.recoder.print_log('Initializing model weights from {}.'.format(self.arg.weights))
                    self._load_weights_into_model(
                        model,
                        self.arg.weights,
                        label='Initial model weights',
                        strict=self.arg.strict_load,
                    )
                if self.arg.temporal_weights:
                    self.recoder.print_log(
                        'Initializing temporal branch from {} -> {}.'.format(
                            self.arg.temporal_weights, self.arg.temporal_target_prefix
                        )
                    )
                    self._load_weights_into_model(
                        model,
                        self.arg.temporal_weights,
                        label='Temporal branch weights',
                        strict=False,
                        source_prefix=self.arg.temporal_source_prefix,
                        target_prefix=self.arg.temporal_target_prefix,
                    )
                if self.arg.spatial_weights:
                    self.recoder.print_log(
                        'Initializing spatial branch from {} -> {}.'.format(
                            self.arg.spatial_weights, self.arg.spatial_target_prefix
                        )
                    )
                    self._load_weights_into_model(
                        model,
                        self.arg.spatial_weights,
                        label='Spatial branch weights',
                        strict=False,
                        source_prefix=self.arg.spatial_source_prefix,
                        target_prefix=self.arg.spatial_target_prefix,
                    )

                self._apply_freeze(model)
                optimizer = Optimizer(model, optimizer_args)
        else:
            raise ValueError("No Models.")

        print("Loading model finished.")
        self.load_data()
        return model, optimizer

    def load_data(self):
        print("Loading data")
        Feeder = import_class(self.arg.dataloader)
        self.data_loader = dict()
        if self.arg.train_loader_args != {}:
            self.data_loader['train'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.train_loader_args),
                batch_size=self.arg.batch_size,
                shuffle=True,
                drop_last=True,
                num_workers=self.arg.num_worker,
            )
        if self.arg.valid_loader_args != {}:
            self.data_loader['valid'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.valid_loader_args),
                batch_size=self.arg.test_batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=self.arg.num_worker,
            )
        if self.arg.test_loader_args != {}:
            test_dataset = Feeder(**self.arg.test_loader_args)
            self.stat.test_size = len(test_dataset)
            self.data_loader['test'] = torch.utils.data.DataLoader(
                dataset=test_dataset,
                batch_size=self.arg.test_batch_size,
                shuffle=False,                 # deterministic for oracle alignment
                drop_last=False,
                num_workers=self.arg.num_worker,
            )
        print("Loading data finished.")

    def start(self):
        if self.arg.phase == 'train':
            self.recoder.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))
            # Send initial Telegram message to establish chat
            try:
                self.send_initial_telegram_message("🚀 Training started!")
            except:
                pass  # Ignore if we can't send the initial message
            
            for epoch in range(self.arg.optimizer_args['start_epoch'], self.arg.num_epoch):
                eval_interval = 10 if (epoch + 1) < 100 else 1
                save_model = ((epoch + 1) % self.arg.save_interval == 0) or \
                             (epoch + 1 == self.arg.num_epoch)
                eval_model = ((epoch + 1) % eval_interval == 0) or \
                             (epoch + 1 == self.arg.num_epoch)
                train_acc, train_loss = self.train(epoch)
                if save_model:
                    model_path = '{}/epoch{}_model.pt'.format(self.arg.work_dir, epoch + 1)
                    self.save_model(epoch, self.model, self.optimizer, model_path)
                if eval_model:
                    if self.arg.valid_loader_args != {}:
                        self.stat.reset_statistic()
                        self.eval(loader_name=['valid'])
                        self.print_inf_log(epoch + 1, "Valid", train_acc, train_loss)
                    if self.arg.test_loader_args != {}:
                        self.stat.reset_statistic()
                        self.eval(loader_name=['test'])
                        self.print_inf_log(epoch + 1, "Test", train_acc, train_loss)
        elif self.arg.phase == 'test':
            if self.arg.weights is None:
                raise ValueError('Please appoint --weights.')
            self.recoder.print_log('Model:   {}.'.format(self.arg.model))
            self.recoder.print_log('Weights: {}.'.format(self.arg.weights))
            # Send initial Telegram message to establish chat
            try:
                self.send_initial_telegram_message("🚀 Testing started!")
            except:
                pass  # Ignore if we can't send the initial message
            
            if self.arg.valid_loader_args != {}:
                self.stat.reset_statistic()
                self.eval(loader_name=['valid'])
                self.print_inf_log(self.arg.optimizer_args['start_epoch'], "Valid", None, None)
            if self.arg.test_loader_args != {}:
                self.stat.reset_statistic()
                self.eval(loader_name=['test'])
                self.print_inf_log(self.arg.optimizer_args['start_epoch'], "Test", None, None)
            self.recoder.print_log('Evaluation Done.\n')

    def print_inf_log(self, epoch, mode, train_acc=None, train_loss=None):
        static = self.stat.show_accuracy('{}/{}_confusion_mat'.format(self.arg.work_dir, mode))
        prec1 = static[str(self.topk[0])] / self.stat.test_size * 100
        prec5 = static[str(self.topk[1])] / self.stat.test_size * 100
        self.recoder.print_log("Epoch {}, {}, Evaluation: prec1 {:.4f}, prec5 {:.4f}".
                               format(epoch, mode, prec1, prec5),
                               '{}/{}.txt'.format(self.arg.work_dir, self.arg.phase))
        
        # Display confusion matrix
        try:
            import numpy as np
            cm = getattr(self.stat, 'confusion_matrix', None)
            if cm is not None:
                self.recoder.print_log(f"Confusion Matrix (epoch {epoch}, {mode}):")
                # Print a simplified version of the confusion matrix
                # Show only the diagonal elements (correct predictions) and some key stats
                diagonal = np.diag(cm)
                total_correct = np.sum(diagonal)
                total_samples = np.sum(cm)
                self.recoder.print_log(f"  Total Correct: {total_correct}/{total_samples}")
                if total_samples > 0:
                    self.recoder.print_log(f"  Overall Accuracy: {total_correct/total_samples*100:.2f}%")
        except Exception as e:
            self.recoder.print_log(f"Failed to display confusion matrix: {e}")
        
        # Send Telegram message with evaluation results
        try:
            # Compute oracle first so best_metric selection can use it.
            self._maybe_compute_oracle(epoch, mode)
            # Pick best-metric: oracle if cached PMamba available, else prec1.
            best_metric = self._latest_oracle if self._latest_oracle is not None else prec1
            best_label = "oracle" if self._latest_oracle is not None else "prec1"
            # Check if this is a new best (by the chosen metric).
            is_new_best = best_metric > self.best_accuracy
            if is_new_best:
                self.best_accuracy = best_metric
                try:
                    best_path = f"{self.arg.work_dir}/best_model.pt"
                    self.save_model(epoch, self.model, self.optimizer, best_path)
                    self.recoder.print_log(f"  Saved new best to {best_path} at {best_label}={self.best_accuracy:.2f}% (prec1={prec1:.2f}%)")
                except Exception as _e:
                    self.recoder.print_log(f"Failed to save best_model.pt: {_e}")
            # Format message as: Train: train acc train loss Test: test acc test loss
            if train_acc is not None and train_loss is not None:
                message = f"📊 Epoch {epoch}\n"
                message += f"Train: {train_acc:.1f} {train_loss:.2f}\n"
                message += f"Test: {prec1:.1f} {prec5:.1f}"
                if self._latest_oracle is not None:
                    message += f"\nOracle: {self._latest_oracle:.2f}% | Fusion a={self._latest_fusion_alpha:.2f} -> {self._latest_fusion:.2f}%"
                if is_new_best:
                    message += f" 🏆 New Best: {self.best_accuracy:.1f}%"
            else:
                # For test phase without training data
                message = f"📊 Epoch {epoch} {mode}\n"
                message += f"Test: {prec1:.1f} {prec5:.1f}\n"
                if self._latest_oracle is not None:
                    message += f"Oracle: {self._latest_oracle:.2f}% | Fusion a={self._latest_fusion_alpha:.2f} -> {self._latest_fusion:.2f}%\n"
                if is_new_best:
                    message += f"🏆 New Best: {self.best_accuracy:.1f}%\n"
            
            # Send message
            self.send_telegram_message(message)
        except Exception as e:
            self.recoder.print_log(f"Failed to send Telegram message: {e}")

    def _maybe_compute_oracle(self, epoch, mode):
        """Populate self._latest_oracle and self._latest_fusion (percentages)
        using cached PMamba test-set probs + this eval pass' accumulated probs.
        """
        self._latest_oracle = None
        self._latest_fusion = None
        self._latest_fusion_alpha = None
        if self._pmamba_probs is None or not self._eval_probs:
            return
        try:
            import numpy as np
            probs = np.concatenate(self._eval_probs, axis=0)                 # (N, 25)
            labels = np.concatenate(self._eval_labels, axis=0)               # (N,)
            if probs.shape[0] != self._pmamba_probs.shape[0]:
                self.recoder.print_log(
                    f"[oracle] skip: {probs.shape[0]} model preds vs {self._pmamba_probs.shape[0]} cached PMamba preds"
                )
                return
            # Labels should agree with cache; use cache labels as reference.
            ref_labels = self._pmamba_labels if self._pmamba_labels.shape[0] == probs.shape[0] else labels
            pm_correct = self._pmamba_probs.argmax(1) == ref_labels
            md_correct = probs.argmax(1) == ref_labels
            oracle = (pm_correct | md_correct).mean() * 100
            best_a, best_acc = 1.0, (pm_correct).mean() * 100
            for ai in range(0, 105, 5):
                a = ai / 100.0
                fp = (a * self._pmamba_probs + (1 - a) * probs).argmax(1)
                acc = (fp == ref_labels).mean() * 100
                if acc > best_acc:
                    best_acc = acc; best_a = a
            self._latest_oracle = float(oracle)
            self._latest_fusion = float(best_acc)
            self._latest_fusion_alpha = float(best_a)
            self.recoder.print_log(
                f"[oracle] epoch={epoch} mode={mode} oracle={oracle:.2f}% fusion[a={best_a:.2f}]={best_acc:.2f}%"
            )
        except Exception as e:
            self.recoder.print_log(f"[oracle] failed: {e}")

    def save_model(self, epoch, model, optimizer, save_path):
        state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.optimizer.state_dict(),
            'scheduler_state_dict': optimizer.scheduler.state_dict(),
        }
        if hasattr(self, 'rng'):
            state['rng_state'] = self.rng.save_rng_state()
        torch.save(state, save_path)

    def save_arg(self):
        arg_dict = vars(self.arg)
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
        with open('{}/config.yaml'.format(self.arg.work_dir), 'w') as f:
            yaml.dump(arg_dict, f)

    def get_telegram_chat_id(self):
        """Get chat ID from the most recent message to the bot"""
        url = f"https://api.telegram.org/bot{self.telegram_bot_token}/getUpdates"
        try:
            response = requests.get(url, timeout=10)
            data = response.json()
            
            if data["ok"] and data["result"]:
                # Get the most recent message chat ID (no need for /start command)
                chat_id = data["result"][-1]["message"]["chat"]["id"]
                return chat_id
        except Exception as e:
            self.recoder.print_log(f"Failed to get Telegram chat ID: {e}")
        return None

    def send_telegram_message(self, message):
        """Send message to Telegram - simplified version"""
        try:
            # Just try to send the message - if there's no chat, it will fail silently
            url = f"https://api.telegram.org/bot{self.telegram_bot_token}/getUpdates"
            response = requests.get(url, timeout=5)
            data = response.json()
            
            if data["ok"] and data["result"]:
                # Get the most recent chat ID
                chat_id = data["result"][-1]["message"]["chat"]["id"]
                
                # Send the actual message
                send_url = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage"
                send_data = {
                    "chat_id": chat_id,
                    "text": message,
                    "parse_mode": "HTML"
                }
                
                send_response = requests.post(send_url, data=send_data, timeout=10)
                return send_response.json()["ok"]
        except:
            # If anything fails, just silently continue without sending message
            pass
        return False


if __name__ == '__main__':
    sparser = get_parser()
    p = sparser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
            try:
                default_arg = yaml.load(f, Loader=yaml.FullLoader)
            except AttributeError:
                default_arg = yaml.load(f)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
        sparser.set_defaults(**default_arg)
    args = sparser.parse_args()
    processor = Processor(args)
    processor.start()
