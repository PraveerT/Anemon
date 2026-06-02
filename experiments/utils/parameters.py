import argparse
from .tools import str2bool


def get_parser():
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # parameter priority: command line > config > default
    parser = argparse.ArgumentParser(
        description='Pytorch version for "An Efficient PointLSTM for Point Clouds Based Gesture Recognition"(CVPR 2020)')
    parser.add_argument(
        '--work-dir',
        default='./work_dir/temp',
        help='the work folder for storing results')
    parser.add_argument(
        '--config',
        default='pointlstm.yaml',
        help='path to the configuration file')
    parser.add_argument(
        '--device',
        type=str,
        default=0,
        help='the indexes of GPUs for training or testing')
    parser.add_argument(
        '--phase', default='train', help='must be train or test')

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # debug
    parser.add_argument(
        '--random_fix',
        type=str2bool,
        default=True,
        help='fix the random seed or not')
    parser.add_argument(
        '--random-seed',
        type=int,
        default=0,
        help='the default value for random seed.')
    parser.add_argument(
        '--save-interval',
        type=int,
        default=200,
        help='the interval for storing models (#epoch)')
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=100,
        help='the interval for evaluating models (#epoch)')
    parser.add_argument(
        '--print-log',
        type=str2bool,
        default=True,
        help='print logging or not')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=20,
        help='the interval for printing messages (#iteration)')

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # dataloader
    parser.add_argument(
        '--dataloader', default='dataloader.dataloader', help='data loader will be used')
    parser.add_argument(
        '--num-worker',
        type=int,
        default=4,
        help='the number of workers for data loader')
    parser.add_argument(
        '--framesize',
        type=int,
        default=32,
        help='the number of sampled frame for each video')
    parser.add_argument(
        '--pts-size',
        type=int,
        default=128,
        help='the number of sampled points for each frame')
    parser.add_argument(
        '--dynamic-pts-size',
        type=str2bool,
        default=True,
        help='dynamically schedule pts_size during training instead of keeping it fixed')
    parser.add_argument(
        '--pts-random-range',
        type=int,
        nargs='*',
        default=None,
        help='if set [lo, hi], sample pts uniformly each epoch in this range (overrides default dynamic past ep 100)')
    parser.add_argument(
        '--pts-ramp-target',
        type=int,
        default=None,
        help='target pts at end of custom linear ramp (used with --pts-ramp-epochs)')
    parser.add_argument(
        '--pts-ramp-epochs',
        type=int,
        default=None,
        help='number of epochs for custom linear ramp from 48 to pts_ramp_target')
    parser.add_argument(
        '--pts-extra-target',
        type=int,
        default=None,
        help='extend ramp past 256 to this target after ep 100')
    parser.add_argument(
        '--pts-extra-epochs',
        type=int,
        default=None,
        help='epochs to spend ramping 256 -> pts_extra_target after ep 100')
    parser.add_argument(
        '--pts-freeze-after-ep',
        type=int,
        default=None,
        help='freeze pts_size at the value it had at this epoch for all subsequent epochs')
    parser.add_argument(
        '--freeze',
        type=str,
        default=None,
        choices=['spatial', 'temporal'],
        help='Freeze spatial or temporal branch during training')
    parser.add_argument(
        '--train-loader-args',
        default=dict(),
        help='the arguments of data loader for training')
    parser.add_argument(
        '--test-loader-args',
        default=dict(),
        help='the arguments of data loader for testing')
    parser.add_argument(
        '--valid-loader-args',
        default=dict(),
        help='the arguments of data loader for validation')

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # model
    parser.add_argument('--model', default=None, help='the model will be used')
    parser.add_argument(
        '--model-args',
        type=dict,
        default=dict(),
        help='the arguments of model')
    parser.add_argument(
        '--weights',
        default=None,
        help='the weights for network initialization')
    parser.add_argument(
        '--resume',
        type=str2bool,
        default=False,
        help='resume optimizer, scheduler, and RNG state from --weights checkpoint')
    parser.add_argument(
        '--min-best-acc',
        type=float,
        default=0.0,
        help='floor for best_accuracy on resume: best_model.pt only overwritten when test_acc exceeds this')
    parser.add_argument(
        '--strict-load',
        type=str2bool,
        default=True,
        help='require exact key coverage and shape compatibility when loading --weights')
    parser.add_argument(
        '--temporal-weights',
        default=None,
        help='optional checkpoint used to initialize the temporal branch of a fused model')
    parser.add_argument(
        '--spatial-weights',
        default=None,
        help='optional checkpoint used to initialize the spatial branch of a fused model')
    parser.add_argument(
        '--temporal-source-prefix',
        type=str,
        default='',
        help='prefix to strip from keys loaded via --temporal-weights before remapping')
    parser.add_argument(
        '--spatial-source-prefix',
        type=str,
        default='',
        help='prefix to strip from keys loaded via --spatial-weights before remapping')
    parser.add_argument(
        '--temporal-target-prefix',
        type=str,
        default='temporal_branch',
        help='target branch prefix inside the fused model for --temporal-weights')
    parser.add_argument(
        '--spatial-target-prefix',
        type=str,
        default='spatial_branch',
        help='target branch prefix inside the fused model for --spatial-weights')
    parser.add_argument(
        '--ignore-weights',
        type=str,
        default=[],
        nargs='+',
        help='the name of weights which will be ignored in the initialization')

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # optim
    parser.add_argument(
        '--batch-size', type=int, default=16, help='training batch size')
    parser.add_argument(
        '--test-batch-size', type=int, default=8, help='test batch size')

    default_optimizer_dict = {
        "base_lr": 1e-2,
        "optimizer": "SGD",
        "nesterov": False,
        "step": [100],
        "weight_decay": 0.00005,
        "start_epoch": 0,
    }

    parser.add_argument(
        '--optimizer-args',
        default=default_optimizer_dict,
        help='the arguments of optimizer')

    parser.add_argument(
        '--num-epoch',
        type=int,
        default=200,
        help='stop training in which epoch')
    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # soft label / knowledge distillation
    parser.add_argument(
        '--use_soft_labels',
        type=str2bool,
        default=False,
        help='use soft labels for knowledge distillation')
    parser.add_argument(
        '--temperature',
        type=float,
        default=3.0,
        help='temperature for softmax scaling in knowledge distillation')
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.7,
        help='weight for soft label loss (0.0-1.0)')
    parser.add_argument(
        '--qcc-weight-schedule',
        default=None,
        help='list of [epoch_threshold, weight] pairs for scheduling qcc_weight')
    return parser
