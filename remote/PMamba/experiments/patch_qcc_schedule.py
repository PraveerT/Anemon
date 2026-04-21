"""Add qcc_weight_schedule support to main.py.

Interprets config key qcc_weight_schedule as a list of [epoch_threshold, weight] pairs.
Applies the highest applicable weight at the start of each training epoch.

Example:
  qcc_weight_schedule: [[0, 0.1], [100, 0.0]]
    epoch 0-99 -> qcc_weight = 0.1
    epoch 100+ -> qcc_weight = 0.0
"""
import re
from pathlib import Path

PATH = Path('main.py')
src = PATH.read_text(encoding='utf-8')

# Insert the schedule logic right after the model_ref assignment in train()
anchor = """    def train(self, epoch):
        self.model.train()
        model_ref = self.model.module if hasattr(self.model, 'module') else self.model
        """

if anchor not in src:
    raise SystemExit('anchor not found')

insertion = """    def train(self, epoch):
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
        """

src = src.replace(anchor, insertion, 1)
PATH.write_text(src, encoding='utf-8')
print('OK: patched main.py (qcc_weight_schedule support)')

# Also register qcc_weight_schedule as a known config field
PARAM = Path('utils/parameters.py')
param_src = PARAM.read_text(encoding='utf-8')
if 'qcc_weight_schedule' not in param_src:
    anchor_param = "    return parser"
    new_arg = '''    parser.add_argument(
        '--qcc-weight-schedule',
        default=None,
        help='list of [epoch_threshold, weight] pairs for scheduling qcc_weight')
    return parser'''
    param_src = param_src.replace(anchor_param, new_arg, 1)
    PARAM.write_text(param_src, encoding='utf-8')
    print('OK: registered qcc_weight_schedule in parameters.py')
else:
    print('qcc_weight_schedule already registered')
