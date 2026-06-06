"""PGCNetBDNQ -- lean PGCNetPruned with the BDN-Q buffered-delta temporal
encoder (recovered from history, commit 80a3fb8) swapped in as the bottleneck.

Replaces the per-frame PGCNetEncoder bottleneck (in 128 -> out 256) with
BDeltaQTemporalEncoder: a short attention KV-buffer (FIFO, capacity W) + a
long-term delta state on a 4-fold quaternion-shape substrate, scanned over
per-point trajectories (scan_axis='T', length T=32). Everything else is the
lean 1.06M base. Tests whether the buffered-delta recurrence beats the
per-frame MLP bottleneck (lean baseline last-50 88.04).
"""
from models.pgcnet_pruned import PGCNetPruned
# vectorized BDN-Q (windowed attention parallelized; verified numerically
# identical to the naive scan, maxdiff ~1e-6). Same params -> checkpoints load.
from models.motion_bdn_q_vec import BDeltaQTemporalEncoder


class PGCNetBDNQ(PGCNetPruned):
    def __init__(self, *args, bdnq_hidden_dim=128, bdnq_num_layers=2,
                 bdnq_num_heads=4, bdnq_n_q=4, bdnq_n_v=8, bdnq_buffer_size=4,
                 bdnq_dropout=0.3, bdnq_bidirectional=True, bdnq_scan_axis='T',
                 **kwargs):
        super().__init__(*args, **kwargs)
        # lean bottleneck is PGCNetEncoder(in_channels=128, output_dim=256)
        self.bottleneck = BDeltaQTemporalEncoder(
            in_channels=128, hidden_dim=bdnq_hidden_dim, output_dim=256,
            num_layers=bdnq_num_layers, num_heads=bdnq_num_heads,
            n_q=bdnq_n_q, n_v=bdnq_n_v, buffer_size=bdnq_buffer_size,
            dropout=bdnq_dropout, bidirectional=bdnq_bidirectional,
            scan_axis=bdnq_scan_axis,
        )
