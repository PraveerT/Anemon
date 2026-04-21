"""Add separate cycle_proj module for cycle_rigidity_side variant.

v13a/b patched rigidity_proj to 2 channels — but even with partial loading,
the Conv1d init noise / interaction with v8a's channel 0 weight cost ~0.62%.

This variant keeps rigidity_proj COMPLETELY UNTOUCHED (1 channel, loads
cleanly from v8a). Adds a brand new module cycle_proj (1 channel input,
D output), init zero. Total modulation = rigidity_proj(geom) + cycle_proj(cyc).

At epoch 1: cycle_proj output = 0 -> model identical to v8a. During training,
cycle_proj gradually learns; geometric rigidity remains exactly as learned.

qcc_variant token: 'cycle_rigidity_side'
"""
from pathlib import Path

PATH = Path('models/reqnn_motion.py')
src = PATH.read_text(encoding='utf-8')

# Add cycle_proj instantiation. Insert right after rigidity_proj setup.
old_rigidity_init = """        self.rigidity_proj = nn.Sequential(
            nn.Conv1d(rig_channels, hidden2, kernel_size=1, bias=True),
            nn.Tanh(),
        )
        nn.init.zeros_(self.rigidity_proj[0].weight)
        nn.init.zeros_(self.rigidity_proj[0].bias)"""

new_rigidity_init = """        self.rigidity_proj = nn.Sequential(
            nn.Conv1d(rig_channels, hidden2, kernel_size=1, bias=True),
            nn.Tanh(),
        )
        nn.init.zeros_(self.rigidity_proj[0].weight)
        nn.init.zeros_(self.rigidity_proj[0].bias)

        # Side-path cycle projection (separate from rigidity_proj so v8a
        # weights stay pristine). Only active when qcc_variant='cycle_rigidity_side'.
        self.cycle_proj = nn.Sequential(
            nn.Conv1d(1, hidden2, kernel_size=1, bias=True),
            nn.Tanh(),
        )
        nn.init.zeros_(self.cycle_proj[0].weight)
        nn.init.zeros_(self.cycle_proj[0].bias)"""

if old_rigidity_init not in src:
    raise SystemExit('ERR: rigidity init block not found')
src = src.replace(old_rigidity_init, new_rigidity_init, 1)

# Update rig_channels: cycle_rigidity_side does NOT expand rigidity_proj.
old_rig_ch = """        is_multiscale = 'multiscale' in self.qcc_variants
        is_cycle_rig = 'cycle_rigidity' in self.qcc_variants
        if is_multiscale:
            rig_channels = len(rigidity_scales)
        elif is_cycle_rig:
            rig_channels = 2
        else:
            rig_channels = 1"""
new_rig_ch = """        is_multiscale = 'multiscale' in self.qcc_variants
        is_cycle_rig = 'cycle_rigidity' in self.qcc_variants
        is_cycle_side = 'cycle_rigidity_side' in self.qcc_variants
        if is_multiscale:
            rig_channels = len(rigidity_scales)
        elif is_cycle_rig:
            rig_channels = 2
        else:
            rig_channels = 1"""
if old_rig_ch not in src:
    raise SystemExit('ERR: rig_channels block not found')
src = src.replace(old_rig_ch, new_rig_ch, 1)

# Update dispatch: cycle_rigidity_side uses same 1-channel geometric rigidity
# as the default but also computes cycle and applies it via cycle_proj.
old_dispatch = """        elif self.qcc_variant == 'cycle_rigidity':
            geom_rig, corr_valid_ratio = _compute_bearing_qcc_aligned(
                sampled, num_frames, knn_k=self.bearing_knn_k,
                corr_matched=corr_matched)
            cyc_rig, _ = _compute_cycle_consistency_rigidity(sampled, num_frames)
            rigidity = torch.cat([geom_rig, cyc_rig], dim=1)
        else:"""
new_dispatch = """        elif self.qcc_variant == 'cycle_rigidity':
            geom_rig, corr_valid_ratio = _compute_bearing_qcc_aligned(
                sampled, num_frames, knn_k=self.bearing_knn_k,
                corr_matched=corr_matched)
            cyc_rig, _ = _compute_cycle_consistency_rigidity(sampled, num_frames)
            rigidity = torch.cat([geom_rig, cyc_rig], dim=1)
        elif self.qcc_variant == 'cycle_rigidity_side':
            rigidity, corr_valid_ratio = _compute_bearing_qcc_aligned(
                sampled, num_frames, knn_k=self.bearing_knn_k,
                corr_matched=corr_matched)
            side_cyc, _ = _compute_cycle_consistency_rigidity(sampled, num_frames)
            # stash for use after rigidity_proj
            self._side_cyc = side_cyc
        else:"""
if old_dispatch not in src:
    raise SystemExit('ERR: dispatch block not found')
src = src.replace(old_dispatch, new_dispatch, 1)

# Modify modulation application: add cycle_proj output when side variant active
old_mod = """        # Modulate with rigidity
        if not self.disable_rigidity:
            modulation = self.rigidity_proj(rigidity)
            encoded = encoded * (1.0 + modulation)"""
new_mod = """        # Modulate with rigidity
        if not self.disable_rigidity:
            modulation = self.rigidity_proj(rigidity)
            if self.qcc_variant == 'cycle_rigidity_side' and hasattr(self, '_side_cyc'):
                modulation = modulation + self.cycle_proj(self._side_cyc)
                self._side_cyc = None
            encoded = encoded * (1.0 + modulation)"""
if old_mod not in src:
    raise SystemExit('ERR: modulation block not found')
src = src.replace(old_mod, new_mod, 1)

PATH.write_text(src, encoding='utf-8')
print('OK: added cycle_proj side module')
print('OK: extended rig_channels handling')
print('OK: wired cycle_rigidity_side dispatch + modulation')
