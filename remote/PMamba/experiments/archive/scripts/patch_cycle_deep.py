"""Add deeper cycle_proj for cycle_rigidity_mlp variant.

v14a had single Conv1d(1, D) for cycle_proj. Hypothesis: 1 channel -> D may
not have enough capacity to learn a useful cycle modulation. Try 2-layer
MLP: 1 -> D/2 -> D with GELU + final-layer zero init.

Final-layer zero init ensures the module output is zero at epoch 1
regardless of the hidden layer's random init, preserving v14a's starting
behavior (identical to v8a).

qcc_variant token: 'cycle_rigidity_mlp'
"""
from pathlib import Path

PATH = Path('models/reqnn_motion.py')
src = PATH.read_text(encoding='utf-8')

# Add cycle_proj_deep alongside cycle_proj
old_cycle_init = """        # Side-path cycle projection (separate from rigidity_proj so v8a
        # weights stay pristine). Only active when qcc_variant='cycle_rigidity_side'.
        self.cycle_proj = nn.Sequential(
            nn.Conv1d(1, hidden2, kernel_size=1, bias=True),
            nn.Tanh(),
        )
        nn.init.zeros_(self.cycle_proj[0].weight)
        nn.init.zeros_(self.cycle_proj[0].bias)"""

new_cycle_init = """        # Side-path cycle projection (separate from rigidity_proj so v8a
        # weights stay pristine). Only active when qcc_variant='cycle_rigidity_side'.
        self.cycle_proj = nn.Sequential(
            nn.Conv1d(1, hidden2, kernel_size=1, bias=True),
            nn.Tanh(),
        )
        nn.init.zeros_(self.cycle_proj[0].weight)
        nn.init.zeros_(self.cycle_proj[0].bias)

        # Deeper cycle projection for cycle_rigidity_mlp variant.
        # 1 -> hidden2//2 -> hidden2, with zero-init final layer so start
        # output is exactly zero (matches v14a starting behavior).
        _cyc_mid = max(hidden2 // 2, 16)
        self.cycle_proj_deep = nn.Sequential(
            nn.Conv1d(1, _cyc_mid, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv1d(_cyc_mid, hidden2, kernel_size=1, bias=True),
            nn.Tanh(),
        )
        # Kaiming init for hidden layer (default); zero init for output layer
        nn.init.zeros_(self.cycle_proj_deep[2].weight)
        nn.init.zeros_(self.cycle_proj_deep[2].bias)"""

if old_cycle_init not in src:
    raise SystemExit('ERR: cycle init block not found')
src = src.replace(old_cycle_init, new_cycle_init, 1)

# Extend dispatch to handle cycle_rigidity_mlp (same as cycle_rigidity_side
# for rigidity path — keeps rigidity_proj pristine)
old_disp = """        elif self.qcc_variant == 'cycle_rigidity_side':
            rigidity, corr_valid_ratio = _compute_bearing_qcc_aligned(
                sampled, num_frames, knn_k=self.bearing_knn_k,
                corr_matched=corr_matched)
            side_cyc, _ = _compute_cycle_consistency_rigidity(sampled, num_frames)
            # stash for use after rigidity_proj
            self._side_cyc = side_cyc
        else:"""
new_disp = """        elif self.qcc_variant in ('cycle_rigidity_side', 'cycle_rigidity_mlp'):
            rigidity, corr_valid_ratio = _compute_bearing_qcc_aligned(
                sampled, num_frames, knn_k=self.bearing_knn_k,
                corr_matched=corr_matched)
            side_cyc, _ = _compute_cycle_consistency_rigidity(sampled, num_frames)
            # stash for use after rigidity_proj
            self._side_cyc = side_cyc
        else:"""
if old_disp not in src:
    raise SystemExit('ERR: dispatch block not found')
src = src.replace(old_disp, new_disp, 1)

# Modify modulation application: select deep vs shallow cycle_proj
old_mod = """        # Modulate with rigidity
        if not self.disable_rigidity:
            modulation = self.rigidity_proj(rigidity)
            if self.qcc_variant == 'cycle_rigidity_side' and hasattr(self, '_side_cyc'):
                modulation = modulation + self.cycle_proj(self._side_cyc)
                self._side_cyc = None
            encoded = encoded * (1.0 + modulation)"""
new_mod = """        # Modulate with rigidity
        if not self.disable_rigidity:
            modulation = self.rigidity_proj(rigidity)
            if self.qcc_variant == 'cycle_rigidity_side' and hasattr(self, '_side_cyc'):
                modulation = modulation + self.cycle_proj(self._side_cyc)
                self._side_cyc = None
            elif self.qcc_variant == 'cycle_rigidity_mlp' and hasattr(self, '_side_cyc'):
                modulation = modulation + self.cycle_proj_deep(self._side_cyc)
                self._side_cyc = None
            encoded = encoded * (1.0 + modulation)"""
if old_mod not in src:
    raise SystemExit('ERR: modulation block not found')
src = src.replace(old_mod, new_mod, 1)

PATH.write_text(src, encoding='utf-8')
print('OK: added cycle_proj_deep (2-layer MLP, zero-init output)')
print('OK: extended dispatch for cycle_rigidity_mlp variant')
