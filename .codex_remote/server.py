"""Tiny stdlib HTTP server: serves PWA static files + /api/status JSON.

JSON shape:
  {
    "ts": "2026-05-17 07:42:11",
    "run": "bdn_buf1_train",
    "log": "/notebooks/Anemon/.../bdn_buf1_train.log",
    "gpu": {"used_gb": 11.2, "total_gb": 47.6, "util_pct": 87},
    "ram": {"used_gb": 14, "total_gb": 64},
    "disk": {"used_gb": 22.1, "total_gb": 50},
    "epochs": [{"ep": 10, "tr_acc": 12.3, "tr_loss": 3.27, "te_p1": 69.5, "te_p5": 94.4}, ...],
    "best": {"ep": 50, "p1": 85.68},
    "now": {"ep": 70, "batch": "12/131", "elapsed_min": 4.3}
  }
"""
import json, os, re, subprocess, time, glob
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
STATIC = os.path.join(HERE, 'static')
LOG_GLOB = '/notebooks/Anemon/experiments/work_dir/**/log.txt'
LEADERBOARD = '/notebooks/Anemon/experiments/LEADERBOARD.md'

_cache = {'ts': 0, 'data': None}
_CACHE_SEC = 5


def latest_log():
    files = sorted(glob.glob(LOG_GLOB, recursive=True), key=os.path.getmtime, reverse=True)
    return files[0] if files else None


def parse_log(path):
    """Re-implements tg_messages.sh awk, returns dict."""
    epochs = []
    cur_ep = None
    cur_batch = None
    ta = None
    tl = None
    al = None
    aux_acc = None
    aux_te = None
    skew_traj = []
    cur_lr = None
    best = None
    try:
        with open(path, 'r', errors='replace') as f:
            for line in f:
                m = re.search(r'Mean training acc:\s+(\d+(?:\.\d+)?)', line)
                if m: ta = float(m.group(1))
                m = re.search(r'Mean training loss:\s+(\d+(?:\.\d+)?)', line)
                if m: tl = float(m.group(1))
                m = re.search(r'Mean auxiliary loss:\s+(\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)', line)
                if m: al = float(m.group(1))
                m = re.search(r'Training epoch:\s+(\d+)', line)
                if m: cur_ep = int(m.group(1))
                m = re.search(r'lr:\s*(\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)', line)
                if m: cur_lr = float(m.group(1))
                m = re.search(r'lr=(\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)', line)
                if m: cur_lr = float(m.group(1))
                m = re.search(r'(\d+)/(\d+) \[', line)
                if m: cur_batch = f"{m.group(1)}/{m.group(2)}"
                m = re.search(r'Batch\((\d+)/(\d+)\)', line)
                if m: cur_batch = f"{m.group(1)}/{m.group(2)}"
                m = re.search(r'Epoch (\d+), Test, Evaluation: prec1 (\d+(?:\.\d+)?), prec5 (\d+(?:\.\d+)?)', line)
                if m:
                    ep = int(m.group(1))
                    p1 = float(m.group(2))
                    p5 = float(m.group(3))
                    row = {
                        'ep': ep, 'tr_acc': ta, 'tr_loss': tl,
                        'aux_loss': al, 'aux_acc': aux_acc, 'aux_te': aux_te,
                        'te_p1': round(p1, 2), 'te_p5': round(p5, 2),
                    }
                    # Same epoch logged twice (stdout + file logger): replace
                    # in place instead of appending a duplicate row.
                    if epochs and epochs[-1].get('ep') == ep:
                        epochs[-1] = row
                    else:
                        epochs.append(row)
                    if best is None or p1 > best['p1']:
                        best = {'ep': ep, 'p1': round(p1, 2)}
                # --- skew-auxce hooks (model.log_hooks + main.py aux metrics) ---
                m = re.search(r'Mean auxiliary acc:\s+(\d+(?:\.\d+)?)', line)
                if m:
                    aux_acc = float(m.group(1))
                m = re.search(r'Aux test acc:\s+(\d+(?:\.\d+)?)', line)
                if m:
                    aux_te = float(m.group(1))
                m = re.search(
                    r'skew_hook scale=([-\d.eE+]+) wu=([-\d.eE+]+) wv=([-\d.eE+]+) '
                    r'head0=([-\d.eE+]+) headf=([-\d.eE+]+) desc_e=([-\d.eE+]+) '
                    r'lag1=([-\d.eE+]+) lag2=([-\d.eE+]+)', line)
                if m and cur_ep is not None:
                    skew_traj.append({
                        'ep': cur_ep,
                        'scale': float(m.group(1)), 'wu': float(m.group(2)),
                        'wv': float(m.group(3)), 'head0': float(m.group(4)),
                        'headf': float(m.group(5)), 'desc_e': float(m.group(6)),
                        'lag1': float(m.group(7)), 'lag2': float(m.group(8)),
                        'aux_acc': aux_acc, 'aux_te': aux_te,
                    })
    except FileNotFoundError:
        pass
    return {'epochs': epochs[-40:], 'best': best,
            'now': {'ep': cur_ep, 'batch': cur_batch, 'lr': cur_lr},
            'skew_traj': skew_traj[-80:]}


def gpu_stats():
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu',
             '--format=csv,noheader,nounits'], text=True, timeout=4).strip().splitlines()[0]
        used, total, util = [float(x.strip()) for x in out.split(',')]
        return {'used_gb': round(used / 1024, 1), 'total_gb': round(total / 1024, 1), 'util_pct': int(util)}
    except Exception:
        return None


def ram_stats():
    try:
        out = subprocess.check_output(['free', '-g'], text=True, timeout=4)
        for line in out.splitlines():
            if line.startswith('Mem:'):
                p = line.split()
                return {'used_gb': int(p[2]), 'total_gb': int(p[1])}
    except Exception:
        return None


def disk_stats():
    try:
        nb = int(os.getxattr('/notebooks', 'ceph.dir.rbytes'))
        return {'used_gb': round(nb / 1e9, 1), 'total_gb': 50}
    except Exception:
        return None


def leaderboard_md():
    try:
        with open(LEADERBOARD, 'r', errors='replace') as f:
            return f.read()
    except FileNotFoundError:
        return None


def _parse_md_table(lines, start):
    """Parse a markdown pipe table starting at start (header line). Returns (rows, end_idx).
    rows is a list of dicts using the header names as keys."""
    if start >= len(lines) or '|' not in lines[start]:
        return [], start
    header_cells = [c.strip() for c in lines[start].strip().strip('|').split('|')]
    # next line should be separator |---|---|
    if start + 1 >= len(lines) or '---' not in lines[start + 1]:
        return [], start
    rows = []
    i = start + 2
    while i < len(lines):
        ln = lines[i].rstrip()
        if '|' not in ln or not ln.strip():
            break
        cells = [c.strip() for c in ln.strip().strip('|').split('|')]
        if len(cells) != len(header_cells):
            break
        rows.append(dict(zip(header_cells, cells)))
        i += 1
    return rows, i


def leaderboard_summary():
    md = leaderboard_md()
    if not md:
        return None
    lines = md.splitlines()
    sections = {}  # heading-text -> list of rows
    current = None
    i = 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith('## '):
            current = ln[3:].strip()
            sections[current] = []
            i += 1
            continue
        # Look for table start under current section
        if current and '|' in ln and i + 1 < len(lines) and '---' in lines[i + 1]:
            rows, end = _parse_md_table(lines, i)
            if rows:
                sections[current].append(rows)
                i = end
                continue
        i += 1
    return sections


def _load_latest_sd(run_dir):
    """Return (epoch, state_dict) of latest epochN_model.pt in run_dir, or (None, None)."""
    if not run_dir or not os.path.isdir(run_dir):
        return None, None
    ckpts = [f for f in os.listdir(run_dir)
             if f.startswith('epoch') and f.endswith('_model.pt')]
    if not ckpts:
        return None, None
    ckpts.sort(key=lambda f: int(f.replace('epoch', '').replace('_model.pt', '')))
    latest = os.path.join(run_dir, ckpts[-1])
    ep = int(ckpts[-1].replace('epoch', '').replace('_model.pt', ''))
    import torch
    sd = torch.load(latest, map_location='cpu')
    sd = sd.get('model_state_dict', sd) if isinstance(sd, dict) else sd
    return ep, sd


def engram_stats(run_dir):
    """engram.out_proj norm (zero-init residual; grows iff model uses engram path)."""
    try:
        ep, sd = _load_latest_sd(run_dir)
        if sd is None:
            return None
        out_w = sd.get('engram.out_proj.weight')
        if out_w is None:
            return None
        return {
            'epoch': ep,
            'out_norm': float(out_w.norm()),
            'out_max': float(out_w.abs().max()),
        }
    except Exception:
        return None


def cluster_stats(run_dir):
    """ST-QNet-C1 cluster-rotation mechanism inspection.
      - cycle_proj_norm: zero-init residual; grows iff mechanism active.
      - cluster_head_norm: aux classifier head weight magnitude.
    """
    try:
        ep, sd = _load_latest_sd(run_dir)
        if sd is None:
            return None
        cycle_proj_w = sd.get('cycle_proj.weight')
        if cycle_proj_w is None:
            return None
        out = {
            'epoch': ep,
            'cycle_proj_norm': float(cycle_proj_w.norm()),
            'cycle_proj_max': float(cycle_proj_w.abs().max()),
        }
        # cluster_head is Sequential: 0=Linear, 2=Linear (final).
        # Use explicit is-None check; `or` on tensors raises ambiguity error.
        ch_w = sd.get('cluster_head.2.weight')
        if ch_w is None:
            ch_w = sd.get('cluster_head.0.weight')
        if ch_w is not None:
            out['cluster_head_norm'] = float(ch_w.norm())
        # Per-cluster mass EMA (if model registered the buffer).
        cm = sd.get('cluster_mass_ema')
        if cm is not None:
            out['cluster_mass'] = [round(float(x), 4) for x in cm.tolist()]
        return out
    except Exception:
        return None


_CNXXL_BASELINE = {  # train,test pairs per epoch from cnxxlquat 91.08 run
    10: (72.29, 74.48), 20: (89.14, 83.82), 30: (91.33, 85.68),
    40: (93.62, 81.74), 50: (94.38, 87.76), 60: (93.62, 87.34),
    70: (93.33, 87.14), 75: (93.52, 88.17), 80: (93.43, 87.14),
    85: (93.24, 87.55), 88: (94.48, 89.63), 90: (93.43, 89.63),
    95: (95.05, 90.66), 96: (96.67, 90.66), 97: (95.71, 90.25),
    98: (96.29, 90.25), 99: (95.90, 90.66), 100: (96.00, 91.08),
}


def param_counts(run_dir):
    """Total / trainable / per-mechanism param counts from latest ckpt."""
    try:
        ep, sd = _load_latest_sd(run_dir)
        if sd is None:
            return None
        total = sum(v.numel() for v in sd.values() if hasattr(v, 'numel'))
        # Group by prefix to bucket aux vs main.
        aux_prefixes = ('cluster_head', 'cycle_gru', 'cycle_proj',
                        'cluster_classifier', 'cluster_clf_scale',
                        'engram', 'qcc_head', 'qcc_scale',
                        'quat_to_coords', 'quat_proj', 'quat_inject_scale')
        aux = 0
        for k, v in sd.items():
            if not hasattr(v, 'numel'): continue
            if any(k.startswith(p) for p in aux_prefixes):
                aux += v.numel()
        return {
            'epoch': ep,
            'total_m': round(total / 1e6, 3),
            'aux_m': round(aux / 1e6, 3),
            'main_m': round((total - aux) / 1e6, 3),
        }
    except Exception:
        return None


_WORK_DIR = '/notebooks/Anemon/experiments/work_dir'
_REF_LABELS = {
    'cn_xxl_quat_head': 'cnxxlquat 91.08',
    'cn_xxl_quat_head_stqnet_c1_gumbel': 'k6 gumbel (ep126) 90.66',
}


def _perclass_from_cm(path):
    """Load confusion matrix and return [(cls_idx_1based, wrong, total), ...]."""
    import numpy as np
    cm = np.load(path)
    out = []
    for c in range(cm.shape[0]):
        tot = int(cm[c].sum())
        if tot == 0: continue
        wr = tot - int(cm[c, c])
        out.append((c + 1, wr, tot))
    return out


def available_refs(exclude_run=None):
    """Scan work_dir for directories with Test_confusion_mat.npy; return their
    per-class breakdowns so the client can pick which to compare against.

    exclude_run: directory path (basename matched) to skip � used to drop the
    currently-tracked run from the list of comparison targets.
    """
    if not os.path.isdir(_WORK_DIR):
        return None
    skip_name = os.path.basename(exclude_run.rstrip('/')) if exclude_run else None
    out = []
    for d in sorted(os.listdir(_WORK_DIR)):
        if skip_name and d == skip_name:
            continue
        p = os.path.join(_WORK_DIR, d)
        if not os.path.isdir(p): continue
        cm = os.path.join(p, 'Test_confusion_mat.npy')
        if not os.path.isfile(cm): continue
        try:
            pc = _perclass_from_cm(cm)
            tot = sum(t for _, _, t in pc)
            cor = sum(t - w for _, w, t in pc)
            out.append({
                'name': d,
                'label': _REF_LABELS.get(d, d),
                'acc': round(100.0 * cor / max(tot, 1), 2),
                'perclass': [{'cls': c, 'wrong': w, 'total': t} for c, w, t in pc],
            })
        except Exception:
            continue
    return out or None


def current_perclass(run_dir):
    """Current run's per-class breakdown from its Test_confusion_mat.npy."""
    if not run_dir:
        return None
    p = os.path.join(run_dir, 'Test_confusion_mat.npy')
    if not os.path.isfile(p):
        return None
    try:
        pc = _perclass_from_cm(p)
        return [{'cls': c, 'wrong': w, 'total': t} for c, w, t in pc]
    except Exception:
        return None


def skew_stats(run_dir):
    """Skew-TCC aux anatomy from the latest ckpt: scale, projector/head norms,
    per-cov-component energy (xx yy zz xy xz yz), per-rank usage, per-lag
    classifier energy, and whether the aux is alive (scale*head0 > eps).
    Works for the self-destructing skew-raw head AND the auxce (fixed-scale) variant."""
    try:
        import numpy as np
        ep, sd = _load_latest_sd(run_dir)
        if sd is None or 'skew_head_scale' not in sd:
            return None

        def t(k):
            v = sd.get(k)
            return v.numpy() if v is not None and hasattr(v, 'numpy') else None

        sc = sd['skew_head_scale']
        scale = float(sc.item() if hasattr(sc, 'item') else sc)
        Wu, Wv = t('Wu.weight'), t('Wv.weight')
        H0, Hf = t('skew_head.0.weight'), t('skew_head.3.weight')
        out = {'epoch': ep, 'scale': scale}
        if Wu is not None: out['wu'] = float(np.linalg.norm(Wu))
        if Wv is not None: out['wv'] = float(np.linalg.norm(Wv))
        if H0 is not None: out['head0'] = float(np.linalg.norm(H0))
        if Hf is not None: out['headf'] = float(np.linalg.norm(Hf))
        out['alive'] = bool(abs(scale) * out.get('head0', 0.0) > 1e-3)
        if Wu is not None:
            uniq = {'xx': [0], 'yy': [4], 'zz': [8], 'xy': [1, 3], 'xz': [2, 6], 'yz': [5, 7]}
            W = np.abs(np.vstack([Wu, Wv])) if Wv is not None else np.abs(Wu)
            e = {k: float(np.sqrt((W[:, idx] ** 2).sum())) for k, idx in uniq.items()}
            tot = sum(e.values()) or 1.0
            out['comp_energy'] = {k: round(100 * v / tot, 1) for k, v in e.items()}
        if H0 is not None:
            r = Wu.shape[0] if Wu is not None else 12
            npairs = r * (r - 1) // 2
            coln = np.linalg.norm(H0, axis=0)
            nlag = max(1, coln.shape[0] // npairs)
            out['lag_energy'] = [round(float(coln[i * npairs:(i + 1) * npairs].sum()), 3) for i in range(nlag)]
            ti = np.tril_indices(r, -1)
            usage = np.zeros(r)
            for d in range(coln.shape[0]):
                k = d % npairs
                usage[ti[0][k]] += coln[d]
                usage[ti[1][k]] += coln[d]
            us = usage.sum() or 1.0
            out['rank_usage'] = [round(float(100 * x / us), 1) for x in usage]
        return out
    except Exception:
        return None


# K=6 v2 run best (per-epoch trajectory lost when this work_dir was reused).
_V2_K6_BEST = 90.04


def reference_baselines(parsed):
    """Return current best + delta vs reference best scores (v2, cnxxlquat)."""
    if not parsed:
        return None
    best = None
    for e in parsed.get('epochs') or []:
        te = e.get('te_p1')
        if te is None: continue
        if best is None or te > best:
            best = te
    if best is None:
        return None
    return {
        'current_best': round(best, 2),
        'refs': [
            {'name': 'skew-tcc', 'value': 91.91, 'delta': round(best - 91.91, 2)},
            {'name': 'quat-head', 'value': 91.29, 'delta': round(best - 91.29, 2)},
            {'name': 'no-aux floor', 'value': 89.83, 'delta': round(best - 89.83, 2)},
        ],
    }


def cnxxl_delta(parsed):
    """For each recent test eval, compute test - cnxxlquat_baseline at same ep.
    parsed: dict with 'epochs' list from parse_log."""
    if not parsed or not parsed.get('epochs'):
        return None
    out = []
    for e in parsed['epochs'][-20:]:
        ep = e.get('ep')
        te = e.get('te_p1')
        if ep is None or te is None: continue
        base = _CNXXL_BASELINE.get(ep)
        if base is None: continue
        out.append({'ep': ep, 'te': te, 'base_te': base[1],
                    'delta': round(te - base[1], 2)})
    return out or None


def qcc_stats(run_dir):
    """Track QCC mechanism usage:
      - qcc_scale: scalar gate for qcc_head aux output (init 0)
      - quat_inject_scale: scalar gate for fea3 quat residual (init 0; v4)
      - quat_inject_norm: norm of the LayerNorm-bounded quat_proj output's
        deepest weight matrix (legacy); for v4 this measures the projection
        MLP weights, not the gated residual.
    """
    try:
        ep, sd = _load_latest_sd(run_dir)
        if sd is None or 'qcc_scale' not in sd:
            return None
        qs = sd['qcc_scale']
        out = {
            'epoch': ep,
            'qcc_scale': float(qs.item() if hasattr(qs, 'item') else qs),
        }
        # v4: gated post-Mamba residual
        if 'quat_inject_scale' in sd:
            qis = sd['quat_inject_scale']
            out['quat_inject_scale'] = float(qis.item() if hasattr(qis, 'item') else qis)
        # MLP weight norm (informational across versions)
        for key in ('quat_proj.2.weight', 'quat_to_coords.weight'):
            if key in sd:
                w = sd[key]
                out['quat_inject_norm'] = float(w.norm())
                out['quat_inject_max'] = float(w.abs().max())
                break
        return out
    except Exception:
        return None


def build_status():
    log = latest_log()
    parsed = parse_log(log) if log else {'epochs': [], 'best': None, 'now': {}}
    run_dir = os.path.dirname(log) if log else None
    return {
        'ts': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'run': os.path.basename(run_dir) if run_dir else None,
        'log': log,
        'gpu': gpu_stats(),
        'ram': ram_stats(),
        'disk': disk_stats(),
        'leaderboard': leaderboard_summary(),
        'engram': engram_stats(run_dir),
        'qcc': qcc_stats(run_dir),
        'cluster': cluster_stats(run_dir),
        'param_counts': param_counts(run_dir),
        'cnxxl_delta': cnxxl_delta(parsed),
        'refs': reference_baselines(parsed),
        'available_refs': available_refs(exclude_run=run_dir),
        'current_perclass': current_perclass(run_dir),
        'skew': skew_stats(run_dir),
        **parsed,
    }


def cached_status():
    now = time.time()
    if _cache['data'] is None or (now - _cache['ts']) > _CACHE_SEC:
        _cache['data'] = build_status()
        _cache['ts'] = now
    return _cache['data']


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *a):  # quieter
        pass

    def _send(self, code, body, ctype='application/json; charset=utf-8'):
        body_b = body if isinstance(body, (bytes, bytearray)) else body.encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body_b)))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body_b)

    def do_GET(self):
        path = self.path.split('?', 1)[0]
        if path == '/api/status':
            self._send(200, json.dumps(cached_status()))
            return
        if path == '/' or path == '':
            path = '/index.html'
        safe = os.path.normpath(path).lstrip(os.sep)
        full = os.path.join(STATIC, safe)
        if not full.startswith(STATIC) or not os.path.isfile(full):
            self._send(404, '{"error":"not found"}')
            return
        ctype = {
            '.html': 'text/html; charset=utf-8',
            '.js':   'application/javascript; charset=utf-8',
            '.json': 'application/json; charset=utf-8',
            '.css':  'text/css; charset=utf-8',
            '.svg':  'image/svg+xml',
            '.png':  'image/png',
        }.get(os.path.splitext(full)[1].lower(), 'application/octet-stream')
        with open(full, 'rb') as f:
            self._send(200, f.read(), ctype)


def main():
    port = int(os.environ.get('SIDEPANEL_PORT', 8765))
    srv = ThreadingHTTPServer(('0.0.0.0', port), Handler)
    print(f'[sidepanel] serving on 0.0.0.0:{port}', flush=True)
    srv.serve_forever()


if __name__ == '__main__':
    main()
