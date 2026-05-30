"""Poll live logits and publish fusion stats for the Anemon sidepanel."""
import json
import os
import re
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
STATE_DIR = os.path.join(HERE, "state")
CACHE = os.path.join(STATE_DIR, "fusion_cache.json")

LIVE_DIRS = {
    "cnxxl": "/notebooks/Anemon/experiments/work_dir/cn_xxl_quat_head",
    "depth_small": "/notebooks/Anemon/experiments/work_dir/depth_small",
}
SNAPSHOTS = {
    "cnxxl": "/notebooks/Anemon/external_ckpts/cnxxl_ep141_91.29_test_logits.npz",
}
LOGITS_DSN = "/notebooks/Anemon/dsn_official_valid_logits.npz"
LOGITS_M = "/notebooks/Anemon/external_ckpts/umdr_M_test_logits.npz"
DSN_T = 9.5
HISTORY_LIMIT = 80


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def sig_of(p):
    m = re.search(r"class_(\d+)/subject(\d+)_r(\d+)", p)
    if not m:
        return str(p)
    return f"class_{m.group(1)}/subject{m.group(2)}_r{m.group(3)}"


def load_logits(path):
    a = np.load(path, allow_pickle=True)
    if "sigs" in a.files:
        sigs = np.array([sig_of(str(s)) for s in a["sigs"]])
    else:
        sigs = np.array([sig_of(str(s)) for s in a["paths"]])
    return a["logits"], a["labels"], sigs


def load_epoch_marker(path):
    try:
        a = np.load(path, allow_pickle=True)
        if "epoch" in a.files:
            val = int(a["epoch"][0])
            return val if val > 0 else None
    except Exception:
        pass
    return None


def resolve_slot(name):
    live = os.path.join(LIVE_DIRS[name], "test_logits.npz")
    snap = SNAPSHOTS.get(name)
    live_mt = os.path.getmtime(live) if os.path.isfile(live) else 0
    snap_mt = os.path.getmtime(snap) if snap and os.path.isfile(snap) else 0
    if live_mt > snap_mt:
        return live, True, load_epoch_marker(live)
    if snap and os.path.isfile(snap):
        return snap, False, None
    return None, False, None


def latest_live_slot():
    best = None
    for name in LIVE_DIRS:
        live = os.path.join(LIVE_DIRS[name], "test_logits.npz")
        if not os.path.isfile(live):
            continue
        snap = SNAPSHOTS.get(name)
        snap_mt = os.path.getmtime(snap) if snap and os.path.isfile(snap) else 0
        mt = os.path.getmtime(live)
        if mt <= snap_mt:
            continue
        if best is None or mt > best[1]:
            best = (name, mt, live)
    return best


def aligned(ref_sigs, logits, sigs):
    by = {s: i for i, s in enumerate(sigs)}
    order = np.array([by[s] for s in ref_sigs])
    return logits[order]


def accuracy(probs, labels):
    return float((probs.argmax(1) == labels).mean() * 100.0)


def fuse_uniform(probs_list, labels):
    return accuracy(np.mean(probs_list, axis=0), labels)


def oracle_any(*probs_list, labels):
    correct = np.zeros_like(labels, dtype=bool)
    for p in probs_list:
        correct |= p.argmax(1) == labels
    return float(correct.mean() * 100.0)


def maybe_load_aligned(path, ref_sigs):
    if not path or not os.path.isfile(path):
        return None
    logits, labels, sigs = load_logits(path)
    return softmax(aligned(ref_sigs, logits, sigs))


def compute_fusion():
    cnxxl_path, cnxxl_live, cnxxl_ep = resolve_slot("cnxxl")
    depth_path, depth_live, depth_ep = resolve_slot("depth_small")
    cnxxl_logits, labels, sigs = load_logits(cnxxl_path)
    p_cnxxl = softmax(cnxxl_logits)
    ref_sigs = sigs

    p_depth = maybe_load_aligned(depth_path, ref_sigs)
    p_dsn = None
    if os.path.isfile(LOGITS_DSN):
        dsn_logits, _, dsn_sigs = load_logits(LOGITS_DSN)
        p_dsn = softmax(aligned(ref_sigs, dsn_logits, dsn_sigs) * DSN_T)
    p_m = maybe_load_aligned(LOGITS_M, ref_sigs)

    out = {
        "solo": {"cnxxl": round(accuracy(p_cnxxl, labels), 2)},
        "fusion": {},
        "oracle": {},
    }
    if p_depth is not None:
        out["solo"]["depth_small"] = round(accuracy(p_depth, labels), 2)
        out["fusion"]["cnxxl_depth"] = round(fuse_uniform([p_cnxxl, p_depth], labels), 2)
        out["oracle"]["cnxxl_depth"] = round(oracle_any(p_cnxxl, p_depth, labels=labels), 2)
    if p_dsn is not None:
        out["solo"]["dsn"] = round(accuracy(p_dsn, labels), 2)
        out["fusion"]["cnxxl_dsn"] = round(fuse_uniform([p_cnxxl, p_dsn], labels), 2)
        out["oracle"]["cnxxl_dsn"] = round(oracle_any(p_cnxxl, p_dsn, labels=labels), 2)
        if p_depth is not None:
            out["fusion"]["cnxxl_dsn_depth"] = round(fuse_uniform([p_cnxxl, p_dsn, p_depth], labels), 2)
            out["oracle"]["cnxxl_dsn_depth"] = round(oracle_any(p_cnxxl, p_dsn, p_depth, labels=labels), 2)
    if p_m is not None:
        out["solo"]["m"] = round(accuracy(p_m, labels), 2)
        if p_dsn is not None:
            out["fusion"]["cnxxl_dsn_m"] = round(fuse_uniform([p_cnxxl, p_dsn, p_m], labels), 2)
            out["oracle"]["cnxxl_dsn_m"] = round(oracle_any(p_cnxxl, p_dsn, p_m, labels=labels), 2)
        if p_dsn is not None and p_depth is not None:
            out["fusion"]["cnxxl_dsn_depth_m"] = round(fuse_uniform([p_cnxxl, p_dsn, p_depth, p_m], labels), 2)
            out["oracle"]["cnxxl_dsn_depth_m"] = round(oracle_any(p_cnxxl, p_dsn, p_depth, p_m, labels=labels), 2)

    if cnxxl_live and depth_live:
        live_name = "cnxxl" if os.path.getmtime(cnxxl_path) >= os.path.getmtime(depth_path) else "depth_small"
    elif cnxxl_live:
        live_name = "cnxxl"
    elif depth_live:
        live_name = "depth_small"
    else:
        live_name = None
    out["live"] = live_name
    out["live_epoch"] = cnxxl_ep if live_name == "cnxxl" else (depth_ep if live_name == "depth_small" else None)
    return out


def load_cache():
    if not os.path.isfile(CACHE):
        return None
    try:
        with open(CACHE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def write_cache(payload):
    os.makedirs(STATE_DIR, exist_ok=True)
    tmp = CACHE + ".tmpjson"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    os.replace(tmp, CACHE)


def append_history(latest):
    prev = load_cache() or {}
    history = list(prev.get("history") or [])
    row = {
        "epoch": latest.get("live_epoch"),
        "ts": latest.get("ts"),
        "live": latest.get("live"),
        "cnxxl": latest["solo"].get("cnxxl"),
        "depth_small": latest["solo"].get("depth_small"),
        "dsn": latest["solo"].get("dsn"),
        "m": latest["solo"].get("m"),
        "cnxxl_depth": latest["fusion"].get("cnxxl_depth"),
        "cnxxl_dsn": latest["fusion"].get("cnxxl_dsn"),
        "cnxxl_dsn_depth": latest["fusion"].get("cnxxl_dsn_depth"),
        "orc_cnxxl_depth": latest["oracle"].get("cnxxl_depth"),
        "orc_cnxxl_dsn_depth": latest["oracle"].get("cnxxl_dsn_depth"),
    }
    if history and history[-1].get("epoch") == row["epoch"] and history[-1].get("live") == row["live"]:
        history[-1] = row
    else:
        history.append(row)
    merged = dict(latest)
    merged["history"] = history[-HISTORY_LIMIT:]
    return merged


def step(last_mtime):
    live = latest_live_slot()
    if live is None:
        return last_mtime
    name, mtime, _path = live
    if mtime == last_mtime:
        return last_mtime
    print(f"[watcher] {name} logits changed mtime={mtime}", flush=True)
    try:
        payload = compute_fusion()
        payload["ts"] = time.strftime("%Y-%m-%d %H:%M:%S")
        merged = append_history(payload)
        write_cache(merged)
        print(
            f"[watcher] cache updated live={payload.get('live')} ep={payload.get('live_epoch')} "
            f"solo={payload['solo']}",
            flush=True,
        )
    except Exception as e:
        print(f"[watcher] fusion error: {e}", flush=True)
        return last_mtime
    return mtime


def main():
    interval = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    print(f"[watcher] polling live dirs every {interval}s: {list(LIVE_DIRS)}", flush=True)
    last = 0.0
    initial = latest_live_slot()
    if initial:
        if not os.path.isfile(CACHE) or os.path.getmtime(CACHE) < initial[1]:
            last = step(0.0)
        else:
            last = initial[1]
    while True:
        try:
            last = step(last)
        except Exception as e:
            print(f"[watcher] step error: {e}", flush=True)
        time.sleep(interval)


if __name__ == "__main__":
    main()
