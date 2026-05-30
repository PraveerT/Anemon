from pathlib import Path


def main():
    p = Path("sidepanel_api/server.py")
    b = p.read_bytes()

    old = b"""                m = re.search(r'lr:\\s*(\\d+(?:\\.\\d+)?(?:[eE][+-]?\\d+)?)', line)
                if m: cur_lr = float(m.group(1))
                m = re.search(r'(\\d+)/(\\d+) \\[', line)
                if m: cur_batch = f"{m.group(1)}/{m.group(2)}"
"""
    new = b"""                m = re.search(r'lr:\\s*(\\d+(?:\\.\\d+)?(?:[eE][+-]?\\d+)?)', line)
                if m: cur_lr = float(m.group(1))
                m = re.search(r'lr=(\\d+(?:\\.\\d+)?(?:[eE][+-]?\\d+)?)', line)
                if m: cur_lr = float(m.group(1))
                m = re.search(r'(\\d+)/(\\d+) \\[', line)
                if m: cur_batch = f"{m.group(1)}/{m.group(2)}"
                m = re.search(r'Batch\\((\\d+)/(\\d+)\\)', line)
                if m: cur_batch = f"{m.group(1)}/{m.group(2)}"
"""
    if old not in b:
        raise SystemExit("first pattern not found")
    b = b.replace(old, new, 1)

    old2 = b"""                    if best is None or p1 > best['p1']:
                        best = {'ep': ep, 'p1': round(p1, 2)}
"""
    insert = b"""                    if best is None or p1 > best['p1']:
                        best = {'ep': ep, 'p1': round(p1, 2)}
                m = re.search(
                    r'ep\\s*(\\d+)\\s+tr_loss=(\\d+(?:\\.\\d+)?)\\s+ce=(\\d+(?:\\.\\d+)?)\\s+'
                    r'q=(\\d+(?:\\.\\d+)?)\\s+cyc=(\\d+(?:\\.\\d+)?)\\s+tr_acc=(\\d+(?:\\.\\d+)?)%\\s+'
                    r'branch=(\\d+(?:\\.\\d+)?)%/(\\d+(?:\\.\\d+)?)%\\s+'
                    r'fused=(\\d+(?:\\.\\d+)?)%/(\\d+(?:\\.\\d+)?)%\\s+'
                    r'w=(\\d+(?:\\.\\d+)?)\\s+tb=(\\d+(?:\\.\\d+)?)\\s+tc=(\\d+(?:\\.\\d+)?)\\s+'
                    r'best_fused=(\\d+(?:\\.\\d+)?)% @ ep(\\d+)',
                    line,
                )
                if m:
                    ep = int(m.group(1))
                    tr_loss = float(m.group(2))
                    ce = float(m.group(3))
                    q_loss = float(m.group(4))
                    cycle_loss = float(m.group(5))
                    tr_acc = float(m.group(6))
                    branch_p1 = float(m.group(7))
                    branch_p5 = float(m.group(8))
                    fused_p1 = float(m.group(9))
                    fused_p5 = float(m.group(10))
                    row = {
                        'ep': ep,
                        'tr_acc': round(tr_acc, 2),
                        'tr_loss': round(tr_loss, 4),
                        'ce_loss': round(ce, 4),
                        'q_loss': round(q_loss, 4),
                        'cycle_loss': round(cycle_loss, 4),
                        'branch_p1': round(branch_p1, 2),
                        'branch_p5': round(branch_p5, 2),
                        'fused_p1': round(fused_p1, 2),
                        'fused_p5': round(fused_p5, 2),
                        'te_p1': round(fused_p1, 2),
                        'te_p5': round(fused_p5, 2),
                        'fusion_w': round(float(m.group(11)), 3),
                        'temp_base': round(float(m.group(12)), 3),
                        'temp_branch': round(float(m.group(13)), 3),
                    }
                    if epochs and epochs[-1].get('ep') == ep:
                        epochs[-1] = row
                    else:
                        epochs.append(row)
                    best_ep = int(m.group(15))
                    best_p1 = float(m.group(14))
                    best = {'ep': best_ep, 'p1': round(best_p1, 2)}
                    cur_ep = ep
"""
    if old2 not in b:
        raise SystemExit("second pattern not found")
    b = b.replace(old2, insert, 1)

    p.write_bytes(b)
    print(f"patched {p}")


if __name__ == "__main__":
    main()
