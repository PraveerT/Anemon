"""Tail both MQAR logs and push each 'Test, Evaluation:' line to telegram."""
import os, time, requests, glob

TOKEN = '8049556095:AAH0c0KB0DmzFtcW0s97ZS_kQ8ux9gX72eE'
CHAT = '7685170478'
WD = '/notebooks/PMamba/experiments/work_dir'

def send(msg):
    try:
        requests.post(f'https://api.telegram.org/bot{TOKEN}/sendMessage',
                      data={'chat_id': CHAT, 'text': msg}, timeout=10)
    except Exception:
        pass

def parse_eval(ln):
    """'Test, Evaluation: Epoch 12 prec1 78.2400, prec5 95.1000' → (12, 78.24, 95.10)"""
    try:
        a = ln.split('Epoch')[1].strip().split()
        ep = int(a[0])
        p1 = float(a[2].rstrip(','))
        p5 = float(a[4])
        return ep, p1, p5
    except Exception:
        return None

def follow(path, tag, send_init):
    if send_init:
        send(f'[mqar:{tag}] watcher attached → {os.path.basename(path)}')
    with open(path) as f:
        f.seek(0, 2)
        while True:
            ln = f.readline()
            if not ln:
                time.sleep(2); continue
            ln = ln.strip()
            if ln.startswith('Test, Evaluation:'):
                p = parse_eval(ln)
                if p:
                    ep, p1, p5 = p
                    send(f'[mqar:{tag}] ep {ep:3d}  p1={p1:5.2f}  p5={p5:5.2f}')
            elif ln.startswith('best:'):
                # only push on improvement
                pass
            elif ln.startswith('run:'):
                send(f'[mqar:{tag}] {ln}')

def main():
    import threading
    targets = []
    for tag in ('deltanet', 'attrd'):
        p = os.path.join(WD, f'mqar_{tag}.log')
        # Wait up to 60s for log to appear
        t0 = time.time()
        while not os.path.exists(p) and time.time() - t0 < 60:
            time.sleep(1)
        if os.path.exists(p):
            targets.append((p, tag))
        else:
            print(f'[watcher] no log: {p}')

    if not targets:
        send('[mqar] no logs found, watcher exiting')
        return

    send('[mqar] starting — DeltaNet vs AttRD head-to-head on associative recall')
    threads = []
    for path, tag in targets:
        t = threading.Thread(target=follow, args=(path, tag, True), daemon=True)
        t.start(); threads.append(t)
    for t in threads:
        t.join()

if __name__ == '__main__':
    main()
