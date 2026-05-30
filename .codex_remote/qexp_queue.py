import subprocess, os, time, shlex

ROOT = '/notebooks/Anemon/experiments/work_dir/qexp'
os.makedirs(ROOT, exist_ok=True)
os.chdir('/notebooks/Anemon')

COMMON = shlex.split(
    "--frames 16 --points 128 --epochs 260 --batch-size 64 --workers 0 "
    "--lr 0.00075 --min-lr 0.000015 --warmup-epochs 10 --wd 0.04 --ema-decay 0.995 "
    "--label-smoothing 0.08 --qcc-weight 0.0 --cycle-weight 0.0 --rot-aug-ce-weight 0.0 "
    "--rot-cycle-prob 1.0 --dropout 0.30 --point-hidden 160 --temporal-hidden 256 "
    "--layers 2 --jitter 0.006 --point-drop 0.08 --no-quat-inject --no-publish-active")

# (name, seed, extra args) -- controls first (decisive), then amp scouts
jobs = []
for s in (29, 31, 37, 43, 47):
    jobs.append((f'clean_s{s}', s, ['--rot-cycle-weight', '0.0']))
    jobs.append((f'qrot_s{s}',  s, ['--rot-cycle-weight', '0.02', '--rot-mode', 'uniform']))
    jobs.append((f'ident_s{s}', s, ['--rot-cycle-weight', '0.02', '--rot-mode', 'z', '--rot-max-angle-deg', '0']))
jobs += [
    ('amp_w004',   29, ['--rot-cycle-weight', '0.04', '--rot-mode', 'uniform']),
    ('amp_w008',   29, ['--rot-cycle-weight', '0.08', '--rot-mode', 'uniform']),
    ('amp_w016',   29, ['--rot-cycle-weight', '0.16', '--rot-mode', 'uniform']),
    ('amp_so3_45', 29, ['--rot-cycle-weight', '0.02', '--rot-mode', 'small-so3', '--rot-max-angle-deg', '45']),
    ('amp_z60',    29, ['--rot-cycle-weight', '0.02', '--rot-mode', 'z', '--rot-max-angle-deg', '60']),
]

MAXN = 6
running = []  # (name, popen, fh)
pending = list(jobs)
print(f'queue: {len(pending)} jobs, max {MAXN} concurrent', flush=True)

while pending or running:
    # reap finished
    still = []
    for name, p, fh in running:
        if p.poll() is None:
            still.append((name, p, fh))
        else:
            fh.close()
            print(f'  done {name} rc={p.returncode}', flush=True)
    running = still
    # launch up to cap
    while pending and len(running) < MAXN:
        name, seed, extra = pending.pop(0)
        wd = os.path.join(ROOT, name)
        os.makedirs(wd, exist_ok=True)
        cmd = ['python', '-u', 'train_corr_qcc_fusion.py',
               '--workdir', wd, '--seed', str(seed)] + COMMON + extra
        fh = open(os.path.join(wd, 'run.log'), 'w')
        p = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT)
        running.append((name, p, fh))
        print(f'  start {name} (pid {p.pid}) [{len(running)} running, {len(pending)} pending]', flush=True)
    time.sleep(5)

print('QUEUE COMPLETE', flush=True)
