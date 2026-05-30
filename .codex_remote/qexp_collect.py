import glob, os, re

root = '/notebooks/Anemon/experiments/work_dir/qexp'
rows = []
for wd in sorted(glob.glob(root + '/*')):
    name = os.path.basename(wd)
    runlog = os.path.join(wd, 'run.log')
    if not os.path.exists(runlog):
        continue
    txt = open(runlog, errors='replace').read()
    done = ('FINAL' in txt) or ('EARLY_STOP' in txt)
    # final best fused + epoch
    best = bep = ep = '-'
    mf = re.search(r'fused_best=([\d.]+)% @ ep(\d+)', txt)
    if mf:
        best, bep = mf.group(1), mf.group(2)
    else:
        eps = re.findall(r'best_fused=([\d.]+)% @ ep(\d+)', txt)
        if eps:
            best, bep = eps[-1]
    eplines = re.findall(r'^ep\s*(\d+)', txt, re.M)
    if eplines:
        ep = eplines[-1]
    err = 'ERR' if ('Traceback' in txt) else ''
    rows.append((name, 'DONE' if done else 'run', ep, best, bep, err))

# group by config for the controls
def cfg(n):
    for c in ('clean', 'qrot', 'ident', 'amp'):
        if n.startswith(c):
            return c
    return '?'

print(f'{"run":14s} {"st":4s} {"ep":>4s} {"bestF":>7s} {"@ep":>5s}  {""}')
for r in rows:
    print(f'{r[0]:14s} {r[1]:4s} {r[2]:>4s} {r[3]:>7s} {r[4]:>5s}  {r[5]}')

# per-config summary of DONE runs only
print('\n--- DONE-run best_fused by config ---')
import statistics as st
agg = {}
for name, status, ep, best, bep, err in rows:
    if status == 'DONE' and best not in ('-',):
        agg.setdefault(cfg(name), []).append(float(best))
for c in ('clean', 'qrot', 'ident', 'amp'):
    if c in agg:
        v = agg[c]
        m = st.mean(v)
        print(f'  {c:6s} n={len(v)} mean={m:.3f} vals={sorted(v)}')
ndone = sum(1 for r in rows if r[1] == 'DONE')
print(f'\n{ndone}/{len(rows)} DONE')
