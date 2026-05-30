import glob, os, re

root = '/notebooks/Anemon/experiments/work_dir/qexp'
rows = []
for wd in sorted(glob.glob(root + '/*')):
    name = os.path.basename(wd)
    log = os.path.join(wd, 'log.txt')
    runlog = os.path.join(wd, 'run.log')
    ep = best = brn = '-'
    crashed = ''
    last = ''
    if os.path.exists(log):
        with open(log) as f:
            lines = [l for l in f if l.startswith('ep')]
        if lines:
            last = lines[-1].strip()
            m = re.match(r'ep\s*(\d+)', last)
            ep = m.group(1) if m else '?'
            mb = re.search(r'best_fused=([\d.]+)%', last)
            best = mb.group(1) if mb else '?'
            mbr = re.search(r'branch=([\d.]+)%', last)
            brn = mbr.group(1) if mbr else '?'
    if os.path.exists(runlog):
        with open(runlog) as f:
            txt = f.read()
        if 'Traceback' in txt or 'Error' in txt:
            crashed = 'ERR'
    rows.append((name, ep, brn, best, crashed))

print(f'{"run":22s} {"ep":>4s} {"branch":>7s} {"bestF":>7s}  flag')
for r in rows:
    print(f'{r[0]:22s} {r[1]:>4s} {r[2]:>7s} {r[3]:>7s}  {r[4]}')
done = sum(1 for r in rows if r[1] == '260')
print(f'\n{done}/{len(rows)} reached ep260')
