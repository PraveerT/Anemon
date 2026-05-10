"""Tail /tmp/finger.log and post each new 'ep ' line to telegram."""
import time, requests, os, re

TOKEN = '8049556095:AAH0c0KB0DmzFtcW0s97ZS_kQ8ux9gX72eE'
CHAT = '7685170478'
LOG = '/tmp/finger.log'

def send(msg):
    try:
        r = requests.post(f'https://api.telegram.org/bot{TOKEN}/sendMessage',
                          data={'chat_id': CHAT, 'text': msg}, timeout=10)
        return r.status_code
    except Exception as e:
        return f'err {e}'

# Send catch-up summary first
seen = set()
if os.path.exists(LOG):
    with open(LOG) as f:
        for ln in f:
            if ln.startswith('ep '): seen.add(ln.strip())
if seen:
    last = sorted(seen, key=lambda s: int(s.split()[1]))[-1]
    send(f'[finger-qcc catchup] {last}')

# Tail forever
with open(LOG) as f:
    f.seek(0, 2)
    while True:
        ln = f.readline()
        if not ln:
            time.sleep(2); continue
        ln = ln.strip()
        if ln.startswith('ep '):
            send(f'[finger-qcc] {ln}')
        elif ln.startswith('BEST'):
            send(f'[finger-qcc DONE] {ln}')
            break
