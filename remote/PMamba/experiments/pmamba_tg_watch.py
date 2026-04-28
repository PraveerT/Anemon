"""Tail PMamba train log and push per-epoch summaries to Telegram.

Usage: python pmamba_tg_watch.py <log_path> <state_path> <run_label>
"""
import os
import re
import sys
import time
import requests

LOG = sys.argv[1] if len(sys.argv) > 1 else "/tmp/dq_train.log"
STATE = sys.argv[2] if len(sys.argv) > 2 else "/tmp/dq_tg_state"
LABEL = sys.argv[3] if len(sys.argv) > 3 else "DQ"

TG_TOKEN = "8049556095:AAH0c0KB0DmzFtcW0s97ZS_kQ8ux9gX72eE"
CHAT_ID = "7685170478"


def send(msg):
    try:
        requests.post(
            f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
            data={"chat_id": CHAT_ID, "text": msg, "parse_mode": "HTML"},
            timeout=10,
        )
    except Exception as e:
        print(f"tg fail: {e}", flush=True)


def main():
    if os.path.exists(STATE):
        last_pos = int(open(STATE).read().strip() or 0)
    elif os.path.exists(LOG):
        last_pos = os.path.getsize(LOG)
    else:
        last_pos = 0

    cur_loss = None
    pending = None

    send(f"<b>{LABEL} watch started</b>")

    while True:
        if not os.path.exists(LOG):
            time.sleep(5)
            continue
        with open(LOG, "r", errors="ignore") as f:
            f.seek(last_pos)
            for line in f:
                m = re.search(r"Mean training loss:\s+([\d.]+)", line)
                if m:
                    cur_loss = m.group(1)
                    continue

                m = re.search(r"Confusion Matrix \(epoch (\d+), Test\)", line)
                if m:
                    pending = {"epoch": m.group(1)}
                    continue

                if pending is not None:
                    m = re.search(r"Total Correct:\s+([\d.]+)/([\d.]+)", line)
                    if m:
                        pending["correct"] = f"{int(float(m.group(1)))}/{int(float(m.group(2)))}"
                        continue
                    m = re.search(r"Overall Accuracy:\s+([\d.]+)%", line)
                    if m:
                        pending["acc"] = m.group(1)
                        continue
                    m = re.search(r"oracle=([\d.]+)% fusion\[a=([\d.]+)\]=([\d.]+)%", line)
                    if m:
                        msg = (
                            f"<b>{LABEL} Ep {pending['epoch']}</b>\n"
                            f"acc {pending.get('acc','?')}%  ({pending.get('correct','?')})\n"
                            f"loss {cur_loss}\n"
                            f"oracle {m.group(1)}%  fusion[a={m.group(2)}]={m.group(3)}%"
                        )
                        send(msg)
                        print(msg, flush=True)
                        pending = None
            last_pos = f.tell()
        with open(STATE, "w") as f:
            f.write(str(last_pos))
        time.sleep(15)


if __name__ == "__main__":
    main()
