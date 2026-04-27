"""Tail UMDR train log and push per-epoch summaries to Telegram.

Train log pattern:
  ... Epoch: N/50  Mini-Batch: ...  (per-iter)
  ********************
  ... Total_loss: ... Acc: ... Acc_top5: ...     <- train summary
  ********************
  ... Epoch: N  Mini-Batch: 0001/0085 ...        (per-val-iter, no /50)
  ********************
  ... Acc_adaptive: ... Acc_all: ... Acc_sm: ...  <- val summary
  ********************
"""
import os
import re
import sys
import time
import requests

LOG = "/tmp/umdr_train.log"
TG_TOKEN = "8049556095:AAH0c0KB0DmzFtcW0s97ZS_kQ8ux9gX72eE"
CHAT_ID = "7685170478"
STATE = "/tmp/umdr_tg_state"


def send(msg):
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
            data={"chat_id": CHAT_ID, "text": msg, "parse_mode": "HTML"},
            timeout=10,
        )
        return r.ok
    except Exception as e:
        print(f"tg fail: {e}", flush=True)
        return False


def parse_floats(line):
    return dict(re.findall(r"(\w+)\s*:\s*([\d.eE+-]+)", line))


def main():
    if os.path.exists(STATE):
        last_pos = int(open(STATE).read().strip() or 0)
    elif os.path.exists(LOG):
        last_pos = os.path.getsize(LOG)
    else:
        last_pos = 0

    cur_epoch = "?"
    in_train = False
    in_val = False
    state = "idle"  # idle -> open -> seen -> idle
    summary_kind = None

    send(f"<b>UMDR watch started</b>\nTailing {LOG}")

    while True:
        if not os.path.exists(LOG):
            time.sleep(5)
            continue
        with open(LOG, "r", errors="ignore") as f:
            f.seek(last_pos)
            for line in f:
                line = line.rstrip("\n")
                m_train = re.search(r"Epoch:\s+(\d+)/\d+\s+Mini-Batch", line)
                m_val = re.search(r"Epoch:\s+(\d+)\s+Mini-Batch:\s+\d+/\d+(?!.*?/\d{2,3})", line)
                if m_train:
                    cur_epoch = m_train.group(1)
                    in_train, in_val = True, False
                elif "Mini-Batch:" in line and re.search(r"Acc_adaptive", line):
                    in_train, in_val = False, True

                if line.strip() == "********************":
                    if state == "idle":
                        state = "open"
                        summary_kind = "val" if in_val else "train"
                    elif state == "seen":
                        state = "idle"
                        summary_kind = None
                    continue

                if state == "open" and ("Total_loss" in line or "Acc_adaptive" in line):
                    d = parse_floats(line)
                    if summary_kind == "train":
                        msg = (
                            f"<b>Ep {cur_epoch}</b>  TRAIN\n"
                            f"loss {d.get('Total_loss','?')}  "
                            f"acc {d.get('Acc','?')}  top5 {d.get('Acc_top5','?')}"
                        )
                    else:
                        msg = (
                            f"<b>Ep {cur_epoch}</b>  VAL\n"
                            f"adaptive {d.get('Acc_adaptive','?')}  "
                            f"all {d.get('Acc_all','?')}  "
                            f"sm {d.get('Acc_sm','?')}  "
                            f"sl {d.get('Acc_sl','?')}  "
                            f"lm {d.get('Acc_lm','?')}"
                        )
                    send(msg)
                    print(msg, flush=True)
                    state = "seen"
            last_pos = f.tell()
        with open(STATE, "w") as f:
            f.write(str(last_pos))
        time.sleep(15)


if __name__ == "__main__":
    main()
