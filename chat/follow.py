#!/usr/bin/env python3
"""Live viewer for the Anemon bus. Human-facing, read-only.

Prints the last --tail messages, then streams new ones as they land using the
server's SSE endpoint. It never touches an agent cursor, so it cannot corrupt
the inbox state of either agent. Ctrl+C to stop.

Usage:
    python follow.py [--tail N] [--no-history]
    BUS_URL=http://127.0.0.1:8787 python follow.py
"""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

BASE = os.environ.get("BUS_URL", "http://127.0.0.1:8787")


def show(m):
    if m.get("body") is None:
        body = "[sealed, waiting on %s]" % ", ".join(m.get("missing") or [])
    else:
        body = m["body"].rstrip()
    head = "#%s  %s -> %s  [%s%s]" % (
        m["seq"], m["from"], m.get("to", "all"), m["kind"],
        " " + m["sealId"] if m.get("sealId") else "")
    for k in ("re", "expects"):
        if m.get(k):
            head += "  %s=%s" % (k, m[k])
    print("=" * 72)
    print(head)
    print("=" * 72)
    print(body)
    print()


def get_state(since):
    req = urllib.request.Request(BASE + "/api/state?since=%d" % since)
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read() or b"{}")


def stream_events():
    """Yield (event, payload) frames from the SSE stream."""
    req = urllib.request.Request(BASE + "/api/events")
    resp = urllib.request.urlopen(req, timeout=60)
    event, data = None, []
    for raw in resp:
        line = raw.decode("utf-8").rstrip("\r\n")
        if line == "":
            if data:
                yield event or "message", "\n".join(data)
            event, data = None, []
        elif line.startswith(":"):
            continue
        elif line.startswith("event:"):
            event = line[len("event:"):].strip()
        elif line.startswith("data:"):
            data.append(line[len("data:"):].strip())


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tail", type=int, default=5,
                   help="recent messages to print at start, default 5")
    p.add_argument("--no-history", action="store_true",
                   help="skip the tail, only stream new messages")
    args = p.parse_args()

    state = get_state(0)
    since = state.get("seq", 0)
    if not args.no_history:
        for m in state.get("messages", [])[-args.tail:]:
            show(m)
        print("[following %s, %d messages in log]" % (BASE, since))

    try:
        while True:
            try:
                # Re-sync after a reconnect so no message is lost in the gap.
                state = get_state(since)
                for m in state.get("messages", []):
                    if m.get("seq", 0) > since:
                        show(m)
                        since = m["seq"]
                for event, payload in stream_events():
                    if event == "message":
                        m = json.loads(payload)
                        if m.get("seq", 0) > since:
                            show(m)
                            since = m["seq"]
                    elif event == "reveal":
                        data = json.loads(payload)
                        print("[seal %s revealed, both halves readable]"
                              % data.get("sealId"))
                        for m in data.get("messages", []):
                            show(m)
            except (urllib.error.URLError, urllib.error.HTTPError,
                    ValueError, ConnectionResetError) as e:
                print("[stream dropped: %s, reconnecting in 2s]" % e,
                      file=sys.stderr)
                time.sleep(2)
    except KeyboardInterrupt:
        print("\n[follow stopped]")


if __name__ == "__main__":
    main()
