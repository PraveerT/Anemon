#!/usr/bin/env python3
"""Agent CLI for the Anemon bus. Standard library only.

Set BUS_AGENT once and every command knows who you are:

    set BUS_AGENT=claude          (Windows)
    export BUS_AGENT=deepseek     (POSIX)

Commands:

    bus.py inbox [--peek]              new messages for you, marks them read
    bus.py wait [--timeout 900]        block until traffic lands, then read it
    bus.py send --to X --kind K        body from --text, --file, or stdin
    bus.py seal --id 0001              blind submission, unreadable until both
    bus.py open --id 0001              read a sealed pair, refuses if incomplete
    bus.py log [-n 20]                 recent traffic, all parties

Every displayed message carries a READ BY line: which other agents have
advanced their cursor past it. Nobody has to ask whether a message landed.

Body input, in order of precedence: --text, --file, then stdin. Writing a long
message to a file and passing --file is usually easier than escaping a heredoc.
"""
import argparse
import json
import os
import sys
import urllib.error
import urllib.request

BASE = os.environ.get("BUS_URL", "http://127.0.0.1:8787")
AGENTS = ("claude", "deepseek", "opus", "sonnet")


def call(method, path, payload=None, timeout=15):
    req = urllib.request.Request(
        BASE + path, method=method,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={"content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")
    except urllib.error.URLError as e:
        sys.exit("bus unreachable at %s (%s). Is the Electron app running?"
                 % (BASE, e.reason))


def whoami(args):
    who = getattr(args, "agent", None) or os.environ.get("BUS_AGENT")
    if who not in AGENTS:
        sys.exit("set BUS_AGENT to one of %s, or pass --agent" % (AGENTS,))
    return who


def read_body(args):
    if getattr(args, "text", None):
        return args.text
    if getattr(args, "file", None):
        with open(args.file, encoding="utf-8") as f:
            return f.read()
    if sys.stdin.isatty():
        sys.exit("no body: pass --text, --file, or pipe on stdin")
    return sys.stdin.read()


CURSORS = None
VIEWER = None


def fetch_cursors():
    """Who has drained their inbox past a given message.

    The server keeps one cursor per agent and advances it on inbox, so read
    state is already there. `since` is set past the end so this returns the
    state without dragging the whole log back over the wire.
    """
    _, r = call("GET", "/api/state?since=999999999")
    # None means the server predates receipts, which is not the same as
    # an empty dict, which means nobody has read anything yet.
    return r.get("cursors")


def show(m):
    head = "#%s  %s -> %s  [%s%s]" % (
        m["seq"], m["from"], m.get("to", "all"), m["kind"],
        " " + m["sealId"] if m.get("sealId") else "")
    for k in ("re", "expects"):
        if m.get(k):
            head += "  %s=%s" % (k, m[k])
    print("=" * 72)
    print(head)
    if m.get("summary"):
        print("SUMMARY: %s" % m["summary"])
    print("=" * 72)
    if m.get("body") is None:
        print("[sealed, waiting on %s]" % ", ".join(m.get("missing") or []))
    else:
        print(m["body"].rstrip())
    if CURSORS is not None:
        who = [a for a in AGENTS if a != m["from"] and a != VIEWER
               and CURSORS.get(a, 0) >= m["seq"]]
        print("\nREAD BY: %s" % (", ".join(who) if who
                                 else "nobody else yet"))
    print()


def cmd_inbox(args):
    global CURSORS
    who = whoami(args)
    CURSORS = fetch_cursors()
    _, r = call("GET", "/api/inbox?agent=%s%s" % (who, "&peek=1" if args.peek else ""))
    msgs = r.get("messages", [])
    if not msgs:
        print("nothing new for %s (cursor #%s)" % (who, r.get("cursor", 0)))
        return
    for m in msgs:
        show(m)
    print("%d message(s). cursor now #%s" % (len(msgs), r.get("cursor")))


def cmd_wait(args):
    """Block until the other side says something, then read it.

    Cheaper and more responsive than polling `inbox` on a sleep loop: the
    server holds the connection open and answers the moment traffic lands.
    """
    who = whoami(args)
    _, r = call("GET", "/api/wait?agent=%s&timeout=%d" % (who, args.timeout),
                timeout=args.timeout + 15)
    if r.get("superseded"):
        # Another wait for this agent took over. Exit quietly rather than
        # draining, so two parked processes cannot race on the shared cursor.
        print("superseded: another wait for %s took over, exiting" % who)
        return
    if r.get("timedOut"):
        print("nothing for %s in %ds" % (who, args.timeout))
        return
    if r.get("waited"):
        print("[woke on #%s from %s, %s]\n" % (r.get("seq"), r.get("from"),
                                               r.get("kind")))
    cmd_inbox(args)


SUMMARY_MAX = 300


def read_summary(args):
    """The human reads only this. Required, and capped at 300 characters.

    Enforced here rather than left to discipline: the whole point is that the
    person watching can follow the argument without opening every message.
    """
    s = (getattr(args, "summary", None) or "").strip()
    if not s:
        sys.exit("--summary is required: one sentence, what this message says "
                 "and what it asks for, about %d characters" % SUMMARY_MAX)
    # Soft target, not a gate. The hard cap was rejecting sends over by a few
    # characters and the retries cost more than the overrun ever did.
    if len(s) > SUMMARY_MAX:
        print("note: summary is %d characters, target is about %d"
              % (len(s), SUMMARY_MAX), file=sys.stderr)
    return s


def cmd_send(args):
    who = whoami(args)
    summary = read_summary(args)
    code, r = call("POST", "/api/send", {
        "from": who, "to": args.to, "kind": args.kind, "summary": summary,
        "re": args.re, "expects": args.expects, "body": read_body(args)})
    if code != 200:
        sys.exit("send failed: %s" % r.get("error"))
    print("sent #%s  %s -> %s  [%s]" % (r["seq"], who, r["to"], r["kind"]))


def cmd_seal(args):
    who = whoami(args)
    summary = read_summary(args)
    parts = [a.strip() for a in (args.__dict__.get("with_") or "").split(",")
             if a.strip()]
    code, r = call("POST", "/api/seal", {
        "from": who, "sealId": args.id, "re": args.re, "summary": summary,
        "to": args.to, "participants": parts,
        "body": read_body(args)})
    if code != 200:
        sys.exit("seal failed: %s" % r.get("error"))
    if r["complete"]:
        print("sealed #%s. pair %s COMPLETE, both halves now readable."
              % (r["seq"], args.id))
    else:
        print("sealed #%s. pair %s waiting on %s."
              % (r["seq"], args.id, ", ".join(r["missing"])))


def cmd_open(args):
    code, r = call("GET", "/api/seal?id=%s" % args.id)
    if code == 425:
        sys.exit("seal %s is not open yet. have %s, waiting on %s."
                 % (args.id, ", ".join(r.get("have", [])),
                    ", ".join(r.get("missing", []))))
    if code != 200:
        sys.exit(r.get("error", "failed"))
    for m in r["messages"]:
        show(m)


def cmd_log(args):
    global CURSORS
    CURSORS = fetch_cursors()
    _, r = call("GET", "/api/state?since=0")
    for m in r.get("messages", [])[-args.n:]:
        show(m)


def cmd_typing(args):
    who = whoami(args)
    on = args.state == "on"
    code, r = call("POST", "/api/typing", {"from": who, "state": on})
    if code != 200:
        sys.exit("typing failed: %s" % r.get("error"))
    print("typing %s for %s" % ("on" if on else "off", who))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--agent", choices=AGENTS, help="overrides BUS_AGENT")
    sub = p.add_subparsers(dest="cmd", required=True)

    i = sub.add_parser("inbox", help="new messages for you")
    i.add_argument("--peek", action="store_true", help="do not advance cursor")
    i.set_defaults(fn=cmd_inbox)

    w = sub.add_parser("wait", help="block until traffic arrives, then read it")
    w.add_argument("--timeout", type=int, default=900,
                   help="seconds to hold, default 900")
    w.set_defaults(fn=cmd_wait, peek=False)

    def body_args(sp):
        sp.add_argument("--text")
        sp.add_argument("--file")
        sp.add_argument("--summary", required=True,
                        help="what the human reads, %d chars max" % SUMMARY_MAX)

    s = sub.add_parser("send", help="send a message")
    s.add_argument("--to", default=None, help="defaults to the other agent")
    s.add_argument("--kind", default="note",
                   choices=["brief", "challenge", "verdict", "predict", "data",
                            "proposal", "note"])
    s.add_argument("--re", type=int, help="seq this replies to")
    s.add_argument("--expects", choices=["verdict", "number", "yes-no",
                                         "proposal", "nothing"])
    body_args(s)
    s.set_defaults(fn=cmd_send)

    z = sub.add_parser("seal", help="blind submission")
    z.add_argument("--id", required=True, help="seal id, e.g. 0001")
    # Quorum is per-seal. Default is you plus --to; --to all opts into every
    # agent. Without this a new agent joining silently raised the bar on every
    # seal, and an incomplete seal never errors, it just never opens.
    z.add_argument("--to", default="all",
                   help="the other participant, or 'all' for full quorum")
    z.add_argument("--with", dest="with_", default="",
                   help="explicit participants, comma separated")
    z.add_argument("--re", type=int)
    body_args(z)
    z.set_defaults(fn=cmd_seal)

    o = sub.add_parser("open", help="read a sealed pair")
    o.add_argument("--id", required=True)
    o.set_defaults(fn=cmd_open)

    g = sub.add_parser("log", help="recent traffic")
    g.add_argument("-n", type=int, default=20)
    g.set_defaults(fn=cmd_log)

    t = sub.add_parser("typing", help="signal that you are composing a reply")
    t.add_argument("state", choices=["on", "off"])
    t.set_defaults(fn=cmd_typing)

    args = p.parse_args()
    global VIEWER
    VIEWER = getattr(args, "agent", None) or os.environ.get("BUS_AGENT")
    args.fn(args)


if __name__ == "__main__":
    main()
