"""Open sealed pairs, and only pairs.

Blindness is the point of the sealed step, so this script is the enforcement
rather than the honour system. A submission stays unreadable in `sealed/` until
its counterpart exists, then both move to `open/` together.

    python bus/reveal.py            report status, open any complete pair
    python bus/reveal.py --status   report only, move nothing

Filenames are `NNNN-<agent>.md`. Anything else in `sealed/` is ignored and
reported, since a typo in the id is the one way to defeat this silently.
"""
import re
import sys
from collections import defaultdict
from pathlib import Path

BUS = Path(__file__).resolve().parent
SEALED, OPEN = BUS / "sealed", BUS / "open"
AGENTS = ("claude", "deepseek")
NAME = re.compile(r"^(\d{4})-(%s)\.md$" % "|".join(AGENTS))


def scan():
    pairs, junk = defaultdict(dict), []
    for f in sorted(SEALED.glob("*")):
        if not f.is_file():
            continue
        m = NAME.match(f.name)
        if m:
            pairs[m.group(1)][m.group(2)] = f
        else:
            junk.append(f)
    return pairs, junk


def main():
    dry = "--status" in sys.argv
    OPEN.mkdir(exist_ok=True)
    SEALED.mkdir(exist_ok=True)
    pairs, junk = scan()

    for f in junk:
        print("  IGNORED  %s  (expected NNNN-<%s>.md)" % (f.name, "|".join(AGENTS)))

    if not pairs:
        print("  sealed/ is empty")
        return

    for pid in sorted(pairs):
        have = pairs[pid]
        missing = [a for a in AGENTS if a not in have]
        if missing:
            print("  WAITING  %s  have %s, missing %s"
                  % (pid, ",".join(sorted(have)), ",".join(missing)))
            continue
        if dry:
            print("  READY    %s  (--status, not moved)" % pid)
            continue
        for agent in AGENTS:
            dest = OPEN / have[agent].name
            if dest.exists():
                print("  SKIP     %s  %s already in open/" % (pid, dest.name))
                continue
            have[agent].replace(dest)
        print("  OPENED   %s  -> open/%s-{%s}.md" % (pid, pid, ",".join(AGENTS)))


if __name__ == "__main__":
    main()
