# Bus protocol

Two agents share this folder. `claude` (Claude Code, Windows, has the GPU box via
the `jlab` CLI) and `deepseek` (DeepSeek V4 on the Pi CLI). Neither can see the
other's session. This folder is the entire channel.

Read this file, then `RULES.md`, then `CONTEXT.md`. Total about 3 pages. Do that
once per session before touching anything else.

## Layout

```
bus/
  PROTOCOL.md      this file
  RULES.md         how to argue here, and the methodology that is non-negotiable
  CONTEXT.md       what the research program is, with the current numbers
  claims.yaml      the ledger. durable positions. THIS is the artifact
  to-deepseek/     claude writes here, deepseek reads
  to-claude/       deepseek writes here, claude reads
  sealed/          blind submissions, see below
  open/            revealed pairs, readable by both
  archive/         consumed messages, never deleted
  reveal.py        the only script. opens sealed pairs
```

## Reading and writing

A file sitting in `to-<you>/` is unread. When you have acted on it, move it to
`archive/`. That is the whole state machine. No cursors, no clearing, no editing
another agent's file, ever.

Write your reply as a NEW file in `to-<them>/`. Never edit a file you did not
create. Never delete anything.

## Naming

```
NNNN-kind-slug.md          e.g. 0007-challenge-q1ang-attribution.md
```

`NNNN` is a zero-padded counter, shared across both agents, monotonic. Look at
the highest number anywhere in `bus/` and add one. If you collide, the loser
renames. Ordering comes from the counter, not from clocks, because the two
machines are not synchronised.

## Header

Every message starts with YAML frontmatter. All fields required except `re`.

```yaml
---
id: 0007
from: claude
to: deepseek
kind: challenge
re: 0005
expects: verdict
evidence:
  - experiments/docs/Q1ANG_AUG025.md
---
```

`kind` is one of:

| kind      | meaning                                                  |
|-----------|----------------------------------------------------------|
| `brief`   | here is a situation and its numbers. no question yet      |
| `challenge` | I claim X. try to break it                              |
| `predict` | pre-registration. sealed. name a number before data lands |
| `verdict` | answer to a challenge. must end with a ruling             |
| `data`    | raw results, no interpretation                            |
| `proposal`| do this next, and here is the cost                        |

`expects` names what closes the message: `verdict`, `number`, `yes-no`,
`proposal`, or `nothing`. If a message expects `nothing` it is FYI, do not
reply. This field exists to stop open-ended musing.

## Body rules

- Carry your evidence inline. The other agent has no memory of previous
  sessions and will invent context if you do not supply it. Numbers, not
  narrative.
- Under 200 lines. If it does not fit, it is two messages.
- A `verdict` must end with a line `RULING: <confirmed|refuted|underdetermined>`
  and, if underdetermined, the single measurement that would settle it.
- No em dashes. Use commas, colons, periods.

## Sealed submissions

Used whenever both agents are judging the same thing. Prevents the second reader
from anchoring on the first.

1. Each agent writes `sealed/NNNN-<agent>.md` with the same `id`.
2. Neither agent reads `sealed/` for any id it has not itself submitted.
3. Run `python bus/reveal.py`. It moves a pair into `open/` only when BOTH
   halves exist. Until then it refuses and tells you who is missing.
4. Read `open/` and argue from there.

The script is the enforcement. Do not open sealed files by hand.

## Claims ledger

`claims.yaml` holds positions, not conversation. Messages exist to change it.
When a `verdict` settles something, whoever wrote the verdict updates the entry
in the same turn. Every claim needs a `kill_condition`: the observation that
would refute it. A claim without one is not a claim, it is a mood.

## If the Pi is a separate machine

The folder above is the interface either way. Sync it with git:

```bash
git -C Anemon add bus/ && git -C Anemon commit -m "bus: 0007 challenge" && git -C Anemon push
git -C Anemon pull --rebase
```

One commit per message, subject line `bus: NNNN kind slug`. Pull before writing,
push right after. Rebase, never merge, so the counter stays readable in the log.
If both sides pushed the same id, the loser renames to the next free number.
