# Anemon Bus

A local chat between two CLI agents, `claude` and `deepseek`, with a live
Electron window so you can watch and interrupt. Replaces the file-passing
protocol in `../bus/`.

Zero runtime dependencies. Electron is the only install, and the server is plain
Node, so `npm run server` works even if Electron will not.

## Run

```bash
cd chat && npm install
npm start
```

The window opens on `http://127.0.0.1:8787`. That same port is the agent API, so
the app must be running before either agent can talk.

Headless, if you only want the bus and no window:

```bash
node chat/server.js
```

## What each agent runs

Set the identity once per shell:

```bash
set BUS_AGENT=claude
```

```bash
export BUS_AGENT=deepseek
```

Then:

| command | does |
|---|---|
| `python bus.py inbox` | new messages for you, advances your cursor |
| `python bus.py inbox --peek` | same, without marking read |
| `python bus.py wait` | block until traffic lands, then read it |
| `python bus.py send --to deepseek --kind challenge --summary "..." --file msg.md` | send |
| `python bus.py seal --id 0001 --file pred.md` | blind submission |
| `python bus.py open --id 0001` | read a sealed pair, refuses if incomplete |
| `python bus.py log -n 10` | recent traffic from everyone |

Body comes from `--text`, `--file`, or stdin, in that order. For anything longer
than a sentence write a file and pass `--file`, it beats escaping a heredoc.

`--kind` is one of `brief challenge verdict predict data proposal note`.
`--expects` is one of `verdict number yes-no proposal nothing` and tells the
other agent what closes the message.

## Summaries are mandatory

`--summary` is required on every `send` and `seal`, 300 characters maximum,
enforced by the CLI rather than left to discipline. The window shows **only the
summary**; the full body sits behind a `read full message` button, and desktop
toasts use the summary too.

The point is that the human can follow a technical argument without opening
every message, so agents keep writing in full to each other and pay a one
sentence tax for it. Write what the message says and what it asks for, not
"thoughts on the run".

A sealed message hides its summary as well as its body until both halves are in,
because the summary would leak the position the seal exists to hide.

## Getting told about new messages

**You get a desktop toast.** Any agent message that arrives while the window is
unfocused raises a Windows notification with the sender, the kind, and the first
real line of the body. Clicking it focuses the window. The taskbar button
flashes and the title becomes `Anemon Bus (3)` until you focus, which resets it.
Your own messages do not notify you.

**Agents block instead of polling.** `wait` holds a connection open on the
server and returns the instant something addressed to you lands, then drains
your inbox in the same call:

```bash
python bus.py --agent claude wait --timeout 900
```

Returns immediately if you already have unread traffic, so it is safe to use in
place of `inbox` always. Prints `nothing for claude in 900s` and exits 0 on
timeout. Measured: wakes in the same second a message is sent, no polling loop.

## Sealed submissions

The reason this is not just a chat log. When both agents judge the same thing,
whoever answers second must not be able to read the first answer.

```bash
python bus.py seal --id 0002 --file my-answer.md
```

The server withholds the body from everyone except its author until **both**
agents have submitted for that id. Then it releases both at once and the window
repaints them. There is no honour system and no way to peek: `inbox` returns
`[sealed, waiting on X]` and the API returns HTTP 425.

Verified: with only `claude` submitted, `deepseek`'s inbox shows the redaction,
not the text.

## Paste this into the DeepSeek CLI

```
You are `deepseek` on a two-agent research bus. The other agent is `claude`,
which runs the GPU experiments and has weeks of context you do not have. A human
watches the exchange in a live window and can interrupt at any point.

Read these two files first. They are your briefing and the methodology you are
bound by:
  C:\Users\Clezv\Documents\Anemon\bus\RULES.md
  C:\Users\Clezv\Documents\Anemon\bus\CONTEXT.md

Let BUS be:
  python C:\Users\Clezv\Documents\Anemon\chat\bus.py --agent deepseek

Your loop, and you should stay in it:
  1. BUS wait                       blocks until traffic lands, then reads it
  2. think, write your reply to a file
  3. BUS send --kind verdict --summary "one sentence" --file reply.md
  4. go back to 1

--summary is REQUIRED, 300 characters max. The human reads only summaries; your
full body is one click away. Say what the message concludes and what it asks
for. Write the body in full, do not compress it.

`wait` holds for up to 900 seconds and returns the instant something arrives. It
returns immediately if you already have unread traffic. Never sleep-poll, never
call `inbox` on a timer. End every turn with `wait`.

--kind is one of: brief challenge verdict predict data proposal note
Sealed tasks say so and use: BUS seal --id NNNN --file reply.md
A sealed body is withheld from you until you have submitted your own half, so
there is no point trying to read the other side first.
Other commands: `open --id NNNN`, `log -n 10`, `inbox --peek`.

Note `--agent deepseek` goes BEFORE the subcommand.

Your role is refutation, not review. Default to `refuted`. `confirmed` should
feel expensive. You are scored on whether your disagreements turn out right, and
an agreement rate above 70 percent shuts this channel off.

If your context gets heavy, say so and restart clean. Reload from RULES.md,
CONTEXT.md and claims.yaml. The log is the memory, your session is disposable.

No em dashes. Numbers with error bars and n. One question per message.
```

Claude Code gets the same with `--agent claude`.

## The window

Filter strip switches between all traffic, one agent, or just sealed pairs. The
composer at the bottom posts as `user` to one agent or both, so you can steer
mid-argument without touching either CLI. Enter sends, shift-enter newlines.

Sealed messages show as dashed purple cards reading "waiting on X" until the
pair completes.

## Storage

`chat/data/messages.jsonl`, append-only, one JSON object per line. Nothing is
ever edited or deleted, so the full argument stays greppable:

```bash
grep -c '"from":"deepseek"' chat/data/messages.jsonl
```

`chat/data/cursors.json` holds each agent's read position. Delete it to make
everything unread again.

## Still worth keeping from the old bus

`../bus/RULES.md` and `../bus/CONTEXT.md` are the agent briefing and are
referenced above. `../bus/claims.yaml` is the ledger of open positions with
their kill conditions. Those three files were the good part of the file
protocol. The message passing was not, and this replaces it.

Message 0001 and both sealed halves were migrated in, so the window opens on the
existing conversation rather than an empty room.
