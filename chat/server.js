'use strict';
// Local message bus for two CLI agents. No dependencies.
//
// Storage is an append-only JSONL log, so a crash loses nothing and the whole
// conversation stays greppable from the shell. Live updates go out over SSE
// rather than a websocket to keep the dependency count at zero.

const http = require('node:http');
const fs = require('node:fs');
const path = require('node:path');
const { EventEmitter } = require('node:events');

const ROOT = __dirname;
// BUS_DATA points the log somewhere else, which is how you run a throwaway bus
// for testing without writing into the real conversation.
const DATA = process.env.BUS_DATA
  ? path.resolve(process.env.BUS_DATA) : path.join(ROOT, 'data');
const LOG = path.join(DATA, 'messages.jsonl');
const CURSORS = path.join(DATA, 'cursors.json');

const AGENTS = ['claude', 'deepseek', 'opus'];
const KINDS = ['brief', 'challenge', 'verdict', 'predict', 'data', 'proposal',
               'note', 'sealed'];
// PORT is what Railway and most hosts inject; BUS_PORT is the local override.
const PORT = Number(process.env.PORT || process.env.BUS_PORT || 8787);

// Phone access. Binding beyond localhost without auth would let anything on the
// LAN POST as any agent, and the log has no other notion of identity, so a
// forged message would be indistinguishable from a real one. Loopback stays
// open so the local agent CLIs need no changes; every non-loopback request must
// carry the token, either as ?t= or in the cookie that a valid ?t= sets.
const HOST = process.env.BUS_HOST || '0.0.0.0';

// Auth is OPT-IN: off unless BUS_TOKEN is set. On a LAN that is the user's call.
// On a public host it means anyone with the URL can read the whole log and post
// as any agent, so set BUS_TOKEN in the platform's env if the deploy is public.
function loadToken() {
  const t = (process.env.BUS_TOKEN || '').trim();
  return t || null;
}

function lanAddress() {
  const nets = require('node:os').networkInterfaces();
  for (const name of Object.keys(nets)) {
    for (const ni of nets[name] || []) {
      if (ni.family === 'IPv4' && !ni.internal) return ni.address;
    }
  }
  return null;
}

function isLoopback(req) {
  const a = req.socket.remoteAddress || '';
  return a === '127.0.0.1' || a === '::1' || a === '::ffff:127.0.0.1';
}

fs.mkdirSync(DATA, { recursive: true });

let messages = [];
let seq = 0;
if (fs.existsSync(LOG)) {
  messages = fs.readFileSync(LOG, 'utf8').split('\n').filter(Boolean)
    .map((l) => JSON.parse(l));
  seq = messages.reduce((m, x) => Math.max(m, x.seq), 0);
}

let cursors = fs.existsSync(CURSORS)
  ? JSON.parse(fs.readFileSync(CURSORS, 'utf8')) : {};

const clients = new Set();

// Who currently has a /api/wait connection open. This is real presence, not a
// heartbeat: an agent is "listening" exactly when it is parked and will be woken.
// The distinction matters because a message sent to an agent that is not parked
// sits unread until someone runs the CLI again.
const listening = new Map();

// agent -> the finish() of its currently parked /api/wait, so a second wait can
// release the first rather than racing it on the shared cursor.
const waiters = new Map();

function presence() {
  return Object.fromEntries(AGENTS.map((a) => [a, (listening.get(a) || 0) > 0]));
}

function setListening(agent, delta) {
  const n = Math.max(0, (listening.get(agent) || 0) + delta);
  const was = (listening.get(agent) || 0) > 0;
  listening.set(agent, n);
  if (was !== (n > 0)) broadcast('presence', presence());
}

// WhatsApp-style typing state. An agent posts /api/typing when it starts
// composing; send and seal clear it; the TTL prunes it if the agent dies.
// Presence says who can be woken, typing says who is writing right now.
const typing = new Map();
const TYPING_TTL = 120 * 1000;

function typingState() {
  const now = Date.now();
  for (const [a, ts] of typing) {
    if (now - ts > TYPING_TTL) typing.delete(a);
  }
  return Object.fromEntries(AGENTS.map((a) => [a, typing.has(a)]));
}

function setTyping(agent, on) {
  const before = typingState();
  if (on) typing.set(agent, Date.now());
  else typing.delete(agent);
  const after = typingState();
  if (JSON.stringify(before) !== JSON.stringify(after)) broadcast('typing', after);
}

// Anything that wants to react to traffic without holding an SSE socket: the
// Electron shell for desktop notifications, and the long-poll route below.
const events = new EventEmitter();
events.setMaxListeners(50);

function broadcast(event, data) {
  const frame = `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
  for (const c of clients) {
    try { c.write(frame); } catch { clients.delete(c); }
  }
}

// The human reads summaries, not bodies. Agents write their own: a truncation
// of the first 300 characters is not a summary, it is a lede.
// Soft target is about 300. This is the hard ceiling that stops a whole body
// being pasted into the summary field; it must sit well above the target so a
// summary that runs slightly long is stored intact rather than silently cut.
const SUMMARY_MAX = 1200;

function summarise(s) {
  if (!s) return null;
  const t = String(s).replace(/\s+/g, ' ').trim();
  return t ? t.slice(0, SUMMARY_MAX) : null;
}

// Same shape as the GPU box's sidepanel publisher: the source of truth pushes
// to the hosted instance, the hosted instance holds no independent state. That
// beats making the agents talk to Railway directly, because the local bus keeps
// working when the network is down and catches up afterwards.
const MIRROR = (process.env.BUS_MIRROR || '').replace(/\/+$/, '');
const MIRROR_TOKEN = process.env.ANEMON_PUBLISH_TOKEN || '';

// Railway is https, but a mirror on the LAN would be http and hardcoding one
// fails with an unhelpful parse error rather than a connection error.
const transport = () =>
  require(MIRROR.startsWith('http://') ? 'node:http' : 'node:https');

function mirrorPush(m) {
  if (!MIRROR) return;
  const body = Buffer.from(JSON.stringify(m));
  const headers = { 'content-type': 'application/json',
                    'content-length': body.length };
  if (MIRROR_TOKEN) headers['authorization'] = `Bearer ${MIRROR_TOKEN}`;
  const req = transport().request(
    `${MIRROR}/api/chat-publish`, { method: 'POST', headers, timeout: 8000 },
    (res) => { res.resume(); });
  // Never let a mirror failure touch the local bus. It catches up on restart.
  req.on('error', (e) => console.log('mirror push failed:', e.message));
  req.on('timeout', () => req.destroy());
  req.end(body);
}

function append(m) {
  m.seq = ++seq;
  m.ts = Date.now();
  messages.push(m);
  fs.appendFileSync(LOG, JSON.stringify(m) + '\n');
  events.emit('message', redact(m, null));
  mirrorPush(m);
  return m;
}

// --- sealed submissions ------------------------------------------------
// A sealed message stays unreadable until every agent has submitted for the
// same sealId. That is the anchoring guard: whoever answers second must not be
// able to read the first answer. The server enforces it, not the agents.

function sealComplete(sealId) {
  const have = new Set(messages
    .filter((m) => m.kind === 'sealed' && m.sealId === sealId)
    .map((m) => m.from));
  return AGENTS.every((a) => have.has(a));
}

function sealMissing(sealId) {
  const have = new Set(messages
    .filter((m) => m.kind === 'sealed' && m.sealId === sealId)
    .map((m) => m.from));
  return AGENTS.filter((a) => !have.has(a));
}

function redact(m, viewer) {
  if (m.kind !== 'sealed') return m;
  if (sealComplete(m.sealId)) return { ...m, sealed: false };
  if (viewer && m.from === viewer) return { ...m, sealed: true };
  // The summary would leak the position the seal exists to hide.
  return { ...m, body: null, summary: null, sealed: true,
           missing: sealMissing(m.sealId) };
}

// --- routing -----------------------------------------------------------

function json(res, code, obj) {
  const b = Buffer.from(JSON.stringify(obj));
  res.writeHead(code, { 'content-type': 'application/json',
                        'content-length': b.length });
  res.end(b);
}

function body(req) {
  return new Promise((resolve, reject) => {
    let s = '';
    req.on('data', (c) => {
      s += c;
      if (s.length > 4e6) reject(new Error('body too large'));
    });
    req.on('end', () => {
      try { resolve(s ? JSON.parse(s) : {}); } catch (e) { reject(e); }
    });
  });
}

const STATIC = {
  '/': ['index.html', 'text/html; charset=utf-8'],
  '/index.html': ['index.html', 'text/html; charset=utf-8'],
  '/renderer.js': ['renderer.js', 'text/javascript; charset=utf-8'],
  '/style.css': ['style.css', 'text/css; charset=utf-8'],
  // Home-screen install. Without the manifest the browser keeps its chrome.
  '/manifest.json': ['manifest.json', 'application/manifest+json'],
  // Must be served from the root, or its scope cannot cover the whole app.
  '/sw.js': ['sw.js', 'text/javascript; charset=utf-8'],
  '/icons/apple-touch-icon.png': ['icons/apple-touch-icon.png', 'image/png'],
  '/icons/icon-192.png': ['icons/icon-192.png', 'image/png'],
  '/icons/icon-512.png': ['icons/icon-512.png', 'image/png'],
};

const TOKEN = loadToken();

async function handle(req, res) {
  const url = new URL(req.url, `http://127.0.0.1:${PORT}`);
  const p = url.pathname;

  if (TOKEN && !isLoopback(req)) {
    const q = url.searchParams.get('t');
    const cookie = /(?:^|;\s*)bus=([a-f0-9]+)/.exec(req.headers.cookie || '');
    if (q === TOKEN) {
      // Sets the cookie so the page's own fetches for /renderer.js and /api/*
      // carry it without a token in every URL.
      res.setHeader('set-cookie',
                    `bus=${TOKEN}; Path=/; Max-Age=31536000; SameSite=Lax`);
    } else if (!cookie || cookie[1] !== TOKEN) {
      res.writeHead(401, { 'content-type': 'text/plain' });
      return res.end('missing or bad token');
    }
  }

  // What the Phone button needs. Loopback only: this hands out the token.
  if (req.method === 'GET' && p === '/api/phone') {
    if (!isLoopback(req)) return json(res, 403, { error: 'local only' });
    const ip = lanAddress();
    return json(res, 200, {
      ip, port: PORT, token: TOKEN,
      // No token configured means no ?t= to append; a literal "?t=null" would
      // both look wrong and be rejected the moment BUS_TOKEN was ever set.
      url: ip ? `http://${ip}:${PORT}/${TOKEN ? '?t=' + TOKEN : ''}` : null,
    });
  }

  if (req.method === 'GET' && STATIC[p]) {
    const [file, type] = STATIC[p];
    return fs.readFile(path.join(ROOT, file), (err, buf) => {
      if (err) { res.writeHead(404); return res.end('not found'); }
      res.writeHead(200, { 'content-type': type });
      res.end(buf);
    });
  }

  // Live stream for the UI.
  if (req.method === 'GET' && p === '/api/events') {
    res.writeHead(200, {
      'content-type': 'text/event-stream',
      'cache-control': 'no-cache',
      connection: 'keep-alive',
    });
    res.write(': connected\n\n');
    clients.add(res);
    const ping = setInterval(() => { try { res.write(': ping\n\n'); } catch {} },
                             25000);
    req.on('close', () => { clearInterval(ping); clients.delete(res); });
    return;
  }

  // Everything the UI needs to paint, with sealed bodies withheld.
  if (req.method === 'GET' && p === '/api/state') {
    const since = Number(url.searchParams.get('since') || 0);
    const viewer = url.searchParams.get('viewer') || null;
    return json(res, 200, {
      agents: AGENTS,
      kinds: KINDS,
      presence: presence(),
      typing: typingState(),
      cursors: { ...cursors },
      seq,
      messages: messages.filter((m) => m.seq > since).map((m) => redact(m, viewer)),
    });
  }

  // An agent draining its inbox. Advances that agent's cursor unless peeking.
  if (req.method === 'GET' && p === '/api/inbox') {
    const agent = url.searchParams.get('agent');
    if (!AGENTS.includes(agent)) {
      return json(res, 400, { error: `agent must be one of ${AGENTS}` });
    }
    const peek = url.searchParams.get('peek') === '1';
    const from = cursors[agent] || 0;
    const out = messages
      .filter((m) => m.seq > from && (m.to === agent || m.to === 'all'))
      .filter((m) => m.from !== agent)
      .map((m) => redact(m, agent));
    if (!peek && seq > from) {
      cursors[agent] = seq;
      fs.writeFileSync(CURSORS, JSON.stringify(cursors));
      broadcast('cursor', { agent, seq });
    }
    return json(res, 200, { agent, cursor: peek ? from : seq, messages: out });
  }

  if (req.method === 'POST' && p === '/api/send') {
    const b = await body(req);
    if (!b.from) return json(res, 400, { error: 'from required' });
    if (!b.body || !String(b.body).trim()) {
      return json(res, 400, { error: 'body required' });
    }
    const kind = b.kind || 'note';
    if (!KINDS.includes(kind)) {
      return json(res, 400, { error: `kind must be one of ${KINDS}` });
    }
    if (kind === 'sealed') {
      return json(res, 400, { error: 'use POST /api/seal for sealed messages' });
    }
    const to = b.to || (b.from === 'claude' ? 'deepseek'
                        : b.from === 'deepseek' ? 'claude' : 'all');
    const m = append({
      from: b.from, to, kind,
      re: b.re == null ? null : Number(b.re),
      title: b.title || null,
      expects: b.expects || null,
      summary: summarise(b.summary),
      body: String(b.body),
    });
    if (AGENTS.includes(b.from)) setTyping(b.from, false);
    broadcast('message', redact(m, null));
    return json(res, 200, m);
  }

  // WhatsApp-style typing signal. Agents post it when composing starts.
  // Send and seal clear it, and the TTL prunes it if the agent dies.
  if (req.method === 'POST' && p === '/api/typing') {
    const b = await body(req);
    if (!AGENTS.includes(b.from)) {
      return json(res, 400, { error: `from must be one of ${AGENTS}` });
    }
    setTyping(b.from, Boolean(b.state));
    return json(res, 200, { typing: typingState() });
  }

  if (req.method === 'POST' && p === '/api/seal') {
    const b = await body(req);
    if (!AGENTS.includes(b.from)) {
      return json(res, 400, { error: `from must be one of ${AGENTS}` });
    }
    if (!b.sealId) return json(res, 400, { error: 'sealId required' });
    if (!b.body || !String(b.body).trim()) {
      return json(res, 400, { error: 'body required' });
    }
    const dup = messages.find((m) => m.kind === 'sealed'
      && m.sealId === b.sealId && m.from === b.from);
    if (dup) {
      return json(res, 409, {
        error: `${b.from} already sealed ${b.sealId} at seq ${dup.seq}`,
      });
    }
    const m = append({
      from: b.from, to: 'all', kind: 'sealed', sealId: String(b.sealId),
      re: b.re == null ? null : Number(b.re),
      title: b.title || null, expects: null,
      summary: summarise(b.summary),
      body: String(b.body),
    });
    const complete = sealComplete(m.sealId);
    if (AGENTS.includes(b.from)) setTyping(b.from, false);
    broadcast('message', redact(m, null));
    if (complete) {
      // Repaint both halves now that they are readable.
      broadcast('reveal', {
        sealId: m.sealId,
        messages: messages
          .filter((x) => x.kind === 'sealed' && x.sealId === m.sealId)
          .map((x) => redact(x, null)),
      });
    }
    return json(res, 200, {
      seq: m.seq, sealId: m.sealId, complete,
      missing: complete ? [] : sealMissing(m.sealId),
    });
  }

  // Read a sealed pair. Refuses until both halves exist.
  if (req.method === 'GET' && p === '/api/seal') {
    const sealId = url.searchParams.get('id');
    if (!sealId) return json(res, 400, { error: 'id required' });
    const subs = messages.filter((m) => m.kind === 'sealed'
      && m.sealId === sealId);
    if (!subs.length) return json(res, 404, { error: `no seal ${sealId}` });
    if (!sealComplete(sealId)) {
      return json(res, 425, {
        error: 'sealed', sealId, missing: sealMissing(sealId),
        have: subs.map((m) => m.from),
      });
    }
    return json(res, 200, { sealId, complete: true, messages: subs });
  }

  // Long poll. Returns immediately if the agent already has unread traffic,
  // otherwise holds the connection until something arrives or the timeout.
  // Deliberately does NOT advance the cursor: the caller follows up with
  // /api/inbox to actually read.
  if (req.method === 'GET' && p === '/api/wait') {
    const agent = url.searchParams.get('agent');
    if (!AGENTS.includes(agent)) {
      return json(res, 400, { error: `agent must be one of ${AGENTS}` });
    }
    const forMe = (m) => m.from !== agent && (m.to === agent || m.to === 'all');
    const from = cursors[agent] || 0;
    const pending = messages.filter((m) => m.seq > from && forMe(m));
    if (pending.length) {
      return json(res, 200, { waited: false, timedOut: false,
                              count: pending.length, seq: pending[0].seq });
    }
    const ms = Math.min(Math.max(Number(url.searchParams.get('timeout')) || 300, 5),
                        3600) * 1000;
    let done = false;
    setListening(agent, +1);
    const finish = (payload) => {
      if (done) return;
      done = true;
      clearTimeout(timer);
      events.off('message', onMsg);
      setListening(agent, -1);
      if (waiters.get(agent) === finish) waiters.delete(agent);
      json(res, 200, payload);
    };

    // Single waiter per agent, newest wins. Two waits for one agent share one
    // cursor, so whichever is woken first drains the message and the other
    // returns empty: the message is not lost from the log, but it never
    // reaches the agent. Observed live. Releasing the older waiter makes an
    // accidental double-park self-healing instead of silently lossy.
    const prev = waiters.get(agent);
    if (prev) prev({ waited: true, timedOut: false, superseded: true, count: 0 });
    waiters.set(agent, finish);
    const onMsg = (m) => {
      if (forMe(m)) {
        finish({ waited: true, timedOut: false, count: 1, seq: m.seq,
                 from: m.from, kind: m.kind });
      }
    };
    const timer = setTimeout(
      () => finish({ waited: true, timedOut: true, count: 0 }), ms);
    events.on('message', onMsg);
    req.on('close', () => {
      if (done) return;
      done = true;
      clearTimeout(timer);
      events.off('message', onMsg);
      setListening(agent, -1);
    });
    return;
  }

  // Receive a mirrored message from the source-of-truth bus. Verbatim: seq and
  // ts are preserved, so the hosted copy and the local one agree on ids and a
  // reply that says "re #137" means the same thing on a phone as on the desk.
  // Idempotent, because catch-up after a restart will resend.
  if (req.method === 'POST' && p === '/api/mirror') {
    const m = await body(req);
    if (!m || typeof m.seq !== 'number' || !m.from) {
      return json(res, 400, { error: 'need a full message record with seq' });
    }
    if (messages.some((x) => x.seq === m.seq)) {
      return json(res, 200, { ok: true, duplicate: true, seq: m.seq });
    }
    messages.push(m);
    messages.sort((a, b) => a.seq - b.seq);
    seq = Math.max(seq, m.seq);
    fs.appendFileSync(LOG, JSON.stringify(m) + '\n');
    events.emit('message', redact(m, null));
    broadcast('message', redact(m, null));
    return json(res, 200, { ok: true, seq: m.seq });
  }

  json(res, 404, { error: 'no route' });
}

const server = http.createServer((req, res) => {
  handle(req, res).catch((e) => json(res, 500, { error: String(e.message) }));
});

// A push that failed while the network was down would otherwise leave a
// permanent hole, since append() only ever fires once per message. On startup,
// ask the mirror how far it got and resend everything after that.
function mirrorCatchUp() {
  if (!MIRROR) return;
  const url = `${MIRROR}/api/chat-publish`;
  const opts = MIRROR_TOKEN
    ? { headers: { authorization: `Bearer ${MIRROR_TOKEN}` } } : {};
  transport().get(url, opts, (res) => {
    let buf = '';
    res.on('data', (c) => { buf += c; });
    res.on('end', () => {
      let theirs = 0;
      try { theirs = JSON.parse(buf).maxSeq || 0; } catch { return; }
      const behind = messages.filter((m) => m.seq > theirs);
      if (!behind.length) return;
      console.log(`mirror is at #${theirs}, pushing ${behind.length} message(s)`);
      behind.forEach(mirrorPush);
    });
  }).on('error', (e) => console.log('mirror catch-up failed:', e.message));
}

function start() {
  return new Promise((resolve) => {
    server.listen(PORT, HOST, () => {
      mirrorCatchUp();
      console.log(`bus on http://127.0.0.1:${PORT}  (${messages.length} messages)`);
      resolve(`http://127.0.0.1:${PORT}`);
    });
  });
}

module.exports = { start, PORT, events };

if (require.main === module) start();
