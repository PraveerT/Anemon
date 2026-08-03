'use strict';
const { app, BrowserWindow, Notification, shell } = require('electron');
const bus = require('./server');

// The window loads over http from the local bus rather than file://, so the
// renderer is same-origin with the API and needs no preload bridge and no
// node integration.

const TITLE = 'Anemon Bus';
let win = null;
let unread = 0;

// Windows will not show a toast from an unpackaged app unless it can attribute
// it to an app id.
app.setAppUserModelId('com.anemon.bus');

// First line worth reading: skip YAML frontmatter, headings and blank lines.
function preview(body) {
  if (!body) return 'sealed submission, unreadable until both halves are in';
  const text = String(body).replace(/\r\n/g, '\n')
    .replace(/^---\n[\s\S]*?\n---\n/, '');
  const line = text.split('\n')
    .map((l) => l.trim())
    .find((l) => l && !l.startsWith('#') && !l.startsWith('---')) || text.trim();
  return line.length > 180 ? line.slice(0, 177) + '...' : line;
}

function notify(m) {
  // The user's own messages are not news to the user.
  if (m.from === 'user') return;
  if (!win || win.isDestroyed()) return;
  if (win.isFocused()) return;

  unread++;
  win.setTitle(`${TITLE} (${unread})`);
  win.flashFrame(true);

  if (!Notification.isSupported()) return;
  const n = new Notification({
    title: `${m.from} to ${m.to}   [${m.kind}${m.sealId ? ' ' + m.sealId : ''}]`,
    body: m.summary || preview(m.body),
    silent: false,
  });
  n.on('click', () => {
    if (win.isMinimized()) win.restore();
    win.show();
    win.focus();
  });
  n.show();
}

async function boot() {
  const url = await bus.start();
  win = new BrowserWindow({
    width: 1180,
    height: 860,
    minWidth: 720,
    backgroundColor: '#f4f6f9',
    title: TITLE,
    autoHideMenuBar: true,
    webPreferences: { contextIsolation: true, nodeIntegration: false },
  });
  win.loadURL(url);

  // loadURL would otherwise let the page's <title> override ours.
  win.on('page-title-updated', (e) => e.preventDefault());

  win.on('focus', () => {
    unread = 0;
    win.setTitle(TITLE);
    win.flashFrame(false);
  });

  win.webContents.setWindowOpenHandler(({ url: target }) => {
    shell.openExternal(target);
    return { action: 'deny' };
  });
}

bus.events.on('message', notify);

app.whenReady().then(boot);

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});

app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) boot();
});
