// Chrome will not offer to install a PWA without a service worker that has a
// fetch handler, even if the manifest is perfect. This is the minimum that
// satisfies that, and it earns its keep by making the shell load offline.
//
// Deliberately NOT caching /api/*: the whole point of this app is live state,
// and a stale cached /api/state would show a frozen conversation that looks
// current. Only the shell is cached; data always goes to the network.

const CACHE = 'anemon-shell-v1';
const SHELL = [
  '/',
  '/style.css',
  '/renderer.js',
  '/manifest.json',
  '/icons/icon-192.png',
];

self.addEventListener('install', (e) => {
  e.waitUntil(caches.open(CACHE).then((c) => c.addAll(SHELL))
    .then(() => self.skipWaiting()));
});

self.addEventListener('activate', (e) => {
  e.waitUntil(caches.keys()
    .then((keys) => Promise.all(keys.filter((k) => k !== CACHE)
      .map((k) => caches.delete(k))))
    .then(() => self.clients.claim()));
});

self.addEventListener('fetch', (e) => {
  const url = new URL(e.request.url);

  // Live data and the event stream never come from cache.
  if (url.pathname.startsWith('/api/')) return;
  if (e.request.method !== 'GET') return;

  // Network first, so a shell edit shows up on the next load rather than
  // being pinned until the cache version is bumped. Cache is the fallback.
  e.respondWith(
    fetch(e.request)
      .then((res) => {
        const copy = res.clone();
        caches.open(CACHE).then((c) => c.put(e.request, copy)).catch(() => {});
        return res;
      })
      .catch(() => caches.match(e.request).then((r) => r || Response.error()))
  );
});
