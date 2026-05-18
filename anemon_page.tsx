"use client";

import { useEffect, useState, useCallback } from "react";

const STORAGE_KEY = "anemon_api_url";
const POLL_MS = 10000;
const RECENT_EPOCHS = 12;

type EpochRow = {
  ep: number;
  tr_acc: number | null;
  tr_loss: number | null;
  aux_loss?: number | null;
  te_p1: number | null;
  te_p5: number | null;
};

type LBRow = Record<string, string>;
type Leaderboard = Record<string, LBRow[][]> | null | undefined;

type Status = {
  ts?: string;
  run?: string;
  log?: string;
  gpu?: { used_gb: number; total_gb: number; util_pct: number } | null;
  ram?: { used_gb: number; total_gb: number } | null;
  disk?: { used_gb: number; total_gb: number } | null;
  epochs?: EpochRow[];
  best?: { ep: number; p1: number } | null;
  now?: { ep?: number; batch?: string; lr?: number | null };
  leaderboard?: Leaderboard;
};

const fmt = (n: number | null | undefined, d = 1) =>
  n == null || Number.isNaN(n) ? "—" : Number(n).toFixed(d);

function stripMd(s: string): string {
  return s
    .replace(/\*\*(.+?)\*\*/g, "$1")
    .replace(/\*(.+?)\*/g, "$1")
    .replace(/\s*\*\([^)]*\)\*/g, "")
    .trim();
}

function Bar({ pct, warn = 70, bad = 90 }: { pct: number; warn?: number; bad?: number }) {
  const clamped = Math.max(0, Math.min(100, pct));
  const color = clamped >= bad ? "#f66" : clamped >= warn ? "#fb6" : "#6bf";
  return (
    <div style={{ background: "#141414", height: 6, borderRadius: 3, overflow: "hidden", marginTop: 3 }}>
      <div style={{ width: `${clamped}%`, height: "100%", background: color, transition: "width 0.3s" }} />
    </div>
  );
}

function KV({ k, v, color }: { k: string; v: string; color?: string }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", minWidth: 70 }}>
      <span style={{ fontSize: 10, color: "#888", textTransform: "uppercase", letterSpacing: "0.6px" }}>{k}</span>
      <span style={{ fontSize: 14, fontWeight: 600, color: color || "#e8e8e8" }}>{v}</span>
    </div>
  );
}

function CompactTable({ rows }: { rows: LBRow[] }) {
  if (!rows || rows.length === 0) return null;
  const cols = Object.keys(rows[0]);
  return (
    <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11, tableLayout: "fixed" }}>
      <thead>
        <tr>
          {cols.map((c, i) => (
            <th key={c} style={{
              padding: "3px 4px",
              textAlign: i === cols.length - 1 ? "right" : "left",
              color: "#888",
              fontWeight: 500,
              textTransform: "uppercase",
              letterSpacing: "0.4px",
              fontSize: 9,
              borderBottom: "1px solid #252525",
            }}>{c}</th>
          ))}
        </tr>
      </thead>
      <tbody>
        {rows.map((r, ri) => (
          <tr key={ri}>
            {cols.map((c, i) => {
              const v = stripMd(r[c] ?? "");
              const isAcc = i === cols.length - 1 && /^[\d.]+$/.test(v);
              return (
                <td key={c} style={{
                  padding: "3px 4px",
                  textAlign: i === cols.length - 1 ? "right" : "left",
                  color: isAcc ? "#6f9" : "#d0d0d0",
                  fontWeight: isAcc ? 600 : 400,
                  wordBreak: "break-word",
                  lineHeight: 1.3,
                  verticalAlign: "top",
                }}>{v}</td>
              );
            })}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export default function AnemonPage() {
  const [status, setStatus] = useState<Status | null>(null);
  const [lastOk, setLastOk] = useState<number | null>(null);
  const [, setErr] = useState<string | null>(null);
  const [ago, setAgo] = useState<string>("…");

  // Lock the browser so the page itself can't scroll.
  useEffect(() => {
    const prevHtml = document.documentElement.style.cssText;
    const prevBody = document.body.style.cssText;
    document.documentElement.style.cssText =
      "height:100%;overflow:hidden;overscroll-behavior:none;background:#0a0a0a;";
    document.body.style.cssText =
      "height:100%;margin:0;overflow:hidden;overscroll-behavior:none;background:#0a0a0a;position:fixed;inset:0;touch-action:none;";
    return () => {
      document.documentElement.style.cssText = prevHtml;
      document.body.style.cssText = prevBody;
    };
  }, []);

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const fromQuery = params.get("api");
    if (fromQuery) {
      localStorage.setItem(STORAGE_KEY, fromQuery);
      window.history.replaceState({}, "", window.location.pathname);
    }
  }, []);

  const fetchStatus = useCallback(async () => {
    try {
      const stored = typeof window !== "undefined" ? localStorage.getItem(STORAGE_KEY) : null;
      const q = stored ? `?api=${encodeURIComponent(stored)}` : "";
      const r = await fetch(`/api/anemon-status${q}`, { cache: "no-store" });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const data: Status = await r.json();
      setStatus(data);
      setLastOk(Date.now());
      setErr(null);
    } catch (e: unknown) {
      setErr(e instanceof Error ? e.message : String(e));
    }
  }, []);

  useEffect(() => {
    fetchStatus();
    const id = setInterval(fetchStatus, POLL_MS);
    return () => clearInterval(id);
  }, [fetchStatus]);

  useEffect(() => {
    const id = setInterval(() => {
      if (!lastOk) { setAgo("…"); return; }
      const s = Math.round((Date.now() - lastOk) / 1000);
      setAgo(s < 60 ? `${s}s ago` : `${Math.floor(s / 60)}m ${s % 60}s ago`);
    }, 1000);
    return () => clearInterval(id);
  }, [lastOk]);

  const last = status?.epochs && status.epochs.length > 0 ? status.epochs[status.epochs.length - 1] : null;
  const bestEp = status?.best?.ep;
  const stale = lastOk != null && (Date.now() - lastOk) / 1000 > 30;
  const recent = (status?.epochs ?? []).slice(-RECENT_EPOCHS).reverse();

  return (
    <main style={{
      position: "fixed",
      inset: 0,
      background: "#0a0a0a",
      color: "#e8e8e8",
      font: "13px/1.45 ui-monospace, 'SF Mono', Menlo, Consolas, monospace",
      padding: "env(safe-area-inset-top) env(safe-area-inset-right) env(safe-area-inset-bottom) env(safe-area-inset-left)",
      boxSizing: "border-box",
      display: "flex",
      flexDirection: "column",
      overflow: "hidden",
      overscrollBehavior: "none",
      touchAction: "none",
    }}>
      <header style={{ padding: "14px 16px 6px", display: "flex", justifyContent: "space-between", alignItems: "baseline", gap: 8, flexShrink: 0 }}>
        <h1 style={{ font: "600 16px/1 inherit", margin: 0, letterSpacing: "0.5px" }}>
          ANEMON · <span style={{ color: "#6bf" }}>{status?.run ?? "—"}</span>
        </h1>
        <span style={{ fontSize: 11, color: stale ? "#fb6" : "#888" }}>{ago}</span>
      </header>

      <section style={{ padding: "8px 16px", borderTop: "1px solid #252525", flexShrink: 0 }}>
        <div style={{ display: "flex", gap: 14, flexWrap: "wrap" }}>
          <KV k="epoch" v={status?.now?.ep != null ? String(status.now.ep) : "—"} />
          <KV k="batch" v={status?.now?.batch ?? "—"} />
          <KV k="lr" v={status?.now?.lr != null ? status.now.lr.toExponential(1) : "—"} />
          <KV k="best p1" v={status?.best ? `${fmt(status.best.p1, 2)}% (ep ${status.best.ep})` : "—"} color="#6f9" />
          <KV k="last p1" v={last ? `${fmt(last.te_p1, 2)}%` : "—"} />
          <KV k="last p5" v={last ? `${fmt(last.te_p5, 2)}%` : "—"} />
        </div>
      </section>

      <section style={{ padding: "8px 16px", borderTop: "1px solid #252525", flexShrink: 0 }}>
        <div style={{ display: "flex", gap: 14, flexWrap: "wrap" }}>
          <div style={{ flex: 1, minWidth: 160 }}>
            <KV k="gpu mem" v={status?.gpu ? `${fmt(status.gpu.used_gb, 1)} / ${fmt(status.gpu.total_gb, 1)} GB` : "—"} />
            {status?.gpu && <Bar pct={(status.gpu.used_gb / status.gpu.total_gb) * 100} warn={80} bad={95} />}
          </div>
          <div style={{ flex: 1, minWidth: 130 }}>
            <KV k="gpu util" v={status?.gpu ? `${status.gpu.util_pct}%` : "—"} />
            {status?.gpu && <Bar pct={status.gpu.util_pct} warn={101} bad={102} />}
          </div>
        </div>
        <div style={{ display: "flex", gap: 14, flexWrap: "wrap", marginTop: 10 }}>
          <div style={{ flex: 1, minWidth: 160 }}>
            <KV k="ram" v={status?.ram ? `${status.ram.used_gb} / ${status.ram.total_gb} GB` : "—"} />
            {status?.ram && <Bar pct={(status.ram.used_gb / status.ram.total_gb) * 100} />}
          </div>
          <div style={{ flex: 1, minWidth: 130 }}>
            <KV k="disk" v={status?.disk ? `${fmt(status.disk.used_gb, 1)} / ${status.disk.total_gb} GB` : "—"} />
            {status?.disk && <Bar pct={(status.disk.used_gb / status.disk.total_gb) * 100} />}
          </div>
        </div>
      </section>

      <section style={{ padding: "8px 16px", borderTop: "1px solid #252525", flex: 1, minHeight: 0, overflow: "hidden", display: "flex", flexDirection: "column" }}>
        {(() => {
          const hasAux = recent.some(e => e.aux_loss != null);
          const headers = hasAux
            ? ["ep", "tr%", "loss", "aux", "te p1", "te p5"]
            : ["ep", "tr%", "loss", "te p1", "te p5"];
          const fmtAux = (v: number | null | undefined) =>
            v == null || Number.isNaN(v) ? "—" : Number(v).toExponential(1);
          return (
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
              <thead>
                <tr>
                  {headers.map((h, i) => (
                    <th key={h} style={{
                      padding: "4px 6px",
                      textAlign: i === 0 ? "left" : "right",
                      color: "#888",
                      fontWeight: 500,
                      textTransform: "uppercase",
                      letterSpacing: "0.4px",
                      fontSize: 10,
                      borderBottom: "1px solid #252525",
                    }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {recent.map((e) => {
                  const isBest = e.ep === bestEp;
                  const color = isBest ? "#6f9" : "#e8e8e8";
                  const weight = isBest ? 700 : 400;
                  const cells = hasAux
                    ? [String(e.ep), fmt(e.tr_acc, 1), fmt(e.tr_loss, 3), fmtAux(e.aux_loss), fmt(e.te_p1, 2), fmt(e.te_p5, 2)]
                    : [String(e.ep), fmt(e.tr_acc, 1), fmt(e.tr_loss, 3), fmt(e.te_p1, 2), fmt(e.te_p5, 2)];
                  return (
                    <tr key={e.ep}>
                      {cells.map((v, i) => (
                        <td key={i} style={{
                          padding: "4px 6px",
                          textAlign: i === 0 ? "left" : "right",
                          color, fontWeight: weight,
                        }}>{v}</td>
                      ))}
                    </tr>
                  );
                })}
              </tbody>
            </table>
          );
        })()}
      </section>

      {status?.leaderboard?.["Top combo per fusion width (with DSN)"]?.[0] && (
        <section style={{ padding: "8px 16px", borderTop: "1px solid #252525", flexShrink: 0 }}>
          <div style={{ fontSize: 9, color: "#888", textTransform: "uppercase", letterSpacing: "0.6px", marginBottom: 6 }}>top by fusion width</div>
          <CompactTable rows={status.leaderboard["Top combo per fusion width (with DSN)"][0]} />
        </section>
      )}
    </main>
  );
}
