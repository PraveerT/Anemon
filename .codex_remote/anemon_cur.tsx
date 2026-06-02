"use client";

import { useEffect, useState, useCallback } from "react";

const STORAGE_KEY = "anemon_api_url";
const VIEW_KEY = "anemon_view_mode";
const POLL_MS = 10000;
const RECENT_EPOCHS = 12;

type EpochRow = {
  ep: number;
  tr_acc: number | null;
  tr_loss: number | null;
  aux_loss?: number | null;
  aux_acc?: number | null;
  aux_te?: number | null;
  te_p1: number | null;
  te_p5: number | null;
};

type SkewTrajRow = {
  ep: number;
  scale: number; wu: number; wv: number;
  head0: number; headf: number; desc_e: number;
  lag1: number; lag2: number;
  aux_acc?: number | null; aux_te?: number | null;
};

type SkewStats = {
  epoch: number;
  scale: number;
  wu?: number; wv?: number; head0?: number; headf?: number;
  alive?: boolean;
  comp_energy?: Record<string, number>;
  rank_usage?: number[];
  lag_energy?: number[];
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
  engram?: { epoch: number; out_norm: number; out_max: number } | null;
  qcc?: {
    epoch: number;
    qcc_scale: number;
    quat_inject_scale?: number;
    quat_inject_norm?: number;
    quat_inject_max?: number;
  } | null;
  cluster?: {
    epoch: number;
    cycle_proj_norm: number;
    cycle_proj_max: number;
    cluster_head_norm?: number;
    cluster_mass?: number[];
  } | null;
  param_counts?: {
    epoch: number;
    total_m: number;
    aux_m: number;
    main_m: number;
  } | null;
  cnxxl_delta?: { ep: number; te: number; base_te: number; delta: number }[] | null;
  refs?: {
    current_best: number;
    refs: { name: string; value: number; delta: number }[];
  } | null;
  available_refs?: {
    name: string;
    label: string;
    acc: number;
    perclass: { cls: number; wrong: number; total: number }[];
    epochs?: { ep: number; te: number }[];
  }[] | null;
  current_perclass?: { cls: number; wrong: number; total: number }[] | null;
  skew?: SkewStats | null;
  skew_traj?: SkewTrajRow[] | null;
};

const REF_KEY = "anemon_ref_name";
const SORT_KEY = "anemon_perclass_sort";
type PerclassSort = "wrong" | "cls" | "delta";

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

function MKV({ k, v, color }: { k: string; v: string; color?: string }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", minWidth: 48 }}>
      <span style={{ fontSize: 8.5, color: "#888", textTransform: "uppercase", letterSpacing: "0.4px" }}>{k}</span>
      <span style={{ fontSize: 11.5, fontWeight: 600, color: color || "#e8e8e8" }}>{v}</span>
    </div>
  );
}

function Chip({ label, value, color, border }: { label: string; value: string; color?: string; border?: string }) {
  return (
    <div style={{
      background: "#0f141b",
      border: `1px solid ${border ?? "#1f2937"}`,
      borderRadius: 3,
      padding: "2px 5px",
      display: "flex", flexDirection: "column",
      minWidth: 48,
    }}>
      <span style={{ fontSize: 8.5, color: "#888", letterSpacing: "0.3px" }}>{label}</span>
      <span style={{ fontSize: 11, color: color || "#e8e8e8", fontWeight: 600 }}>{value}</span>
    </div>
  );
}

function SubHead({ children }: { children: React.ReactNode }) {
  return (
    <div style={{ fontSize: 8.5, color: "#666", textTransform: "uppercase", letterSpacing: "0.5px", marginBottom: 4 }}>
      {children}
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
  const [viewMode, setViewMode] = useState<"compact" | "detailed">("compact");
  const [selectedRef, setSelectedRef] = useState<string | null>(null);
  const [perclassSort, setPerclassSort] = useState<PerclassSort>("wrong");
  useEffect(() => {
    if (typeof window !== "undefined") {
      const saved = window.localStorage.getItem(VIEW_KEY);
      if (saved === "detailed" || saved === "compact") setViewMode(saved);
      const r = window.localStorage.getItem(REF_KEY);
      if (r) setSelectedRef(r);
      const s = window.localStorage.getItem(SORT_KEY);
      if (s === "wrong" || s === "cls" || s === "delta") setPerclassSort(s);
    }
  }, []);
  useEffect(() => {
    if (typeof window !== "undefined" && selectedRef) {
      window.localStorage.setItem(REF_KEY, selectedRef);
    }
  }, [selectedRef]);
  useEffect(() => {
    if (typeof window !== "undefined") {
      window.localStorage.setItem(SORT_KEY, perclassSort);
    }
  }, [perclassSort]);
  const toggleView = useCallback(() => {
    setViewMode((m) => {
      const next = m === "compact" ? "detailed" : "compact";
      if (typeof window !== "undefined") window.localStorage.setItem(VIEW_KEY, next);
      return next;
    });
  }, []);
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
  const allEpochs = status?.epochs ?? [];
  const detailed = viewMode === "detailed";
  const tableRows = detailed ? [...allEpochs].reverse() : allEpochs.slice(-RECENT_EPOCHS).reverse();
  const teValues = allEpochs.map((e) => e.te_p1 ?? 0);
  const lastGap = last && last.tr_acc != null && last.te_p1 != null ? last.tr_acc - last.te_p1 : null;

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
      overflow: detailed ? "auto" : "hidden",
      overscrollBehavior: "none",
      touchAction: detailed ? "auto" : "none",
    }}>
      <header style={{ padding: "14px 16px 6px", display: "flex", justifyContent: "space-between", alignItems: "center", gap: 8, flexShrink: 0 }}>
        <h1 style={{ font: "600 16px/1 inherit", margin: 0, letterSpacing: "0.5px" }}>
          ANEMON · <span style={{ color: "#6bf" }}>{status?.run ?? "—"}</span>
        </h1>
        <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
          <button
            onClick={toggleView}
            style={{
              background: detailed ? "#1e3a5f" : "#0f141b",
              color: detailed ? "#bfe1ff" : "#9ca3af",
              border: "1px solid " + (detailed ? "#2d5a8a" : "#1f2937"),
              borderRadius: 5,
              padding: "4px 10px",
              fontSize: 10,
              fontFamily: "inherit",
              cursor: "pointer",
              letterSpacing: "0.6px",
              textTransform: "uppercase",
            }}
            aria-label={detailed ? "switch to compact view" : "switch to detailed view"}
          >
            {detailed ? "compact" : "details"}
          </button>
          <span style={{ fontSize: 11, color: stale ? "#fb6" : "#888" }}>{ago}</span>
        </div>
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

      {detailed && allEpochs.length > 0 && (() => {
        const auxVals = allEpochs.map(e => e.aux_loss).filter((v): v is number => v != null);
        const lossVals = allEpochs.map(e => e.tr_loss).filter((v): v is number => v != null);
        const lastAux = last?.aux_loss;
        const lastTrLoss = last?.tr_loss;
        // Mechanism contribution = aux_loss / total_loss at last epoch.
        // tr_loss in main.py is the MEAN training loss = mean(CE + aux). So
        // aux_share = aux / tr_loss approximates the share aux takes of the
        // optimizer's gradient signal.
        const auxShare = (lastAux != null && lastTrLoss != null && lastTrLoss > 0)
          ? lastAux / lastTrLoss : null;
        const teMin = Math.min(...teValues);
        const teMax = Math.max(...teValues);
        const gaps = allEpochs
          .filter(e => e.tr_acc != null && e.te_p1 != null)
          .map(e => (e.tr_acc! - e.te_p1!));
        const maxGap = gaps.length ? Math.max(...gaps) : null;
        const minGap = gaps.length ? Math.min(...gaps) : null;
        const tot = (auxVals.length ? auxVals.reduce((a,b)=>a+b,0)/auxVals.length : null);
        const cyc = status?.cluster;
        const qcc = status?.qcc;
        const eng = status?.engram;
        return (
          <section style={{ padding: "6px 16px", borderTop: "1px solid #252525", flexShrink: 0 }}>
            <SubHead>detailed metrics</SubHead>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(64px, 1fr))", gap: 6 }}>
              <MKV k="te range" v={teValues.length ? `${teMin.toFixed(1)}–${teMax.toFixed(1)}` : "—"} />
              <MKV k="last gap" v={lastGap != null ? lastGap.toFixed(2) : "—"} color={lastGap != null && lastGap > 8 ? "#fb6" : "#e8e8e8"} />
              <MKV k="max gap" v={maxGap != null ? maxGap.toFixed(2) : "—"} color={maxGap != null && maxGap > 15 ? "#fb6" : "#e8e8e8"} />
              <MKV k="min gap" v={minGap != null ? minGap.toFixed(2) : "—"} />
              <MKV k="best−last" v={status?.best && last?.te_p1 != null ? (status.best.p1 - last.te_p1).toFixed(2) : "—"} />
              <MKV k="aux/total" v={auxShare != null ? `${(auxShare*100).toFixed(1)}%` : "—"} color={auxShare != null && auxShare > 0.5 ? "#fb6" : auxShare != null && auxShare > 0.1 ? "#bfe1ff" : "#e8e8e8"} />
              <MKV k="mean aux" v={tot != null ? tot.toExponential(1) : "—"} />
              <MKV k="last loss" v={lastTrLoss != null ? lastTrLoss.toFixed(2) : "—"} />
              <MKV k="last aux" v={lastAux != null ? lastAux.toExponential(1) : "—"} />
            </div>
            {(cyc || qcc || eng) && (
              <div style={{ marginTop: 6, paddingTop: 6, borderTop: "1px dashed #252525" }}>
                <SubHead>mechanism weights (zero-init)</SubHead>
                <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(64px, 1fr))", gap: 6 }}>
                  {cyc && (
                    <>
                      <MKV k="cycle" v={cyc.cycle_proj_norm.toFixed(3)} color={cyc.cycle_proj_norm > 0.01 ? "#6bf" : "#f88"} />
                      <MKV k="cyc max" v={cyc.cycle_proj_max.toFixed(3)} />
                      {cyc.cluster_head_norm != null && (
                        <MKV k="cluster_h" v={cyc.cluster_head_norm.toFixed(2)} />
                      )}
                    </>
                  )}
                  {qcc && (
                    <>
                      <MKV k="qcc" v={qcc.qcc_scale.toFixed(3)} color={Math.abs(qcc.qcc_scale) > 0.005 ? "#6bf" : "#f88"} />
                      {qcc.quat_inject_scale !== undefined && (
                        <MKV k="qinj" v={qcc.quat_inject_scale.toFixed(3)} color={Math.abs(qcc.quat_inject_scale) > 0.005 ? "#6bf" : "#f88"} />
                      )}
                    </>
                  )}
                  {eng && (
                    <>
                      <MKV k="engram" v={eng.out_norm.toFixed(3)} color={eng.out_norm > 0.01 ? "#6bf" : "#f88"} />
                      <MKV k="eng max" v={eng.out_max.toFixed(3)} />
                    </>
                  )}
                </div>
              </div>
            )}
            {status?.refs && (
              <div style={{ marginTop: 6, paddingTop: 6, borderTop: "1px dashed #252525" }}>
                <SubHead>best vs refs</SubHead>
                <div style={{ display: "flex", flexWrap: "wrap", gap: 4 }}>
                  <Chip label="current" value={status.refs.current_best.toFixed(2)} color="#6f9" border="#2d5a8a" />
                  {status.refs.refs.map(r => {
                    const dc = r.delta > 0.5 ? "#6f9" : r.delta > 0 ? "#bfe1ff" : r.delta < -1.5 ? "#f88" : "#fc9";
                    return (
                      <Chip key={r.name} label={`vs ${r.name} ${r.value}`} value={`${r.delta > 0 ? "+" : ""}${r.delta.toFixed(2)}`} color={dc} />
                    );
                  })}
                </div>
              </div>
            )}
            {status?.cnxxl_delta && status.cnxxl_delta.length > 0 && (
              <div style={{ marginTop: 6, paddingTop: 6, borderTop: "1px dashed #252525" }}>
                <SubHead>delta vs cnxxlquat 91.08 (per ep)</SubHead>
                <div style={{ display: "flex", flexWrap: "wrap", gap: 4 }}>
                  {status.cnxxl_delta.slice(-10).map(d => {
                    const dc = d.delta > 1 ? "#6f9" : d.delta > 0 ? "#bfe1ff" : d.delta < -2 ? "#f88" : "#fc9";
                    return (
                      <Chip key={d.ep} label={`ep ${d.ep}`} value={`${d.delta > 0 ? "+" : ""}${d.delta.toFixed(2)}`} color={dc} />
                    );
                  })}
                </div>
              </div>
            )}
            {status?.param_counts && (
              <div style={{ marginTop: 6, paddingTop: 6, borderTop: "1px dashed #252525" }}>
                <SubHead>param counts (M)</SubHead>
                <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(56px, 1fr))", gap: 6 }}>
                  <MKV k="total" v={`${status.param_counts.total_m}M`} />
                  <MKV k="main" v={`${status.param_counts.main_m}M`} />
                  <MKV k="aux" v={`${status.param_counts.aux_m}M`} color="#bfe1ff" />
                  <MKV k="aux %" v={`${(100 * status.param_counts.aux_m / Math.max(status.param_counts.total_m, 1e-9)).toFixed(1)}%`} />
                </div>
              </div>
            )}
            {cyc?.cluster_mass && cyc.cluster_mass.length > 0 && (
              <div style={{ marginTop: 6, paddingTop: 6, borderTop: "1px dashed #252525" }}>
                <SubHead>per-cluster mass (1/K = {fmt(1.0 / cyc.cluster_mass.length, 3)})</SubHead>
                <div style={{ display: "flex", gap: 3, alignItems: "flex-end", height: 36 }}>
                  {cyc.cluster_mass.map((m, i) => {
                    const target = 1.0 / cyc.cluster_mass!.length;
                    const ratio = m / target;
                    const color = ratio > 2 ? "#fb6" : ratio > 1.4 ? "#fc9" : ratio < 0.4 ? "#f88" : "#6bf";
                    const h = Math.max(2, Math.min(36, m * 120));
                    return (
                      <div key={i} style={{ flex: 1, display: "flex", flexDirection: "column", alignItems: "center", gap: 1 }}>
                        <div style={{ fontSize: 8, color: "#888" }}>{(m * 100).toFixed(0)}</div>
                        <div style={{ width: "100%", background: color, height: `${h}px`, borderRadius: 2 }} />
                        <div style={{ fontSize: 8, color: "#666" }}>k{i}</div>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}
            {status?.skew && (() => {
              const sk = status.skew!;
              const traj = status.skew_traj ?? [];
              const lastT = traj.length ? traj[traj.length - 1] : null;
              const spark = (key: keyof SkewTrajRow, color: string) => {
                const vals = traj.map(t => t[key]).filter((v): v is number => v != null);
                if (vals.length < 2) return null;
                const mx = Math.max(...vals), mn = Math.min(...vals);
                const pts = vals.map((v, i) =>
                  `${(i / Math.max(1, vals.length - 1)) * 100},${30 - ((v - mn) / (mx - mn || 1)) * 28}`).join(" ");
                return (
                  <svg viewBox="0 0 100 30" preserveAspectRatio="none" style={{ width: "100%", height: 26 }}>
                    <polyline points={pts} fill="none" stroke={color} strokeWidth={1.4} />
                  </svg>
                );
              };
              const accColor = (v?: number | null) => v == null ? "#666" : v > 30 ? "#6f9" : v > 10 ? "#fb6" : "#f88";
              return (
                <div style={{ marginTop: 6, paddingTop: 6, borderTop: "1px dashed #252525" }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                    <SubHead>skew-tcc aux · ep {sk.epoch}</SubHead>
                    <span style={{ fontSize: 9, fontWeight: 700, letterSpacing: "0.5px", color: sk.alive ? "#6f9" : "#f88" }}>
                      {sk.alive ? "● ALIVE" : "○ DEAD (decayed)"}
                    </span>
                  </div>
                  <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(56px, 1fr))", gap: 6 }}>
                    <MKV k="scale" v={fmt(sk.scale, 3)} color={Math.abs(sk.scale) > 0.01 ? "#6bf" : "#f88"} />
                    <MKV k="head0" v={sk.head0 != null ? fmt(sk.head0, 3) : "—"} color={(sk.head0 ?? 0) > 0.05 ? "#6bf" : "#f88"} />
                    <MKV k="wu" v={sk.wu != null ? fmt(sk.wu, 3) : "—"} />
                    <MKV k="wv" v={sk.wv != null ? fmt(sk.wv, 3) : "—"} />
                    <MKV k="headf" v={sk.headf != null ? fmt(sk.headf, 3) : "—"} />
                    {lastT && <MKV k="aux tr%" v={lastT.aux_acc != null ? fmt(lastT.aux_acc, 1) : "—"} color={accColor(lastT.aux_acc)} />}
                    {lastT && <MKV k="aux te%" v={lastT.aux_te != null ? fmt(lastT.aux_te, 1) : "—"} color={accColor(lastT.aux_te)} />}
                    {sk.lag_energy && sk.lag_energy.map((v, i) => <MKV key={i} k={`lag${i + 1}`} v={fmt(v, 2)} />)}
                  </div>
                  {sk.comp_energy && (
                    <div style={{ marginTop: 6 }}>
                      <SubHead>projector energy by cov component %</SubHead>
                      <div style={{ display: "flex", gap: 3, alignItems: "flex-end", height: 34 }}>
                        {Object.entries(sk.comp_energy).map(([k, v]) => (
                          <div key={k} style={{ flex: 1, display: "flex", flexDirection: "column", alignItems: "center", gap: 1 }}>
                            <div style={{ fontSize: 8, color: "#888" }}>{v}</div>
                            <div style={{ width: "100%", background: k[0] === k[1] ? "#6bf" : "#fb6", height: `${Math.max(2, v * 0.35)}px`, borderRadius: 2 }} />
                            <div style={{ fontSize: 7.5, color: "#666" }}>{k}</div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                  {sk.rank_usage && sk.rank_usage.length > 0 && (
                    <div style={{ marginTop: 6 }}>
                      <SubHead>per-rank usage % (r={sk.rank_usage.length})</SubHead>
                      <div style={{ display: "flex", gap: 2, alignItems: "flex-end", height: 26 }}>
                        {sk.rank_usage.map((v, i) => (
                          <div key={i} title={`r${i}: ${v}%`} style={{ flex: 1, background: v > 12 ? "#6bf" : v > 4 ? "#9c9" : "#3a4252", height: `${Math.max(2, v * 1.4)}px`, borderRadius: 1 }} />
                        ))}
                      </div>
                    </div>
                  )}
                  {traj.length > 1 && (
                    <div style={{ marginTop: 6 }}>
                      <SubHead>trajectory over epochs (does the aux survive?)</SubHead>
                      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
                        <div><div style={{ fontSize: 8, color: "#6bf" }}>scale</div>{spark("scale", "#6bf")}</div>
                        <div><div style={{ fontSize: 8, color: "#fb6" }}>head0 (norm)</div>{spark("head0", "#fb6")}</div>
                        <div><div style={{ fontSize: 8, color: "#9c9" }}>desc energy</div>{spark("desc_e", "#9c9")}</div>
                        <div><div style={{ fontSize: 8, color: "#6f9" }}>aux test %</div>{spark("aux_te", "#6f9")}</div>
                      </div>
                    </div>
                  )}
                </div>
              );
            })()}
            {status?.current_perclass && status.current_perclass.length > 0 && (() => {
              const refs = status.available_refs ?? [];
              const cur = status.current_perclass!;
              const activeRefName = selectedRef && refs.some(r => r.name === selectedRef)
                ? selectedRef
                : (refs[0]?.name ?? null);
              const refObj = refs.find(r => r.name === activeRefName) ?? null;
              const refMap = new Map((refObj?.perclass ?? []).map(p => [p.cls, p]));
              type Row = {
                cls: number; cur_w: number; cur_t: number; cur_pct: number;
                ref_w: number | null; ref_t: number | null; ref_pct: number | null; delta_pct: number | null;
              };
              const rows: Row[] = cur.map(c => {
                const r = refMap.get(c.cls);
                const cur_pct = c.total > 0 ? 100 * c.wrong / c.total : 0;
                const ref_pct = r && r.total > 0 ? 100 * r.wrong / r.total : null;
                return {
                  cls: c.cls, cur_w: c.wrong, cur_t: c.total,
                  cur_pct: Math.round(cur_pct * 10) / 10,
                  ref_w: r?.wrong ?? null, ref_t: r?.total ?? null,
                  ref_pct: ref_pct != null ? Math.round(ref_pct * 10) / 10 : null,
                  delta_pct: ref_pct != null ? Math.round((cur_pct - ref_pct) * 10) / 10 : null,
                };
              });
              if (perclassSort === "wrong") rows.sort((a, b) => b.cur_w - a.cur_w);
              else if (perclassSort === "cls") rows.sort((a, b) => a.cls - b.cls);
              else rows.sort((a, b) => (b.delta_pct ?? -999) - (a.delta_pct ?? -999));
              return (
                <div style={{ marginTop: 6, paddingTop: 6, borderTop: "1px dashed #252525" }}>
                  <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 6, marginBottom: 4, flexWrap: "wrap" }}>
                    <SubHead>per-class (current · ref)</SubHead>
                    <div style={{ display: "flex", gap: 4, alignItems: "center" }}>
                      {refs.length > 0 && (
                        <select
                          value={activeRefName ?? ""}
                          onChange={(e) => setSelectedRef(e.target.value)}
                          style={{
                            background: "#0f141b", color: "#bfe1ff", border: "1px solid #1f2937",
                            borderRadius: 3, padding: "1px 3px", fontSize: 10, fontFamily: "inherit",
                          }}
                        >
                          {refs.map(r => (
                            <option key={r.name} value={r.name}>{r.label} ({r.acc}%)</option>
                          ))}
                        </select>
                      )}
                      <div style={{ display: "flex", gap: 2 }}>
                        {(["wrong", "cls", "delta"] as const).map(s => (
                          <button key={s}
                            onClick={() => setPerclassSort(s)}
                            style={{
                              background: perclassSort === s ? "#1e3a5f" : "#0f141b",
                              color: perclassSort === s ? "#bfe1ff" : "#888",
                              border: "1px solid " + (perclassSort === s ? "#2d5a8a" : "#1f2937"),
                              borderRadius: 3, padding: "1px 4px", fontSize: 9, fontFamily: "inherit", cursor: "pointer",
                            }}
                          >{s}</button>
                        ))}
                      </div>
                    </div>
                  </div>
                  <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(70px, 1fr))", gap: 3 }}>
                    {rows.map(r => {
                      const curC = r.cur_pct > 50 ? "#f88" : r.cur_pct > 25 ? "#fb6" : r.cur_pct > 5 ? "#fc9" : "#9c9";
                      const dC = r.delta_pct == null ? "#666"
                        : r.delta_pct > 5 ? "#f88" : r.delta_pct > 0 ? "#fb6"
                        : r.delta_pct < -5 ? "#6f9" : "#bfe1ff";
                      return (
                        <div key={r.cls} style={{
                          background: "#0f141b", border: "1px solid #1f2937", borderRadius: 3,
                          padding: "1px 4px", display: "flex", flexDirection: "column",
                        }}>
                          <span style={{ fontSize: 8.5, color: "#888" }}>cls {r.cls}</span>
                          <div style={{ display: "flex", alignItems: "baseline", gap: 3 }}>
                            <span style={{ fontSize: 10.5, color: curC, fontWeight: 600 }}>{r.cur_pct}%</span>
                            {r.ref_pct != null && (
                              <>
                                <span style={{ fontSize: 8.5, color: "#888" }}>{r.ref_pct}</span>
                                <span style={{ fontSize: 8.5, color: dC, fontWeight: 600 }}>
                                  {r.delta_pct! > 0 ? "+" : ""}{r.delta_pct}
                                </span>
                              </>
                            )}
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              );
            })()}
          </section>
        );
      })()}

      <section style={{ padding: "8px 16px", borderTop: "1px solid #252525", flex: detailed ? "0 0 auto" : 1, minHeight: 0, overflow: detailed ? "visible" : "hidden", display: "flex", flexDirection: "column" }}>
        {(() => {
          const hasAux = tableRows.some(e => e.aux_loss != null);
          const refs = status?.available_refs ?? [];
          const activeRef = (selectedRef ? refs.find(r => r.name === selectedRef) : null) ?? refs[0] ?? null;
          const refMap = new Map((activeRef?.epochs ?? []).map(p => [p.ep, p.te] as [number, number]));
          const showRef = (activeRef?.epochs?.length ?? 0) > 0;
          const headers = [
            "ep", "tr%", "loss",
            ...(hasAux ? ["aux"] : []),
            "te p1",
            ...(showRef ? ["ref", "Δ"] : []),
            "te p5",
            ...(detailed ? ["gap"] : []),
          ];
          const fmtAux = (v: number | null | undefined) =>
            v == null || Number.isNaN(v) ? "—" : Number(v).toExponential(1);
          return (
            <>
              {showRef && (
                <div style={{ fontSize: 9.5, color: "#888", marginBottom: 4 }}>
                  ref = <span style={{ color: "#7aa2c0" }}>{activeRef!.label}</span>
                  <span style={{ color: "#555" }}> · Δ = current − ref (per epoch)</span>
                </div>
              )}
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
                  {tableRows.map((e) => {
                    const isBest = e.ep === bestEp;
                    const baseColor = isBest ? "#6f9" : "#e8e8e8";
                    const weight = isBest ? 700 : 400;
                    const gap = e.tr_acc != null && e.te_p1 != null ? e.tr_acc - e.te_p1 : null;
                    const refTe = refMap.get(e.ep);
                    const dRef = (refTe != null && e.te_p1 != null) ? e.te_p1 - refTe : null;
                    const cells: { v: string; color: string }[] = [
                      { v: String(e.ep), color: baseColor },
                      { v: fmt(e.tr_acc, 1), color: baseColor },
                      { v: fmt(e.tr_loss, 3), color: baseColor },
                      ...(hasAux ? [{ v: fmtAux(e.aux_loss), color: baseColor }] : []),
                      { v: fmt(e.te_p1, 2), color: baseColor },
                      ...(showRef ? [
                        { v: refTe != null ? refTe.toFixed(2) : "—", color: "#7aa2c0" },
                        {
                          v: dRef != null ? `${dRef > 0 ? "+" : ""}${dRef.toFixed(2)}` : "—",
                          color: dRef == null ? "#666" : dRef > 0 ? "#6f9" : dRef < 0 ? "#f88" : "#bbb",
                        },
                      ] : []),
                      { v: fmt(e.te_p5, 2), color: baseColor },
                      ...(detailed ? [{
                        v: gap != null ? gap.toFixed(2) : "—",
                        color: (gap != null && gap > 10) ? "#fb6" : (gap != null && gap > 5) ? "#fc9" : baseColor,
                      }] : []),
                    ];
                    return (
                      <tr key={e.ep}>
                        {cells.map((c, i) => (
                          <td key={i} style={{
                            padding: "4px 6px",
                            textAlign: i === 0 ? "left" : "right",
                            color: c.color, fontWeight: weight,
                          }}>{c.v}</td>
                        ))}
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </>
          );
        })()}
      </section>

      {status?.engram && (
        <section style={{ padding: "8px 16px", borderTop: "1px solid #252525", flexShrink: 0 }}>
          <div style={{ fontSize: 9, color: "#888", textTransform: "uppercase", letterSpacing: "0.6px", marginBottom: 6 }}>engram aux contribution</div>
          <div style={{ display: "flex", gap: 16, fontSize: 12, color: "#e5e7eb" }}>
            <div>ep {status.engram.epoch}</div>
            <div>
              out_proj norm{" "}
              <span style={{ color: status.engram.out_norm > 0.01 ? "#6bf" : "#f88", fontWeight: 600 }}>
                {status.engram.out_norm.toFixed(4)}
              </span>
            </div>
            <div style={{ color: "#888" }}>max {status.engram.out_max.toFixed(4)}</div>
          </div>
          <div style={{ fontSize: 10, color: "#6b7280", marginTop: 4 }}>
            zero-init residual; blue = contributing, red = suppressed
          </div>
        </section>
      )}

      {status?.qcc && (
        <section style={{ padding: "8px 16px", borderTop: "1px solid #252525", flexShrink: 0 }}>
          <div style={{ fontSize: 9, color: "#888", textTransform: "uppercase", letterSpacing: "0.6px", marginBottom: 6 }}>qcc mechanism usage</div>
          <div style={{ display: "flex", gap: 16, fontSize: 12, color: "#e5e7eb", flexWrap: "wrap" }}>
            <div>ep {status.qcc.epoch}</div>
            <div>
              qcc_scale{" "}
              <span style={{ color: Math.abs(status.qcc.qcc_scale) > 0.005 ? "#6bf" : "#f88", fontWeight: 600 }}>
                {status.qcc.qcc_scale.toFixed(4)}
              </span>
            </div>
            {status.qcc.quat_inject_scale !== undefined && (
              <div>
                quat_inject_scale{" "}
                <span style={{ color: Math.abs(status.qcc.quat_inject_scale) > 0.005 ? "#6bf" : "#f88", fontWeight: 600 }}>
                  {status.qcc.quat_inject_scale.toFixed(4)}
                </span>
              </div>
            )}
            {status.qcc.quat_inject_norm !== undefined && (
              <div style={{ color: "#888" }}>
                proj_norm {status.qcc.quat_inject_norm.toFixed(4)}
              </div>
            )}
          </div>
          <div style={{ fontSize: 10, color: "#6b7280", marginTop: 4 }}>
            qcc_scale: aux head gate (init 0). quat_inject_scale: post-Mamba quat residual gate (init 0). blue = used.
          </div>
        </section>
      )}

      <section style={{ padding: "10px 16px", borderTop: "1px solid #252525", flexShrink: 0 }}>
        <a
          href="/notes"
          style={{
            display: "block",
            textAlign: "center",
            color: "#9ca3af",
            textDecoration: "none",
            fontSize: 12,
            padding: "8px 12px",
            border: "1px solid #1f2937",
            borderRadius: 6,
            background: "#0f141b",
          }}
        >
          notes →
        </a>
      </section>
    </main>
  );
}
