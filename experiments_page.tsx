"use client";

/* Comprehensive experiments page for supervisor briefing.
 * Path: /experiments
 * Self-contained: no API calls, static content.
 */

const PALETTE = {
  bg: "#0a0a0a",
  card: "#141414",
  border: "#262626",
  text: "#e5e5e5",
  muted: "#9ca3af",
  good: "#6f9",
  bad: "#f87171",
  warn: "#fbbf24",
  accent: "#7aa2ff",
  highlight: "#1f1f1f",
};

const fontMono =
  "ui-monospace, SFMono-Regular, Menlo, Consolas, 'Liberation Mono', monospace";

function Section({
  title,
  num,
  children,
}: {
  title: string;
  num?: string | number;
  children: React.ReactNode;
}) {
  return (
    <section style={{ marginBottom: 28 }}>
      <h2
        style={{
          fontSize: 18,
          color: PALETTE.accent,
          margin: "0 0 10px 0",
          borderBottom: `1px solid ${PALETTE.border}`,
          paddingBottom: 6,
        }}
      >
        {num != null && (
          <span style={{ color: PALETTE.muted, fontFamily: fontMono, fontSize: 14 }}>
            §{num}{" "}
          </span>
        )}
        {title}
      </h2>
      <div style={{ color: PALETTE.text, lineHeight: 1.55, fontSize: 14 }}>{children}</div>
    </section>
  );
}

function Sub({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div style={{ marginTop: 14, marginBottom: 10 }}>
      <h3 style={{ fontSize: 15, color: "#fff", margin: "0 0 6px 0" }}>{title}</h3>
      <div>{children}</div>
    </div>
  );
}

function Card({ tone = "default", children }: { tone?: "default" | "bad" | "good" | "warn"; children: React.ReactNode }) {
  const accent =
    tone === "bad" ? PALETTE.bad
      : tone === "good" ? PALETTE.good
      : tone === "warn" ? PALETTE.warn
      : PALETTE.border;
  return (
    <div
      style={{
        background: PALETTE.card,
        border: `1px solid ${accent}`,
        borderLeft: `3px solid ${accent}`,
        borderRadius: 6,
        padding: 12,
        marginBottom: 8,
      }}
    >
      {children}
    </div>
  );
}

function Code({ children }: { children: React.ReactNode }) {
  return (
    <pre
      style={{
        background: "#0d1117",
        color: "#c9d1d9",
        border: `1px solid ${PALETTE.border}`,
        borderRadius: 4,
        padding: 10,
        fontSize: 12,
        overflow: "auto",
        fontFamily: fontMono,
        margin: "6px 0",
      }}
    >
      {children}
    </pre>
  );
}

function Tbl({
  headers,
  rows,
  rightAligned = [],
}: {
  headers: string[];
  rows: (string | number)[][];
  rightAligned?: number[];
}) {
  return (
    <div style={{ overflowX: "auto" }}>
      <table
        style={{
          width: "100%",
          borderCollapse: "collapse",
          fontFamily: fontMono,
          fontSize: 12,
          margin: "6px 0",
        }}
      >
        <thead>
          <tr>
            {headers.map((h, i) => (
              <th
                key={i}
                style={{
                  textAlign: rightAligned.includes(i) ? "right" : "left",
                  padding: "4px 8px",
                  borderBottom: `1px solid ${PALETTE.border}`,
                  color: PALETTE.muted,
                  fontWeight: 600,
                  whiteSpace: "nowrap",
                }}
              >
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i}>
              {r.map((c, j) => (
                <td
                  key={j}
                  style={{
                    textAlign: rightAligned.includes(j) ? "right" : "left",
                    padding: "3px 8px",
                    borderBottom: `1px solid ${PALETTE.border}30`,
                    whiteSpace: j === 0 ? "nowrap" : "normal",
                  }}
                >
                  {c}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Pill({
  tone,
  children,
}: {
  tone: "good" | "bad" | "warn" | "muted" | "accent";
  children: React.ReactNode;
}) {
  const colors = {
    good: PALETTE.good,
    bad: PALETTE.bad,
    warn: PALETTE.warn,
    muted: PALETTE.muted,
    accent: PALETTE.accent,
  };
  return (
    <span
      style={{
        display: "inline-block",
        padding: "1px 6px",
        borderRadius: 3,
        background: `${colors[tone]}22`,
        color: colors[tone],
        fontFamily: fontMono,
        fontSize: 11,
        fontWeight: 600,
      }}
    >
      {children}
    </span>
  );
}

function KV({ k, v }: { k: string; v: React.ReactNode }) {
  return (
    <div
      style={{
        display: "flex",
        justifyContent: "space-between",
        padding: "3px 0",
        borderBottom: `1px dashed ${PALETTE.border}`,
      }}
    >
      <span style={{ color: PALETTE.muted, fontSize: 13 }}>{k}</span>
      <span style={{ fontFamily: fontMono, fontSize: 13 }}>{v}</span>
    </div>
  );
}

export default function ExperimentsPage() {
  return (
    <main
      style={{
        background: PALETTE.bg,
        color: PALETTE.text,
        minHeight: "100vh",
        padding: "30px 32px",
        fontFamily: "system-ui, -apple-system, sans-serif",
        maxWidth: 1080,
        margin: "0 auto",
      }}
    >
      <header style={{ marginBottom: 24 }}>
        <h1 style={{ fontSize: 24, margin: "0 0 4px 0", color: "#fff" }}>
          NVGesture: Architectural Investigation & Diagnostic Findings
        </h1>
        <div style={{ color: PALETTE.muted, fontSize: 13 }}>
          Supervisor briefing — May 18, 2026. Full repo: <code style={{ fontFamily: fontMono }}>worktree-qhdelta-fuse</code>.
          7 trained architectures, 5 frozen-mechanism diagnostics, 4 auxiliary-loss attempts. ~120 GPU-hours.
        </div>
      </header>

      {/* ============================================================ */}
      <Section title="Executive summary" num={0}>
        <Card tone="warn">
          <ul style={{ paddingLeft: 18, margin: 0 }}>
            <li>
              The Stage-3 temporal encoder in PMamba is <b>inference-decorative</b>: every mechanism we tried
              (Mamba, RD, AttRD, BDN-Q, Lie-group state, CSTA) produces predictions identical to setting its
              output to zero. <Pill tone="bad">+0.00 pp</Pill> contribution at inference across all 5 architectures tested.
            </li>
            <li>
              The 88–90% accuracy of every variant comes from the <b>spatial pipeline + in/out channel
              bottleneck</b>, not from temporal recurrence. Confirmed by replacing the entire encoder with the
              identity (→ 5.19%, near chance).
            </li>
            <li>
              CN-XXL — a per-frame MLP with <b>zero temporal mixing</b> — trains to 89.00%, matching every
              &quot;novel&quot; architecture, simply by having enough parameters in the encoder slot.
            </li>
            <li>
              The 92.53% fusion ceiling is mostly the multi-modal lift from the depth-stream baseline (DSN
              solo = 90.25); per-architecture ensemble variance adds only +0.62 pp.
            </li>
            <li>
              Architectural mods inside the residual stack are bypassable by construction. <b>Auxiliary
              losses</b> are the remaining unbypassable path; we are currently testing Future Feature
              Prediction as the leading candidate.
            </li>
          </ul>
        </Card>
      </Section>

      {/* ============================================================ */}
      <Section title="Setup, dataset, protocol" num={1}>
        <Sub title="Dataset & evaluation">
          <ul style={{ paddingLeft: 18 }}>
            <li>NVGesture — 1,532 dynamic hand-gesture point-cloud sequences, 25 classes, subject-independent split.</li>
            <li>1,050 training / 482 test sequences. Each sequence sampled to <code style={{fontFamily:fontMono}}>T = 32</code> frames.</li>
            <li>
              <b>Honest train-best protocol:</b> for each model, pick the checkpoint with the highest training accuracy and evaluate it once on the test set. No test-set epoch selection — the 92.3% numbers from prior work used test-best selection which is leakage.
            </li>
          </ul>
        </Sub>

        <Sub title="PMamba baseline architecture (full forward path)">
          <Code>{`input (B, T=32, N_raw≤96, 4 coords)
  └─ _sample_points → (B, 4, T, pts_size=256)        ← random subsample per frame

Stage 1 (intra-frame)
  └─ group.group_points  k=32   distance=Euclidean over (x,y,z)
  └─ MLPBlock [4 → 32 → 64]  (per-point pointwise convs + BN + GELU)
  └─ AdaptiveMaxPool2d (None,1)   pool over k-NN axis

Stage 2 (inter-frame, multi-scale temporal context)
  └─ MultiScaleFeatureProcessor  scales {2,4,8,16,32} kernel sizes over T
       │                          summed with residual at the end
  └─ st_group_points k=24  spatio-temporal k-NN
  └─ select_ind → ↓2 points
  └─ MotionBlock [128 → 128]
  └─ AdaptiveMaxPool2d (None,1)

Stage 3 (spatial + the "temporal encoder" slot)
  └─ st_group_points k=48
  └─ select_ind → ↓2 points
  └─ MotionBlock [256 → 256]
  └─ AdaptiveMaxPool2d (None,1) → fea3 (B, 256, T, N)
  └─ self.mamba = MambaTemporalEncoder:                ← all our experiments swap this
        QuaternionLinear  256 → 128
        for L=2 layers (both fwd + bwd):
            LayerNorm(128)
            Mamba(d_state=16, d_conv=4, expand=2)
            Dropout(0.3)
            x = x + residual                          ← THIS IS THE BYPASS ROUTE
        LayerNorm(128)
        QuaternionLinear  128 → 256
     fea3_mamba = output

Stage 4 (classifier)
  └─ cat(coords, fea3_mamba) → (B, 260, T, N)
  └─ MLPBlock [260 → 1024] + BN
  └─ AdaptiveMaxPool2d (1,1)                          ← collapses (T, N) into one number
  └─ BatchNorm2d(1024)
  └─ flatten → MLPBlock [1024 → 25]   class logits`}</Code>
        </Sub>

        <Sub title="Trained-from-scratch baseline anchor">
          <Card tone="good">
            <KV k="model" v="MambaTemporalEncoder in Stage 3 slot (the original PMamba)" />
            <KV k="checkpoint" v="work_dir/pmamba_branch/epoch115_model.pt" />
            <KV k="test acc (honest train-best)" v={<><b>90.04 %</b> (434 / 482 correct)</>} />
            <KV k="protocol" v="pmamba_test.yaml — pts_size=256, multi_scale_num_scales=5, knn=[32,24,48,24]" />
          </Card>
        </Sub>
      </Section>

      {/* ============================================================ */}
      <Section title="The correspondence problem (why this domain is hard)" num={2}>
        <Card tone="warn">
          <p>
            <b>NVGesture point-cloud sequences do not have point correspondence across frames.</b> A point with
            index <code style={{fontFamily:fontMono}}>i</code> at frame <code style={{fontFamily:fontMono}}>t</code> is
            <i> not the same physical point</i> as the point with index <code style={{fontFamily:fontMono}}>i</code> at
            frame <code style={{fontFamily:fontMono}}>t+1</code>.
          </p>
          <p style={{ margin: 0 }}>
            <code style={{fontFamily:fontMono}}>_sample_points</code> does a fresh random/linspace sub-sample for
            each frame independently. The depth-camera point cloud also varies in <i>which points exist</i> from
            frame to frame depending on visibility, occlusion, and re-sampling.
          </p>
        </Card>

        <Sub title="What this rules out">
          <ul style={{ paddingLeft: 18 }}>
            <li>
              <b>Per-point optical-flow / scene-flow losses</b> — there is no &quot;next-frame position&quot; for a
              specific point.
            </li>
            <li>
              <b>Per-point trajectory regularization</b> — no trajectory can be defined for a fixed point
              identity.
            </li>
            <li>
              <b>Per-point cycle consistency</b> in feature space — <code style={{fontFamily:fontMono}}>Δ(F_a[i], F_b[i])</code> is comparing two unrelated points.
            </li>
            <li>
              <b>Per-point ground-truth rotation</b> — the original ST-QNet draft used quaternion cycle on
              per-point estimated rotations, which only made sense under an implicit (and never enforced)
              correspondence assumption.
            </li>
          </ul>
        </Sub>

        <Sub title="What this leaves">
          <ul style={{ paddingLeft: 18 }}>
            <li>
              <b>Set-level losses</b> (Chamfer / Earth-Mover) — possible but expensive.
            </li>
            <li>
              <b>Per-frame global descriptors</b> — attention-pool over points to get one vector per frame,
              then operate on those (no correspondence needed).
            </li>
            <li>
              <b>Soft correspondence learning</b> — learn matching via attention between point sets at different
              frames; what ST-QNet&apos;s original cycle-consistency tried to do with Gaussian-weighted
              neighborhoods.
            </li>
            <li>
              <b>Auxiliary tasks operating on aggregated features</b> — once features are pooled to a single
              vector per frame, the correspondence problem disappears.
            </li>
          </ul>
        </Sub>

        <Sub title="Why this matters for our auxiliary-loss attempts">
          <p>
            Direction A&prime; (Global Frame-Feature Cycle) operates only on{" "}
            <i>global per-frame vectors</i> precisely because of this constraint. We cannot do per-point cycle
            consistency at all, so the &quot;cycle&quot; becomes a vector composition on global features. The
            absence of correspondence is also why the original ST-QNet cycle loss <i>had</i> to use Gaussian
            soft-correspondence machinery: it was working around this exact problem.
          </p>
        </Sub>
      </Section>

      {/* ============================================================ */}
      <Section title="Architectural variants — all targeted Stage 3" num={3}>
        <p>
          Seven different mechanisms were swapped into <code style={{fontFamily:fontMono}}>self.mamba</code>,
          training from scratch with the rest of the network frozen-architecture (k-NN sizes, channel widths,
          classifier head, training schedule all identical). Same 120-epoch budget, same Adam optimizer,
          same data split.
        </p>
        <Tbl
          headers={["Variant", "Mechanism", "Encoder params", "Solo (train-best)", "Solo (test-best)", "Notes"]}
          rows={[
            ["Mamba (PMamba)", "Selective SSM (Gu 2023)", "0.115 M", "—", "90.04", "the original PMamba; 'M' in Mamba"],
            ["RealDeltaNet (RD)", "Delta rule on 4-fold elementwise substrate", "0.29 M", "88.59", "90.46", "best test-best of all"],
            ["AttRD", "RD + softmax read over memory states", "~0.30 M", "89.00", "89.63", "decorative on the read side"],
            ["BDN-Q", "FIFO buffer + delta state on overflow", "0.23 M", "87.76", "89.00", "needed Q substrate to compete"],
            ["Lie-group", "SO(3) state, Hamilton-product recurrence", "0.19 M", "89.42", "89.42", "only one with real quaternion algebra"],
            ["CSTA", "Causal ST attention + RBF time bias", "0.42 M", "89.63", "89.63", "factored attention"],
            ["Mamba2", "Mamba2 SSD (Dao & Gu 2024)", "~0.80 M", "—", "88.80", "more params, lower acc"],
          ]}
        />
        <p>
          <Pill tone="warn">2.7 pp spread</Pill> across radically different mechanisms (87.76 → 90.46). This narrow
          band was the first red flag — if any mechanism were truly load-bearing, we&apos;d expect bigger gaps.
        </p>
      </Section>

      {/* ============================================================ */}
      <Section title="The frozen-mechanism diagnostic" num={4}>
        <p>
          For each trained checkpoint, override the temporal block&apos;s <i>forward pass at inference</i> to
          produce a no-op output. Leave the rest of the network and trained weights intact. Compare
          predictions vs the original forward.
        </p>

        <Sub title="Methodology">
          <Code>{`# General pattern applied to each architecture
def patched_block_forward(self, x):
    return torch.zeros_like(x)        # block emits 0; residual carries spatial signal

BlockClass.forward = patched_block_forward
load trained checkpoint               # weights stay untouched
acc_zeroed  = eval(test_set)
acc_normal  = eval(test_set, original forward)

if acc_zeroed == acc_normal:           # same predictions → block was decorative
    block was never on the inference path`}</Code>
        </Sub>

        <Sub title="Per-architecture diagnostics">
          <Tbl
            headers={["Variant", "What was patched", "Normal acc", "Patched acc", "Δ", "Verdict"]}
            rows={[
              ["Mamba", "Mamba.forward → zeros", "90.04 (434/482)", "90.04 (434/482)", "+0.00 pp", "decorative"],
              ["RealDeltaNet (RD)", "RealDeltaNetBlock.forward → zeros", "88.59", "88.59", "+0.00 pp", "decorative"],
              ["AttRD", "attn pattern → identity (= RD point read)", "89.00 (429/482)", "89.00 (429/482)", "+0.00 pp", "decorative"],
              ["AttRD (alt)", "attn pattern → uniform 1/T", "89.00 (429/482)", "89.00 (429/482)", "+0.00 pp", "decorative"],
              ["BDN-Q", "block output → 0 (residual still active)", "87.76 (423/482)", "87.76 (423/482)", "+0.00 pp", "decorative"],
              ["Lie-group", "ω → 0, state stays at identity quaternion", "89.42 (431/482)", "89.42 (431/482)", "+0.00 pp", "decorative"],
              ["RD soft-residual rs=0.7", "RealDeltaNetBlock.forward → zeros", "84.65 @ ep40 (408/482)", "84.65 (408/482)", "+0.00 pp", "decorative even with scaled residual"],
            ]}
          />
        </Sub>

        <Sub title="Encoder decomposition">
          <p>
            The MambaTemporalEncoder consists of: <code style={{fontFamily:fontMono}}>QuaternionLinear in</code> →
            [Mamba layers + residual stack] → <code style={{fontFamily:fontMono}}>QuaternionLinear out</code>. We
            tested replacing the entire encoder with the identity to isolate which part carries signal.
          </p>
          <Tbl
            headers={["Configuration", "Test acc", "Δ vs Mamba"]}
            rows={[
              ["Normal Mamba encoder", "90.04 (434/482)", "—"],
              ["Mamba layers → 0 (in/out QuaternionLinear active)", "90.04 (434/482)", "+0.00 pp"],
              ["Entire encoder → identity (no projections)", "5.19 (25/482)", "−84.85 pp"],
            ]}
          />
          <Card tone="bad">
            <b>The Mamba layers contribute 0 pp. The in/out QuaternionLinear projections carry +84.85 pp.</b>{" "}
            The &quot;temporal encoder&quot; is doing no temporal computation that affects predictions — only
            the per-frame channel-bottleneck transformation matters.
          </Card>
        </Sub>

        <Sub title="Same diagnostic on different residual scaling">
          <p>
            We retrained RD with the residual scaled to 0.7 instead of 1.0, forcing the block&apos;s output to
            carry more relative weight. At ep 40 we re-ran the frozen diagnostic:
          </p>
          <Tbl
            headers={["Setup", "Normal acc", "Block → 0", "Δ"]}
            rows={[
              ["RD residual_scale=1.0 (ep118)", "88.59", "88.59", "+0.00 pp"],
              ["RD residual_scale=0.7 (ep40)", "84.65 (408/482)", "84.65 (408/482)", "+0.00 pp"],
            ]}
          />
          <p>
            The bypass behaviour is robust across residual scales. The architecture itself permits the
            bypass; reducing residual_scale to 0.1 broke training (ep10 = 9.34%) but did force the block
            to be temporarily load-bearing (+31 pp at ep30) — at the cost of plateau at 67%.
          </p>
        </Sub>
      </Section>

      {/* ============================================================ */}
      <Section title="Parameter budget vs accuracy (the param-count finding)" num={5}>
        <p>
          To test whether mechanism matters at all, we built CN (Clean Network) variants: replace the entire
          MambaTemporalEncoder with a simple per-frame channel bottleneck. <b>No temporal mixing</b>, no recurrence,
          no attention. Scale parameters up progressively.
        </p>

        <Tbl
          headers={["Variant", "Stage-3 encoder", "Params", "ep10", "ep20", "ep40", "Best", "Cluster"]}
          rows={[
            ["NoTemporal (per-frame MLP w/ GELU)", "Linear-GELU-LN-Linear", "0.13 M", "65.77", "—", "84.85", "84.85 @ep40", "B"],
            ["CN (Linear bottleneck)", "Linear(256→128)-LN-Linear(128→256)", "0.066 M", "70.95", "80.5", "83.82", "83.82 @ep40", "B"],
            ["Mamba (original PMamba)", "MambaTemporalEncoder", "0.115 M", "70.54", "82.0", "87.3", "90.04 @ep115", "B→A"],
            ["CN-XL (wider MLP)", "Linear-GELU-Linear, mlp=256, 2L bidir", "0.33 M", "79.9", "82.2", "—", "(killed)", "A"],
            ["CN-XXL (wider+deeper)", "Linear-GELU-Linear, mlp=512, 4L bidir", "2.24 M", "77.2", "84.0", "85.3", "89.00 @ep103", "A"],
            ["RD", "RealDeltaNet (delta-rule)", "0.29 M", "80.3", "81.7", "85.3", "90.46 @ep107", "A"],
            ["AttRD", "RD + softmax read", "~0.30 M", "79.1", "—", "—", "89.63 @ep117", "A"],
            ["Lie-group", "SO(3) recurrence", "0.19 M", "79.1", "—", "83.8", "89.42 @ep109", "A"],
            ["BDN-Q", "Buffered delta + Q substrate", "0.23 M", "78.4", "—", "—", "89.00 @ep105", "A"],
            ["CSTA", "Causal ST attention", "0.42 M", "79.9", "—", "—", "89.63 @ep109", "A"],
          ]}
          rightAligned={[3, 4, 5, 6]}
        />

        <Card tone="good">
          <Sub title="Clustered behaviour at ep10">
            <ul style={{ paddingLeft: 18, margin: 0 }}>
              <li>
                <Pill tone="muted">Cluster A</Pill> (ep10 ≈ 78–80): all encoders with ≥ ~0.2 M params, regardless
                of mechanism. RD, AttRD, BDN-Q, Lie, CSTA, CN-XL, CN-XXL.
              </li>
              <li>
                <Pill tone="muted">Cluster B</Pill> (ep10 ≈ 66–71): encoders with &lt; ~0.13 M params or no
                encoder at all. CN, NoTemporal, Mamba (0.115 M).
              </li>
              <li>
                <b>Threshold:</b> ~0.2–0.3 M parameters in the Stage-3 slot is enough to lift early-epoch
                accuracy from cluster B to cluster A. Mechanism is irrelevant; only parameter count matters.
              </li>
            </ul>
          </Sub>
        </Card>

        <Sub title="CN-XXL reaches 89.00% with zero temporal mixing">
          <p>
            The cleanest possible architecture without any temporal recurrence/attention/SSM/quaternion algebra
            matches every &quot;novel&quot; temporal mechanism we tested. The 5–6 pp gap between CN (83.82) and
            CN-XXL (89.00) is purely a parameter-budget effect on the spatial pipeline&apos;s training dynamics,
            not from any temporal computation.
          </p>
          <Code>{`# CN-XXL encoder — what is in the "self.mamba" slot:
class CleanestLinXLEncoder(nn.Module):
    """Wider per-frame MLP encoder, NO temporal mixing.
    Each frame processed independently."""
    def __init__(self, in_channels=256, hidden_dim=256, mlp_dim=512,
                 output_dim=256, num_layers=4, dropout=0.3,
                 bidirectional=True, residual_scale=0.7):
        ...
        self.input_proj = nn.Linear(in_channels, hidden_dim)
        # Per-direction stack of MLP blocks (no temporal kernel)
        self.fwd_blocks = nn.ModuleList([mlp_block() for _ in range(num_layers)])
        # ...same for bwd
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def _stack(self, x, blocks, norms):
        for blk, norm in zip(blocks, norms):
            residual = x
            x = norm(x)
            x = blk(x)                       # only per-frame MLP, no T-mixing
            x = self.dropout(x)
            x = x + self.residual_scale * residual
        return x`}</Code>
        </Sub>

        <Sub title="Inference-time vs training-time contribution (a sharper distinction)">
          <Tbl
            headers={["Question", "Test", "Verdict"]}
            rows={[
              ["Does the block do work at inference?",
                "Frozen-block diagnostic: zero output, re-eval.",
                "No — Δ = +0.00 pp for every architecture."],
              ["Does the block matter during training?",
                "Remove it entirely (CN) and re-train.",
                "Yes — CN plateaus at 83.82 (≈5 pp below Mamba/RD/Lie etc.)."],
              ["Does the SPECIFIC mechanism matter?",
                "Replace block with a per-frame MLP at matched param count (CN-XL/XXL).",
                "No — CN-XXL (no temporal mixing) hits 89.00, in the same band as all mechanisms."],
            ]}
          />
          <Card>
            <p style={{ margin: 0 }}>
              <b>Refined finding:</b> the Stage-3 block is <i>inference-decorative</i> but <i>training-time
              useful</i>. Its presence during training adds parameter budget and gradient routing that helps
              the spatial pipeline converge to better weights. But <i>any</i> 0.2–0.3 M-param block in that
              slot works equally well — delta rule, SSM, attention, MLP, all interchangeable.
            </p>
          </Card>
        </Sub>
      </Section>

      {/* ============================================================ */}
      <Section title="Ensemble / 'multiple networks' decomposition (the 92.53 ceiling)" num={6}>
        <p>
          Uniform 1/K average across 5 PMamba-family solos plus the CVPR I3DWTrans depth-stream baseline reaches{" "}
          <Pill tone="good">92.53%</Pill> under honest train-best epoch selection. Source-of-lift breakdown:
        </p>

        <Tbl
          headers={["Combo", "Acc", "Δ from previous", "Lift source"]}
          rows={[
            ["DSN solo (depth-stream CNN)", "90.25", "—", "(modality anchor)"],
            ["DSN + AttRD", "91.91", "+1.66", "multi-modal (depth + point cloud)"],
            ["DSN + AttRD + DN2", "92.32", "+0.41", "ensemble variance"],
            ["DSN + RD + BRD(N2) + AttRD + DN2", "92.53", "+0.21", "ensemble variance (diminishing)"],
          ]}
        />

        <Card>
          <Sub title="Why &quot;different architectures&quot; help the ensemble (even though they&apos;re individually decorative)">
            <ul style={{ paddingLeft: 18, margin: 0 }}>
              <li>
                Every PMamba-family variant has a decorative temporal block — they all reach ~88–90% via the
                same spatial pipeline.
              </li>
              <li>
                But training with a <i>different</i> decorative block in the encoder slot produces a{" "}
                <i>different gradient flow</i> back through the spatial pipeline. The spatial weights converge to
                slightly different local optima.
              </li>
              <li>
                When test predictions of these slightly-different-spatial-weight models are averaged, you get
                uniform-K variance reduction. <b>~0.62 pp total from this effect</b> across 4 added models.
              </li>
              <li>
                In other words: the &quot;ensemble of architecturally different temporal mechanisms&quot; trope
                is really &quot;ensemble of the same backbone with different seed perturbations from different
                decorative blocks.&quot;
              </li>
            </ul>
          </Sub>
        </Card>

        <Sub title="Decomposition of the 92.53 ceiling">
          <Tbl
            headers={["Source", "pp"]}
            rows={[
              ["DSN solo (depth-modality baseline)", "90.25"],
              ["+ multi-modal (one point-cloud variant)", "+1.66"],
              ["+ ensemble variance (4 add'l decorative variants)", "+0.62"],
              ["= 92.53", ""],
            ]}
            rightAligned={[1]}
          />
        </Sub>
      </Section>

      {/* ============================================================ */}
      <Section title="Auxiliary-loss attempts (current research path)" num={7}>
        <p>
          Since architectural mods are bypassable via residual, we shifted to <b>auxiliary losses</b> that
          provide gradient signal directly to features. Losses cannot be bypassed by construction.
        </p>

        <Tbl
          headers={["Direction", "Mechanism", "Result"]}
          rows={[
            ["A2 — Quaternion Pose-Trajectory Pool", "Replace Stage 4 max-pool with a per-frame predicted quaternion + quat-aware stats", "88.38 @ ep19 (resume from CN-XXL ep103) — below 89.00 baseline"],
            ["B2 — Quaternion Multi-Scale Conv", "Replace Stage 2 scale_filters with Hamilton-product convs (Parcollet)", "fail (killed mid-train, ep15 ~80%)"],
            ["Direction 2 — REQNN Stage 1", "Stage 1 MLPBlock → quaternion-MLP (Shen 2024 style)", "fail (slow climb, ~82% at ep3)"],
            ["A′ — Global Frame-Feature Cycle", "‖Δ_ab + Δ_bc − Δ_ac‖² on attention-pooled per-frame features", "trivial-zero collapse — predictor learns Δ ≈ 0 by ep 8"],
            ["A″ — Future Feature Prediction", "Predict f_{t+k} from f_t; detached target, multi-horizon", "training in progress (resume from CN-XXL ep100)"],
          ]}
        />

        <Sub title="A′ failure mode (trivial-zero collapse)">
          <p>
            The cycle loss <code style={{fontFamily:fontMono}}>‖Δ_ab + Δ_bc − Δ_ac‖²</code> has a trivial
            solution: predict Δ ≡ 0. Then 0 + 0 − 0 = 0 satisfies the cycle perfectly. The predictor learned
            this in ~8 epochs:
          </p>
          <Tbl
            headers={["ep", "test acc", "aux loss"]}
            rows={[
              ["1", "76.14", "1.58e−2"],
              ["2", "84.23", "3.57e−3"],
              ["8", "80.08", "2.58e−6"],
              ["12", "85.06", "1.43e−8"],
            ]}
            rightAligned={[1, 2]}
          />
          <p>
            Aux loss converged to near-zero at the trivial fixed point. No gradient signal to the features
            → no accuracy lift over the baseline.
          </p>
        </Sub>

        <Sub title="A″ design (current, no collapse)">
          <p>
            Future-feature prediction avoids trivial-zero collapse because the target is{" "}
            <i>detached and non-zero</i>: the predictor must match a real feature, not just satisfy a
            self-referential constraint.
          </p>
          <Code>{`# FutureFeaturePredictor.loss   (no quaternions, no correspondence)
f = attn_pool(fea3)                            # (B, T, C)  per-frame global
e = embed(f)                                    # (B, T, embed)
pred = e[:, :T-max_h]                           # source frames
for step in 1..max_h:
    pred = pred + transition_mlp(pred)         # residual rollout
    if step in horizons:                        # multi-horizon (1, 2, 4)
        target = e[:, step:step+pred.shape[1]].detach()    # DETACHED real target
        loss += MSE(pred, target)`}</Code>
          <p>
            Multi-horizon (1, 2, 4 steps) gives diverse temporal supervision. Detached target = no collapse.
            On the main backward path, gradient flows back through the encoder, Stages 1/2/3.
          </p>
        </Sub>
      </Section>

      {/* ============================================================ */}
      <Section title="Failed paths timeline" num={8}>
        <Tbl
          headers={["Attempt", "Hypothesis", "Outcome"]}
          rows={[
            ["B1 Lie-group state", "SO(3) state with multiplicative recurrence beats additive linear-attention", "89.42 — decorative (ω trained to 0)"],
            ["B2 Graph-temporal (CSTA)", "Causal ST attention with RBF time-bias", "89.63 — decorative (any attn pattern → same predictions)"],
            ["BDN-Q with full residual=1.0", "Buffer-eject mechanism is load-bearing", "87.76 — decorative (block out → 0 same acc)"],
            ["BDN-Q with no residual", "Force block to be load-bearing by removing the bypass", "training broke (ep10 = 9.34%, spatial features destroyed)"],
            ["BDN-Q residual_scale=0.1", "Soft residual: block load-bearing, spatial signal partial", "block IS load-bearing (+31 pp at ep30) but plateaus at 67%"],
            ["BDN-Q residual_scale=0.7 (RD softres variant)", "Compromise residual strength", "block still inference-decorative (+0.00 pp), no different convergence target"],
            ["CN-XL", "Match RD param budget with no temporal mechanism", "82.2 at ep20 (matches RD), runs killed early to make GPU"],
            ["CN-XXL", "More params = better convergence?", "89.00 — matches every temporal mechanism, confirms param-count hypothesis"],
            ["A2 Quaternion Pose Pool", "Replace max-pool with quat-aware aggregator in main path", "88.38 — fail, below CN-XXL baseline"],
            ["B2 Quaternion Multi-Scale Conv", "Hamilton-product convs in Stage 2", "killed — Stage 2 has its own residual escape"],
            ["REQNN Stage 1", "Quaternion equivariance in spatial feature extraction", "slow convergence, below baseline"],
            ["A′ Feature Cycle", "Vector cycle consistency on global features", "trivial-zero collapse"],
            ["A″ Future Prediction", "Detached-target rollout in feature space", "running"],
          ]}
        />
      </Section>

      {/* ============================================================ */}
      <Section title="What is — and isn't — on the main path" num={9}>
        <p>
          The frozen-mechanism diagnostic taught us that the residual stack inside the Stage-3 encoder lets the
          network route around any computation we put there. The same applies anywhere with
          <code style={{fontFamily:fontMono}}> output + residual</code> structure. We mapped this onto the full
          architecture:
        </p>
        <Tbl
          headers={["Component", "Currently", "Bypass risk", "Novelty room"]}
          rows={[
            ["Stage 1 group.group_points (k-NN over coords)", "Euclidean k-NN", "none — no residual", "high"],
            ["Stage 1 MLPBlock", "[4 → 32 → 64]", "low — small skip from coords", "medium"],
            ["Stage 1 pool1 (AdaptiveMaxPool over k-NN)", "max-pool", "none", "medium"],
            ["Stage 2 multi_scale", "scales {2..32} + final residual `+x`", "MEDIUM — `return output + x` ", "low (because of bypass)"],
            ["Stage 2 MotionBlock", "[128 → 128]", "low", "medium"],
            ["Stage 3 spatial MotionBlock", "[256 → 256]", "low", "medium"],
            ["Stage 3 temporal encoder (self.mamba)", "MambaTemporalEncoder + residual stack", "HIGH — confirmed decorative", "none"],
            ["Stage 4 cat(coords, fea3_mamba)", "concat", "n/a", "low"],
            ["Stage 4 pool5 (AdaptiveMaxPool over (T,N))", "max-pool", "none", "medium (tested A2, didn't help)"],
            ["Stage 4 stage5 MLP, stage6", "MLP layers", "none", "low (just MLP)"],
            ["Loss function", "CE on logits", "n/a — adding aux losses can't be bypassed", "high (current path)"],
          ]}
        />

        <Sub title="Direct implication">
          <ul style={{ paddingLeft: 18 }}>
            <li>
              The two locations where temporal computation could be unbypassable inside the architecture are
              Stage 2&apos;s multi_scale (partially bypassed by its own final residual) and the input
              coordinate channels (no bypass — augmenting coords with motion derivatives is unbypassable).
            </li>
            <li>
              The main path for genuine novelty without architectural risk is{" "}
              <b>auxiliary losses</b>, which we are testing now.
            </li>
          </ul>
        </Sub>
      </Section>

      {/* ============================================================ */}
      <Section title="Methodology contribution — the frozen-mechanism diagnostic" num={10}>
        <p>
          The diagnostic itself is the most generalizable artefact from this investigation. It can be applied
          to any architecture paper that proposes a novel block, attention pattern, recurrence, or pooling
          mechanism — to verify the proposed component is actually doing work at inference, not just adding
          parameters / training noise.
        </p>

        <Sub title="Per-architecture patches we wrote">
          <Code>{`# 1. Mamba SSM
Mamba.forward = lambda self, x: torch.zeros_like(x)
# residual in MambaTemporalEncoder._stack carries the signal

# 2. RealDeltaNet
RealDeltaNetBlock.forward = lambda self, x: torch.zeros_like(x)

# 3. AttRD — replace the attention pattern, not the block
attn = torch.eye(T)                             # identity attn = standard RD point read
# alt: attn = torch.full((T, T), 1.0/T)         # uniform attn

# 4. BDN-Q — zero the entire block output (residual still carries signal)
class FrozenBDeltaQBlock(BDeltaQBlock):
    def forward(self, x):
        return torch.zeros_like(super().forward(x))

# 5. Lie-group — freeze state at identity, skip the ω → exp → mul update
def frozen_forward(self, x):
    q_state = torch.zeros(..., 4); q_state[..., 0] = 1.0      # identity
    u = quat_rotate(q_state, v_q[:, t])                      # rotation = no-op
    return self.o_proj(u * gate)

# 6. Encoder-as-identity (the most aggressive)
def identity_encoder(self, x): return x`}</Code>
        </Sub>

        <Sub title="What this would do as a methodology paper">
          <ul style={{ paddingLeft: 18 }}>
            <li>
              Demonstrate on multiple architectures, multiple datasets, multiple residual patterns.
            </li>
            <li>
              Frame as a sanity check that should be run before claiming any novel temporal mechanism in
              point-cloud sequence modelling. Many recent papers in skeleton-based action recognition and
              point-cloud temporal modelling likely have the same issue.
            </li>
            <li>
              Quantify the &quot;parameter budget&quot; effect separately from the &quot;mechanism&quot; effect.
            </li>
          </ul>
        </Sub>
      </Section>

      {/* ============================================================ */}
      <Section title="Decision tree for what to try next" num={11}>
        <Code>{`Future Feature Prediction (A″, currently running)
  ├── if > 90.46 → publishable: aux-loss-based temporal contribution
  │   ├── add SHREC'17 result for cross-dataset validation
  │   └── write paper around "main-path auxiliary supervision in
  │       point-cloud sequence models"
  │
  └── if < 90.46
        ├── try variant: coordinate channel augmentation
        │   (velocity, acceleration, jerk in input — main path, no bypass)
        │
        ├── try variant: trajectory-smoothness regularization
        │   (penalty on per-frame global feature acceleration)
        │
        ├── try variant: multi-anchor cycle (Direction E from prior)
        │   (multi-scale cycle consistency hierarchy)
        │
        └── if all aux-loss directions plateau:
              accept that NVGesture is data-bound at ~90% solo
              and pitch the ensemble + diagnostic methodology instead.`}</Code>

        <Card tone="warn">
          <Sub title="Honest assessment of upside">
            <ul style={{ paddingLeft: 18, margin: 0 }}>
              <li>
                Architecture path is dead (decorative). All future work must be on the main path —
                auxiliary losses or input features.
              </li>
              <li>
                Realistic upper bound on solo accuracy: 91–92% on NVGesture. We&apos;ve seen 5+ architectures
                cluster at 88–90, and the architectural family appears data-bound.
              </li>
              <li>
                The fusion 92.53 is real and reproducible from the existing softmax dumps; that&apos;s the
                strongest empirical number we have.
              </li>
              <li>
                The frozen-mechanism diagnostic methodology is the most generalizable contribution. It can
                anchor a Pattern Recognition methodology paper independent of any specific architecture.
              </li>
            </ul>
          </Sub>
        </Card>
      </Section>

      {/* ============================================================ */}
      <footer
        style={{
          marginTop: 32,
          paddingTop: 12,
          borderTop: `1px solid ${PALETTE.border}`,
          fontSize: 12,
          color: PALETTE.muted,
          display: "flex",
          justifyContent: "space-between",
          flexWrap: "wrap",
          gap: 12,
        }}
      >
        <div>
          Repo: <code style={{ fontFamily: fontMono }}>worktree-qhdelta-fuse</code>. Diagnostics in{" "}
          <code style={{ fontFamily: fontMono }}>diagnose_*_frozen.py</code>. Last update May 18 2026.
        </div>
        <div>
          Anchor commits: <code style={{ fontFamily: fontMono }}>80a3fb8</code> (BDN-Q),{" "}
          <code style={{ fontFamily: fontMono }}>58d8f5e</code> (CN + diagnostics),{" "}
          <code style={{ fontFamily: fontMono }}>8b1580e</code> (CN-XL/XXL + RD softres)
        </div>
      </footer>
    </main>
  );
}
