"""MQAR (Multi-Query Associative Recall) — head-to-head AttRD vs DeltaNet.

Task: a sequence of (key, value) pairs followed by queries; predict the value
for each query. Standard benchmark from Zoology / DeltaNet / Mamba-2 papers.

Log format matches tg_messages.sh parser:
    Training epoch: N
    Mean training acc: X
    Mean training loss: X
    Test, Evaluation: Epoch N ... prec1 X, prec5 Y
"""
import argparse, math, os, sys, time, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# MQAR data generator
# -----------------------------------------------------------------------------
def make_mqar_batch(B, T, vocab, num_kv, num_q, device, rng):
    """One batch of MQAR.

    Layout per row: [k1 v1 k2 v2 ... k_n v_n  q1 ? q2 ? ... q_m ?]
        - keys/values drawn iid from [2, vocab-1]; 0 = pad, 1 = ?-placeholder
        - queries are sampled WITH replacement from existing keys
        - target at the '?' position is the value for that key
        - loss only on '?' positions
    """
    assert T >= 2 * num_kv + 2 * num_q
    x = torch.zeros(B, T, dtype=torch.long, device=device)
    y = torch.full((B, T), -100, dtype=torch.long, device=device)
    QP = 1  # query placeholder token
    for b in range(B):
        keys = rng.choice(vocab - 2, num_kv, replace=False) + 2
        vals = rng.integers(2, vocab, num_kv)
        kv_seq = np.empty(2 * num_kv, dtype=np.int64)
        kv_seq[0::2] = keys
        kv_seq[1::2] = vals
        # Random pad before kv block
        kv_start = 0
        x[b, kv_start:kv_start + 2 * num_kv] = torch.from_numpy(kv_seq).to(device)
        # Queries: sample with replacement
        idx = rng.integers(0, num_kv, num_q)
        q_start = 2 * num_kv
        for j, ki in enumerate(idx):
            pos = q_start + 2 * j
            x[b, pos]     = int(keys[ki])
            x[b, pos + 1] = QP
            y[b, pos + 1] = int(vals[ki])
    return x, y


# -----------------------------------------------------------------------------
# DeltaNet block (real-valued, canonical Schlag/Yang formulation, chunkwise scan)
# -----------------------------------------------------------------------------
class DeltaNetBlock(nn.Module):
    def __init__(self, d_model, num_heads=4, head_dim=32, dropout=0.1):
        super().__init__()
        self.H = num_heads
        self.D = head_dim
        d_inner = num_heads * head_dim
        self.q = nn.Linear(d_model, d_inner, bias=False)
        self.k = nn.Linear(d_model, d_inner, bias=False)
        self.v = nn.Linear(d_model, d_inner, bias=False)
        self.beta = nn.Linear(d_model, num_heads)
        self.o = nn.Linear(d_inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, _ = x.shape
        H, D = self.H, self.D
        q = self.q(x).view(B, T, H, D)
        k = F.normalize(self.k(x).view(B, T, H, D), dim=-1)
        v = self.v(x).view(B, T, H, D)
        beta = torch.sigmoid(self.beta(x)).view(B, T, H, 1)
        q = F.silu(q)

        # DeltaRule: S_t = S_{t-1} (I - beta_t k_t k_t^T) + beta_t k_t v_t^T
        # Causal sequential scan (slow but correct). T ~ 64-256 fine.
        S = torch.zeros(B, H, D, D, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(T):
            kt = k[:, t]          # B,H,D
            vt = v[:, t]          # B,H,D
            bt = beta[:, t]       # B,H,1
            # Sk: (B,H,D) <- (B,H,D,D) @ k_t
            Sk = torch.einsum('bhij,bhj->bhi', S, kt)
            err = vt - Sk
            S = S + bt.unsqueeze(-1) * torch.einsum('bhi,bhj->bhij', err, kt)
            yt = torch.einsum('bhij,bhj->bhi', S, q[:, t])
            ys.append(yt)
        y = torch.stack(ys, dim=1).reshape(B, T, H * D)
        return self.o(self.dropout(y))


# -----------------------------------------------------------------------------
# AttRD block: same DeltaRule write, attention read over the full {S_τ} sequence
# -----------------------------------------------------------------------------
class AttRDBlock(nn.Module):
    def __init__(self, d_model, num_heads=4, head_dim=32, d_read=32, dropout=0.1):
        super().__init__()
        self.H = num_heads
        self.D = head_dim
        self.d_read = d_read
        d_inner = num_heads * head_dim
        self.q = nn.Linear(d_model, d_inner, bias=False)
        self.k = nn.Linear(d_model, d_inner, bias=False)
        self.v = nn.Linear(d_model, d_inner, bias=False)
        self.beta = nn.Linear(d_model, num_heads)
        self.read_q = nn.Linear(d_model, num_heads * d_read, bias=False)
        self.read_k = nn.Linear(d_model, num_heads * d_read, bias=False)
        self.o = nn.Linear(d_inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, _ = x.shape
        H, D, d_r = self.H, self.D, self.d_read
        q = self.q(x).view(B, T, H, D)
        k = F.normalize(self.k(x).view(B, T, H, D), dim=-1)
        v = self.v(x).view(B, T, H, D)
        beta = torch.sigmoid(self.beta(x)).view(B, T, H, 1)
        q = F.silu(q)

        # Same DeltaRule, but stash every S_t  (B, T, H, D, D)
        S = torch.zeros(B, H, D, D, device=x.device, dtype=x.dtype)
        S_seq = []
        for t in range(T):
            kt = k[:, t]; vt = v[:, t]; bt = beta[:, t]
            Sk = torch.einsum('bhij,bhj->bhi', S, kt)
            err = vt - Sk
            S = S + bt.unsqueeze(-1) * torch.einsum('bhi,bhj->bhij', err, kt)
            S_seq.append(S)
        S_seq = torch.stack(S_seq, dim=1)   # B,T,H,D,D

        # V_pair[b,t,s,h,d] = q_t^T S_s  →  (B,T,T,H,D)
        V_pair = torch.einsum('bthd,bshde->btshe', q, S_seq)

        rq = self.read_q(x).view(B, T, H, d_r)
        rk = self.read_k(x).view(B, T, H, d_r)
        scores = torch.einsum('bthd,bshd->bths', rq, rk) / math.sqrt(d_r)

        # Causal mask: query at t can only read S_s for s <= t. scores: B,T_q,H,T_s
        mask = torch.ones(T, T, device=x.device).tril().bool()
        scores = scores.masked_fill(~mask.view(1, T, 1, T), float('-inf'))
        attn = self.attn_dropout(F.softmax(scores, dim=-1))   # softmax over s

        Y = torch.einsum('bths,btshe->bthe', attn, V_pair)
        Y = Y.reshape(B, T, H * D)
        return self.o(self.dropout(Y))


# -----------------------------------------------------------------------------
# Common wrapper: embed → stack of blocks (residual + norm) → LM head
# -----------------------------------------------------------------------------
class SeqModel(nn.Module):
    def __init__(self, arch, vocab, d_model=128, num_layers=2, num_heads=4,
                 head_dim=32, d_read=32, dropout=0.1):
        super().__init__()
        self.emb = nn.Embedding(vocab, d_model)
        blocks = []
        for _ in range(num_layers):
            if arch == 'deltanet':
                blocks.append(DeltaNetBlock(d_model, num_heads, head_dim, dropout))
            elif arch == 'attrd':
                blocks.append(AttRDBlock(d_model, num_heads, head_dim, d_read, dropout))
            else:
                raise ValueError(arch)
        self.blocks = nn.ModuleList(blocks)
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.final_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
        self.head.weight = self.emb.weight  # tied

    def forward(self, x):
        h = self.emb(x)
        for blk, norm in zip(self.blocks, self.norms):
            h = h + blk(norm(h))
        return self.head(self.final_norm(h))


# -----------------------------------------------------------------------------
# Train / eval
# -----------------------------------------------------------------------------
def run(arch, args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    model = SeqModel(arch, args.vocab, args.d_model, args.layers,
                     args.heads, args.head_dim, args.d_read, dropout=0.1).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'run: mqar_{arch}', flush=True)
    print(f'arch: {arch}  params: {n_params/1e6:.2f}M  T={args.T} kv={args.kv} q={args.q} vocab={args.vocab}', flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_p1 = 0.0; best_ep = 0
    for ep in range(args.epochs):
        model.train()
        tot_loss = 0.0; tot_correct = 0; tot_tokens = 0
        print(f'Training epoch: {ep}', flush=True)
        for step in range(args.steps_per_epoch):
            x, y = make_mqar_batch(args.bs, args.T, args.vocab, args.kv, args.q, device, rng)
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, args.vocab), y.reshape(-1), ignore_index=-100)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            with torch.no_grad():
                pred = logits.argmax(-1)
                m = (y != -100)
                tot_correct += ((pred == y) & m).sum().item()
                tot_tokens  += m.sum().item()
                tot_loss    += loss.item()
        sched.step()
        tr_acc = 100 * tot_correct / max(1, tot_tokens)
        tr_loss = tot_loss / args.steps_per_epoch
        print(f'Mean training acc: {tr_acc:.4f}', flush=True)
        print(f'Mean training loss: {tr_loss:.4f}', flush=True)

        # Eval
        model.eval()
        p1 = 0; p5 = 0; ntot = 0
        rng_eval = np.random.default_rng(123456 + ep)  # held-out seed family
        with torch.no_grad():
            for _ in range(args.eval_steps):
                x, y = make_mqar_batch(args.bs, args.T, args.vocab, args.kv, args.q, device, rng_eval)
                logits = model(x)
                m = (y != -100)
                top5 = logits.topk(5, dim=-1).indices       # B,T,5
                correct1 = (top5[..., 0] == y) & m
                correct5 = (top5 == y.unsqueeze(-1)).any(-1) & m
                p1 += correct1.sum().item()
                p5 += correct5.sum().item()
                ntot += m.sum().item()
        p1_pct = 100 * p1 / max(1, ntot)
        p5_pct = 100 * p5 / max(1, ntot)
        print(f'Test, Evaluation: Epoch {ep} prec1 {p1_pct:.4f}, prec5 {p5_pct:.4f}', flush=True)
        if p1_pct > best_p1:
            best_p1 = p1_pct; best_ep = ep
        print(f'best: ep {best_ep} p1={best_p1:.2f}%', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--arch', choices=['deltanet', 'attrd'], required=True)
    p.add_argument('--vocab', type=int, default=256)
    p.add_argument('--T', type=int, default=64)
    p.add_argument('--kv', type=int, default=8)
    p.add_argument('--q', type=int, default=16)
    p.add_argument('--d_model', type=int, default=128)
    p.add_argument('--layers', type=int, default=2)
    p.add_argument('--heads', type=int, default=4)
    p.add_argument('--head_dim', type=int, default=32)
    p.add_argument('--d_read', type=int, default=32)
    p.add_argument('--bs', type=int, default=64)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--steps_per_epoch', type=int, default=50)
    p.add_argument('--eval_steps', type=int, default=20)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    run(args.arch, args)
