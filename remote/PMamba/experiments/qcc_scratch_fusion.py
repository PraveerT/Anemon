"""Fusion analysis with qcc_scratch latest checkpoint + pmamba_base.

1. Load epoch102_model.pt (MotionRigidResFBQ, current best-near-100 region)
2. Run test inference -> qcc_scratch logits
3. Load pmamba_base logits from /tmp/tinyknn_fuse.npz
4. Compute solo, oracle, alpha-sweep, error correlation
"""
import sys
sys.path.insert(0, '/notebooks/PMamba/experiments')
import torch, torch.nn.functional as F, numpy as np, requests
import nvidia_dataloader
from models import motion

TOKEN = "8049556095:AAH0c0KB0DmzFtcW0s97ZS_kQ8ux9gX72eE"
def tg(msg):
    try:
        r = requests.get(f"https://api.telegram.org/bot{TOKEN}/getUpdates", timeout=5).json()
        if r.get("ok") and r.get("result"):
            chat_id = r["result"][-1]["message"]["chat"]["id"]
            requests.post(f"https://api.telegram.org/bot{TOKEN}/sendMessage",
                data={"chat_id": chat_id, "text": msg, "parse_mode": "HTML"}, timeout=10)
    except Exception: pass


CKPT = '/notebooks/PMamba/experiments/work_dir/pmamba_qcc_scratch/epoch102_model.pt'
print(f'Loading {CKPT}')
qcc = motion.MotionRigidResFBQ(num_classes=25, pts_size=96, knn=[32,24,48,24], topk=8).cuda()
sd = torch.load(CKPT, map_location='cuda')
qcc.load_state_dict(sd['model_state_dict'], strict=False)
qcc.eval()

print('Running qcc_scratch inference on test...')
loader = nvidia_dataloader.NvidiaQuaternionQCCParityLoader(
    framerate=32, phase='test', return_correspondence=True)
N = len(loader)
qcc_logits = torch.zeros(N, 25)
labels = torch.zeros(N, dtype=torch.long)
with torch.no_grad():
    for i in range(N):
        s = loader[i]; pts_d = s[0]; label = s[1]
        inputs = {}
        for k, v in pts_d.items():
            if isinstance(v, np.ndarray): v = torch.from_numpy(v)
            inputs[k] = v.cuda().unsqueeze(0)
        out = qcc(inputs)
        qcc_logits[i] = out[0].cpu()
        labels[i] = int(label)
        if (i+1) % 100 == 0: print(f'  {i+1}/{N}')

qcc_solo = (qcc_logits.argmax(-1) == labels).float().mean().item()
print(f'\nqcc_scratch ep102 solo: {qcc_solo*100:.2f}%')

# Load pmamba_base logits
try:
    d = np.load('/tmp/tinyknn_fuse.npz')
    pm_logits = torch.from_numpy(d['pm_logits'])
    pm_labels = torch.from_numpy(d['labels'])
    assert torch.equal(labels, pm_labels), "label order mismatch"
    print(f'pmamba_base logits loaded from cache')
except Exception as e:
    print(f"need to re-run pmamba_base: {e}")
    pm = motion.Motion(num_classes=25, pts_size=256, knn=[32,24,48,24], topk=8).cuda()
    sd2 = torch.load('/notebooks/PMamba/experiments/work_dir/pmamba_branch/epoch110_model.pt',
                     map_location='cuda')
    pm.load_state_dict(sd2['model_state_dict'], strict=False)
    pm.eval()
    loader2 = nvidia_dataloader.NvidiaLoader(framerate=32, phase='test')
    pm_logits = torch.zeros(N, 25)
    pm_labels = torch.zeros(N, dtype=torch.long)
    with torch.no_grad():
        for i in range(N):
            pts, lab, _ = loader2[i]
            pts_t = (pts if torch.is_tensor(pts) else torch.from_numpy(pts)).float().cuda().unsqueeze(0)
            out = pm(pts_t)
            pm_logits[i] = out[0].cpu(); pm_labels[i] = int(lab)

pm_p = F.softmax(pm_logits, -1); qc_p = F.softmax(qcc_logits, -1)
pm_right = pm_p.argmax(-1) == labels
qc_right = qc_p.argmax(-1) == labels

pm_solo = pm_right.float().mean().item()
qcc_solo2 = qc_right.float().mean().item()
oracle = (pm_right | qc_right).float().mean().item()

# Alpha sweep
best_a = 0; best_a_acc = 0
for a in np.arange(0.0, 1.001, 0.01):
    f = a * pm_p + (1 - a) * qc_p
    acc = (f.argmax(-1) == labels).float().mean().item()
    if acc > best_a_acc: best_a_acc = acc; best_a = a

# Error correlation
ep = (~pm_right).float(); et = (~qc_right).float()
cov = ((ep - ep.mean()) * (et - et.mean())).mean().item()
r = cov / (ep.std().item() * et.std().item() + 1e-9)

both_r = (pm_right & qc_right).sum().item()
pm_only = (pm_right & ~qc_right).sum().item()
qc_only = (~pm_right & qc_right).sum().item()
both_w = (~pm_right & ~qc_right).sum().item()

msg = f"""
=== pmamba_qcc_scratch ep102 + pmamba_base ep110 fusion ===
pmamba_base solo:    {pm_solo*100:.2f}%
qcc_scratch solo:     {qcc_solo2*100:.2f}%
oracle:               {oracle*100:.2f}%  headroom +{(oracle-pm_solo)*100:.2f}pp
alpha-blend:          {best_a_acc*100:.2f}% at a={best_a:.2f}  gain={(best_a_acc-pm_solo)*100:+.2f}pp
error correlation r:  {r:.3f}

breakdown (vs base):
  both right    = {both_r}
  pm_only       = {pm_only}
  qcc_only      = {qc_only}  (recovery ceiling if routed)
  both_wrong    = {both_w}

Reference points:
  TinyKNN 82.57:    fuse 90.66 (+0.83), r=0.44, qcc_only=16
  tiny no-knn 78:   fuse 90.04 (+0.21), r=0.39, qcc_only=14
"""
print(msg); tg(msg)

np.savez('/tmp/qcc_scratch_fuse.npz',
         qcc_logits=qcc_logits.numpy(),
         pm_logits=pm_logits.numpy(),
         labels=labels.numpy(),
         qcc_solo=qcc_solo2, pm_solo=pm_solo, oracle=oracle,
         fuse=best_a_acc, fuse_a=best_a, r=r,
         pm_only=pm_only, qc_only=qc_only, both_wrong=both_w)
print("saved /tmp/qcc_scratch_fuse.npz")
