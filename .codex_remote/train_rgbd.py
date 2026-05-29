"""RGBD R(2+1)D for NVGesture: strong orthogonal fusion partner, no DSN, with a
quaternion component on the 4-channel RGBD input (canonical quaternion use).

RGB(3) + depth(1) = 4-channel, both depth-fg-cropped & aligned. Pretrained
R(2+1)D stem expanded 3->4 (RGB weights copied, depth = mean-init). --head quat
inserts a Hamilton-product bottleneck before the classifier; --head real is the
matched control. Dumps train+test logits (sigs) for honest fusion.
"""
import os, time, random, argparse
import numpy as np
from PIL import Image
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

RGB='/notebooks/cvpr_data/rgb'; DEPTH='/notebooks/cvpr_data/depth'
SPLITS='/notebooks/cvpr_data/dataset_splits'
CACHE='/notebooks/Anemon/dataset/Nvidia/Processed/rgbd_fgcrop_cache'
KMEAN=np.array([0.43216,0.394666,0.37645,0.5],np.float32)
KSTD=np.array([0.22803,0.22145,0.216989,0.25],np.float32)


def read_split(p):
    return [(a[0].strip('/'),int(a[1]),int(a[2])) for a in (ln.split() for ln in open(p)) if len(a)>=3]


def fg_bbox(dep,pad=0.18):
    m=np.zeros(dep[0].shape,bool)
    for a in dep: m|=a>10
    ys,xs=np.where(m); h,w=dep[0].shape
    if len(xs)==0: return (0,0,w,h)
    y0,y1,x0,x1=ys.min(),ys.max()+1,xs.min(),xs.max()+1
    p=int(max(y1-y0,x1-x0)*pad)+4
    return (max(0,x0-p),max(0,y0-p),min(w,x1+p),min(h,y1+p))


def build_cache(split,phase,size=128):
    os.makedirs(CACHE,exist_ok=True); base=os.path.join(CACHE,f'{phase}_s{size}')
    if os.path.exists(base+'.npy'):
        return np.load(base+'.npy',mmap_mode='r'),np.load(base+'_lab.npy'),np.load(base+'_sig.npy',allow_pickle=True)
    recs=read_split(os.path.join(SPLITS,split)); maxf=max(r[1] for r in recs)
    clips=np.zeros((len(recs),maxf,size,size,4),np.uint8)
    labs=np.array([r[2] for r in recs],np.int64); sigs=np.array([r[0] for r in recs],dtype=object)
    t0=time.time()
    for i,(rel,nf,_) in enumerate(recs):
        dd,rd=os.path.join(DEPTH,rel),os.path.join(RGB,rel)
        dep=[np.asarray(Image.open(os.path.join(dd,f'{t:06d}.jpg')).convert('L'),np.uint8) for t in range(nf)]
        bb=fg_bbox(dep)
        for t in range(nf):
            r=Image.open(os.path.join(rd,f'{t:06d}.jpg')).convert('RGB').crop(bb).resize((size,size),Image.BILINEAR)
            d=Image.fromarray(dep[t]).crop(bb).resize((size,size),Image.BILINEAR)
            clips[i,t,:,:,:3]=np.asarray(r,np.uint8); clips[i,t,:,:,3]=np.asarray(d,np.uint8)
        if (i+1)%200==0 or i+1==len(recs): print(f'[cache {phase}] {i+1}/{len(recs)} {time.time()-t0:.0f}s',flush=True)
    np.save(base+'.npy',clips); np.save(base+'_lab.npy',labs); np.save(base+'_sig.npy',sigs)
    return np.load(base+'.npy',mmap_mode='r'),labs,sigs


class DS(Dataset):
    def __init__(self,c,l,s,frames=40,crop=112,train=False):
        self.c,self.l,self.s,self.frames,self.crop,self.train=c,l,s,frames,crop,train
        self.cache=c.shape[2]; self.maxf=c.shape[1]
    def __len__(self): return len(self.l)
    def _tidx(self):
        if self.train:
            span=random.randint(max(self.frames,int(self.maxf*0.6)),self.maxf); st=random.randint(0,self.maxf-span)
            idx=np.linspace(st,st+span-1,self.frames); return np.clip(np.rint(idx+np.random.uniform(-0.4,0.4,self.frames)),0,self.maxf-1).astype(np.int64)
        return np.linspace(0,self.maxf-1,self.frames).round().astype(np.int64)
    def __getitem__(self,i):
        x=np.asarray(self.c[i,self._tidx()],np.float32)/255.0
        lim=self.cache-self.crop
        if self.train and lim>0: y0,x0=random.randint(0,lim),random.randint(0,lim)
        else: y0=x0=max(0,lim//2)
        x=x[:,y0:y0+self.crop,x0:x0+self.crop,:]
        if self.train and random.random()<0.5: x=x[:,:,::-1,:].copy()
        x=(x-KMEAN)/KSTD
        return torch.from_numpy(x).permute(3,0,1,2).float(),int(self.l[i]),str(self.s[i])


class QuatLinear(nn.Module):
    def __init__(self,iq,oq):
        super().__init__(); self.iq,self.oq=iq,oq
        self.w=nn.ParameterList([nn.Parameter(torch.empty(oq,iq)) for _ in range(4)])
        for w in self.w: nn.init.kaiming_uniform_(w,a=5**0.5)
        self.b=nn.Parameter(torch.zeros(4*oq))
    def forward(self,x):
        xr,xi,xj,xk=x.split(self.iq,-1); wr,wi,wj,wk=self.w
        cr=F.linear(xr,wr)-F.linear(xi,wi)-F.linear(xj,wj)-F.linear(xk,wk)
        ci=F.linear(xr,wi)+F.linear(xi,wr)+F.linear(xj,wk)-F.linear(xk,wj)
        cj=F.linear(xr,wj)-F.linear(xi,wk)+F.linear(xj,wr)+F.linear(xk,wi)
        ck=F.linear(xr,wk)+F.linear(xi,wj)-F.linear(xj,wi)+F.linear(xk,wr)
        return torch.cat([cr,ci,cj,ck],-1)+self.b


@torch.no_grad()
def evaluate(m,loader,dev,dump=None):
    m.eval(); L,Y,S=[],[],[]
    for x,y,s in loader:
        with torch.autocast('cuda'):
            o=m(x.to(dev))+m(torch.flip(x,dims=[4]).to(dev))
        L.append(o.float().cpu().numpy()); Y.append(y.numpy()); S+=list(s)
    L=np.concatenate(L); Y=np.concatenate(Y)
    if dump: np.savez(dump,logits=L,labels=Y,sigs=np.array(S,dtype=object))
    return (L.argmax(1)==Y).mean()*100


class RGBDNet(nn.Module):
    def __init__(self,head='real'):
        super().__init__()
        from torchvision.models.video import r2plus1d_18,R2Plus1D_18_Weights
        b=r2plus1d_18(weights=R2Plus1D_18_Weights.KINETICS400_V1)
        old=b.stem[0]  # Conv3d(3,45,(1,7,7))
        new=nn.Conv3d(4,old.out_channels,old.kernel_size,old.stride,old.padding,bias=False)
        with torch.no_grad():
            new.weight[:,:3]=old.weight; new.weight[:,3:]=old.weight.mean(1,keepdim=True)
        b.stem[0]=new
        self.feat_dim=b.fc.in_features; b.fc=nn.Identity(); self.backbone=b
        self.head_kind=head
        if head=='quat':
            self.q=QuatLinear(self.feat_dim//4, 64); self.cls=nn.Linear(256,25)
        else:
            self.cls=nn.Linear(self.feat_dim,25)
    def forward(self,x):
        f=self.backbone(x)
        if self.head_kind=='quat':
            f=F.gelu(self.q(f)); return self.cls(f)
        return self.cls(f)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--epochs',type=int,default=60); ap.add_argument('--bs',type=int,default=5)
    ap.add_argument('--lr',type=float,default=4e-4); ap.add_argument('--blr',type=float,default=4e-5)
    ap.add_argument('--frames',type=int,default=40); ap.add_argument('--head',default='real')
    a=ap.parse_args()
    WD=f'/notebooks/Anemon/experiments/work_dir/rgbd_{a.head}'; os.makedirs(WD,exist_ok=True)
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    tr=build_cache('train.txt','train'); va=build_cache('valid.txt','valid')
    mk=lambda ds,sh: DataLoader(ds,batch_size=a.bs,shuffle=sh,num_workers=6,drop_last=sh,pin_memory=True,persistent_workers=True)
    dtr=mk(DS(*tr,frames=a.frames,train=True),True); dva=mk(DS(*va,frames=a.frames),False); dtre=mk(DS(*tr,frames=a.frames),False)
    dev='cuda'; m=RGBDNet(a.head).to(dev)
    head=[p for n,p in m.named_parameters() if n.startswith('cls') or n.startswith('q.') or n.startswith('backbone.stem.0')]
    hids={id(p) for p in head}; body=[p for p in m.parameters() if id(p) not in hids]
    opt=torch.optim.AdamW([{'params':body,'lr':a.blr},{'params':head,'lr':a.lr}],weight_decay=0.02)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,a.epochs); sc=torch.cuda.amp.GradScaler()
    lf=nn.CrossEntropyLoss(label_smoothing=0.1); best=0.0
    for ep in range(a.epochs):
        m.train(); t0=time.time(); tot=cor=seen=0
        for x,y,_ in dtr:
            x,y=x.to(dev),y.to(dev); opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda'): o=m(x); loss=lf(o,y)
            sc.scale(loss).backward(); sc.step(opt); sc.update()
            tot+=loss.item()*y.numel(); cor+=(o.argmax(1)==y).sum().item(); seen+=y.numel()
        sch.step(); acc=evaluate(m,dva,dev,dump=os.path.join(WD,'test_logits.npz'))
        msg=f'ep{ep:3d} tr_loss={tot/seen:.4f} tr_acc={cor/seen*100:.2f}% te_acc={acc:.2f}% best={best:.2f} dt={time.time()-t0:.0f}s'
        if acc>best:
            best=acc; evaluate(m,dva,dev,dump=os.path.join(WD,'best_logits.npz')); evaluate(m,dtre,dev,dump=os.path.join(WD,'train_logits.npz')); msg+=' *'
        print(msg,flush=True); open(os.path.join(WD,'run.log'),'a').write(msg+'\n')
    print(f'DONE best={best:.2f}',flush=True)


if __name__=='__main__': main()
