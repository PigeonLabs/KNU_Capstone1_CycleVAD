"""E18 fixed low-motion controls and causal prefix alarm adaptation.

All temporal units are source frames. Fitting accepts explicit FIT/REF banks;
neither test labels nor edit boundaries are inputs to the detector.
"""
from collections import deque
import math
import numpy as np
from .model import TailCalibrator
from .process_experiment import weighted_quantile

VARIANTS = ['S0', 'S1', 'S2', 'S3']

def normalize(g):
    g=np.asarray(g,dtype=np.float64)
    return g / np.maximum(np.sqrt(np.sum(g*g,axis=-1,keepdims=True)),1e-12)

def motion(g, stride=2):
    if stride not in (1,2): raise ValueError('Source stride must be 1 or 2')
    g=normalize(g); out=np.full((len(g),2),np.nan)
    for j,h in enumerate([2,8]):
        k=h//stride
        if len(g)>k: out[k:,j]=np.clip(1-np.sum(g[k:]*g[:-k],axis=1),0,2)
    return out

def contexts(m,theta,gate,stride):
    n=len(m); active=np.isfinite(m[:,0]) & (m[:,0]<=gate)
    locked=np.asarray(theta).copy(); duration=np.zeros(n); start=None
    for j in range(n):
        if not active[j]: start=None; continue
        if start is None: start=j
        locked[j]=theta[start]; duration[j]=(j-start+1)*stride
    return active,locked,duration

def pooled(arrays):
    arrays=[np.asarray(a) for a in arrays if len(a)]
    if not arrays: raise ValueError('No valid observations')
    return np.concatenate(arrays), np.concatenate([np.full(len(a),1/len(a)) for a in arrays])

class LowerCDF:
    def fit(self, arrays):
        values,w=pooled(arrays);order=np.argsort(values,kind='stable')
        self.values=values[order]; self.cum=np.r_[0,np.cumsum(w[order]/w.sum())]
        self.cum[-1]=1.; self.n=len(values); return self
    def cdf(self,values):
        # Right ties: genuinely repeated normal stills are not maximally rare.
        rank=self.cum[np.searchsorted(self.values,values,side='right')]
        return (self.n*rank+1)/(self.n+1)

class StallModel:
    def fit(self, fit, reference, period):
        """Banks map source IDs to fresh stride-2 (global_features, theta)."""
        if set(fit)&set(reference): raise ValueError('FIT/REF sources overlap')
        self.period=float(period)
        fm={i:motion(v[0]) for i,v in fit.items()}
        values,w=pooled([m[np.isfinite(m[:,0]),0] for m in fm.values()])
        self.gate=weighted_quantile(values,w,.1)
        self.global_cdf=[LowerCDF().fit([m[np.isfinite(m[:,j]),j] for m in fm.values()]) for j in range(2)]
        self.local={}; self.coverage={}
        for variant in ['S2','S3']:
            bucket={b:[] for b in range(4)}; sources={b:0 for b in range(4)}
            for i,(_,theta) in fit.items():
                m=fm[i]; _,locked,_=contexts(m,theta,self.gate,2)
                context=theta if variant=='S2' else locked
                bins=np.floor((np.asarray(context)%1)*4).astype(int)
                valid=np.isfinite(m).all(1)
                for b in range(4):
                    v=m[(bins==b)&valid]
                    if len(v): bucket[b].append(v);sources[b]+=1
            self.local[variant]=[];self.coverage[variant]=[]
            for b in range(4):
                n=sum(len(v) for v in bucket[b]);eligible=sources[b]>=3 and n>=200
                self.local[variant].append([LowerCDF().fit([v[:,j] for v in bucket[b]]) for j in range(2)] if eligible else None)
                self.coverage[variant].append({'bin':b,'sources':sources[b],'observations':n,'local_used':eligible})
        self.k={}; self.cal={}
        rm={i:motion(v[0]) for i,v in reference.items()}
        for variant in VARIANTS:
            vals=[]
            if variant!='S0':
                for i,(_,theta) in reference.items():
                    m=rm[i];valid=np.isfinite(m).all(1)
                    vals.append(self.evidence(m,theta,variant,2)[valid])
                x,w=pooled(vals);mean=np.average(x,weights=w)
                self.k[variant]=float(mean+np.sqrt(np.average((x-mean)**2,weights=w)))
            else: self.k[variant]=0.
            raw=[self.raw(rm[i],theta,variant,2)[np.isfinite(rm[i]).all(1)] for i,(_,theta) in reference.items()]
            self.cal[variant]=TailCalibrator().fit(np.concatenate(raw))
        return self

    def evidence(self,m,theta,variant,stride):
        valid=np.isfinite(m).all(1); s=np.zeros(len(m))
        if not valid.any():return s
        safe=np.where(np.isfinite(m),m,0)
        cdf=np.column_stack([self.global_cdf[j].cdf(safe[:,j]) for j in range(2)])
        if variant in ['S2','S3']:
            _,locked,_=contexts(m,theta,self.gate,stride)
            context=theta if variant=='S2' else locked
            bins=np.floor((np.asarray(context)%1)*4).astype(int)
            for b,cals in enumerate(self.local[variant]):
                mask=bins==b
                if cals is not None and mask.any():
                    local=np.column_stack([cals[j].cdf(safe[mask,j]) for j in range(2)])
                    cdf[mask]=.5*cdf[mask]+.5*local
        s[valid]=np.min(-np.log(np.maximum(cdf[valid],np.finfo(float).tiny)),axis=1)
        return s

    def raw(self,m,theta,variant,stride=2):
        valid=np.isfinite(m).all(1);active,_,duration=contexts(m,theta,self.gate,stride)
        if variant=='S0':return np.where(valid,duration/self.period,0)
        evidence=self.evidence(m,theta,variant,stride); out=np.zeros(len(m));u=0.
        for j in range(len(m)):
            u=max(0.,u+(stride/2)*(evidence[j]-self.k[variant])) if active[j] and valid[j] else 0.
            out[j]=u
        return out

    def score(self,g,theta,variants=VARIANTS,stride=2):
        m=motion(g,stride);valid=np.isfinite(m).all(1)
        raw=np.column_stack([self.raw(m,theta,v,stride) for v in variants])
        scores=np.column_stack([self.cal[v].score(raw[:,j]) for j,v in enumerate(variants)])
        scores[~valid]=0.
        return {'raw':raw,'scores':scores,'motion':m,'valid':valid}

class StallState:
    """Bounded-state S0/S1/S2/S3 inference; one fresh observation per step."""
    def __init__(self, model, stride=2, variant='S3'):
        if stride not in (1,2) or variant not in VARIANTS: raise ValueError('Invalid cadence/variant')
        self.model=model;self.stride=stride;self.variant=variant;self.reset()
    def reset(self):
        self.history=deque(maxlen=8//self.stride+1);self.count=0;self.entry=None;self.duration=0.;self.u=0.
    def step(self,g,theta):
        model=self.model;self.history.append(normalize(g)); m=[]
        for h in [2,8]:
            k=h//self.stride
            m.append(float(np.clip(1-np.sum(self.history[-1]*self.history[-1-k]),0,2)) if self.count>=k else np.nan)
        self.count+=1
        active=np.isfinite(m[0]) and m[0]<=model.gate
        if active:
            if self.entry is None:self.entry=float(theta)
            self.duration+=self.stride
        else:self.entry=None;self.duration=0.;self.u=0.
        valid=bool(np.isfinite(m).all());evidence=0.
        if valid and self.variant!='S0':
            cdf=np.array([model.global_cdf[j].cdf(m[j]) for j in range(2)])
            if self.variant in ['S2','S3']:
                context=self.entry if self.variant=='S3' and self.entry is not None else theta
                local=model.local[self.variant][int((context%1)*4)]
                if local is not None:cdf=.5*cdf+.5*np.array([local[j].cdf(m[j]) for j in range(2)])
            evidence=float(np.min(-np.log(cdf)))
            self.u=max(0.,self.u+self.stride/2*(evidence-model.k[self.variant])) if active else 0.
        elif self.variant=='S0':self.u=self.duration/model.period if valid else 0.
        else:self.u=0.
        score=float(model.cal[self.variant].score(self.u)) if valid else 0.
        return {'score':score,'raw':self.u,'valid':valid,'motion':m,'evidence':evidence,'entry_theta':self.entry}

def low_stat(values):
    values=np.asarray(values)
    if not len(values):raise ValueError('Empty prefix')
    return np.median(np.sort(values,axis=0)[:max(1,math.ceil(.2*len(values)))],axis=0)

def adaptive_thresholds(scores, fixed, reference_low, stride=2):
    """Held-frame scores in, full [frame, quantile, arm] thresholds out."""
    scores=np.asarray(scores);out=np.broadcast_to(fixed,(len(scores),*np.shape(fixed))).copy()
    if len(scores)>48:
        delta=low_stat(scores[:48:stride])-reference_low
        out[48:]=fixed+.3*np.clip(delta,0,fixed)
    return out

class PrefixAlarm:
    def __init__(self,threshold,reference_low,adaptive=True):
        self.fixed=float(threshold);self.reference_low=float(reference_low);self.adaptive=adaptive;self.reset()
    def reset(self):self.prefix=[];self.threshold=self.fixed;self.count=0
    def step(self,score,fresh=True):
        if self.count<48 and fresh:self.prefix.append(float(score))
        if self.count==48 and self.adaptive:
            self.threshold=self.fixed+.3*float(np.clip(low_stat(self.prefix)-self.reference_low,0,self.fixed))
        self.count+=1
        return {'threshold':self.threshold,'alarm':bool(score>self.threshold)}

class StreamingStallDetector:
    """Frozen CP + S3 adapter. Six readouts share only current causal features."""
    arms=['C0','C1','B0','B1','B2','B3']
    def __init__(self,checkpoint,fitted,stride=2):
        from .streaming import StreamingDetector
        self.base=StreamingDetector(checkpoint,fitted['thresholds'][3,2],stride,'CP')
        self.stall=StallState(fitted['stall'],stride,'S3');self.stride=stride
        cols=[0,1,2,3,2,3];self.policies=[PrefixAlarm(fitted['thresholds'][3,j],fitted['reference_low'][j],k in [4,5]) for k,j in enumerate(cols)]
        self.last_stall=None
    def reset(self):
        self.base.reset();self.stall.reset();self.last_stall=None
        for policy in self.policies:policy.reset()
    def step(self,global_z=None,patches=None):
        result=self.base.step(global_z,patches)
        if result['fresh']:
            g=np.asarray(global_z).astype(np.float16).astype(np.float32)
            self.last_stall=self.stall.step(g,result['theta'])
        c,p=result['C'],result['P'];u=self.last_stall['score'];cp=max(c,p);joint=max(cp,u)
        scores=np.array([c,max(c,u),cp,joint,cp,joint])
        decisions=[policy.step(score,result['fresh']) for policy,score in zip(self.policies,scores)]
        return {'source_index':result['source_index'],'fresh':result['fresh'],'base':result,'stall':self.last_stall,'scores':scores,
                'thresholds':np.array([v['threshold'] for v in decisions]),'alarms':np.array([v['alarm'] for v in decisions])}
