"""Frozen E18B/C experiment runner: fit first, then evaluate fixed candidates."""
from pathlib import Path
import time
import joblib
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from .data import hold, labels_for, write_json, fingerprint
from .model import descriptor
from .paper_experiments import read, calibrate, conditional
from .paper_calibration import suffix
from .paper_diagnostics import sha
from .pipeline import load_rows
from .process_experiment import starts, edit_map, halfup
from .phase_experiment import compact
from .stall import StallModel, VARIANTS, low_stat, adaptive_thresholds

BASE_ARMS=['C0','C1','B0','B1','Full','P']
EXTRA=['CP_S0','CP_S1','CP_S2']
ARMS=['C0','C1','B0','B1','B2','B3','B0_T110','B0_T130','B1_T110','B1_T130','Full','P']
COLS=[0,1,2,3,2,3,2,2,3,3,4,5]

def baseline(root,rel,identity):
    tag='_'.join(identity.split('/')[1:])
    with np.load(root/'runs/E11B'/rel/f'{tag}.npz') as d:s=d['scores'].copy()
    with np.load(root/'runs/E15A'/rel/f'{tag}.npz') as d:theta=d['theta'].copy()
    return s,theta

def score_bank(stall,g,theta,old,seed,stride=2):
    ss=stall.score(g[::stride],theta[::stride],stride=stride)
    st=hold(np.arange(0,len(g),stride),ss['scores'],len(g))
    c,p,cp=old[:,4],old[:,5],old[:,6]
    scores=np.column_stack([c,np.maximum(c,st[:,3]),cp,np.maximum(cp,st[:,3]),old[:,3],p])
    if seed==42:scores=np.column_stack([scores,*[np.maximum(cp,st[:,j]) for j in range(3)]])
    return scores,st

def alarm_bank(scores,thresholds,lref,stride=2):
    cols=COLS+list(range(6,scores.shape[1]));allscore=scores[:,cols]
    fixed=thresholds[:,cols].copy()
    fixed[:,6]*=1.1;fixed[:,7]*=1.3;fixed[:,8]*=1.1;fixed[:,9]*=1.3
    dynamic=np.broadcast_to(fixed,(len(scores),*fixed.shape)).copy()
    adapted=adaptive_thresholds(scores,thresholds,lref,stride)
    dynamic[:,:,4]=adapted[:,:,2];dynamic[:,:,5]=adapted[:,:,3]
    alarm=allscore[:,None,:]>dynamic
    assert not np.any(alarm[:,:,4]&~alarm[:,:,2])
    assert not np.any(alarm[:,:,5]&~alarm[:,:,3])
    return allscore,dynamic,alarm

def records(identity,labels,scores,alarms,T,D,normal=False):
    frames=[];events=[]
    prior=np.cumsum(labels==1)>0
    for name,mask in [('normal',labels==0),('anomaly',labels==1),('normal_before',(labels==0)&~prior),('normal_after',(labels==0)&prior)]:
        if mask.any():frames.append({'id':identity,'stratum':name,'frames':int(mask.sum()),'alarm_count':alarms[mask].sum(0).tolist()})
    delta=np.diff(np.r_[False,labels==1,False].astype(int))
    for j,(lo,hi) in enumerate(zip(np.flatnonzero(delta==1),np.flatnonzero(delta==-1))):
        a=alarms[lo:min(hi,lo+D)];hit=a.any(0)
        events.append({'id':f'{identity}:event{j}','source_id':identity,'onset':int(lo),'end':int(hi),'deadline_frames':len(a),'hit':hit.astype(int).tolist(),'delay':np.where(hit,a.argmax(0),-1).tolist()})
    record=None
    if normal:
        loc=starts(len(labels),T,D);hits=np.stack([alarms[o:o+D].any(0) for o in loc])
        record={'id':identity,'frames':len(labels),'windows':len(loc),'grid_far':hits.mean(0).tolist(),'frame_fpr':alarms.mean(0).tolist(),'early_frame_fpr':alarms[:32].mean(0).tolist()}
    return frames,events,record

def run_unit(root,rows,seed,scene,fold,synthetic=False):
    began=time.perf_counter();rel=suffix(seed,scene,fold);cfg=read(root/'configs/experiments/followup/e18_v1.json')
    r=read(root/'results/E11B'/rel/'run.json');splits=r['splits'];T,D,W=[r[k] for k in ['T','D','W']]
    modelpath=root/'runs/E11'/rel/'model.joblib';model=joblib.load(modelpath);lookup={v['id']:v for v in rows}
    local=root/'runs/E18B'/rel;local.mkdir(parents=True,exist_ok=True)
    trained=local/'stall.joblib';frozen=root/'results/E18B'/rel/'fitted.json'
    # Only training IDs are loaded during fitting/calibration. Test starts below.
    trainids=sum(splits.values(),[])
    g={};theta={};old={};signature={}
    for i in trainids:
        data=load_rows(root/'runs/stride1_seed42',[lookup[i]])[0]
        g[i]=data['global'];signature[i]=str(data['signature'])
        if i in splits['fit']:
            z=descriptor(data)[::2];x=model['base']['tracker'].reducer.transform(z)/model['base']['tracker'].scale
            theta[i]=hold(np.arange(0,len(data['global']),2),model['base']['tracker'].predict_latent(x,np.arange(len(x))*2)['angle'],len(data['global']))
        elif i not in splits['validation']:
            old[i],theta[i]=baseline(root,rel,i)
    fit={i:(g[i][::2],theta[i][::2]) for i in splits['fit']}
    ref={i:(g[i][::2],theta[i][::2]) for i in splits['reference']}
    stall=StallModel().fit(fit,ref,T)
    scored={};stalls={}
    for role in ['reference','threshold']:
        for i in splits[role]:scored[i],stalls[i]=score_bank(stall,g[i],theta[i],old[i],seed)
    thresholds,_,counts=calibrate(scored,splits,T,D)
    np.testing.assert_array_equal(thresholds[:,[0,2,4,5]],np.asarray(r['window_thresholds'])[:,[4,6,3,5]])
    lref=np.median([low_stat(scored[i][:48:2]) for i in splits['reference'] if len(scored[i])>=48],axis=0)
    joblib.dump({'stall':stall,'thresholds':thresholds,'reference_low':lref},trained,compress=3)
    write_json(frozen,{'status':'fitted_before_test','config_sha256':fingerprint(cfg),'checkpoint_sha256':sha(modelpath),'stall_sha256':sha(trained),'splits':splits,'gate':stall.gate,'k':stall.k,'coverage':stall.coverage,'thresholds':thresholds.tolist(),'reference_low':lref.tolist(),'threshold_windows':counts,'feature_signatures':signature})
    # From here onward parameters are frozen; no test feedback is used.
    evaluated=splits['normal_evaluation']+([v['id'] for v in rows if v['scene']==scene and v['partition']=='testing'] if fold==0 else [])
    for i in evaluated:
        if i not in g:
            data=load_rows(root/'runs/stride1_seed42',[lookup[i]])[0];g[i]=data['global'];signature[i]=str(data['signature'])
            old[i],theta[i]=baseline(root,rel,i)
        scored[i],stalls[i]=score_bank(stall,g[i],theta[i],old[i],seed)
    b_arms=BASE_ARMS+(EXTRA if seed==42 else []);c_arms=ARMS+(EXTRA if seed==42 else [])
    bf=[];be=[];bn=[];cf=[];ce=[];cn=[];ranking=[];hist=[]
    for i in evaluated:
        y=labels_for(lookup[i]);s=scored[i];a=s[:,None,:]>thresholds
        full,th,ca=alarm_bank(s,thresholds,lref)
        normal=i in splits['normal_evaluation']
        fr,ev,nr=records(i,y,s,a,T,D,normal);bf+=fr;be+=ev
        if nr:bn.append(nr)
        fr,ev,nr=records(i,y,full,ca,T,D,normal);cf+=fr;ce+=ev
        if nr:cn.append(nr)
        if not normal:hist.append((y,s))
        tag='_'.join(i.split('/')[1:])
        np.savez_compressed(local/f'{tag}.npz',labels=y,scores=s,stall=stalls[i],theta=theta[i])
        out=root/'runs/E18C'/rel;out.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(out/f'{tag}.npz',labels=y,scores=full,thresholds=th,alarms=ca)
    if hist:
        y=np.concatenate([v[0] for v in hist]);s=np.concatenate([v[1] for v in hist]);valid=y>=0
        for j,arm in enumerate(b_arms):ranking.append({'arm':arm,'auroc':float(roc_auc_score(y[valid],s[valid,j])),'ap':float(average_precision_score(y[valid],s[valid,j]))})
    common={'status':'complete','seed':seed,'scene':scene,'fold':fold,'T':T,'D':D,'W':W,'splits':splits,'config_sha256':fingerprint(cfg),'stall_sha256':sha(trained),'baseline_sha256':sha(root/'results/E11B'/rel/'run.json'),'coverage':stall.coverage,'gate':stall.gate,'k':stall.k,'wall_seconds':time.perf_counter()-began}
    for stage,arms,fr,ev,nr in [('E18B',b_arms,bf,be,bn),('E18C',c_arms,cf,ce,cn)]:
        dest=root/'results'/stage/rel
        for name,val in [('frames',fr),('events',ev),('normal',nr)]:compact(dest/f'{name}.json',val)
        write_json(dest/'run.json',{**common,'arms':arms,'thresholds':thresholds.tolist(),'reference_low':lref.tolist()})
    compact(root/'results/E18B'/rel/'ranking.json',ranking)
    print(f'E18B/C {rel}: {len(evaluated)} evaluations, {time.perf_counter()-began:.1f}s',flush=True)
    if synthetic:run_aux(root,rows,seed,scene,fold,model,stall,thresholds,lref,g,theta,old,scored)

def run_aux(root,rows,seed,scene,fold,model=None,stall=None,thresholds=None,lref=None,g=None,theta=None,old=None,scored=None):
    rel=suffix(seed,scene,fold);r=read(root/'results/E11B'/rel/'run.json');T,D,W=[r[k] for k in ['T','D','W']];ids=r['splits']['normal_evaluation']
    if model is None:
        model=joblib.load(root/'runs/E11'/rel/'model.joblib');fit=joblib.load(root/'runs/E18B'/rel/'stall.joblib')
        stall,thresholds,lref=fit['stall'],fit['thresholds'],fit['reference_low']
        lookup={v['id']:v for v in rows};g={};theta={};old={};scored={}
        for i in ids:
            g[i]=load_rows(root/'runs/stride1_seed42',[lookup[i]])[0]['global'];old[i],theta[i]=baseline(root,rel,i)
            scored[i],_=score_bank(stall,g[i],theta[i],old[i],seed)
    specs={s['id']:s for s in read(root/'configs/experiments/followup/e9_cases_v2.json') if s['eligible']}
    previous=read(root/'results/E11B'/rel/'cases.json')
    archive=np.load(root/'runs/E15BC'/rel/'edited_raw.npz',allow_pickle=True)
    oldwindows=np.load(root/'runs/E11B'/rel/'edited_windows.npz',allow_pickle=True)
    assert archive['ids'].tolist()==[v['id'] for v in previous]==oldwindows['ids'].tolist()
    output=[]
    def evaluate(spec,fullold,fulltheta,globalfeatures,group):
        i=spec['source_id'];o,L=spec['onset'],spec['length']
        s,st=score_bank(stall,globalfeatures,fulltheta,fullold,seed)
        full,th,a=alarm_bank(s,thresholds,lref)
        np.testing.assert_allclose(s[:o],scored[i][:o],atol=1e-9,rtol=1e-9)
        _,_,orig=alarm_bank(scored[i],thresholds,lref)
        length=min(L,D) if spec['edit']=='freeze' else D
        end=min(o+length,len(s));window=a[o:end];legacy=a[o:min(o+D,len(a))]
        hit=window.any(0)
        output.append({'id':spec['id'],'source_id':i,'edit':spec['edit'],'group':group,'onset':o,'length':L,'evaluated_frames':len(window),'hit':hit.astype(int).tolist(),'legacy_D_hit':legacy.any(0).astype(int).tolist(),'matched_hit':orig[o:min(o+length,len(orig))].any(0).astype(int).tolist(),'delay':np.where(hit,window.argmax(0),-1).tolist(),'theta_shift_cycles':float(((fulltheta[min(o+L,len(fulltheta))-1]-fulltheta[o]+.5)%1)-.5) if spec['edit']=='freeze' else None})
    for j,case in enumerate(previous):
        spec=specs[case['id']];i=spec['source_id'];o,L=spec['onset'],spec['length'];n=o+W
        mapping=edit_map(len(g[i]),o,L,spec['edit'])[:n]
        fullold=np.concatenate([old[i][:o],oldwindows['scores'][j]])
        fulltheta=np.r_[theta[i][:o],archive['theta'][j]]
        evaluate(spec,fullold,fulltheta,g[i][mapping],'existing')
    # Early-prefix stress requires replaying the actual frozen tracker, not using
    # a ground-truth onset reset or splicing calibrated base scores.
    lookup={v['id']:v for v in rows}
    for i in ids:
        data=load_rows(root/'runs/stride1_seed42',[lookup[i]])[0];z=descriptor(data)
        x=model['base']['tracker'].reducer.transform(z)/model['base']['tracker'].scale
        for o in [1,8,24,48]:
            L=max(2,halfup(.1*T));n=min(o+W,len(z)+L);mapping=edit_map(len(z),o,L,'freeze')[:n];take=np.arange(0,n,2);src=mapping[take]
            tr=model['base']['tracker'].predict_latent(x[src],take)
            c=conditional(z[src],tr['angle'],model['coef'],model['h'],model['space'],model['ccal']);p=np.maximum.reduce([model['pcal'][k].score(tr[k]) for k in model['pcal']])
            # A is framewise and can be indexed directly, including the Full control.
            a=old[i][src,0];cp=np.maximum(c,p)
            ss=np.column_stack([a,np.maximum(a,c),np.maximum(a,p),np.maximum(a,cp),c,p,cp])
            fullold=hold(take,ss,n);fulltheta=hold(take,tr['angle'],n)
            spec={'id':f'{i}:early_freeze_{o}','source_id':i,'onset':o,'length':L,'edit':'freeze'}
            evaluate(spec,fullold,fulltheta,g[i][mapping],'early_prefix')
    compact(root/'results/E18C'/rel/'auxiliary.json',output)
    print(f'E18 auxiliary {rel}: {len(output)} cases',flush=True)
