"""Source-paired E18 actual-test statistics; all fixed candidates are retained."""
from pathlib import Path
import json
import numpy as np
from threadpoolctl import threadpool_limits
from cycle_vad.data import write_json
from cycle_vad.paper_experiments import read
from cycle_vad.paper_calibration import suffix
from cycle_vad.stall_experiment import ARMS, EXTRA
from bootstrap_followup import weighted_metrics

ROOT=Path(__file__).resolve().parents[1];SCENES=['R01','R02','R03','R04'];Q=3

def ratio(a,b):return float(a/b) if b else None

def main():
    output={};arrays42={};events42={}
    for seed in [42,43,44]:
        arms=ARMS+(EXTRA if seed==42 else []);scene={};normal={};aux={};ledgers=[]
        for sc in SCENES:
            rel=suffix(seed,sc,0);meta=read(ROOT/'results/E18C'/rel/'run.json');assert meta['arms']==arms
            data={p.stem:dict(np.load(p)) for p in sorted((ROOT/'runs/E18C'/rel).glob('testing_*.npz'))}
            y=np.concatenate([d['labels'] for d in data.values()]);s=np.concatenate([d['scores'] for d in data.values()]);a=np.concatenate([d['alarms'] for d in data.values()]);valid=y>=0
            events=read(ROOT/'results/E18C'/rel/'events.json');hits=np.array([e['hit'] for e in events]);scene[sc]={}
            # Unknown files are still processed, but never treated as negatives.
            assert sum(len(d['labels']) for d in data.values())==len(y)
            np.testing.assert_array_equal(s[:,2],s[:,4]);np.testing.assert_array_equal(s[:,3],s[:,5])
            from sklearn.metrics import roc_auc_score, average_precision_score
            for j,arm in enumerate(arms):
                normalmask=y==0;anomaly=y==1
                delays=[e['delay'][Q][j] for e in events if e['hit'][Q][j]]
                scene[sc][arm]={'auroc':float(roc_auc_score(y[valid],s[valid,j])),'ap':float(average_precision_score(y[valid],s[valid,j])),
                    'fpr':a[normalmask,Q,j].mean().item(),'recall':a[anomaly,Q,j].mean().item(),'event_recall':hits[:,Q,j].mean().item(),'event_hits':int(hits[:,Q,j].sum()),'event_total':len(events),
                    'normal_frames':int(normalmask.sum()),'anomaly_frames':int(anomaly.sum()),'detected_delay_median_frames':float(np.median(delays)) if delays else None,
                    'quantiles':[{'fpr':float(a[normalmask,q,j].mean()),'recall':float(a[anomaly,q,j].mean()),'event_recall':float(hits[:,q,j].mean())} for q in range(5)]}
            old=read(ROOT/f'results/E11B/seed{seed}/historical_summary.json')['by_scene'][sc]
            for arm,col in [('C0',4),('B0',6),('Full',3),('P',5)]:
                for metric in ['auroc','ap','fpr','recall']:np.testing.assert_allclose(scene[sc][arm][metric],old[col][metric],atol=1e-12)
            for hi,lo in [('B1','B0'),('B2','B0'),('B3','B0'),('B3','B1'),('C1','C0'),('B3','C0')]:
                hj,lj=arms.index(hi),arms.index(lo)
                for e in events:
                    new,base=e['hit'][Q][hj],e['hit'][Q][lj]
                    ledgers.append({'contrast':f'{hi}-{lo}','scene':sc,'event_id':e['id'],'source_id':e['source_id'],'onset':e['onset'],'end':e['end'],'outcome':('retained' if base else 'added') if new else ('lost' if base else 'missed'),'base_delay':e['delay'][Q][lj],'new_delay':e['delay'][Q][hj]})
            normals=sum([read(ROOT/'results/E18C'/suffix(seed,sc,f)/'normal.json') for f in range(5)],[])
            assert len(normals)==len(set(n['id'] for n in normals))
            normal[sc]={arm:{'grid_far':float(np.mean([n['grid_far'][Q][j] for n in normals])),
                             'frame_fpr':float(sum(n['frames']*n['frame_fpr'][Q][j] for n in normals)/sum(n['frames'] for n in normals)),
                             'early_frame_fpr':float(np.mean([n['early_frame_fpr'][Q][j] for n in normals]))} for j,arm in enumerate(arms)}
            files=[ROOT/'results/E18C'/suffix(seed,sc,f)/'auxiliary.json' for f in range(5)]
            if all(p.exists() for p in files):
                cases=sum([read(p) for p in files],[]);aux[sc]={}
                for group in ['existing','early_prefix']:
                    aux[sc][group]={}
                    for edit in sorted(set(c['edit'] for c in cases if c['group']==group)):
                        cs=[c for c in cases if c['group']==group and c['edit']==edit]
                        aux[sc][group][edit]={arm:{'n':len(cs),'hit':float(np.mean([c['hit'][Q][j] for c in cs])),
                           'legacy_D_hit':float(np.mean([c['legacy_D_hit'][Q][j] for c in cs])),
                           'matched_far':float(np.mean([c['matched_hit'][Q][j] for c in cs]))} for j,arm in enumerate(arms)}
            if seed==42:arrays42[sc]=data;events42[sc]=events
        metrics=['auroc','ap','fpr','recall','event_recall']
        macro={arm:{m:float(np.mean([scene[sc][arm][m] for sc in SCENES])) for m in metrics} for arm in arms}
        for arm in arms:
            macro[arm]['event_hits']=sum(scene[sc][arm]['event_hits'] for sc in SCENES)
            macro[arm]['event_total']=sum(scene[sc][arm]['event_total'] for sc in SCENES)
            macro[arm]['normal_grid_far']=float(np.mean([normal[sc][arm]['grid_far'] for sc in SCENES]))
        contrasts={}
        for name in sorted(set(v['contrast'] for v in ledgers)):
            counts={k:sum(v['contrast']==name and v['outcome']==k for v in ledgers) for k in ['retained','added','lost','missed']}
            contrasts[name]=counts
        output[str(seed)]={'arms':arms,'scene':scene,'macro':macro,'normal_oof':normal,'auxiliary':aux,'event_changes':contrasts,
           'leave_one_scene_out':{omit:{arm:{m:float(np.mean([scene[sc][arm][m] for sc in SCENES if sc!=omit])) for m in metrics} for arm in arms} for omit in SCENES}}
        write_json(ROOT/f'results/E18C/seed{seed}/event_ledger.json',ledgers)
    arms=output['42']['arms'];rng=np.random.default_rng(20261012);samples={}
    for sc in SCENES:
        data={i:d for i,d in arrays42[sc].items() if (d['labels']>=0).any()};keys=list(data);vs=list(data.values());n=len(vs)
        counts=np.column_stack([np.ones(n,dtype=int),rng.multinomial(n,np.full(n,1/n),size=2000).T])
        y=np.concatenate([v['labels'][v['labels']>=0] for v in vs]);s=np.concatenate([v['scores'][v['labels']>=0] for v in vs]);owner=np.concatenate([np.full((v['labels']>=0).sum(),i) for i,v in enumerate(vs)])
        normalcounts=np.array([(v['labels']==0).sum() for v in vs]);anomalycounts=np.array([(v['labels']==1).sum() for v in vs])
        eventcounts=np.array([sum(e['source_id'].endswith('/'+i.split('_')[1]) for e in events42[sc]) for i in keys])
        samples[sc]={};rankcache={}
        for j,arm in enumerate(arms):
            key=s[:,j].tobytes()
            if key not in rankcache:rankcache[key]=weighted_metrics(y,s[:,j],owner,counts)
            ap,auc=rankcache[key]
            def bootratio(num,den):
                a=np.asarray(num)@counts;b=np.asarray(den)@counts
                return np.divide(a,b,out=np.full(a.shape,np.nan),where=b>0)[1:]
            hits=np.array([sum(e['hit'][Q][j] for e in events42[sc] if e['source_id'].endswith('/'+i.split('_')[1])) for i in keys])
            fpr=bootratio([v['alarms'][v['labels']==0,Q,j].sum() for v in vs],normalcounts)
            recall=bootratio([v['alarms'][v['labels']==1,Q,j].sum() for v in vs],anomalycounts)
            er=bootratio(hits,eventcounts)
            samples[sc][arm]={'ap':ap[1:],'auroc':auc[1:],'fpr':fpr,'recall':recall,'event_recall':er}
            np.testing.assert_allclose([ap[0],auc[0]],[output['42']['scene'][sc][arm]['ap'],output['42']['scene'][sc][arm]['auroc']],atol=1e-12)
        print(f'E18 bootstrap {sc} complete',flush=True)
    pairs=[('B1','B0'),('B2','B0'),('B3','B0'),('B3','B1'),('C1','C0'),('B3','C0'),('B3','B1_T110'),('B3','B1_T130'),('B2','B0_T110'),('B2','B0_T130'),('B1','CP_S1'),('B1','CP_S0'),('B1','CP_S2')]
    intervals={}
    for hi,lo in pairs:
        intervals[f'{hi}-{lo}']={}
        for omit in [None,*SCENES]:
            keep=[sc for sc in SCENES if sc!=omit];group='macro' if omit is None else 'without_'+omit;intervals[f'{hi}-{lo}'][group]={}
            for m in ['ap','auroc','fpr','recall','event_recall']:
                delta=100*np.mean([samples[sc][hi][m]-samples[sc][lo][m] for sc in keep],axis=0);good=np.isfinite(delta)
                point=100*np.mean([output['42']['scene'][sc][hi][m]-output['42']['scene'][sc][lo][m] for sc in keep])
                intervals[f'{hi}-{lo}'][group][m]={'delta_pp':float(point),'ci95_pp':np.quantile(delta[good],[.025,.975]).tolist(),'valid_resamples':int(good.sum())}
    summary={'status':'complete','primary':'actual R01-R04 test; all fixed arms; seed42 paired intervals; seeds43/44 robustness, not extra samples','test_sources':66,'valid_test_sources':63,'valid_test_frames':31550,'seeds':output,'bootstrap':{'resamples':2000,'unit':'source video','stratified_by':'scene','conditional_on':'frozen models; recording independence unverified; pointwise exploratory intervals','contrasts':intervals}}
    write_json(ROOT/'results/E18C/summary.json',summary)
    write_json(ROOT/'results/E18B/summary.json',{'status':'complete','source':'../E18C/summary.json','arms':['C0','C1','B0','B1','Full','P',*EXTRA],'seeds':{seed:{'macro':{k:v for k,v in s['macro'].items() if k in ['C0','C1','B0','B1','Full','P',*EXTRA]}} for seed,s in output.items()},'bootstrap':{k:v for k,v in intervals.items() if k in ['B1-B0','C1-C0','B1-CP_S1','B1-CP_S0','B1-CP_S2']}})

if __name__=='__main__':
    with threadpool_limits(limits=4):main()
