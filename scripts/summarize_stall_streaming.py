"""E18D source-level accuracy and paired cadence effects, separate from speed."""
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score,roc_auc_score
from threadpoolctl import threadpool_limits
from cycle_vad.data import write_json
from cycle_vad.paper_experiments import read
from cycle_vad.paper_diagnostics import sha
from bootstrap_followup import weighted_metrics

ROOT=Path(__file__).resolve().parents[1];SCENES=['R01','R02','R03','R04'];ARMS=['C0','C1','B0','B1','B2','B3']

def main():
    metadata=[read(p) for p in sorted((ROOT/'results/E18D/videos').glob('*.json'))]
    assert len(metadata)==66 and sum(m['frames'] for m in metadata)==33462
    assert sum(m['valid_frames'] for m in metadata)==31550 and sum(m['unknown_frames'] for m in metadata)==1912
    arrays={}
    for m in metadata:
        path=ROOT/'runs/E18D'/f'{m["source_id"].replace("/","_")}.npz'
        assert sha(path)==m['scores_sha256'];a=dict(np.load(path));arrays[m['source_id']]=a
        sc,_,seq=m['source_id'].split('/')
        with np.load(ROOT/f'runs/E18C/seed42/{sc}/fold0/testing_{seq}.npz') as cached:
            np.testing.assert_array_equal(a['labels'],cached['labels']);a['cached']=cached['scores'][:,:6].copy();a['cached_alarm']=cached['alarms'][:,3,:6].copy()
        for stride in [1,2]:
            np.testing.assert_array_equal(a[f'alarm{stride}'],a[f'stride{stride}']>a[f'threshold{stride}'])
            if stride==2:np.testing.assert_array_equal(a['stride2'][1::2],a['stride2'][::2][:len(a['stride2'][1::2])])
            for name in ['alarm_parity','prefix_exact','threshold_parity']:assert m['parity'][str(stride)][name]
    byscene={};sample={};rng=np.random.default_rng(20261012)
    for sc in SCENES:
        ms=[m for m in metadata if m['scene']==sc];vs=[arrays[m['source_id']] for m in ms];valid=[v for v in vs if (v['labels']>=0).any()]
        byscene[sc]={};sample[sc]={};y=np.concatenate([v['labels'][v['labels']>=0] for v in valid]);owner=np.concatenate([np.full((v['labels']>=0).sum(),j) for j,v in enumerate(valid)])
        n=len(valid);counts=np.column_stack([np.ones(n,dtype=int),rng.multinomial(n,np.full(n,1/n),size=2000).T])
        for mode,alarmkey in [('cached','cached_alarm'),('stride1','alarm1'),('stride2','alarm2')]:
            scores=np.concatenate([v[mode][v['labels']>=0] for v in valid]);alarms=np.concatenate([v[alarmkey][v['labels']>=0] for v in valid]);byscene[sc][mode]={};sample[sc][mode]={}
            for j,arm in enumerate(ARMS):
                ap,auc=weighted_metrics(y,scores[:,j],owner,counts)
                np.testing.assert_allclose([ap[0],auc[0]],[average_precision_score(y,scores[:,j]),roc_auc_score(y,scores[:,j])],atol=1e-12)
                if mode=='cached':
                    es=read(ROOT/f'results/E18C/seed42/{sc}/fold0/events.json');hit=sum(e['hit'][3][j] for e in es);total=len(es)
                else:
                    es=sum([m['events'][mode[-1]] for m in ms],[]);hit=sum(e['hit'][0][j] for e in es);total=len(es)
                byscene[sc][mode][arm]={'ap':float(ap[0]),'auroc':float(auc[0]),'fpr':float(alarms[y==0,j].mean()),'recall':float(alarms[y==1,j].mean()),'event_hits':hit,'event_total':total,'event_recall':hit/total}
                sample[sc][mode][arm]={'ap':ap[1:],'auroc':auc[1:]}
        print(f'E18D summary {sc}',flush=True)
    macro={mode:{arm:{m:float(np.mean([byscene[sc][mode][arm][m] for sc in SCENES])) for m in ['ap','auroc','fpr','recall','event_recall']} for arm in ARMS} for mode in ['cached','stride1','stride2']}
    for mode in macro:
        for arm in ARMS:macro[mode][arm]['event_hits']=sum(byscene[sc][mode][arm]['event_hits'] for sc in SCENES)
    intervals={}
    for hi,lo in [('stride1','stride2'),('stride2','cached')]:
        name=f'{hi}-{lo}';intervals[name]={}
        for arm in ARMS:
            intervals[name][arm]={}
            for metric in ['ap','auroc']:
                d=100*np.mean([sample[sc][hi][arm][metric]-sample[sc][lo][arm][metric] for sc in SCENES],axis=0);good=np.isfinite(d)
                intervals[name][arm][metric]={'delta_pp':100*(macro[hi][arm][metric]-macro[lo][arm][metric]),'ci95_pp':np.quantile(d[good],[.025,.975]).tolist(),'valid_resamples':int(good.sum())}
    maxerr={k:max(m['parity'][s][k] for m in metadata for s in ['1','2']) for k in ['max_trace_error','max_C_error','max_P_error','stall_max_error']}
    out={'status':'complete','arms':ARMS,'seed':42,'fold':0,'videos':66,'source_frames':33462,'valid_frames':31550,'unknown_frames':1912,'execution':'fresh GPU image features; two independent causal states sharing only current encoding; not a speed benchmark','scene':byscene,'macro':macro,'paired_bootstrap':intervals,'resamples':2000,'parity_max_errors':maxerr,'all_video_causal_parity_passed':True}
    write_json(ROOT/'results/E18D/accuracy.json',out)

if __name__=='__main__':
    with threadpool_limits(limits=4):main()
