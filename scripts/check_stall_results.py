"""Independent E18 counts, causal policy invariants, and artifact coverage."""
from pathlib import Path
import re
import numpy as np
from cycle_vad.paper_experiments import read
from cycle_vad.paper_calibration import suffix
from cycle_vad.data import write_json
ROOT=Path(__file__).resolve().parents[1]

def main():
    summary=read(ROOT/'results/E18C/summary.json');units=0;cases=0;checked=0;greatest=0.
    for seed in [42,43,44]:
        ncase=0;hits=np.zeros(len(summary['seeds'][str(seed)]['arms']),int)
        for sc in ['R01','R02','R03','R04']:
            for fold in range(5):
                rel=suffix(seed,sc,fold);r=read(ROOT/'results/E18C'/rel/'run.json');fitted=read(ROOT/'results/E18B'/rel/'fitted.json')
                assert r['config_sha256']==read(ROOT/'results/E18A/summary.json')['config_sha256']
                assert r['stall_sha256']==fitted['stall_sha256']
                events=read(ROOT/'results/E18C'/rel/'events.json');by={e['id']:e for e in events}
                for path in (ROOT/'runs/E18C'/rel).glob('*.npz'):
                    d=dict(np.load(path));s,th,y=d['scores'],d['thresholds'],d['labels'];a=d['alarms'];checked+=1
                    assert np.isfinite(s).all() and np.isfinite(th).all()
                    np.testing.assert_array_equal(a,s[:,None,:]>th)
                    np.testing.assert_array_equal(s[:,2],s[:,4]);np.testing.assert_array_equal(s[:,3],s[:,5])
                    assert not np.any(a[:,:,4]&~a[:,:,2]) and not np.any(a[:,:,5]&~a[:,:,3])
                    np.testing.assert_array_equal(s[1::2],s[::2][:len(s[1::2])])
                    np.testing.assert_array_equal(th[:48],np.broadcast_to(th[0],th[:48].shape))
                    if len(s)>48:
                        np.testing.assert_array_equal(th[48:],np.broadcast_to(th[48],th[48:].shape))
                    if path.stem.startswith('testing'):
                        identity=f'{sc}/testing/{path.stem.split("_")[1]}'
                        # Independent contiguous interval parser, not event_records.
                        intervals=[];start=None
                        for t,lab in enumerate(np.r_[y,-1]):
                            if lab==1 and start is None:start=t
                            if lab!=1 and start is not None:intervals.append((start,t));start=None
                        for j,(lo,hi) in enumerate(intervals):
                            e=by[f'{identity}:event{j}'];window=a[lo:min(hi,lo+r['D'])]
                            actual=np.zeros(window.shape[1:],bool);delay=np.full(window.shape[1:],-1)
                            for t,alarm in enumerate(window):
                                delay[(~actual)&alarm]=t;actual|=alarm
                            np.testing.assert_array_equal(actual,e['hit']);np.testing.assert_array_equal(delay,e['delay'])
                            hits+=actual[3]
                aux=read(ROOT/'results/E18C'/rel/'auxiliary.json');ncase+=len(aux)
                for c in aux:
                    assert c['source_id'] in r['splits']['normal_evaluation']
                    if c['edit']=='freeze':assert c['evaluated_frames']==min(c['length'],r['D'])
                    h=np.asarray(c['hit']);assert not np.any(h[:,4]&~h[:,2]) and not np.any(h[:,5]&~h[:,3])
                units+=1
        assert ncase==3910+111*4;cases+=ncase
        np.testing.assert_array_equal(hits,[v['event_hits'] for v in summary['seeds'][str(seed)]['macro'].values()])
    # Ensure the older E0-E17 result region survives and every published plot exists.
    text=(ROOT/'README.md').read_text()
    assert text.count('<!-- BEGIN EXPERIMENT RESULTS -->')==1
    for anchor in ['result-e0','result-e11b','result-e15b1','result-e16','result-e17','result-e18a','result-e18b','result-e18c']:assert f'id="{anchor}"' in text
    figs=re.findall(r'!\[[^\]]*\]\((docs/figures/[^)]+)\)',text)
    assert all((ROOT/p).exists() for p in figs)
    result={'status':'passed','model_units':units,'evaluated_source_arrays':checked,'auxiliary_cases_across_seeds':cases,'checks':['frozen config and checkpoint linkage','independent actual event hit/delay counts','strict comparison to per-frame thresholds','held stride2 outputs','no adaptation before frame48','adaptive alarms subset of fixed alarms','AUROC/AP score equality for threshold-only policies','OOF-only auxiliary source membership','freeze interval clipping','baseline E0-E17 README anchors preserved','all README figure paths exist'],'all_readme_figure_references':len(figs)}
    write_json(ROOT/'results/E18C/validation.json',result);print(result)

if __name__=='__main__':main()
