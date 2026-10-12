"""E18A: verify frozen baselines, source roles, and CUDA feature provenance."""
from pathlib import Path
import json, time
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from cycle_vad.data import inventory, labels_for, write_json, fingerprint
from cycle_vad.paper_experiments import calibrate
from cycle_vad.paper_calibration import branches, load_bank, suffix
from cycle_vad.paper_diagnostics import sha

ROOT = Path(__file__).resolve().parents[1]
def read(p): return json.loads(p.read_text())

def main():
    began=time.perf_counter(); cfg=read(ROOT/'configs/experiments/followup/e18_v1.json')
    rows=inventory(ROOT.parent/'IPAD_dataset',cfg['scenes']); lookup={r['id']:r for r in rows}
    folds=read(ROOT/'configs/experiments/followup/normal_folds.json')['scenes']
    caches={}; units=[]
    for row in rows:
        p=ROOT/'runs/stride1_seed42/cache'/f'{row["id"]}.npz'
        with np.load(p) as a:
            identity=json.loads(str(a['encoder_identity']))
            assert identity['device']=='cuda'
            assert identity['resolved_revision']=='f9e44c814b77203eaa57a6bdbbd535f21ede1415'
            np.testing.assert_array_equal(a['indices'],np.arange(row['frames']))
            caches[row['id']]={'signature':str(a['signature']),'frames':row['frames'],'encoder':identity}
    for sc in cfg['scenes']:
        oo=sum([f['splits']['normal_evaluation'] for f in folds[sc]],[])
        expected=[r['id'] for r in rows if r['scene']==sc and r['partition']=='training']
        assert sorted(oo)==sorted(expected) and len(oo)==len(set(oo))
    with threadpool_limits(limits=4):
        for seed in cfg['seeds']:
            for sc in cfg['scenes']:
                for fold in cfg['folds']:
                    rel=suffix(seed,sc,fold); r=read(ROOT/'results/E11B'/rel/'run.json'); splits=r['splits']
                    assert splits==folds[sc][fold]['splits']
                    ids=sum(splits.values(),[]); assert len(ids)==len(set(ids)) and all('/training/' in i for i in ids)
                    modelpath=ROOT/'runs/E11'/rel/'model.joblib'; m=joblib.load(modelpath)
                    old=load_bank(ROOT,rel); raw=load_bank(ROOT,rel,True)
                    scoreerr=0.; alarmer=0
                    for i,d in old.items():
                        np.testing.assert_array_equal(d['labels'],labels_for(lookup[i]))
                        rep=branches(raw[i]['raw'][:,3:],m['ccal']+[m['pcal'][k] for k in ['alignment','innovation','progress']])
                        target=d['scores'][:,[4,5]]
                        np.testing.assert_allclose(rep,target,atol=1e-9,rtol=1e-9)
                        scoreerr=max(scoreerr,float(abs(rep-target).max()))
                        th=np.asarray(r['window_thresholds'])[3,[4,6]]
                        rep=np.column_stack([rep[:,0],rep.max(1)])
                        target=d['scores'][:,[4,6]]
                        alarmer+=int(np.count_nonzero((rep>th)!=(target>th)))
                    assert alarmer==0
                    th,_,counts=calibrate({i:d['scores'] for i,d in old.items()},splits,r['T'],r['D'])
                    np.testing.assert_array_equal(th,r['window_thresholds'])
                    units.append({'seed':seed,'scene':sc,'fold':fold,'score_max_error':scoreerr,'alarm_mismatches':alarmer,'threshold_exact':True,'threshold_windows':counts,'checkpoint_sha256':sha(modelpath),'baseline_sha256':sha(ROOT/'results/E11B'/rel/'run.json')})
                    print(f'E18A {rel}: baseline and split passed',flush=True)
    test=[r for r in rows if r['partition']=='testing']
    result={'status':'complete','config_sha256':fingerprint(cfg),'units':units,'feature_caches':caches,'normal_sources':len(rows)-len(test),'test_sources':len(test),'test_frames':sum(r['frames'] for r in test),'valid_test_frames':sum(int((labels_for(r)>=0).sum()) for r in test),'excluded_test_sources':[r['id'] for r in test if np.all(labels_for(r)<0)],'wall_seconds':time.perf_counter()-began}
    write_json(ROOT/'results/E18A/summary.json',result)

if __name__=='__main__':main()
