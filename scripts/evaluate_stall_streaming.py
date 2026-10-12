"""E18D: fresh batch-one GPU replay and separately measured JPEG-to-alarm speed."""
import argparse,time,platform
from pathlib import Path
import joblib
import numpy as np
from PIL import Image
import torch
from threadpoolctl import threadpool_limits
from benchmark_streaming import ROOT,read,encode,check_sequence,stats
from cycle_vad.features import VisualEncoder
from cycle_vad.streaming import StreamingDetector
from cycle_vad.stall import StreamingStallDetector,adaptive_thresholds,low_stat
from cycle_vad.data import inventory,frame_paths,labels_for,write_json,hold
from cycle_vad.paper_diagnostics import sha
from cycle_vad.stall_experiment import records
from cycle_vad.metrics import frame_metrics

OUT=ROOT/'results/E18D';LOCAL=ROOT/'runs/E18D';ARMS=StreamingStallDetector.arms

def replay(encoder,models,fitted,rows,resume):
    for number,row in enumerate(rows,1):
        tag=row['id'].replace('/','_');dest=OUT/'videos'/f'{tag}.json';local=LOCAL/f'{tag}.npz'
        if resume and dest.exists() and local.exists():continue
        sc=row['scene'];detectors={s:StreamingStallDetector(models[sc],fitted[sc],s) for s in [1,2]}
        saved={s:[] for s in [1,2]};fresh={s:[] for s in [1,2]};gs={s:[] for s in [1,2]};began=time.perf_counter()
        for index,path in enumerate(frame_paths(row['directory'])):
            with Image.open(path) as f:im=f.convert('RGB')
            g,p,_=encode(encoder,im);im.close()
            for stride,det in detectors.items():
                out=det.step(g,p) if index%stride==0 else det.step()
                saved[stride].append(out)
                if out['fresh']:fresh[stride].append(out['base']);gs[stride].append(np.asarray(g).astype(np.float16).astype(np.float32))
        elapsed=time.perf_counter()-began;parity={};arrays={};metrics={};events={};counts={}
        y=labels_for(row);meta=read(ROOT/f'results/E18B/seed42/{sc}/fold0/run.json')
        for stride in [1,2]:
            pred=saved[stride];scores=np.array([x['scores'] for x in pred]);th=np.array([x['thresholds'] for x in pred]);alarm=np.array([x['alarms'] for x in pred])
            basecheck=check_sequence(fresh[stride],models[sc],stride)
            theta=np.array([v['theta'] for v in fresh[stride]])
            ref=fitted[sc]['stall'].score(np.array(gs[stride]),theta,['S3'],stride)
            actual=np.array([v['stall']['score'] for v in pred if v['fresh']])
            np.testing.assert_allclose(actual,ref['scores'][:,0],atol=1e-9,rtol=1e-9)
            # Truncation and a future-feature perturbation cannot change the prefix.
            cut=max(1,len(theta)//2);refpre=fitted[sc]['stall'].score(np.array(gs[stride])[:cut],theta[:cut],['S3'],stride)
            np.testing.assert_array_equal(refpre['scores'],ref['scores'][:cut])
            fixed=fitted[sc]['thresholds'][3,[0,1,2,3,2,3]];expected=np.broadcast_to(fixed,th.shape).copy()
            if len(scores)>48:
                delta=low_stat(scores[:48:stride])-fitted[sc]['reference_low'][[0,1,2,3,2,3]]
                expected[48:,4:]=fixed[4:]+.3*np.clip(delta[4:],0,fixed[4:])
            np.testing.assert_allclose(th,expected,atol=1e-12)
            np.testing.assert_array_equal(alarm,scores>expected)
            np.testing.assert_array_equal(scores[:,2],scores[:,4]);np.testing.assert_array_equal(scores[:,3],scores[:,5])
            fr,ev,_=records(row['id'],y,scores,alarm[:,None,:],meta['T'],meta['D'])
            events[str(stride)]=ev;counts[str(stride)]=fr
            metrics[str(stride)]={}
            for j,arm in enumerate(ARMS):
                metric=frame_metrics(y,scores[:,j],float(th[0,j]))
                metric.update(fpr=float(alarm[y==0,j].mean()) if (y==0).any() else None,
                              recall=float(alarm[y==1,j].mean()) if (y==1).any() else None,
                              threshold='causal per-frame threshold' if j in [4,5] else float(th[0,j]))
                metrics[str(stride)][arm]=metric
            arrays.update({f'stride{stride}':scores,f'threshold{stride}':th,f'alarm{stride}':alarm})
            parity[str(stride)]={**basecheck,'stall_max_error':float(abs(actual-ref['scores'][:,0]).max()),'prefix_exact':True,'threshold_parity':True,'alarm_parity':True}
        LOCAL.mkdir(parents=True,exist_ok=True);np.savez_compressed(local,labels=y,**arrays)
        write_json(dest,{'source_id':row['id'],'scene':sc,'frames':len(y),'valid_frames':int((y>=0).sum()),'unknown_frames':int((y<0).sum()),'joint_replay_seconds':elapsed,'parity':parity,'metrics':metrics,'events':events,'frame_counts':counts,'scores_sha256':sha(local)})
        print(f'E18D {number}/{len(rows)} {row["id"]}: {elapsed:.1f}s, both strides parity passed',flush=True)

def benchmark(encoder,models,fitted,rows,resume):
    selected=[]
    for sc in models:selected += [r for r in rows if r['scene']==sc][:4]
    jobs=[(arm,s,repeat,None) for arm in ['B0','B3'] for s in [1,2] for repeat in range(3)]
    jobs += [('B3',s,0,hz) for s in [1,2] for hz in [30,60]]
    for arm,stride,repeat,hz in jobs:
        name=f'{arm}_s{stride}_r{repeat}'+(f'_paced{hz}' if hz else '')
        dest=OUT/'benchmarks'/f'{name}.json'
        if resume and dest.exists():continue
        observations=[];wall=0.;sources=[];torch.cuda.reset_peak_memory_stats()
        # Paced cases use one file per scene, 512 total frames, like E16.
        chosen=selected if hz is None else [next(r for r in selected if r['scene']==sc) for sc in models]
        for row in chosen:
            sc=row['scene'];paths=frame_paths(row['directory'])[:128];sources.append(row['id']);start=time.perf_counter()
            det=None
            for j,path in enumerate(paths):
                due=start+j/hz if hz else None
                if due is not None and due>time.perf_counter():time.sleep(due-time.perf_counter())
                began=time.perf_counter()
                if det is None:det=StreamingStallDetector(models[sc],fitted[sc],stride) if arm=='B3' else StreamingDetector(models[sc],fitted[sc]['thresholds'][3,2],stride,'CP')
                with Image.open(path) as f:im=f.convert('RGB')
                fresh=j%stride==0
                if fresh:g,p,_=encode(encoder,im)
                h=time.perf_counter();out=det.step(g,p) if fresh else det.step();done=time.perf_counter();im.close()
                observations.append({'fresh':fresh,'service_ms':1000*(done-began),'head_ms':1000*(done-h),'arrival_ms':1000*(done-due) if due else None,'missed_period':bool(due is not None and done-due>1/hz)})
            wall+=time.perf_counter()-start
        fresh=[v for v in observations if v['fresh']]
        result={'id':name,'arm':arm,'stride':stride,'repeat':repeat,'paced_fps':hz,'source_ids':sources,'input_frames':len(observations),'fresh_updates':len(fresh),'wall_seconds':wall,
          'input_fps':len(observations)/wall,'fresh_updates_per_second':len(fresh)/wall,'service_ms':stats([v['service_ms'] for v in observations]),'fresh_service_ms':stats([v['service_ms'] for v in fresh]),'head_ms':stats([v['head_ms'] for v in fresh]),
          'arrival_ms':stats([v['arrival_ms'] for v in observations]) if hz else None,'missed_periods':sum(v['missed_period'] for v in observations),'peak_allocated_mb':torch.cuda.max_memory_allocated()/1024**2,'scope':'JPEG decode + preprocess + fresh CUDA encoder + causal score/alarm; no camera/codec/network/display; no prefetch; B3 adapter computes six shared readouts'}
        write_json(dest,result);print(f'E18D speed {name}: {result["input_fps"]:.1f} input FPS, p95 {result["fresh_service_ms"]["p95"]:.2f} ms',flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--benchmark',action='store_true');p.add_argument('--resume',action='store_true');args=p.parse_args()
    torch.set_num_threads(4);torch.set_num_interop_threads(1);assert torch.cuda.is_available()
    config=read(ROOT/'configs/experiments/followup/e18_v1.json');rows=inventory(ROOT.parent/'IPAD_dataset',config['scenes']);tests=[r for r in rows if r['partition']=='testing']
    with threadpool_limits(limits=4):
        encoder=VisualEncoder(read(ROOT/'configs/experiments/replay_pinned.json')['encoder'])
        models={sc:joblib.load(ROOT/f'runs/E11/seed42/{sc}/fold0/model.joblib') for sc in config['scenes']}
        fitted={sc:joblib.load(ROOT/f'runs/E18B/seed42/{sc}/fold0/stall.joblib') for sc in config['scenes']}
        # Explicit FIT-only warmup, excluded from timing and evaluation.
        warm=read(ROOT/'results/E18B/seed42/R01/fold0/run.json')['splits']['fit'][0];row=next(r for r in rows if r['id']==warm)
        for path in frame_paths(row['directory'])[:100]:
            with Image.open(path) as f:im=f.convert('RGB')
            encode(encoder,im);im.close()
        write_json(OUT/('benchmark_environment.json' if args.benchmark else 'environment.json'),{'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,'cuda':torch.version.cuda,'python':platform.python_version(),'encoder_identity':encoder.identity,'runner_sha256':sha(Path(__file__)),'stall_sha256':sha(ROOT/'src/cycle_vad/stall.py'),'fit_warmup_source':warm,'checkpoint_sha256':{sc:sha(ROOT/f'runs/E11/seed42/{sc}/fold0/model.joblib') for sc in models},'stall_checkpoint_sha256':{sc:sha(ROOT/f'runs/E18B/seed42/{sc}/fold0/stall.joblib') for sc in models}})
        if args.benchmark:benchmark(encoder,models,fitted,tests,args.resume)
        else:replay(encoder,models,fitted,tests,args.resume)

if __name__=='__main__':main()
