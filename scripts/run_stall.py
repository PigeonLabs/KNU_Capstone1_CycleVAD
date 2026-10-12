import argparse
from pathlib import Path
from threadpoolctl import threadpool_limits
from cycle_vad.data import inventory
from cycle_vad.paper_experiments import read
from cycle_vad.paper_calibration import suffix
from cycle_vad.stall_experiment import run_unit, run_aux

ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seeds',type=int,nargs='+',default=[42,43,44]);p.add_argument('--folds',type=int,nargs='+',default=list(range(5)))
    p.add_argument('--scenes',nargs='+',default=['R01','R02','R03','R04']);p.add_argument('--resume',action='store_true');p.add_argument('--auxiliary',action='store_true');a=p.parse_args()
    assert set(a.scenes)<=set(['R01','R02','R03','R04'])
    assert read(ROOT/'results/E18A/summary.json')['status']=='complete'
    rows=inventory(ROOT.parent/'IPAD_dataset',a.scenes)
    with threadpool_limits(limits=4):
        for seed in a.seeds:
            for fold in a.folds:
                for sc in a.scenes:
                    dest=ROOT/'results/E18C'/suffix(seed,sc,fold)
                    if a.auxiliary:
                        if not (a.resume and (dest/'auxiliary.json').exists()):run_aux(ROOT,rows,seed,sc,fold)
                    elif not (a.resume and (dest/'run.json').exists()):run_unit(ROOT,rows,seed,sc,fold)
