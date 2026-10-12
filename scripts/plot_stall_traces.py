"""Fixed first eligible freeze per scene: mechanism illustration, not selection."""
from pathlib import Path
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from cycle_vad.paper_experiments import read
from cycle_vad.data import write_json
from cycle_vad.process_experiment import edit_map
from cycle_vad.stall_experiment import baseline,score_bank,alarm_bank

ROOT=Path(__file__).resolve().parents[1]
def main():
    manifest={c['id']:c for c in read(ROOT/'configs/experiments/followup/e9_cases_v2.json') if c['eligible']}
    fig,axes=plt.subplots(4,2,figsize=(12,12),layout='constrained');selected=[]
    plt.rcParams['svg.fonttype']='none'
    for row,sc in enumerate(['R01','R02','R03','R04']):
        rel=Path('seed42')/sc/'fold0';cases=read(ROOT/'results/E11B'/rel/'cases.json')
        j=next(j for j,c in enumerate(cases) if c['edit']=='freeze');case=manifest[cases[j]['id']];selected.append(case['id'])
        i=case['source_id'];o,L,W=case['onset'],case['length'],case['window'];n=o+W
        old,theta=baseline(ROOT,rel,i);scores=np.load(ROOT/'runs/E11B'/rel/'edited_windows.npz',allow_pickle=True)['scores'][j]
        tr=np.load(ROOT/'runs/E15BC'/rel/'edited_raw.npz',allow_pickle=True)['theta'][j]
        with np.load(ROOT/'runs/stride1_seed42/cache'/f'{i}.npz') as d:g=d['global'].copy()
        fitted=joblib.load(ROOT/'runs/E18B'/rel/'stall.joblib');fullold=np.concatenate([old[:o],scores]);fulltheta=np.r_[theta[:o],tr]
        mapping=edit_map(len(g),o,L,'freeze')[:n];s,st=score_bank(fitted['stall'],g[mapping],fulltheta,fullold,42)
        _,threshold,_=alarm_bank(s,fitted['thresholds'],fitted['reference_low'])
        lo=max(0,o-32);xx=np.arange(lo,n)-o
        a,b=axes[row];a.plot(xx,np.unwrap(2*np.pi*fulltheta)[lo:]/(2*np.pi),label='Edited posterior theta')
        a.plot(xx,np.unwrap(2*np.pi*theta)[lo:n]/(2*np.pi),label='Unedited posterior theta',ls='--')
        b.plot(xx,s[lo:,2],label='CP');b.plot(xx,st[lo:,3],label='Stall S3')
        b.plot(xx,threshold[lo:,3,3],label='Fixed joint threshold',ls=':');b.plot(xx,threshold[lo:,3,5],label='Adaptive joint threshold',ls='--')
        for ax in [a,b]:ax.axvspan(0,L,color='#D8B277',alpha=.2);ax.grid(alpha=.15);ax.set_xlabel('Source frames relative to edit onset')
        a.set_ylabel('Unwrapped latent cycles');b.set_ylabel('Calibrated score');a.set_title(f'{i} | freeze {L} frames');b.set_title('Shaded interval: repeated source frame')
        if row==0:a.legend(fontsize=8);b.legend(fontsize=8)
    fig.suptitle('E18 auxiliary mechanism check | First eligible freeze per scene\nFixed seed 42 / fold 0; no selection by score or detection outcome',fontsize=13)
    for ext in ['svg','png']:fig.savefig(ROOT/f'docs/figures/E18_stall_traces.{ext}',dpi=140,metadata={'Date':None} if ext=='svg' else None)
    write_json(ROOT/'results/E18C/trace_examples.json',{'selection':'first eligible existing freeze in original manifest order, per scene, seed42 fold0; fixed illustrative cases','case_ids':selected})

if __name__=='__main__':main()
