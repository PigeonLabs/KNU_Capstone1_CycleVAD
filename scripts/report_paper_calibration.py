"""Publish E15 calibration experiments with inspectable tables and figures."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from cycle_vad.paper_calibration import SCENES,POLICIES,read,deserialize_thresholds
from cycle_vad.data import write_json
ROOT=Path(__file__).resolve().parents[1]
plt.rcParams.update({'font.size':10,'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','svg.hashsalt':'cyclevad-calibration','figure.facecolor':'white'})
def table(head,rows):return '| '+' | '.join(head)+' |\n| '+' | '.join(['---']*len(head))+' |\n'+''.join('| '+' | '.join(map(str,r))+' |\n' for r in rows)+'\n'
def save(fig,name):
    fig.savefig(ROOT/f'docs/figures/{name}.svg',bbox_inches='tight',metadata={'Date':None})
    fig.savefig(ROOT/f'docs/figures/{name}.png',bbox_inches='tight',dpi=140);plt.close(fig)
def pct(x):return f'{100*x:.2f}'
def ci(v):return f'{v["estimate"]:+.2f} [{v["ci95"][0]:+.2f}, {v["ci95"][1]:+.2f}]'
def b0():
    result=[];lost=[];allruns=[]
    for seed in [42,43,44]:
        paths=sorted((ROOT/f'results/E15B0/seed{seed}').glob('R*/fold*/run.json'));assert len(paths)==20
        allruns += [read(p) for p in paths]
        for sc in SCENES:
            subset=[p for p in paths if p.parent.parent.name==sc];normal=[];events=[]
            for p in subset:normal+=read(p.with_name('normal.json'));events+=read(p.with_name('events.json'))
            assert len({v['id'] for v in normal})==len(normal)
            hit=np.array([v['hit'] for v in events])[:,0,:].astype(bool)
            item={'seed':seed,'scene':sc,'events':len(events),'detected':hit.sum(0).tolist(),
                  'added_over_C':(hit&~hit[:,0,None]).sum(0).tolist(),'lost_from_C':(~hit&hit[:,0,None]).sum(0).tolist(),
                  'normal_sources':len(normal),'grid_far':np.mean([v['grid_far'] for v in normal],axis=0).tolist(),
                  'overlap':{k:float(np.mean([v['overlap_q99'][k] for v in normal])) for k in normal[0]['overlap_q99']}}
            result.append(item)
            if seed==42:
                for e,h in zip(events,hit):
                    if h[0] and not h[1]:lost.append({'scene':sc,**e})
    collapsed=sum(all(j['q95_equals_q995'] for j in r['ecdf']) for r in allruns)
    write_json(ROOT/'results/E15B0/summary.json',{'policies':POLICIES,'rows':result,'q95_q995_all_three_equal_units':collapsed,'units':len(allruns),'seed42_lost_events':lost})
    r=[v for v in result if v['seed']==42];total=np.sum([v['detected'] for v in r],axis=0);far=np.mean([v['grid_far'] for v in r],axis=0)[0]
    text='### E15B0 경보 손실·보정 해상도 진단 완료\n\n고정 E11B 점수로 60개 단위를 검사했습니다. 아래는 seed 42, α=1%입니다. ORfree는 C/P 각자의 q99 임계값을 유지하는 진단이며 전체 1% 예산 방법이 아닙니다.\n\n'
    text+=table(['정책','실제 이벤트 / 66','C 대비 추가','C 대비 손실','정상 OOF grid FAR %'],[[POLICIES[j],int(total[j]),sum(v['added_over_C'][j] for v in r),sum(v['lost_from_C'][j] for v in r),pct(far[j])] for j in [0,1,4]])
    text+=table(['장면','C / CP / ORfree 탐지','C grid FAR %','ORfree grid FAR %','P만 추가한 정상 창 %'],[[v['scene'],' / '.join(str(v['detected'][j]) for j in [0,1,4]),pct(v['grid_far'][0][0]),pct(v['grid_far'][0][4]),pct(v['overlap']['P_only'])] for v in r])
    text+=f'원 임계값 OR는 C의 탐지를 보존하지만 추가 정상 경보를 발생시킵니다. 60개 단위 중 **{collapsed}개**에서 C/P/CP 모두 q95=q99.5입니다. 임계값 보정 영상은 3–4개, 겹치는 창은 21–33개이므로 높은 분위수의 구분 능력이 제한됩니다. 창 수를 독립 표본 수로 해석하지 않습니다. 원본 하나 제외 임계값과 ECDF jump도 각 run에 공개했습니다.\n\n'
    text+='![경보와 정상 비용](docs/figures/E15B0_tradeoff.svg)\n\n![C가 탐지했으나 CP가 놓친 7개 이벤트](docs/figures/E15B0_lost_events.svg)\n\n회색 영역은 이벤트 onset부터 탐지 deadline까지입니다. 패널별 점수 범위는 다릅니다.\n\n[원본 수치](results/E15B0/summary.json) · [재현 검증](results/E15B0/validation.txt)\n\n'
    fig,ax=plt.subplots(1,2,figsize=(10.5,4));names=['C','joint max','ORfree'];js=[0,1,4];colors=['#3873a1','#e09b38','#555555']
    ax[0].bar(names,total[js],color=colors);ax[0].set_ylim(0,66);ax[0].set_ylabel('Detected historical events / 66')
    ax[1].bar(names,100*far[js],color=colors);ax[1].set_ylabel('Held-out normal grid-window FAR (%)');ax[1].set_ylim(0,max(15,100*far[js].max()*1.2))
    for a,vs in zip(ax,[total[js],100*far[js]]):
        for k,v in enumerate(vs):a.text(k,v+.5,f'{v:.2f}' if a==ax[1] else str(int(v)),ha='center')
    fig.suptitle('Original branch thresholds: event preservation and normal alarm cost');fig.tight_layout();save(fig,'E15B0_tradeoff')
    fig,axes=plt.subplots(4,2,figsize=(12,12));axes=axes.ravel()
    for ax,e in zip(axes,lost):
        _,part,seq=e['source_id'].split('/');rel=Path('seed42')/e['scene']/'fold0';r0=read(ROOT/'results/E15B0'/rel/'run.json');th=deserialize_thresholds(r0['thresholds'])
        with np.load(ROOT/'runs/E11B'/rel/f'{part}_{seq}.npz') as d:s=d['scores'][:,[4,5]]
        lo=max(0,e['onset']-20);hi=min(len(s),e['onset']+e['deadline_frames']+20);xx=np.arange(lo,hi)-e['onset']
        ax.plot(xx,s[lo:hi,0],color='#3873a1',label='C');ax.plot(xx,s[lo:hi,1],color='#e09b38',label='P')
        ax.axhline(th[0,0,0],ls='--',color='#3873a1',label='C threshold');ax.axhline(th[0,1,0],ls=':',color='#555',label='joint threshold')
        ax.axvspan(0,e['deadline_frames'],alpha=.08,color='#555');ax.set_title(e['id']);ax.set_xlabel('Source frames from event onset');ax.set_ylabel('Calibrated score')
    for ax in axes[len(lost):]:ax.axis('off')
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,ncol=4,loc='lower center',frameon=False);fig.tight_layout(rect=(0,.035,1,1));save(fig,'E15B0_lost_events')
    return text

def publish(text):
    title='## 보정과 경보 결합 E15B0·E15B/C\n\n'
    header='[사전 설계](docs/PAPER_NEXT_EXPERIMENTS_20261009.md) · [고정 설정](configs/experiments/followup/e15bc_v1.json) · [코드 검사](results/paper_calibration_tests.txt)\n\n'
    block=title+header+text
    (ROOT/'docs/PAPER_CALIBRATION_RESULTS.md').write_text(block.replace('](docs/','](').replace('](results/','](../results/').replace('](configs/','](../configs/'))
    p=ROOT/'README.md';old=p.read_text();start=old.find(title)
    if start>=0:
        end=old.find('\n## ',start+len(title));end=len(old) if end<0 else end
        old=old[:start]+old[end+1:]
    position=old.index('## 최소 구성과 오경보 진단');p.write_text(old[:position]+block+old[position:])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--include-bc',action='store_true');a=p.parse_args();text=b0()
    if a.include_bc:raise RuntimeError('E15BC report is pending validation')
    else:text+='E15B/C는 실행·검증 중입니다. 아직 결합 개선 결과를 확정하지 않았습니다.\n\n'
    publish(text)
