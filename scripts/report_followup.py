"""Evidence-driven follow-up README section and static research plots."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from cycle_vad.data import write_json
ROOT = Path(__file__).resolve().parents[1]
SCENES = ['R01', 'R02', 'R03', 'R04']
FIG = ROOT/'docs/figures'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
def read(p): return json.loads((ROOT/p).read_text())
def table(h, rows): return '| '+' | '.join(h)+' |\n| '+' | '.join(['---']*len(h))+' |\n'+''.join('| '+' | '.join(map(str,r))+' |\n' for r in rows)+'\n'
def fmt(x): return '—' if x is None else f'{100*x:.2f}'
def save(fig, name):
    fig.savefig(FIG/f'{name}.svg',bbox_inches='tight');fig.savefig(FIG/f'{name}.png',bbox_inches='tight',dpi=150);plt.close(fig)
def scorefile(scene, fold, identity):
    _, part, seq = identity.split('/')
    return ROOT/'runs/followup'/scene/f'fold{fold}'/'scores'/f'{part}_{seq}.npz'

def report_e6():
    records=[];quant=[]
    fig, axes=plt.subplots(2,2,figsize=(12,8))
    parts=['reference','threshold','normal_evaluation','historical_test']
    colors=['#6b7280','#d49b38','#3676b5','#c75a56']
    box, bax=plt.subplots(1,4,figsize=(14,4),sharey=True)
    for scene,ax,bx in zip(SCENES,axes.flat,bax):
        ms=read(f'results/E8/{scene}/fold0/metrics.json'); pv=read(f'results/E8/{scene}/fold0/per_video.json')
        for part,color in zip(parts,colors):
            m=next(x for x in ms if x['partition']==part and x['cell']=='C-00' and x['readout']=='full')
            records.append(m)
            videos=[x for x in pv if x['partition']==part and x['cell']=='C-00' and x['readout']=='full']
            values=[]
            for v in videos:
                with np.load(scorefile(scene,0,v['id'])) as d:
                    values.append(d['C-00__full'][d['labels']==0])
            x=np.sort(np.concatenate(values));ids=np.unique(np.linspace(0,len(x)-1,min(1200,len(x))).astype(int))
            ax.plot(x[ids],(ids+1)/len(x),label=part,color=color,lw=1.8)
            quant.append({'scene':scene,'partition':part,'normal_frames':len(x),'p50_p90_p99':np.quantile(x,[.5,.9,.99]).tolist()})
        ax.axvline(m['threshold'],ls='--',c='black',lw=1,label='normal threshold q99')
        ax.set(title=scene,xlabel='C-00 full score (symlog)',ylabel='Normal-frame ECDF');ax.set_xscale('symlog',linthresh=1);ax.grid(alpha=.15)
        vals=[[x['fpr'] for x in pv if x['partition']==part and x['cell']=='C-00' and x['readout']=='full' and x['fpr'] is not None] for part in parts]
        bx.boxplot(vals,tick_labels=['Ref','Thr','Eval','Hist'],showfliers=True)
        bx.set(title=scene,ylim=(-.02,1.02));bx.yaxis.set_major_formatter(PercentFormatter(1));bx.grid(axis='y',alpha=.2)
    axes[0,0].legend(fontsize=8);fig.suptitle('E6 / Fold 0: calibration transfer on normal frames');fig.tight_layout();save(fig,'E6_normal_ecdf')
    bax[0].set_ylabel('Per-video normal false-positive rate');box.suptitle('E6 / Fold 0: file-level FPR distribution');box.tight_layout();save(box,'E6_video_fpr')
    write_json(ROOT/'results/E6/calibration_summary.json', {'metrics':records,'normal_score_quantiles':quant,
        'status':'complete','diagnostic_scope':'fold0, C-00 full; historical test normal frames are diagnostic only',
        'no_threshold_retuning':True,'annotations_completed':0})
    rows=[]
    for scene in SCENES:
        selected=[next(x for x in records if x['scene']==scene and x['partition']==part) for part in parts]
        rows.append([scene]+[fmt(m['fpr']) for m in selected]+[fmt(selected[2]['video_mean_fpr'])])
    return ('### E6 — 정상 분할·보정 이동 진단 완료\n\n'
        '정상 111개 영상의 5개 분할(장면×fold 20개)에서 fit/validation/reference/threshold/evaluation 중복이 없고 각 영상이 정확히 한 번 평가됨을 확인했습니다. 파일 간 원본 recording 독립성은 아직 미확인입니다. 40개 영상의 모델 예측을 숨긴 로컬 프레임 뷰어와 접촉시트를 준비했습니다. **실제 cycle 주석은 0개이며 위치 정확도는 미측정**입니다.\n\n'
        +table(['장면','Reference FPR %','Threshold FPR %','Normal eval FPR %','Historical normal FPR %','Eval 영상 평균 FPR %'],rows)
        +'표는 fold 0의 C-00 Full, 각 분할의 정상 프레임 기준입니다. Reference/threshold는 보정에 사용했으므로 일반화 성능이 아닙니다. Historical은 기존 테스트의 정상 라벨 구간입니다. 정상 q99라도 새로운 정상 영상에서 1% FPR을 보장하지 않습니다.\n\n'
        +'![정상 점수 분포](docs/figures/E6_normal_ecdf.svg)\n\n![영상별 오경보](docs/figures/E6_video_fpr.svg)\n\n'
        +'[E6 수치](results/E6/calibration_summary.json) · [분할 검사](results/E6/fold_audit.json) · [주석 자료 상태](results/E6/annotation_packet.json)\n\n')

def report_e8(complete):
    all_metrics=[];all_pv=[];runs=[]
    for p in sorted((ROOT/'results/E8').glob('R*/fold*/run.json')):
        runs.append(json.loads(p.read_text()))
        all_metrics.extend(json.loads((p.parent/'metrics.json').read_text()))
        all_pv.extend(json.loads((p.parent/'per_video.json').read_text()))
    primary='appearance_conditional'
    cells=[f'C-{r}{g}' for r in range(3) for g in range(3)]
    hist=[m for m in all_metrics if m['partition']=='historical_test' and m['readout']==primary]
    normal=[m for m in all_pv if m['partition']=='normal_evaluation' and m['readout']==primary]
    summary=[]
    for cell in cells:
        h=[m for m in hist if m['cell']==cell]
        fpr={s:sum(m['fpr']*m['valid_frames'] for m in normal if m['cell']==cell and m['scene']==s)/sum(m['valid_frames'] for m in normal if m['cell']==cell and m['scene']==s) for s in SCENES}
        summary.append({'cell':cell,'historical_macro_ap':float(np.mean([m['ap'] for m in h])),
            'historical_macro_auroc':float(np.mean([m['auroc'] for m in h])), 'normal_oof_fpr':fpr,
            'normal_macro_fpr':float(np.mean(list(fpr.values())))})
    base=summary[0];cand=next(x for x in summary if x['cell']=='C-21')
    decision={'candidate':'C-21','baseline':'C-00','primary_readout':primary,
        'historical_macro_ap_gain_pp':100*(cand['historical_macro_ap']-base['historical_macro_ap']),
        'normal_fpr_increase_pp':{s:100*(cand['normal_oof_fpr'][s]-base['normal_oof_fpr'][s]) for s in SCENES}}
    decision['point_target_met']=decision['historical_macro_ap_gain_pp']>=1 and all(v<=1 for v in decision['normal_fpr_increase_pp'].values())
    write_json(ROOT/'results/E8/summary.json', {'status':'complete_seed42' if complete else 'running',
        'completed_scene_folds':len(runs),'statistical_models':len(runs)*3,'seed':42,
        'rows':summary,'decision':decision,'historical_test_diagnostic_only':True,
        'robustness_seeds_43_44':'not_run','tracking_accuracy':None,'wall_seconds':sum(x['wall_seconds'] for x in runs)})
    fig,axes=plt.subplots(1,2,figsize=(10,4.6))
    for ax,key,title in zip(axes,['historical_macro_ap','normal_macro_fpr'],['Historical test: macro AP (%)','Normal evaluation: macro FPR (%)']):
        a=np.array([x[key]*100 for x in summary]).reshape(3,3)
        im=ax.imshow(a,cmap='Blues' if key.endswith('ap') else 'Oranges')
        for r in range(3):
            for g in range(3):ax.text(g,r,f'{a[r,g]:.2f}',ha='center',va='center',color='black')
        ax.set_xticks(range(3),['r × agreement','r only','No gate']);ax.set_yticks(range(3),['Legacy weights','r weights','Uniform weights'])
        ax.set(title=title,xlabel='Inference gate',ylabel='Mean-fitting weights');fig.colorbar(im,ax=ax,shrink=.7)
    fig.suptitle('E8 / Primary readout: max(appearance, conditional)');fig.tight_layout();save(fig,'E8_confidence_factorial')
    rows=[[x['cell'],fmt(x['historical_macro_auroc']),fmt(x['historical_macro_ap'])]+[fmt(x['normal_oof_fpr'][s]) for s in SCENES] for x in summary]
    text=('### E8 — Confidence 3×3 실험 '+('seed 42 완료' if complete else '진행 중')+'\n\n'
        +f'{len(runs)}개 장면×fold, 통계 모델 {len(runs)*3}개를 학습하고 각 모델의 추론 게이트 3종과 readout 3종을 평가했습니다. 각 fold의 추적기·θ·pooled/local 분기와 C-00에서 선택한 Fourier 차수·ridge를 고정했습니다. C-00 재학습 및 원래 점수와의 동등성 검사를 모두 통과했습니다.\n\n'
        +'C-행열: 학습 행 0=legacy, 1=위치 신뢰도 r, 2=균등; 추론 열 0=legacy, 1=r, 2=gate 없음. 아래 주지표는 max(appearance, conditional)입니다. 모든 조합과 readout은 각각 정상 q99 임계값을 사용합니다.\n\n'
        +table(['조합','Historical macro AUROC %','Historical macro AP %']+[f'{s} 정상 OOF FPR %' for s in SCENES],rows)
        +'![Confidence factorial](docs/figures/E8_confidence_factorial.svg)\n\n'
        +f'사전에 고정한 C-21의 C-00 대비 historical macro AP 변화는 **{decision["historical_macro_ap_gain_pp"]:+.3f} pp**입니다. AP +1 pp 및 장면별 정상 FPR 증가 ≤1 pp라는 점추정 기준은 **'+('충족' if decision['point_target_met'] else '미충족')+'**입니다. 기존 테스트는 탐색적 진단이며 확증 자료가 아닙니다.\n\n'
        +'[전체 요약](results/E8/summary.json) · 각 장면/fold의 metrics.json, per_video.json, mechanism.json에는 27개 readout과 gate 활성 비율·점수 분포가 포함됩니다. seed 43/44 재검증은 아직 실행하지 않았습니다.\n\n')
    ci_path=ROOT/'results/E8/paired_bootstrap.json'
    if complete and ci_path.exists():
        ci=json.loads(ci_path.read_text())
        text+=f'C-21−C-00 historical macro AP의 파일 단위 paired bootstrap 95% 구간: **[{ci["macro_ap_gain_pp_ci95"][0]:+.3f}, {ci["macro_ap_gain_pp_ci95"][1]:+.3f}] pp** (2,000회). 영상 원본 그룹 독립성은 미검증입니다. [불확실성 수치](results/E8/paired_bootstrap.json)\n\n'
    return text

def main():
    p=argparse.ArgumentParser();p.add_argument('--complete-e8',action='store_true');p.add_argument('--e6-only',action='store_true');args=p.parse_args()
    text='## 후속 실험 E6–E10\n\n[사전 고정 실험 명세](docs/FOLLOWUP_EXPERIMENTS.md) · [후속 검사 로그](results/followup_tests.txt)\n\n'
    text+=report_e6()
    text+=('E8: Confidence 3×3 × readout 3종의 5-fold 실험 진행 중입니다.\n\n' if args.e6_only else report_e8(args.complete_e8))
    text+='E7: T0/T1/T2 인과적 예측을 로컬에 저장했으며 독립적인 실제 cycle/anchor 주석이 필요합니다. E9: 사전 계획에 따라 E7 진단 후 진행합니다. E10: 신규 독립 촬영 자료가 필요합니다. 미실행 항목을 완료로 표시하지 않습니다.\n\n'
    (ROOT/'docs/FOLLOWUP_RESULTS.md').write_text(text.replace('](docs/', '](').replace('](results/', '](../results/'))
    path=ROOT/'README.md';old=path.read_text()
    start=old.index('## 후속 실험');end=old.index('## 실험 상태',start)
    path.write_text(old[:start]+text+old[end:])
    write_json(ROOT/'results/followup_status.json',{'E6':'complete','E7':'predictions_ready_independent_annotations_required',
        'E8':'seed42_complete_robustness_pending' if args.complete_e8 else 'running',
        'E9':'pending_E7_diagnosis','E10':'new_independent_recordings_required'})

if __name__=='__main__':main()
