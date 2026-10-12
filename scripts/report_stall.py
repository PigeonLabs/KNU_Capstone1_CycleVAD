"""Publish every fixed E18 arm inline, including unfavorable outcomes."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from cycle_vad.paper_experiments import read
from cycle_vad.data import write_json
from report_fragments import stall_fragment,write_stall_fragment
ROOT=Path(__file__).resolve().parents[1]
SCENES=['R01','R02','R03','R04']

def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|',*['| '+' | '.join(map(str,row))+' |' for row in rows]])+'\n'
def pct(v):return f'{100*v:.2f}'
def ci(v):return f'{v["delta_pp"]:+.2f} [{v["ci95_pp"][0]:+.2f}, {v["ci95_pp"][1]:+.2f}]'

def main():
    s=read(ROOT/'results/E18C/summary.json');seed=s['seeds']['42'];macro=seed['macro'];boot=s['bootstrap']['contrasts']
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','svg.hashsalt':'CycleVAD-E18'})
    fig,axes=plt.subplots(1,2,figsize=(12,4.7),layout='constrained');x=np.arange(4)
    for hi,lo,c,label in [('C1','C0','#337BA9','C + stall minus C'),('B1','B0','#B37322','CP + stall minus CP')]:
        axes[0].plot(x,[100*(seed['scene'][sc][hi]['ap']-seed['scene'][sc][lo]['ap']) for sc in SCENES],marker='o',label=label,color=c)
        axes[1].plot(x,[seed['scene'][sc][hi]['event_hits']-seed['scene'][sc][lo]['event_hits'] for sc in SCENES],marker='o',label=label,color=c)
    for ax in axes:ax.axhline(0,color='#777',lw=.8);ax.set_xticks(x,SCENES);ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('AP difference (percentage points)');axes[1].set_ylabel('Net additional actual events');axes[0].legend(fontsize=9)
    fig.suptitle('E18B | Stall gains depend on the scene\nActual R01-R04 test; frozen seed 42 models',fontsize=13)
    for ext in ['svg','png']:fig.savefig(ROOT/f'docs/figures/E18B_scene.{ext}',dpi=150,metadata={'Date':None} if ext=='svg' else None)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.7),layout='constrained')
    for arm in ['C0','C1','B0','B1','B2','B3','B0_T110','B0_T130','B1_T110','B1_T130']:
        m=macro[arm];axes[0].scatter(100*m['fpr'],m['event_hits'],s=50)
        offset={'B0':(5,-13),'B1':(5,6),'B2':(5,-13),'B3':(5,6),'C0':(-25,-13),'C1':(-25,5)}.get(arm,(5,5))
        axes[0].annotate(arm,(100*m['fpr'],m['event_hits']),xytext=offset,textcoords='offset points',fontsize=8)
    axes[0].set(xlabel='Actual normal frame FPR, scene macro (%)',ylabel='Detected actual events / 66',xlim=(0,18),ylim=(0,46));axes[0].grid(alpha=.2)
    arms=['B0','B1','B2','B3'];xx=np.arange(4)
    axes[1].bar(xx-.17,[100*macro[a]['fpr'] for a in arms],width=.34,label='Normal frame FPR')
    axes[1].bar(xx+.17,[100*macro[a]['recall'] for a in arms],width=.34,label='Anomaly frame recall')
    axes[1].set_xticks(xx,arms);axes[1].set_ylabel('Scene macro (%)');axes[1].set_ylim(0,50);axes[1].legend(fontsize=9);axes[1].grid(axis='y',alpha=.2)
    fig.suptitle('E18C | Fewer false alarms come with reduced anomaly-frame recall\nFixed candidate policies; no test-FPR matching',fontsize=13)
    for ext in ['svg','png']:fig.savefig(ROOT/f'docs/figures/E18C_tradeoff.{ext}',dpi=150,metadata={'Date':None} if ext=='svg' else None)
    plt.close(fig)
    path=ROOT/'docs/E18_RESULTS.md';pre=stall_fragment(ROOT).split('\n## E18B')[0]
    pre=pre.replace('E18B 정지 점수, E18C 적응형 경보, E18D 온라인 검증은 진행 중입니다.','E18B·E18C는 완료했으며 E18D 온라인 검증은 별도 진행 상태를 확인하세요.')
    lines=[pre.rstrip(),'','## E18B 정지 신호와 위치 조건 ablation 완료','',
      '**주 평가는 R01–R04 실제 test입니다.** 66개 영상 중 strict-v1 유효 영상은 63개, 유효 프레임은 31,550개입니다. 3개 seed × 4장면 × 5-fold의 60개 단위를 평가했습니다. 아래 수치는 seed 42이며, AUROC/AP/FPR/Recall은 장면별 값의 단순 평균입니다. 이벤트 수는 66개 실제 이벤트의 합계이므로 Macro event recall과 분모가 다릅니다.','',
      '정지 신호 S3를 CP에 추가하면 기존 29개 이벤트를 모두 유지하고 6개를 추가해 **35/66**을 탐지합니다. AP는 **73.43→73.93%**이나 차이의 신뢰구간은 0을 포함합니다. C+정지는 **41/66**을 탐지하지만 C의 기존 이벤트 2개를 잃고 10개를 추가하며, 정상 OOF grid FAR도 증가합니다. 자동으로 최종 모델을 교체하지 않습니다.','']
    names={'C0':'C','C1':'C+S3','B0':'CP','B1':'CP+S3','Full':'A+C+P','P':'P','CP_S0':'CP+전역 지속시간','CP_S1':'CP+전역 CUSUM','CP_S2':'CP+현재 위치 CUSUM'}
    barms=list(names)
    lines += [table(['모델','구성','AUROC %','AP %','실제 이벤트 /66','정상 FPR %','이상 Recall %','OOF grid FAR %'],[[a,names[a],pct(macro[a]['auroc']),pct(macro[a]['ap']),macro[a]['event_hits'],pct(macro[a]['fpr']),pct(macro[a]['recall']),pct(macro[a]['normal_grid_far'])] for a in barms]),
      '![장면별 정지 신호의 효과](docs/figures/E18B_scene.svg)','',
      '#### 장면별 실제 test와 정지 ablation','',
      table(['장면','모델','AUROC %','AP %','이벤트 탐지/전체','정상 FPR %','이상 Recall %'],[[sc,a,pct(m['auroc']),pct(m['ap']),f'{m["event_hits"]}/{m["event_total"]}',pct(m['fpr']),pct(m['recall'])] for sc in SCENES for a in barms for m in [seed['scene'][sc][a]]]),
      '#### 영상 단위 paired bootstrap','',
      '원본 영상 단위, 장면 층화, 2,000회, 고정 모델에 조건부인 탐색적 95% 구간입니다. seed를 독립 표본으로 세지 않습니다. 아래 event recall 차이는 **장면 Macro**입니다. 파일 간 촬영 독립성은 확인되지 않았습니다.','',
      table(['비교','AP Δ pp [95% CI]','AUROC Δ pp [95% CI]','Event recall Δ pp [95% CI]'],[[k,ci(boot[k]['macro']['ap']),ci(boot[k]['macro']['auroc']),ci(boot[k]['macro']['event_recall'])] for k in ['B1-B0','C1-C0','B1-CP_S0','B1-CP_S1','B1-CP_S2']]),
      '전역 지속시간·전역 CUSUM·현재 위치·진입 위치 조건 모두 총 35개 이벤트를 탐지하지만 탐지한 이벤트 구성이 같다는 뜻은 아닙니다. S3의 S1 대비 AP 이득은 약 0.21%p로 작습니다. 이를 위치 추적의 실제 정답 정확도나 강한 정지 유형 구분 증거로 확대하지 않습니다.','',
      '#### 장면 하나 제외 분석','',
      table(['제외 장면','C1−C0 AP Δ pp','B1−B0 AP Δ pp','B1−B0 AUROC Δ pp'],[[sc,ci(boot['C1-C0']['without_'+sc]['ap']),ci(boot['B1-B0']['without_'+sc]['ap']),ci(boot['B1-B0']['without_'+sc]['auroc'])] for sc in SCENES]),
      'R02 의존성이 남는지 위 표로 확인합니다. FIT 위치별 분포는 4-bin이며, 표본 기준을 충족하지 못한 bin은 전역 분포로 대체했습니다. 각 60개 단위의 사용 여부·gate·k·임계값은 run/fitted JSON에 공개했습니다.','',
      '[E18B 요약·신뢰구간](results/E18B/summary.json) · [고정 프로토콜](configs/experiments/followup/e18_v1.json)','',
      '## E18C 적응형 경보와 고정 임계값 대조 완료','',
      '**적응형 임계값은 원래 이상 점수를 바꾸지 않습니다.** B0=B2, B1=B3의 AUROC/AP가 정확히 같습니다. 최초 48개 원본 프레임은 고정 임계값을 사용하며, frame 48부터 과거 prefix의 하위 점수로 최대 30% 상향합니다.','',
      table(['모델','AUROC %','AP %','이벤트 /66','정상 FPR %','이상 Recall %','OOF grid FAR %'],[[a,pct(macro[a]['auroc']),pct(macro[a]['ap']),macro[a]['event_hits'],pct(macro[a]['fpr']),pct(macro[a]['recall']),pct(macro[a]['normal_grid_far'])] for a in ARMS_FOR_REPORT()]),
      '![오경보와 탐지 손실](docs/figures/E18C_tradeoff.svg)','',
      '**B2는 정상 FPR을 12.37→4.61%로 줄이지만 실제 이벤트 2개를 잃고, 이상 프레임 Recall이 약 19.21%p 감소합니다. B3는 B1 대비 이벤트 3개를 잃어 seed 42의 사전 허용 목표(2개 이하)를 충족하지 못합니다.** B3의 CP 대비 이벤트 순증은 +3개지만 C 기준으로는 9개를 잃습니다. 오경보 감소만으로 탐지 보존에 성공했다고 결론 내리지 않습니다.','',
      '#### 이벤트 유지·추가·손실','',table(['비교','유지','추가','손실','둘 다 미탐지'],[[k,*[v[x] for x in ['retained','added','lost','missed']]] for k,v in seed['event_changes'].items()]),
      '전체 이벤트별 ID·시작/종료·탐지 지연은 [seed 42 이벤트 원장](results/E18C/seed42/event_ledger.json)에 있습니다. 이벤트 정답은 평가에만 사용하며 추적 상태 초기화에 사용하지 않았습니다.','',
      '#### 적응형 경보의 차이와 불확실성','',
      table(['비교','정상 FPR Δ pp [95% CI]','이상 Recall Δ pp [95% CI]','Event recall Δ pp [95% CI]'],[[k,ci(boot[k]['macro']['fpr']),ci(boot[k]['macro']['recall']),ci(boot[k]['macro']['event_recall'])] for k in ['B2-B0','B3-B0','B3-B1','B3-C0','B2-B0_T110','B2-B0_T130','B3-B1_T110','B3-B1_T130']]),
      '고정 1.1배·1.3배와 적응형 정책은 서로 다른 탐지–오경보 운영점입니다. test 정상 라벨로 FPR을 맞추거나 가장 좋은 정책을 재선택하지 않았습니다.','',
      '#### 장면별 적응형 정책·고정 상승 대조','',
      table(['장면','모델','이벤트 탐지/전체','정상 FPR %','이상 Recall %','탐지된 이벤트 지연 중앙값 (frame)'],[[sc,a,f'{m["event_hits"]}/{m["event_total"]}',pct(m['fpr']),pct(m['recall']),m['detected_delay_median_frames']] for sc in SCENES for a in ['B2','B3','B0_T110','B0_T130','B1_T110','B1_T130'] for m in [seed['scene'][sc][a]]]),
      '지연 중앙값은 탐지된 이벤트에 조건부입니다. 옆의 탐지 수와 함께 읽어야 하며, 미탐지를 지연 0으로 대체하지 않았습니다.','',
      '#### 3-seed 전체 주 비교군','',
      table(['Seed','모델','AUROC %','AP %','이벤트 /66','정상 FPR %','이상 Recall %','OOF grid FAR %'],[[sd,a,pct(m['auroc']),pct(m['ap']),m['event_hits'],pct(m['fpr']),pct(m['recall']),pct(m['normal_grid_far'])] for sd,v in s['seeds'].items() for a,m in v['macro'].items() if a not in ['CP_S0','CP_S1','CP_S2']]),
      '#### 모든 사전 지정 분위수 운영점 — seed 42','',
      '장면별 비율을 Macro로 집계했습니다. 명목 q99는 실제 FPR 1%를 보장하지 않으며, 고분위수 동률 때문에 동일한 행이 나타날 수 있습니다.','',
      table(['모델','q','정상 FPR %','이상 Recall %','Macro event recall %'],[[a,q,*[pct(np.mean([seed['scene'][sc][a]['quantiles'][j][m] for sc in SCENES])) for m in ['fpr','recall','event_recall']]] for a in seed['arms'] for j,q in enumerate([.90,.95,.975,.99,.995])]),
      '#### 합성 이상 보조 진단 — 실제 test와 분리','',
      '기존 3,910개 편집과 정상 OOF 111개 원본 × 4개 onset의 초기 freeze 444개를 seed별로 평가했습니다. 아래는 seed 42, 장면별 편집 평균의 Macro입니다. 동일 원본의 파생 편집을 독립 원본처럼 세지 않습니다. Freeze는 실제 정지 내부이면서 0.2T 이내의 경보만 집계하며, 종전 D-window는 재개 이후 경보를 포함할 수 있는 별도 값입니다.','']
    auxrows=[]
    for group in ['existing','early_prefix']:
        edits=sorted(seed['auxiliary']['R01'][group])
        for edit in edits:
            for a in seed['arms']:
                vals=[seed['auxiliary'][sc][group][edit][a] for sc in SCENES]
                auxrows.append([group,edit,a,sum(v['n'] for v in vals),pct(np.mean([v['hit'] for v in vals])),pct(np.mean([v['legacy_D_hit'] for v in vals])),pct(np.mean([v['matched_far'] for v in vals]))])
    lines += [table(['구분','편집','모델','사례 수','탐지 %','기존 D 탐지 %','대응 정상 경보 %'],auxrows),
      '합성 정지는 동일 영상 프레임 반복에 대한 진단입니다. 실제 기계 정지 유형의 정답이 없으므로 이 결과를 실제 정지 Recall로 부르지 않습니다. 초기 prefix가 이상을 포함할 때의 손실도 위 early_prefix 결과에 포함합니다. CP+S3는 freeze 내부 탐지율 100%를 기록하지만 reverse·skip·swap 탐지율은 기존 CP보다 낮아집니다. 정지 신호를 추가한 뒤 결합 임계값을 다시 보정하므로 모든 기존 경보가 보존되는 것은 아닙니다.','',
      '![고정 사례의 위치·정지 점수 궤적](docs/figures/E18_stall_traces.svg)','',
      '장면별 manifest의 첫 eligible freeze를 고정 예시로 선택했습니다. 성능이 좋은 사례를 고른 것이 아니며, 원본과 편집 영상의 posterior 위치 및 CP·정지 점수·임계값을 보여줍니다. [선정된 사례 ID](results/E18C/trace_examples.json).','',
      '#### 해석과 채택 판단','',
      '정지 신호는 실제 이벤트 탐지를 보완하지만, AP 이득의 불확실성과 R02 의존성이 남습니다. 적응형 경보는 정상 오경보를 줄이는 대신 이상 프레임 Recall과 일부 이벤트를 희생합니다. **기존 C 중심 기여를 유지하고, 정지 신호는 보완 요소, 적응형 임계값은 운영점 선택으로 제시합니다.** C+정지를 후속 후보로 기록하되 test 결과로 새 주 설정을 선택하거나 재튜닝하지 않습니다.','',
      '[전체 장면·seed·운영점·paired CI](results/E18C/summary.json) · [E18B/C 실행 코드](scripts/run_stall.py) · [집계 코드](scripts/summarize_stall.py)','',
      '재현: `PYTHONPATH=src python scripts/check_e18_preflight.py`, `PYTHONPATH=src python scripts/run_stall.py --resume`, `PYTHONPATH=src python scripts/run_stall.py --auxiliary --resume`, `PYTHONPATH=src python scripts/summarize_stall.py`, `PYTHONPATH=src python scripts/report_stall.py`. 로컬 특징·모델은 Git에 포함하지 않습니다.','']
    content='\n'.join(lines)
    # Preserve a separately completed online section when rebuilding B/C.
    existing=stall_fragment(ROOT);online='\n## E18D'+existing.split('\n## E18D',1)[1] if '\n## E18D' in existing else ''
    write_stall_fragment(ROOT,content+online)
    status=read(ROOT/'results/followup_status.json');status.update(E18B='complete_mixed_actual_test_effect',E18C='complete_false_alarm_recall_tradeoff');write_json(ROOT/'results/followup_status.json',status)

def ARMS_FOR_REPORT():return ['C0','C1','B0','B1','B2','B3','B0_T110','B0_T130','B1_T110','B1_T130','Full','P']

if __name__=='__main__':main()
