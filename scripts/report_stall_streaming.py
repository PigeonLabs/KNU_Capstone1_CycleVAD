"""E18D report: accuracy, operating alarms, and separately measured latency."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from cycle_vad.paper_experiments import read
from cycle_vad.data import write_json
from report_stall import table,pct,ci
from report_fragments import stall_fragment,write_stall_fragment
ROOT=Path(__file__).resolve().parents[1];SCENES=['R01','R02','R03','R04']

def main():
    a=read(ROOT/'results/E18D/accuracy.json');runs=[read(p) for p in sorted((ROOT/'results/E18D/benchmarks').glob('*.json'))]
    assert len(runs)==16
    speed=[];paced=[]
    for arm in ['B0','B3']:
        for stride in [1,2]:
            group=[r for r in runs if r['arm']==arm and r['stride']==stride and r['paced_fps'] is None]
            assert len(group)==3 and {r['repeat'] for r in group}=={0,1,2}
            assert all(r['input_frames']==2048 for r in group)
            speed.append({'arm':arm,'stride':stride,'repeats':3,'input_fps':sum(r['input_frames'] for r in group)/sum(r['wall_seconds'] for r in group),'fresh_fps':sum(r['fresh_updates'] for r in group)/sum(r['wall_seconds'] for r in group),
                          'fresh_p95_ms_mean':float(np.mean([r['fresh_service_ms']['p95'] for r in group])),
                          'fresh_p50_ms_mean':float(np.mean([r['fresh_service_ms']['p50'] for r in group])),
                          'head_ms_mean':float(np.mean([r['head_ms']['mean'] for r in group])),
                          'repeat_input_fps_range':[min(r['input_fps'] for r in group),max(r['input_fps'] for r in group)]})
    paced=[r for r in runs if r['paced_fps'] is not None];assert len(paced)==4 and all(r['input_frames']==512 for r in paced)
    write_json(ROOT/'results/E18D/summary.json',{'status':'complete','accuracy':a,'speed':speed,'paced':paced,'caveat':'Frozen stride2 calibrators at stride1; no threshold retuning. Accuracy replay shares current encoder output between independent states; speed is measured separately.'})
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','svg.hashsalt':'CycleVAD-E18D'})
    fig,axes=plt.subplots(1,2,figsize=(12,4.6),layout='constrained');x=np.arange(4)
    for mode,label,marker in [('cached','Cached stride 2','x'),('stride2','Online stride 2','o'),('stride1','Online stride 1','s')]:
        axes[0].plot(x,[100*a['scene'][sc][mode]['B1']['ap'] for sc in SCENES],marker=marker,label=label)
    axes[0].set_xticks(x,SCENES);axes[0].set_ylabel('CP + stall AP (%)');axes[0].set_ylim(40,100);axes[0].legend(fontsize=9);axes[0].grid(axis='y',alpha=.2)
    labels=[f'{r["arm"]}\nstride {r["stride"]}' for r in speed];x=np.arange(len(speed))
    axes[1].bar(x-.18,[r['input_fps'] for r in speed],width=.36,label='Input frames/s');axes[1].bar(x+.18,[r['fresh_fps'] for r in speed],width=.36,label='Fresh scores/s')
    axes[1].set_xticks(x,labels);axes[1].set_ylabel('Throughput (per second)');axes[1].legend(fontsize=9);axes[1].grid(axis='y',alpha=.2)
    fig.suptitle('E18D | Causal accuracy and separately measured GPU throughput\nRTX PRO 6000; batch 1; no future frames or prefetch',fontsize=13)
    for ext in ['svg','png']:fig.savefig(ROOT/f'docs/figures/E18D_online.{ext}',dpi=150,metadata={'Date':None} if ext=='svg' else None)
    plt.close(fig)
    lines=['## E18D 전체 test 온라인 검증·실시간 성능 완료','',
      'RTX PRO 6000에서 66개 test 영상·33,462프레임을 JPEG 입력부터 새 GPU 특징으로 순차 처리했습니다. 유효 평가 프레임은 31,550개이며, 라벨은 영상 추론이 끝난 뒤 지표 계산에만 사용했습니다. Stride 1·2는 독립 인과적 상태를 사용하고 현재 프레임의 인코딩만 공유했습니다. 속도는 별도 실행으로 측정했습니다.','',
      '#### 실제 test 온라인 정확도 — seed 42 / fold 0','',
      table(['입력/간격','모델','AUROC %','AP %','이벤트 /66','정상 FPR %','이상 Recall %'],[[mode,arm,pct(m['auroc']),pct(m['ap']),m['event_hits'],pct(m['fpr']),pct(m['recall'])] for mode,models in a['macro'].items() for arm,m in models.items()]),
      '온라인 stride 2의 CP+정지는 AP 약 73.92%, 실제 이벤트 35개로 캐시 평가를 거의 재현합니다. **온라인 적응형 B3는 33개, 캐시 기반 B3는 32개**입니다. R04/testing/08의 event 0이 온라인에서 새로 탐지됐으며, batch 추출 캐시와 새 batch-one GPU 특징의 작은 수치 차이에 영향을 받은 경계 사례입니다. 동일한 새 특징을 사용하는 순차/참조 계산의 경보 일치는 별도로 검증했습니다. [변경 이벤트](results/E18D/cache_online_event_changes.json).','',
      'Stride 1은 기존 stride 2 모델·보정값을 유지하고 원본 시간 간격만 맞춘 민감도 실험입니다. 새 판별 빈도가 높아졌다고 AP가 개선되는 것은 아닙니다.','',
      '![온라인 정확도와 처리량](docs/figures/E18D_online.svg)','',
      '#### 장면별 온라인 정확도','',
      table(['장면','Stride','모델','AUROC %','AP %','이벤트 탐지/전체','정상 FPR %','이상 Recall %'],[[sc,stride,arm,pct(m['auroc']),pct(m['ap']),f'{m["event_hits"]}/{m["event_total"]}',pct(m['fpr']),pct(m['recall'])] for sc in SCENES for stride in [1,2] for arm,m in a['scene'][sc][f'stride{stride}'].items()]),
      '#### 온라인 cadence·캐시 차이의 paired CI','',
      table(['비교','모델','AP Δ pp [95% CI]','AUROC Δ pp [95% CI]'],[[contrast,arm,ci(v['ap']),ci(v['auroc'])] for contrast,models in a['paired_bootstrap'].items() for arm,v in models.items()]),
      '원본 영상 단위·장면 층화 2,000회, 고정 모델에 조건부인 탐색적 구간입니다.','',
      '#### JPEG 입력부터 경보까지 처리 속도','',
      'R01–R04에서 각 4개 영상의 첫 128프레임, 조건당 2,048프레임을 3회 처리했습니다. 입력 FPS와 fresh FPS는 반복 전체 처리량이며 p50/p95는 각 반복의 fresh 지연 분위수를 평균한 값입니다. B3 어댑터는 공유 계산으로 6개 비교 출력까지 산출합니다.','',
      table(['모델','Stride','입력 FPS','새 점수/초','fresh p50 ms','fresh p95 ms','평균 head ms'],[[r['arm'],r['stride'],f'{r["input_fps"]:.2f}',f'{r["fresh_fps"]:.2f}',f'{r["fresh_p50_ms_mean"]:.2f}',f'{r["fresh_p95_ms_mean"]:.2f}',f'{r["head_ms_mean"]:.3f}'] for r in speed]),
      '#### 30/60 FPS 입력 재생','',
      table(['Stride','입력 FPS','프레임','처리 기간 초과','도착→출력 p95 ms','도착→출력 최대 ms'],[[r['stride'],r['paced_fps'],r['input_frames'],r['missed_periods'],f'{r["arrival_ms"]["p95"]:.2f}',f'{r["arrival_ms"]["max"]:.2f}'] for r in paced]),
      '30 FPS 입력의 두 stride는 마감 초과가 없었습니다. 60 FPS에서는 stride 1이 5/512, stride 2가 8/512프레임에서 16.67ms를 넘었습니다. 평균 처리량은 충분하지만 모든 프레임의 60 FPS 마감을 보장한 결과는 아닙니다.','',
      'FIT 영상 100프레임 warmup은 측정에서 제외했습니다. JPEG 읽기·디코딩·전처리·GPU 인코딩·CPU 추적·정지 점수·경보를 포함하고, 미래 프레임 읽기나 prefetch는 사용하지 않았습니다. 카메라 수신·H.264/H.265 디코딩·네트워크·화면 출력은 측정 범위 밖입니다.','',
      '#### 인과성·재현 검증','',
      table(['검증','결과'],[['검증 영상/stride','66 × 2, 모두 통과'],['추적 최대 차이',f'{a["parity_max_errors"]["max_trace_error"]:.3g}'],['C 점수 최대 차이',f'{a["parity_max_errors"]["max_C_error"]:.3g}'],['P 점수 최대 차이',f'{a["parity_max_errors"]["max_P_error"]:.3g}'],['정지 점수 최대 차이',f'{a["parity_max_errors"]["stall_max_error"]:.3g}'],['동일 특징 경보·임계값 일치','모든 영상 통과'],['미래 변경/초기 lag/짧은 영상/상태 초기화','단위 테스트 및 전체 재생 통과']]),
      '[온라인 전체 수치·속도](results/E18D/summary.json) · [GPU·체크포인트 환경](results/E18D/environment.json) · [실행 코드](scripts/evaluate_stall_streaming.py)','',
      '재현: `PYTHONPATH=src python scripts/evaluate_stall_streaming.py --resume`, `PYTHONPATH=src python scripts/summarize_stall_streaming.py`, `PYTHONPATH=src python scripts/evaluate_stall_streaming.py --benchmark --resume`, `PYTHONPATH=src python scripts/report_stall_streaming.py`. 속도 실행은 다른 실험 연산과 겹치지 않게 수행합니다.','']
    text=stall_fragment(ROOT).split('\n## E18D',1)[0]+'\n'+'\n'.join(lines)
    write_stall_fragment(ROOT,text)
    status=read(ROOT/'results/followup_status.json');status['E18D']='complete_causal_gpu_replay_and_latency';write_json(ROOT/'results/followup_status.json',status)

if __name__=='__main__':main()
