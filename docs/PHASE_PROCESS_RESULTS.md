## 위치·진행 검증 E8B / E9S

[고정 실험 명세](PHASE_PROCESS_EXPERIMENTS.md) · [35개 코드 검사](../results/phase_process_tests.txt)

### E8B — 위치별 정상 기준 검증 완료

60개 장면×fold×seed, 640개 비교 모델을 평가했습니다. Confidence 가중치와 추론 게이트를 제거하고 진행 점수를 제외했습니다. M0=θ 없는 pooled PCA, M1=첫 관측 위치+정상 FIT 시간, M2=관측별 위치, M3=누적 추적 위치(C-22), M4=영상별 고정 무작위 위치 이동입니다. M4의 다섯 offset은 별도 학습·보정한 대조군이며 독립 영상으로 세지 않았습니다.

| 조건 | Historical AP % | AUROC % | R01 정상 FPR % | R02 정상 FPR % | R03 정상 FPR % | R04 정상 FPR % |
| --- | --- | --- | --- | --- | --- | --- |
| M0 | 67.99 | 76.26 | 5.49 | 1.14 | 4.84 | 1.77 |
| M1 | 65.51 | 75.06 | 5.75 | 3.39 | 3.38 | 2.36 |
| M2 | 68.77 | 76.53 | 5.38 | 1.14 | 4.89 | 1.23 |
| M3 | 71.10 | 77.82 | 5.40 | 1.18 | 4.08 | 2.22 |
| M4_1101 | 67.76 | 76.00 | 5.30 | 1.04 | 4.80 | 1.44 |
| M4_1102 | 68.13 | 76.34 | 5.38 | 1.03 | 4.67 | 2.03 |
| M4_1103 | 67.82 | 76.04 | 5.39 | 1.12 | 4.89 | 1.77 |
| M4_1104 | 67.76 | 76.03 | 5.49 | 1.18 | 3.95 | 1.91 |
| M4_1105 | 68.09 | 76.31 | 5.05 | 1.08 | 4.85 | 1.85 |

표는 seed 42, 외형+조건부 점수입니다. Historical은 fold 0의 기존 테스트, 정상 FPR은 5-fold OOF 전체 정상 프레임 기준입니다. 라벨 길이 불일치 R02 테스트 12/13/14는 전체 제외했습니다. M0의 결합 점수는 외형 점수와 정확히 같습니다. M3의 균등 μ 재학습 동등성을 확인했습니다.

![위치 조건 비교](figures/E8B_phase_comparison.svg)

| 사전 지정 비교 | Macro AP 차이 pp [95% CI] |
| --- | --- |
| M3-M0 | +3.11 [+1.78, +4.94] |
| M3-M1 | +5.59 [+1.04, +9.11] |
| M3-M2 | +2.33 [+1.03, +4.48] |
| M3-M4_mean | +3.19 [+1.79, +5.16] |

신뢰구간은 seed 42 모델을 고정한 장면별 원본 영상 단위 paired bootstrap 2,000회입니다. 파일 간 원 촬영의 독립성은 미확인이고 다중 비교 보정은 하지 않았습니다. 현재 관측을 이용한 μ의 정상 재구성 오차는 위치 GT 정확도나 미래 예측 정확도가 아닙니다.

| Seed | M0 AP % | M3 AP % | M3−M0 pp | M3−M4 평균 pp |
| --- | --- | --- | --- | --- |
| 42 | 67.99 | 71.10 | +3.11 | +3.19 |
| 43 | 68.43 | 71.77 | +3.35 | +3.31 |
| 44 | 68.59 | 71.45 | +2.87 | +3.05 |

사전 점추정 기준(AP +1 pp, 각 장면 정상 FPR 증가 ≤1 pp)은 **충족**입니다. 최대 장면 FPR 증가는 +0.45 pp입니다. M3−M0 및 M3−M4 CI의 0 포함 여부는 각각 **미포함 / 미포함**입니다. 이 결과는 위치 조건의 추가 가치를 탐색적으로 뒷받침하지만, 기존 테스트를 사용했으므로 독립 확증 결과는 아닙니다.

| 민감도 조건 (seed 42) | 외형+조건부 AP % | 조건부 단독 AP % |
| --- | --- | --- |
| M0_fixedrank | 67.84 | 68.05 |
| M3_fixedrank | 71.10 | 72.43 |
| M1_retuned | 65.45 | 67.33 |
| M2_retuned | 68.94 | 70.11 |
| M3_retuned | 71.10 | 72.43 |

고정 rank에서는 두 모델의 축 수를 동일하게 맞췄고, retuned에서는 각 조건에 동일한 정상 validation 9개 조합 탐색을 적용했습니다. 주 비교와 분리하여 보고합니다. 개별 장면의 μ MSE, subspace 외부 MSE, 유지 rank, 영상 평균 FPR과 historical recall·활성화는 아래 JSON에 공개했습니다.

[E8B 요약](../results/E8B/summary.json) · [paired CI](../results/E8B/paired_bootstrap.json) · [점수 재검증](../results/E8B/validation.txt) · [분할별 원자료](../results/E8B/)

E9S는 전체 seed·stride 실행과 결과 검증을 진행 중입니다. 완료 수치와 오경보 분석은 다음 단계 커밋으로 공개합니다.

재실행: `PYTHONPATH=src python scripts/run_phase_process.py --stage E8B --resume`, `--stage E9S --resume`, `--stage E9S --stride 1 --seeds 42 --resume`. 이어 `scripts/summarize_phase_process.py --stage E8B` 및 E9S의 각 `--seed`/`--stride`를 실행합니다. 주 E9S는 `--bootstrap`을 붙입니다. `scripts/check_phase_process.py --stage E8B`/`E9S`로 검증하고 `scripts/report_phase_process.py --include-e9`로 문서를 재생성합니다. 특징 추출은 로컬 RTX PRO 6000의 고정 캐시를 재사용했고 새 통계 모델은 CPU에서 학습했습니다.

