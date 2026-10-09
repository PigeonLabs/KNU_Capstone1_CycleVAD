## 후속 실험 E6–E10

[사전 고정 실험 명세](FOLLOWUP_EXPERIMENTS.md) · [후속 검사 로그](../results/followup_tests.txt)

### E6 — 정상 분할·보정 이동 진단 완료

정상 111개 영상의 5개 분할(장면×fold 20개)에서 fit/validation/reference/threshold/evaluation 중복이 없고 각 영상이 정확히 한 번 평가됨을 확인했습니다. 파일 간 원본 recording 독립성은 아직 미확인입니다. 40개 영상의 모델 예측을 숨긴 로컬 프레임 뷰어와 접촉시트를 준비했습니다. **실제 cycle 주석은 0개이며 위치 정확도는 미측정**입니다.

| 장면 | Reference FPR % | Threshold FPR % | Normal eval FPR % | Historical normal FPR % | Eval 영상 평균 FPR % |
| --- | --- | --- | --- | --- | --- |
| R01 | 0.00 | 0.89 | 0.00 | 47.92 | 0.00 |
| R02 | 1.51 | 0.76 | 0.97 | 1.18 | 0.96 |
| R03 | 2.07 | 0.94 | 7.12 | 4.28 | 7.29 |
| R04 | 0.00 | 0.90 | 1.54 | 4.08 | 1.55 |

표는 fold 0의 C-00 Full, 각 분할의 정상 프레임 기준입니다. Reference/threshold는 보정에 사용했으므로 일반화 성능이 아닙니다. Historical은 기존 테스트의 정상 라벨 구간입니다. 정상 q99라도 새로운 정상 영상에서 1% FPR을 보장하지 않습니다.

![정상 점수 분포](figures/E6_normal_ecdf.svg)

![영상별 오경보](figures/E6_video_fpr.svg)

[E6 수치](../results/E6/calibration_summary.json) · [분할 검사](../results/E6/fold_audit.json) · [주석 자료 상태](../results/E6/annotation_packet.json)

E8: Confidence 3×3 × readout 3종의 5-fold 실험 진행 중입니다.

E7: T0/T1/T2 인과적 예측을 로컬에 저장했으며 독립적인 실제 cycle/anchor 주석이 필요합니다. E9: 사전 계획에 따라 E7 진단 후 진행합니다. E10: 신규 독립 촬영 자료가 필요합니다. 미실행 항목을 완료로 표시하지 않습니다.

