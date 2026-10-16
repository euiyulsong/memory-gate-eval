# Memory search binary gate — GPT-6 Luna

메모리 검색 필요 100개 + 불필요 100개, 총 200개. 실제 모델 실행은 API 키 부재로 수행하지 않았습니다. 포함된 output/cases.jsonl은 실제 공개 데이터에서 seed=42로 추출한 고정 평가셋입니다. 원본 데이터는 zip에 포함하지 않습니다.

## 실행
```bash
cd memory_gate_eval
python -m pip install -r requirements.txt
export OPENAI_API_KEY='본인 키'
python eval.py run
```
기본 모델 gpt-6-luna, reasoning effort none, 동시성 4. --model, --workers로 변경 가능. 모델 출력은 0 또는 1 한 글자만 지시합니다. 엄격한 검사에서 다른 문자열은 invalid 처리하고 정확도에서는 오답 처리합니다. JSON이나 regex로 숫자를 추출해 보정하지 않습니다.

## 라벨 정의
1: 현재 보이는 문맥만으로 부족하며, 이전 대화에 저장된 사실/선호/사건 검색이 필요.
0: 현재 제공된 문맥 또는 일반 지식/추론으로 처리 가능.

- Positive: LongMemEval의 answer_session_ids가 있는 과거 대화 질문. 과거 대화 원문은 분류기에 전달하지 않음. abstention은 제외(검색이 필요 없다는 의미가 아니므로 음성에 넣지 않음).
- Negative: SQuAD v1.1 dev의 질문과 정답이 포함된 지문. 지문을 현재 문맥으로 제공.
- 분류기에는 출처, 정답, 라벨, annotation을 전달하지 않음.
- 라벨은 본 실험의 규칙으로 파생한 것으로 원 데이터의 memory-gate 정답 라벨이 아님. 수동 검수는 수행하지 않음.

## 결과
output/predictions.jsonl: 원문, 예측, raw 출력, 오류, 호출 ID, 실제 모델명, usage, 지연시간.
output/summary.json: 전체/출처별 정확도, precision/recall/F1, TP/TN/FP/FN, invalid, 불필요 검색률(FP/음성 수), 검색 누락률(FN/양성 수), latency mean/p95, always/never search 기준선.
invalid는 별도 집계. 양성 invalid는 recall 분모에 포함되지만 FN에는 포함하지 않으므로 missed_search_rate와 positive_invalid를 함께 확인할 것. API 오류는 invalid로 집계하며 지연시간 통계에서 제외. SDK 재시도 성공 시 지연시간은 재시도 포함.

## 해석 한계
이것은 1차 smoke benchmark. 출처와 라벨이 일치하고 지문 유무/길이가 다르므로 shortcut 가능. 높은 점수만으로 실제 개인화 질의의 검색 판단 성능을 주장할 수 없음. 특히 한국어, 최근 대화에 이미 답이 있는 개인 질문, 개인화가 선택적인 추천, 단순 지시/인사 등 hard negative는 별도 수동 라벨 세트가 필요. 본 평가는 retrieval 여부만 평가하며 retrieval 품질/최종 답변 EM은 측정하지 않음. 항상 검색/항상 미검색 정확도는 각 50%.

동일 평가셋을 반복 튜닝에 사용하지 말고 최종 검증용 새 샘플을 별도로 확보할 것. 100개 클래스당 비율의 불확실성도 고려할 것.

## 재생성
python eval.py prepare --out new_output --seed 42
다운로드 URL과 실제 원본 SHA256은 output/manifest.json에 기록. 기존 raw 파일은 재사용. 원본 main 변경 시 hash와 샘플이 달라질 수 있으므로 포함된 고정 cases.jsonl을 우선 사용할 것.

## 출처
https://github.com/xiaowu0162/LongMemEval
https://huggingface.co/datasets/xiaowu0162/longmemeval
https://github.com/rajpurkar/SQuAD-explorer
https://developers.openai.com/api/docs/models/gpt-6-luna
공개 데이터의 이용/재배포 조건은 원 출처를 따릅니다.

## v2: 싱글턴 / 멀티턴 모두 포함 (이 절이 위의 v1 설명보다 우선)
총 400개: single/1 100, single/0 100, multi/1 100, multi/0 100.
싱글턴은 기존 구성(LongMemEval 질문 / SQuAD 질문+지문).
멀티턴은 LongMemEval의 동일한 질문 100개를 두 조건으로 짝지어 구성:
- multi/0: 해당 질문의 전체 oracle 세션 대화를 visible recent_messages로 제공. 정답 근거가 현재 문맥에 있으므로 외부 메모리 검색 불필요.
- multi/1: 다른 질문의 세션 대화를 제공. target의 근거 세션과 세션 ID가 겹치지 않도록 검사. target 근거는 외부 메모리에만 있다는 가정으로 검색 필요.
원본 공개 대화 텍스트를 그대로 사용하며 세션 날짜를 덧붙임. question_date도 현재 문맥에 제공.
멀티턴 길이는 원본 세션 길이를 따르며 고정 5턴이 아님. 여러 oracle 세션이 있으면 이어서 제공. 마지막 질문에 대한 0/1만 평가.
이 조합은 원본에 있던 자연스러운 연속 대화가 아니라 평가용으로 재구성한 문맥 조건임. multi/1에서 다른 세션의 우연한 답변 가능성은 수동 검수하지 않았음. multi/0의 전체 근거 노출은 recent-history 충분성 조건이며 원래 LongMemEval QA 평가와 다름. 두 모드의 난이도/문맥 분포가 달라 단순 차이를 순수 멀티턴 효과로 해석할 수 없음.
summary.json의 by_mode에 single/multi 성능, by_condition에 네 조합별 정확도/invalid/지연시간을 저장. 클래스 하나뿐인 그룹의 precision/F1 등은 비교 목적에 적합하지 않으며 정확도와 오류율을 확인할 것.
기존 데이터 원본 2개는 zip에서 제외하고 고정 400개와 manifest를 포함.

## 실험 결과 (2026-10-16)
gpt-6-luna, reasoning effort none, 400건 평가. `output/summary.json`에 전체 metric이 있고 로컬의 `output/predictions.jsonl`에는 개별 API 응답이 저장됩니다(요청 원문/모델 응답을 포함하므로 공개 저장소에는 올리지 않음).

| 모드 | 엄격 정확도 | Precision | Recall | Invalid |
|---|---:|---:|---:|---:|
| 전체 (400) | 65.0% | 79.3% | 95.5% | 81 (20.25%) |
| single (200) | 68.0% | 88.2% | 97.0% | 48 |
| multi (200) | 62.0% | 71.8% | 94.0% | 33 |

81건의 invalid 응답은 모두 음성(검색 불필요) 사례에서 발생했습니다. strict accuracy는 invalid를 오답 처리합니다. 출처와 라벨/문맥이 confounded된 smoke benchmark이므로 이 결과를 실제 사용자 질의 성능으로 일반화하면 안 됩니다.
