# 03. 음성 명령 체인 — 발화 종료 → 명령 판정

기준: **발화가 끝난 뒤 1초 이내에 명령을 판정** (등록 화자 + 명령어).

## 명령어 (configs/voice.toml `[commands.<키>]`)

| 키 | 말 | 비고 |
|---|---|---|
| chulbal | 출발 | '가자' 대체 후보 1 |
| geotja | 걷자 | '가자' 대체 후보 2 — 실제 녹음으로 출발과 비교 후 하나로 확정 |
| meomchwo | 멈춰 | |
| anja | 앉아 | |
| ireona | 일어나 | |
| dowajwo | 도와줘 | 🚨 위급 명령 — 더 엄격한 규칙 (아래) |

'가자'는 Vosk small-ko 어휘에 없어 제외했다 ('가 자' 두 단어 우회는 TTS 5개 중 0개 인식).

### 명령별 판정 기준
| | 기본 (도와줘 외) | 도와줘 |
|---|---|---|
| 명령 단어 신뢰도 | ≥ 0.7 | **≥ 0.9** |
| 등록 화자 코사인 | ≥ 0.25 | **≥ 0.35** |
| 다른 단어 섞임 | 허용 ("오늘 멈춰" → 멈춰) | **거부** ("자 도와줘" → 거부, "도와줘 도와줘" 는 허용) |

잘못된 '도와줘' = 거짓 위급 알림이므로 놓치는 위험을 일부 감수하고 엄격하게 둔다. 보고서에는
명령별 인식률·놓침(이유)·잘못 수락과 **"명령 아닌 말 → 도와줘"** 건수가 따로 나오고,
엄격한 규칙 때문에 진짜 '도와줘'를 놓친 건수(엄격함의 비용)도 함께 표시된다.

## 구조

```
마이크(지금은 wav 를 실시간 속도로 투입)
  │ 32ms 청크
  ▼
[audio 스레드] Silero VAD ── 발화 중 청크 ──▶ 큐 ──▶ [speech 스레드]
               (무거운 일 안 함)                       Vosk 에 계속 흘려 넣음 (스트리밍)
                    │ 'end' (300ms 무음)                 │ 'end' 수신 시
                    └──────────────────────────────────▶ Vosk 최종 결과 → 명령어
                                                         ECAPA 임베딩 → 등록 화자와 코사인 (명령별 기준)
                                                         → 판정 (명령 && 화자)
```
- 모델: Silero VAD v5 (ONNX), Vosk `vosk-model-small-ko-0.22` (Kaldi, 문법 제한),
  ECAPA-TDNN `speechbrain/spkrec-ecapa-voxceleb` (ONNX, 192차원)
- 설정: [configs/voice.toml](../configs/voice.toml)
- 코드: [python/pdbench/scenarios/voice_chain.py](../python/pdbench/scenarios/voice_chain.py)

## 지연 정의

| 항목 | 정의 |
|---|---|
| **발화→판정 (e2e)** | 판정 시각 − 정답 발화 끝이 녹음된 시각. **기준 판정에 사용 (명령 발화 전부의 최대값 ≤ 1000 ms)** |
| VAD 종료 판정 | VAD 'end' 시각 − 정답 발화 끝 (min_silence 300ms 대기가 대부분) |
| 큐 대기 | VAD 'end' → speech 스레드가 처리 시작. 커지면 speech 스레드가 밀리는 것 |
| Vosk 최종 / ECAPA | 각 단계 처리 시간 (ECAPA = 특징 추출 + 임베딩) |

정답 발화 끝 = `data/tools/build_command_stream.py` 가 에너지 기준(최대 −35 dB, 최소 −55 dBFS)으로 정한 마지막 음성 프레임 끝.

## ECAPA 변환 (노트북)

```powershell
.venv-convert\Scripts\python -m pip install torchaudio --index-url https://download.pytorch.org/whl/cpu
.venv-convert\Scripts\python -m pip install speechbrain
.venv-convert\Scripts\python models\convert\export_ecapa.py
```
- 신경망만 ONNX 로 내보내고, 특징 추출(STFT·멜·log·평균 정규화)은 numpy 로 구현 (Rust 도 같은 방식 예정)
- 멜 필터 행렬 등 상수는 SpeechBrain 에서 그대로 뽑아 `models/ecapa_fbank.npz` 에 저장 → 두 언어가 같은 상수 사용
- 검증: numpy 특징 + ONNX 결과가 SpeechBrain 원본과 **코사인 1.000000, 최대 오차 7e-4** (길이 다른 입력 7개)

## 테스트 음성 세트

| 세트 | 만드는 법 | 용도 |
|---|---|---|
| `tts_heami` (현재) | 노트북 Windows TTS: `powershell -ExecutionPolicy Bypass -File data\tools\make_tts_commands.ps1` | **지연 측정용**. 등록·시험 화자가 같은 합성 음성이라 화자 정확도는 의미 없음 |
| 직접 녹음 `my_voice` | 노트북 마이크: [04-recording.md](04-recording.md) | **정확도 측정용** (명령 인식률, 잘못 수락, 화자 점수) |

어느 세트든 마지막에 `python data/tools/build_command_stream.py data/voice/<세트>` → `stream.wav` + `stream_labels.csv`.
측정할 세트는 `configs/voice.toml` 의 `[pipeline] dataset` 으로 지정.

`tts_heami` 구성: 등록 문장 5, 명령 6종 × 말하기 속도 5단계 = 30, 등록 화자의 명령 아닌 말 8
(그중 4개는 '도와줘'와 비슷한 말: 더워·도라지·돈 좀 줘·도와줄까), 다른 화자(영어 TTS) 5 → 43 발화, 143초.
측정할 세트는 실행 시 `--voice-set data/voice/<세트>` 로 고른다 (기본 `configs/voice.toml [pipeline] dataset`).

## 알려진 한계 (모델 선정에 영향)

1. **'가자' 가 Vosk small-ko 어휘에 없음** → '출발'·'걷자' 두 후보로 교체, 실제 녹음으로 비교 후 확정.
2. **small-ko 모델에 `[unk]`(미등록어) 가 없음** → 명령 아닌 말도 가장 가까운 명령으로 끼워 맞춰짐 (예: "Stop" → "앉아").
   완화: 명령과 발음이 겹치지 않는 일반 단어 17개를 문법에 섞어 흡수 + 명령 단어 신뢰도 ≥ 0.7.
   '도와줄까' 는 처음에 '도와줘'(신뢰도 1.0)로 인식됐음 → 어휘에 있는 '도와줄'·'도와준'·'도와주' 를 문법에 넣어 흡수.
   TTS 세트 오프라인 시험(파일 단위): 명령 30개 모두 정답(출발 5/5, 걷자 5/5), 등록 화자의 명령 아닌 말 8개 모두 거부,
   영어 "Stop" 1개만 '멈춰' 로 인식 (다른 화자라 실제 체인에서는 화자 확인에서 거부됨).
   다만 스트리밍에서는 "오늘 뭐 먹지" → "오늘 멈춰" 처럼 잘못 수락되는 경우가 있음 → 실제 녹음으로 비율 확인 필요.
3. 화자 기준 0.25(도와줘 0.35)는 VoxCeleb 일반값 기반 — 파킨슨 환자의 작은 목소리·발음 변화에 맞춰 **실제 녹음으로 보정** 필요.
