# pd-dog-edge-bench

Hiwonder MechDog(ESP32) + Raspberry Pi 5 기반 파킨슨병 재활돌봄로봇을 위한 엣지 AI 벤치마크.
**AI 가속기 없이 Pi 5(4GB) CPU 만으로 실사용 기준을 만족하는지** 수치로 검증하고,
모델 선정과 언어(Python vs Rust) 선택의 근거를 만든다.

## 현재 범위 (1차)
- Python + onnxruntime(CPU), **YOLO11n-pose** FP32 (640/416/320)
- 음성 명령 체인: **Silero VAD → Vosk 한국어 small(명령 6개: 출발·걷자·멈춰·앉아·일어나·도와줘) + ECAPA-TDNN(화자 확인)** — [docs/03-voice.md](docs/03-voice.md)
- 시나리오 A(포즈 단독, 해상도 × 스레드), V(음성 체인 단독), B(포즈 + 음성 동시, 포즈 FPS 상한 비교 포함),
  C(B 를 30분, 액티브 쿨러 장착 후)
- 냉각 상태(`--cooler none|active`)를 모든 결과에 기록해 쿨러 유무별로 구분 비교
- 이후: Rust, INT8, 프레임 건너뛰기, 실시간 카메라/마이크, MediaPipe 참고

## 시작하기
1. [docs/01-pi-setup.md](docs/01-pi-setup.md) — OS 설치부터 SSH 접속, Python 환경까지
2. [docs/02-today-runbook.md](docs/02-today-runbook.md) — 모델/데이터 준비, 측정, 결과 정리
3. [docs/03-voice.md](docs/03-voice.md) — 음성 명령 체인 구조, 명령별 판정 기준, 지연 정의, 알려진 한계
4. [docs/04-recording.md](docs/04-recording.md) — 노트북 마이크로 명령어 녹음 → 16kHz 변환 → Pi 로 옮기기

## 구조
```
configs/   두 언어가 공유하는 설정 (pose/vad/scenarios/criteria .toml)
models/    manifest(출처·sha256), fetch.py(Pi), convert/(노트북 전용 export, PyTorch 는 여기서만)
data/      fetch_data.py + tools/(영상→.rgb 무압축, wav→16k mono). 원본·변환 파일은 git 제외
python/    pdbench 패키지 (common / sources / models / scenarios, CLI: python -m pdbench)
tools/     monitor.py — 언어 무관 시스템 모니터 (CPU·RSS·온도·클럭·throttled·팬)
bench/     summarize.py — 세션 결과 → REPORT.md (표 + 통과 기준 판정)
results/   측정 세션별 결과 (summary.json·REPORT.md·env.json 만 커밋)
rust/      (예정)
```

## 공정 비교 원칙
- 같은 ONNX 파일(sha256 기록), 같은 입력 파일(무압축 RGB 프레임 / 16kHz wav), 같은 설정 파일
- 실행 시점에는 onnxruntime 만 사용 (PyTorch 는 변환 단계에서만)
- 시스템 지표는 외부 모니터 프로세스가 같은 정의로 측정
- 측정 환경(OS·라이브러리·카메라·쿨러·메모)을 결과와 함께 `env.json` 에 기록

## 데이터·모델 라이선스
- YOLO11n-pose: Ultralytics, AGPL-3.0 — 제품화 시 검토 필요
- Silero VAD: MIT
- ECAPA-TDNN (SpeechBrain spkrec-ecapa-voxceleb): Apache-2.0
- Vosk vosk-model-small-ko-0.22: Apache-2.0
- UR Fall Detection Dataset: 연구 목적 공개 (B. Kwolek, M. Kepski, 2014). 레포에 포함하지 않음
