# 02. 오늘 런북 — Python + CPU 로 YOLO11n-pose 1차 판정

질문: **AI 가속기 없이 Python + CPU 만으로 YOLO11n-pose 가 통과 기준을 만족하는가?**
**오늘 범위: 쿨러 없이 시나리오 A + B(짧게)**. 시나리오 C(30분)는 액티브 쿨러 장착 후 별도 세션으로 측정.
예상 소요: 약 40분~1.5시간 (대부분 자동 실행. 쿨러가 없어 실행 사이 50°C 냉각 대기가 길어질 수 있음).

오늘의 Pi 환경 (2026-11-12 확인): 사용자 `<pi-user>`, 접속 `ssh -4 <pi-user>@pd-dog.local`, 카메라 `imx708`(일반형, 오늘 미사용),
throttled=0x0, 유휴 42°C, 메모리 4GB 중 213MB 사용, **액티브 쿨러 없음**.

## 0. 노트북 (이미 완료된 상태로 레포에 포함)

```powershell
python -m venv .venv-convert
.venv-convert\Scripts\python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv-convert\Scripts\python -m pip install -r models\convert\requirements-convert.txt
.venv-convert\Scripts\python models\convert\export_yolo_pose.py     # → models/yolo11n-pose_{640,416,320}_fp32.onnx
.venv-convert\Scripts\python data\fetch_data.py                     # → data/video/*.rgb, data/audio/*_16k.wav
```
- export 결과의 sha256·버전은 `models/yolo11n-pose_export.json` 에 기록됨
- 검증 완료: 우리 전처리/후처리 결과가 Ultralytics 공식 추론과 키포인트까지 일치(차이 0 px),
  Silero VAD 래퍼가 공식 `silero_vad` 패키지와 확률이 일치(차이 0)

그다음 [01-pi-setup.md](01-pi-setup.md) 6절의 '레포 옮기기'로 Pi 에 복사.

## 1. Pi: 준비 확인 (5분)

```bash
cd ~/pd-dog-edge-bench
source ~/venvs/pdbench/bin/activate
python models/fetch.py          # silero_vad OK, yolo11n-pose 3개 present 확인
python data/fetch_data.py       # 노트북에서 data 를 복사했으면 'exists' 만 출력
python -m pdbench env --cooler none | head -40   # throttled=0x0, camera_sensor=imx708, onnxruntime 버전, numpy 경로(/usr/lib/...) 확인
```

### 30초 동작 확인
```bash
python -m pdbench pose --res 320 --threads 4 --out results/_sanity --measure-frames 100
```
`[A] res=320 threads=4: xx FPS ...` 한 줄이 나오면 정상. (`results/_sanity` 는 지워도 됨)

## 2. Pi: 1차 측정 — A + B(VAD 만), 쿨러 없음 (완료)

```bash
sudo apt install -y tmux        # 처음 한 번
tmux new -s bench
source ~/venvs/pdbench/bin/activate && cd ~/pd-dog-edge-bench
python -m pdbench matrix AB --cooler none --note "쿨러 없음, 케이스 없음, 실내 __°C, 공식 27W 전원"
```
- `--cooler` 는 필수. `none` 이 env.json·각 summary.json·REPORT.md·세션 폴더 이름(`…_cooler-none`)에 기록됨
- **측정 중에는 VS Code Remote-SSH 를 끊어 두세요** (Pi 의 RAM·CPU 사용).
- `Ctrl+b` → `d` 로 빠져나와도 계속 실행됨. 다시 보기: `tmux attach -t bench`
- 순서: A(12조합, 각 ≤45초) → B(6조합 × 1분)
- 각 실행 전 CPU 가 **50°C 이하로 식을 때까지 대기** (최대 10분, 초과하면 그대로 시작하고 경고 표시)
- 실행 중 아래 상황이 생기면 그 결과에 **경고(⚠)** 가 붙음. 콘솔에는 `WARN:` 줄로 출력되고 결과는 그대로 남음
  - 최고 온도 **80°C 초과**
  - `throttled` 값이 **0x0 이 아님**. 이전 실행에서 남은 값이면 "측정 전부터 남아 있던 값"으로 구분
  - 시작 온도가 50°C 를 넘음 (냉각 대기 시간 초과)
- 중간에 끊겼으면 같은 세션 폴더에 이어서: `python -m pdbench matrix B --cooler none --session results/<세션폴더>`

> 1차 측정 결과: `results/20261112-0412_pd-dog_python_cooler-none/REPORT.md`
> (4코어를 모두 쓰는 실행 9개에서 저전압 감지 → 전원 점검 후 재측정)

## 2-2. 다음 측정 — 전원 점검 후 A + V + B(포즈 2스레드·FPS 상한 포함), 쿨러 없음

### (1) 전원 점검
```bash
vcgencmd pmic_read_adc EXT5V_V      # 5V 입력 전압. 정상 ≈ 5.0~5.2 V, 4.8 V 아래로 떨어지면 저전압
vcgencmd get_throttled              # 재부팅 직후 0x0 이어야 함 (sticky 비트는 재부팅 시 초기화)
```
- 공식 27W(5V 5A) 어댑터인지, 케이블이 어댑터 동봉품(굵고 짧은 것)인지 확인
- **측정 전에 재부팅** → 1차 측정에서 남은 sticky 비트(0xf0000)를 지우고 시작
- 이번 버전부터 모니터가 1초마다 **실제 ARM 클럭**(`vcgencmd measure_clock arm`)과 **입력 전압(EXT5V)** 을 기록
  (1차 측정의 클럭 2400 MHz 는 OS 요청값이라 펌웨어 스로틀이 반영되지 않았음)

### (2) 파일 갱신 (노트북 → Pi)
[01-pi-setup.md](01-pi-setup.md) '레포 옮기기'의 1)·2)·4) 다시 실행 (코드, ECAPA 모델, 음성 세트). 그다음 Pi 에서:
```bash
source ~/venvs/pdbench/bin/activate && cd ~/pd-dog-edge-bench
pip install -r python/requirements-pi.txt     # vosk 추가됨
python models/fetch.py                         # Vosk 한국어 모델 받기 + 압축 해제, ECAPA present 확인
sha256sum -c transfer.sha256
python -m pdbench voice --duration 30 --out results/_sanity_voice   # 30초 동작 확인 (판정 줄이 나오면 정상)
```

### (3) 측정 (tmux 안에서)
```bash
python -m pdbench matrix AVB --cooler none --note "쿨러 없음, 전원 점검 후(어댑터/케이블: ____), 실내 __°C"
```
- 순서: **A**(12조합) → **V**(음성 체인 단독 2.5분) → **B**(해상도 3 × 포즈 스레드 2/3/4 = 9조합 × 2.5분, 음성 체인 전체)
  → **FPS 상한**(10 FPS × 해상도 416/320 × 스레드 2/3 = 4조합 × 2.5분)
- 예상 소요: 약 2~3시간 (쿨러가 없어 실행 사이 냉각 대기가 대부분)
- 음성 세트 기본값은 TTS 합성 세트(지연 측정용). 실제 목소리 정확도는 [04-recording.md](04-recording.md) 로 녹음한 세트를
  `--voice-set data/voice/my_voice` 로 지정해 측정
- B 는 이제 VAD 만이 아니라 **Vosk 명령 인식 + ECAPA 화자 확인까지** 실행 → '발화 종료 → 명령 판정 ≤ 1초' 를 끝까지 판정
- A 를 건너뛰려면 `matrix VB` (단, 전원 점검 후 A 4스레드도 다시 재는 것을 권장)
- 음성 체인 설명·한계: [03-voice.md](03-voice.md)

## 2-1. (나중에) 액티브 쿨러 장착 후

```bash
cat /sys/devices/platform/cooling_fan/hwmon/*/fan1_input      # 팬 RPM 이 나오면 인식됨
python -m pdbench matrix all --cooler active --note "액티브 쿨러, 케이스 없음, 실내 __°C"
```
- **새 세션**(`…_cooler-active`)으로 A→V→B→C 전체 측정 → 쿨러 없는 결과와 표끼리 비교
- C 에서 실제 운용 설정(예: 10 FPS 상한)으로 오래 돌리려면 `configs/scenarios.toml [C] fps_cap = 10`
- C 설정은 같은 세션의 B 결과에서 자동 선택: 동시 실행 5 FPS 이상인 가장 높은 해상도, 그중 가장 빠른 스레드 수
- 쿨러 없는 세션에 `--cooler active` 로 이어 붙이려 하면 오류로 막힘 (결과 섞임 방지)
- `--cooler` 지정값과 실제 팬 감지 결과가 다르면 env.json 에 `cooler_warning` 이 기록되고 시작 시 경고가 출력됨

## 3. 결과 정리

```bash
python bench/summarize.py results/<세션폴더>      # → results/<세션폴더>/REPORT.md
```
노트북으로 가져오기 (PowerShell):
```powershell
scp -4 -r <pi-user>@pd-dog.local:pd-dog-edge-bench/results/<세션폴더> results\
```

## 결과 폴더 구조

```
results/<YYYYMMDD-HHMM>_<host>_python_cooler-<none|active>/
├── env.json            # cooler(none/active)·OS·커널·펌웨어·라이브러리·모델 sha256·카메라·메모
├── REPORT.md           # 판정 요약 + A/B/C 표
├── A_r640_t1/ … A_r320_t4/
│   ├── summary.json    # 지연 p50/p95, FPS, 최대 RSS, CPU, 온도, cooler, flags(경고)
│   ├── frames.csv      # 프레임별 read/pre/infer/post/total ms   (git 제외)
│   ├── system.csv      # 1초 간격 CPU·RSS·온도·클럭·throttled·팬  (git 제외)
│   └── kpts.npy        # 프레임별 키포인트 (해상도 간 정확도 비교용) (git 제외)
├── B_r640_t3/ …        # + audio.csv(청크별 VAD 지연/lag), events.csv(발화 시작/종료 판정)
└── C_r…/               # + 5분 구간별 요약(summary.json 의 buckets/system_buckets)
```

## 지표 정의

| 지표 | 정의 |
|---|---|
| FPS | 측정 프레임 수 / 첫 프레임 시작~마지막 프레임 완료 시간. 포즈는 항상 최대 속도로 돌림 = **이 조건에서 낼 수 있는 처리 FPS** |
| 1초 구간 FPS p5 | 1초 단위로 자른 FPS 의 하위 5% — 순간 끊김 확인 |
| 추론 지연 | `session.run` 만 |
| 전체 지연 | 전처리(letterbox·정규화) + 추론 + 후처리(NMS·좌표 복원). raw 파일 읽기는 따로(read_ms) |
| 최대 메모리 | `/proc/self/status` 의 VmHWM (프로세스 최대 RSS) |
| CPU 사용률 | 외부 모니터(`tools/monitor.py`)가 1초 간격 측정. 프로세스 CPU 는 코어 합(최대 400%) |
| 스로틀링 | `vcgencmd get_throttled` 의 현재 상태 비트(저전압·클럭 제한·스로틀·소프트 온도 제한)가 한 번이라도 켜졌는지 |
| VAD lag | 청크 처리 완료 - 청크 도착(마지막 샘플이 녹음됐을 시각). 32ms 이상이면 실시간 지연 |
| 발화 종료 판정 지연 | 마지막 음성 샘플 녹음 시각 → VAD 'end' 판정. `min_silence_ms`(300ms) 대기 포함 |

## 오늘 측정의 한계 (보고서 해석 시)
- 입력이 UR Fall 데이터셋(실내, 1인, 640×480)이라 실제 카메라 모듈 3 영상과 장면이 다름 → 속도에는 영향이 작고, 키포인트 차이는 참고용
- 음성 기준(발화 종료 → 명령 판정 1초)은 VAD 까지만 측정 → '부분 측정'. ECAPA·Vosk 가 남은 예산 안에 들어가야 최종 통과
- 반복 1회, 실행 순서 고정 (계획서의 3회 반복·순서 섞기는 다음 단계)
- 포즈를 최대 속도로 돌리므로 CPU 사용률은 최악 조건. 실제 로봇에서 카메라 FPS 로 상한을 걸면 더 낮아짐
