# 01. 라즈베리파이 5 세팅 (최소 절차)

목표: 노트북에서 `ssh -4 <pi-user>@pd-dog.local`로 접속해 벤치마크를 돌릴 수 있는 상태까지.
소요: 약 30~40분 (대부분 다운로드/업데이트 대기).

## 준비물
- Raspberry Pi 5 (4GB). 액티브 쿨러는 있으면 장착 (없이도 측정 가능, 결과에 `cooler: none` 으로 구분 기록)
- microSD 32GB 이상 (A2 등급 권장)
- **공식 27W(5V 5A) USB-C 전원** — 일반 충전기는 저전압 경고(throttled 비트)로 측정값이 왜곡됨
- 카메라 모듈 3 (오늘 벤치마크엔 안 쓰지만 연결 확인만 함)

---

## 1. 노트북: SSH 키 만들기 (PowerShell)

```powershell
ssh-keygen -t ed25519          # 엔터 3번 (기본 경로, 암호 없음 가능)
Get-Content $env:USERPROFILE\.ssh\id_ed25519.pub   # 이 한 줄을 복사해 둠
```

## 2. 노트북: Raspberry Pi Imager로 OS 굽기

1. https://www.raspberrypi.com/software/ 에서 Imager 설치 후 실행
2. **장치**: Raspberry Pi 5
3. **OS**: Raspberry Pi OS (other) → **Raspberry Pi OS Lite (64-bit)**
   - 데스크톱 없음 → RAM 여유, 측정 잡음 감소. 모든 작업은 SSH로 함
4. **저장소**: microSD 선택
5. **OS 커스터마이징(설정 편집)** — 반드시 입력:
   | 항목 | 값 |
   |---|---|
   | 호스트명 | `pd-dog` |
   | 사용자 이름 / 비밀번호 | `<pi-user>` / (임의) |
   | Wi-Fi SSID / 비밀번호 | 노트북과 **같은 공유기** (2.4GHz/5GHz 무관) |
   | Wi-Fi 국가 | **KR** |
   | 시간대 / 키보드 | Asia/Seoul / us |
   | SSH | 사용 → **공개키 인증만 허용** → 1번에서 복사한 키 붙여넣기 |
6. 쓰기 → 완료 후 SD를 Pi에 꽂고 전원 연결. 첫 부팅은 1~2분 걸림(자동 재부팅 1회).

## 3. 노트북에서 접속

```powershell
ssh -4 <pi-user>@pd-dog.local    # -4: IPv4 강제 (mDNS 가 IPv6 주소를 줘서 접속이 안 될 때 필요)
```
- `Could not resolve hostname` → 공유기 관리 페이지의 DHCP 목록에서 `pd-dog`의 IP를 찾아 `ssh -4 <pi-user>@192.168.x.x`
- 이후 편의를 위해 노트북 `~/.ssh/config`에 추가:
  ```
  Host pd-dog
      HostName pd-dog.local
      User <pi-user>
      AddressFamily inet          # = -4. 이렇게 두면 `ssh pd-dog` / `scp ... pd-dog:...` 만으로 접속
  ```

## 4. Pi: 업데이트 & 기본 패키지

```bash
sudo apt update && sudo apt full-upgrade -y
sudo rpi-eeprom-update -a
sudo apt install -y git python3-venv python3-dev python3-numpy python3-opencv \
                    python3-psutil htop
sudo reboot
```

## 5. Pi: 하드웨어 상태 확인 (재접속 후)

```bash
vcgencmd get_throttled         # 반드시 throttled=0x0 이어야 함 (0x50000/0x50005 등 → 전원 문제)
vcgencmd measure_temp          # 유휴 시 보통 40~50°C
cat /sys/devices/platform/cooling_fan/hwmon/*/fan1_input   # 팬 RPM (숫자가 나오면 쿨러 인식됨)
free -h                        # 총 메모리 ~4GB 확인
```
`get_throttled`가 0이 아니면 **측정 전에 전원부터 해결**하세요.

### 카메라 종류 확인 (일반형/Wide형)
```bash
rpicam-hello --list-cameras
```
| 출력 센서명 | 의미 |
|---|---|
| `imx708` | 카메라 모듈 3 일반형 (수평 화각 약 66°) |
| `imx708_wide` | Wide형 (약 102°) |
| `imx708_noir` / `imx708_wide_noir` | 적외선 필터 없는 버전 |

`No cameras available!` → 전원을 끄고 FPC 케이블 방향(Pi 5는 22핀 소형 커넥터, 금속 접점 방향)을 확인.
이 결과는 벤치마크 실행 시 `env.json`에 자동 기록됩니다.

## 6. Pi: 레포와 Python 환경

먼저 레포를 Pi 로 옮긴다 (아래 '레포 옮기기'). 그다음 Pi 에서:

```bash
cd ~/pd-dog-edge-bench
python3 -m venv --system-site-packages ~/venvs/pdbench
source ~/venvs/pdbench/bin/activate
pip install -r python/requirements-pi.txt
pip install -e python --no-deps

python -c "import numpy, cv2, onnxruntime as ort; print(numpy.__file__); print(ort.__version__)"
```
- `numpy.__file__`이 **`/usr/lib/python3/dist-packages/...`** 이어야 정상 (apt 버전 사용).
  `~/venvs/...` 아래라면 pip가 numpy를 따로 깔아 버린 것 → `pip uninstall numpy`.
- 매 접속마다 `source ~/venvs/pdbench/bin/activate` 필요 (`~/.bashrc`에 추가해도 됨).

### 레포 옮기기 (원격 저장소 없이, 노트북 PowerShell 에서)
옮기는 것: 코드 + ONNX 5개(+ECAPA 특징 상수) + 변환된 데이터(.rgb/.json, 16k wav, 음성 명령 세트) + 체크섬 파일. 합계 약 430MB.
Vosk 한국어 모델(87MB)은 Pi 에서 `python models/fetch.py` 로 직접 받는다.
옮기지 않는 것: `.venv-convert`(수 GB), 원본 zip/wav(변환본이 있으면 불필요), `models/convert/_work`(.pt)

```powershell
cd C:\Users\<you>\Desktop\pd-dog-edge-bench\pd-dog-edge-bench
$PI = "<pi-user>@pd-dog.local"
$DST = "pd-dog-edge-bench"     # Pi 홈 폴더 기준 상대경로 (= ~/pd-dog-edge-bench)

# 0) 폴더 만들기
ssh -4 $PI "mkdir -p $DST/models/convert $DST/data/video $DST/data/audio"

# 1) 코드: .git 만 복사하고 Pi 에서 커밋된 그대로 풀기
#    → 추적 파일이 빠짐없이 커밋과 같아져 결과 env.json 의 git_dirty=false (커밋하지 않은 변경은 옮겨지지 않음)
#    → 모델·데이터·results 의 미추적 파일은 그대로 둠
scp -4 -r .git transfer.sha256 "${PI}:${DST}/"
ssh -4 $PI "cd $DST && git reset -q --hard HEAD && git log -1 --format='%h %s' && git status --short"

# 2) 모델: 스크립트·메타 + ONNX 5개 + ECAPA 특징 상수
scp -4 models/manifest.toml models/fetch.py models/yolo11n-pose_export.json models/ecapa_export.json models/yolo11n-pose_640_fp32.onnx models/yolo11n-pose_416_fp32.onnx models/yolo11n-pose_320_fp32.onnx models/silero_vad.onnx models/ecapa_tdnn_voxceleb_fp32.onnx models/ecapa_fbank.npz "${PI}:${DST}/models/"
scp -4 models/convert/export_yolo_pose.py models/convert/export_ecapa.py models/convert/requirements-convert.txt "${PI}:${DST}/models/convert/"

# 3) 데이터: 도구 + 변환본
scp -4 -r data/fetch_data.py data/tools "${PI}:${DST}/data/"
scp -4 data/video/urfd_fall-01-cam0.rgb data/video/urfd_fall-01-cam0.json data/video/urfd_adl-01-cam0.rgb data/video/urfd_adl-01-cam0.json "${PI}:${DST}/data/video/"
scp -4 data/audio/silero_test_16k.wav "${PI}:${DST}/data/audio/"

# 4) 음성 명령 세트 (노트북에서 만든 TTS 세트: enroll/, utts/, stream.wav, stream_labels.csv)
ssh -4 $PI "mkdir -p $DST/data/voice"
scp -4 -r data/voice/tts_heami "${PI}:${DST}/data/voice/"
```

Pi 에서 확인:
```bash
cd ~/pd-dog-edge-bench && sha256sum -c transfer.sha256    # 모든 줄 OK
```
- `transfer.sha256` 은 노트북에서 만든 체크섬 목록(git 제외 파일)
- 옮긴 뒤 노트북에서 코드를 고치면 **커밋한 다음** 1)만 다시 실행 (마지막 `git status --short` 가 아무것도 출력하지 않으면 깨끗함)
- GitHub 원격 저장소를 쓸 경우: 코드는 `git clone`, 2)의 ONNX 와 3)의 데이터 변환본만 scp

## 7. 측정 → [02-today-runbook.md](02-today-runbook.md) 로 이어서 진행

---

## 측정할 때 지킬 것
- 측정 중에는 **VS Code Remote-SSH 연결을 끊고** 일반 `ssh` 터미널만 사용 (Remote 서버가 RAM 수백 MB·CPU 사용)
- 장시간 측정은 `tmux` 안에서 실행해 SSH가 끊겨도 계속 돌게 함 (`sudo apt install -y tmux`)
- 쿨러 상태는 `--cooler none|active` 로 반드시 지정 (env.json·결과·세션 폴더 이름에 기록). 케이스 상태, 실내 온도는 `--note` 로 기록
