# 04. 노트북 마이크로 명령어 녹음 → Pi 로 옮기기

USB 마이크가 생기기 전까지, 실제 목소리 정확도(명령 인식률·잘못 수락·화자 점수)는 이 방법으로 측정한다.
녹음은 노트북에서, 측정은 Pi 에서 같은 파일로 한다.

## 1. 준비 (노트북, 한 번)
```powershell
cd C:\Users\<you>\Desktop\pd-dog-edge-bench\pd-dog-edge-bench
.venv-convert\Scripts\python -m pip install sounddevice
.venv-convert\Scripts\python data\tools\record_commands.py --list-devices    # 마이크 확인 (번호 메모)
.venv-convert\Scripts\python data\tools\record_commands.py --test           # 3초 녹음해 음량 확인
```
- 음량: peak −20 ~ −3 dBFS 가 적당. "너무 작음"이면 Windows 설정 > 시스템 > 소리 > 입력 볼륨을 올림
- 조용한 방, 노트북에서 30~50cm. 로봇 사용 거리와 비슷하게 하려면 1m 도 한 번 녹음해 볼 것
- 기본 마이크가 아니면 `--device <번호>` 추가

## 2. 녹음 (노트북)
```powershell
# 등록 화자(로봇 사용자): 등록 문장 5개 + 명령 6종 × 5회 + 명령 아닌 말 10개  ≈ 10분
.venv-convert\Scripts\python data\tools\record_commands.py data\voice\my_voice --speaker target --enroll 5 --reps 5

# 다른 사람 (가족 등, 화자 거부 확인용): 명령 6종 × 3회 + 명령 아닌 말
.venv-convert\Scripts\python data\tools\record_commands.py data\voice\my_voice --speaker guest1 --reps 3
```
- 마이크 기본 샘플레이트(이 노트북은 44.1 kHz)로 녹음 → **16 kHz mono 16-bit wav 로 변환해 저장**
- 각 항목: Enter → 말하기(명령 2.5초, 등록 문장 5초) → 음량 표시 → Enter=저장 / r=다시
- 명령어: 출발 · 걷자 · 멈춰 · 앉아 · 일어나 · 도와줘 (`configs/voice.toml [commands]`)
- 명령 아닌 말 뒤쪽 5개(더워, 도라지, 돈 좀 줘, 도와줄까, 도와준대)는 '도와줘'와 비슷한 말 → 거짓 위급 알림 확인용
- 평소 말투로. 피곤할 때·작게 말할 때도 몇 개 섞으면 실제 사용에 가까움

## 3. 스트림 만들기 (노트북)
```powershell
.venv-convert\Scripts\python data\tools\build_command_stream.py data\voice\my_voice
```
→ `stream.wav`(발화 사이 무음을 넣어 이어 붙인 실시간 투입용 음성) + `stream_labels.csv`(정답 발화 구간). 출력된 길이(초)를 메모.

## 4. Pi 로 옮기기 (노트북 PowerShell)
```powershell
$PI = "<pi-user>@pd-dog.local"; $DST = "pd-dog-edge-bench"
ssh -4 $PI "mkdir -p $DST/data/voice"
scp -4 -r data/voice/my_voice "${PI}:${DST}/data/voice/"
```

## 5. 측정 (Pi)
```bash
source ~/venvs/pdbench/bin/activate && cd ~/pd-dog-edge-bench
# 음성 체인 단독. --duration 은 3. 에서 메모한 스트림 길이 + 10초 이상
python -m pdbench voice --voice-set data/voice/my_voice --duration 400 --cooler none --out results/<세션>/V_my_voice
# 또는 매트릭스 전체를 이 세트로
python -m pdbench matrix VB --cooler none --voice-set data/voice/my_voice --note "실제 녹음 세트"
python bench/summarize.py results/<세션폴더>
```
보고서의 **명령별 결과** 표에서 '출발' vs '걷자' 인식률·잘못 수락을 비교하고, **🚨 명령 아닌 말 → 도와줘** 건수를 확인.
`configs/scenarios.toml` 의 V/B `duration_s` 는 TTS 세트(143초) 기준 → 녹음 세트가 길면 matrix 전에 늘릴 것.

## 개인정보
녹음 wav(`data/voice/*/utts`, `enroll`, `stream.wav`)는 `.gitignore` 로 git 에 올라가지 않는다. 정답 라벨 csv 만 커밋됨.
