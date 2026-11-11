"""[노트북] 노트북 마이크로 명령어 세트를 녹음 → 16kHz mono wav 로 저장 (실제 목소리로 정확도 측정용).

    pip install sounddevice                                   # 노트북 venv 에서 한 번
    python data/tools/record_commands.py --list-devices       # 마이크 장치 번호 확인
    python data/tools/record_commands.py data/voice/my_voice --test                        # 3초 녹음해 음량 확인
    python data/tools/record_commands.py data/voice/my_voice --speaker target --enroll 5 --reps 5
    python data/tools/record_commands.py data/voice/my_voice --speaker guest1 --reps 3    # 다른 사람(거부 확인용)

- 마이크 기본 샘플레이트(보통 48kHz)로 녹음 → data/tools/resample.py 로 16kHz 변환 → 16-bit 모노 저장
  (노트북 마이크는 16kHz 직접 녹음을 지원하지 않는 경우가 많아 이렇게 함)
- target = 등록 화자(로봇 사용자). 다른 사람은 아무 이름이나 (guest1, family2 …)
- 각 항목: Enter → 말하기 → 자동 종료. 음량 확인 후 Enter=저장, r=다시
- 명령어 목록은 configs/voice.toml [commands] 에서 읽음 (현재: 출발·걷자·멈춰·앉아·일어나·도와줘)
- 결과 폴더 규칙: <set>/enroll/<speaker>_<nn>.wav, <set>/utts/<speaker>__<command>__<nn>.wav
- 끝나면: python data/tools/build_command_stream.py data/voice/my_voice   → docs/04-recording.md
"""
import argparse
import sys
import tomllib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from resample import resample  # noqa: E402
from wav_to_16k import TARGET_SR, write_16k  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
ENROLL_SENTENCES = [
    "안녕하세요, 저는 오늘도 로봇과 함께 운동을 합니다.",
    "아침에 일어나서 물을 한 잔 마시고 산책을 나갔어요.",
    "손을 천천히 들어 올리고 다섯까지 세어 봅시다.",
    "오늘 날씨가 맑아서 기분이 좋습니다.",
    "약은 식사 후 삼십 분에 먹으면 됩니다.",
    "가족들과 함께 저녁을 먹고 텔레비전을 봤습니다.",
]
# 명령 아닌 말. 뒤쪽은 '도와줘'와 발음이 비슷한 말 → 거짓 위급 알림 확인용
NONE_PHRASES = ["안녕하세요", "물 좀 줄래", "오늘 뭐 먹지", "텔레비전 켜 줘", "가방 어디 있어",
                "더워", "도라지", "돈 좀 줘", "도와줄까", "도와준대"]


class Recorder:
    def __init__(self, device):
        import sounddevice as sd
        self.sd = sd
        self.device = device
        info = sd.query_devices(device, "input")
        self.sr = int(info["default_samplerate"])
        print(f"마이크: {info['name']}  ({self.sr} Hz → {TARGET_SR} Hz 로 변환해 저장)")

    def record(self, seconds: float) -> np.ndarray:
        x = self.sd.rec(int(seconds * self.sr), samplerate=self.sr, channels=1, dtype="float32", device=self.device)
        self.sd.wait()
        return resample(x[:, 0], self.sr, TARGET_SR)


def level_report(y: np.ndarray) -> str:
    peak = float(np.abs(y).max()) if y.size else 0.0
    rms = float(np.sqrt(np.mean(y ** 2))) if y.size else 0.0
    msg = f"peak {20 * np.log10(peak + 1e-9):.0f} dBFS, RMS {20 * np.log10(rms + 1e-9):.0f} dBFS"
    if peak < 0.05:
        msg += "  ⚠ 너무 작음 (마이크에 가까이 / 윈도우 '소리 설정 > 입력' 볼륨 올리기)"
    elif peak > 0.98:
        msg += "  ⚠ 잘림(클리핑) — 조금 떨어져서"
    return msg


def take(rec: Recorder, prompt: str, path: Path, seconds: float) -> None:
    while True:
        input(f"\n[{path.name}]  \"{prompt}\"  — Enter 후 말하세요 ({seconds:.1f}초)")
        y = rec.record(seconds)
        print("  " + level_report(y))
        if input("  저장=Enter, 다시=r : ").strip().lower() != "r":
            write_16k(path, y)
            return


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("set_dir", type=Path, nargs="?")
    ap.add_argument("--list-devices", action="store_true")
    ap.add_argument("--device", default=None, help="입력 장치 번호 또는 이름 일부 (기본: 시스템 기본 마이크)")
    ap.add_argument("--test", action="store_true", help="3초 녹음해 음량만 확인 (저장 안 함)")
    ap.add_argument("--speaker", default="target")
    ap.add_argument("--enroll", type=int, default=0, help="등록 문장 개수 (target 만, 문장당 5초)")
    ap.add_argument("--reps", type=int, default=5, help="명령어당 반복 횟수")
    ap.add_argument("--none", type=int, default=len(NONE_PHRASES), help="명령 아닌 말 개수")
    ap.add_argument("--seconds", type=float, default=2.5, help="명령어 1개 녹음 길이")
    args = ap.parse_args()

    if args.list_devices:
        import sounddevice as sd
        print(sd.query_devices())
        return 0
    device = int(args.device) if args.device and args.device.isdigit() else args.device
    rec = Recorder(device)
    if args.test:
        input("Enter 후 3초 동안 평소 목소리로 말하세요")
        print(level_report(rec.record(3.0)))
        return 0
    if args.set_dir is None:
        ap.error("set_dir 필요 (예: data/voice/my_voice)")

    cfg = tomllib.loads((ROOT / "configs" / "voice.toml").read_text(encoding="utf-8"))["commands"]
    display = {k: (v["phrase"] if isinstance(v, dict) else v).replace(" ", "") for k, v in cfg.items()}
    print(f"녹음 목록: 등록 {args.enroll} + 명령 {len(display)}×{args.reps} + 명령 아닌 말 {args.none}")
    for i in range(args.enroll):
        take(rec, ENROLL_SENTENCES[i % len(ENROLL_SENTENCES)], args.set_dir / "enroll" / f"{args.speaker}_{i + 1:02d}.wav", 5.0)
    for k, word in display.items():
        for n in range(args.reps):
            take(rec, word, args.set_dir / "utts" / f"{args.speaker}__{k}__{n + 1:02d}.wav", args.seconds)
    for n in range(args.none):
        take(rec, NONE_PHRASES[n % len(NONE_PHRASES)], args.set_dir / "utts" / f"{args.speaker}__none__{n + 1:02d}.wav",
             args.seconds)
    print(f"\n완료. 다음: python data/tools/build_command_stream.py {args.set_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
