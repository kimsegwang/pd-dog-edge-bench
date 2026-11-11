"""pdbench CLI.

  python -m pdbench env [--note "..."]                       측정 환경 출력
  python -m pdbench pose --res 416 --threads 4               시나리오 A 한 조합
  python -m pdbench concurrent --res 416 --pose-threads 3 --voice full [--fps-cap 10]   시나리오 B 한 조합
  python -m pdbench voice                                     시나리오 V: 음성 명령 체인 단독
  python -m pdbench matrix AVB --cooler none --note "..."     A → V → B(+FPS 상한) 순서로 실행
  python -m pdbench matrix all --cooler active --note "..."   A → V → B → C 전체
"""
import argparse
import json
import sys
from pathlib import Path

from .common import config
from .common.sysinfo import collect_env

COOLERS = ["none", "active"]


def _steps(v: str) -> str:
    v = {"all": "AVBC", "AB": "AVB"}.get(v, v).upper()
    if not v or any(ch not in "AVBC" for ch in v):
        raise argparse.ArgumentTypeError("A, V, B, C 조합 (예: AVB) 또는 all")
    return v


def main(argv: list[str] | None = None) -> int:
    sc = config.load("scenarios")
    c = sc["common"]
    ap = argparse.ArgumentParser(prog="pdbench")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_env = sub.add_parser("env")
    p_env.add_argument("--note", default="")
    p_env.add_argument("--cooler", default="unknown", choices=COOLERS + ["unknown"])

    def common_args(p, res_required=True):
        p.add_argument("--res", type=int, required=res_required, choices=[640, 416, 320])
        p.add_argument("--frames", type=Path, default=config.repo_path(c["frames"]))
        p.add_argument("--out", type=Path, required=True)
        p.add_argument("--warmup", type=int, default=c["warmup_frames"])
        p.add_argument("--cooldown", action="store_true", help="시작 전 CPU 온도가 cooldown_temp_c 아래로 내려갈 때까지 대기")
        p.add_argument("--cooler", default="unknown", choices=COOLERS + ["unknown"])

    p_pose = sub.add_parser("pose", help="시나리오 A: 포즈 단독")
    common_args(p_pose)
    p_pose.add_argument("--threads", type=int, required=True)
    p_pose.add_argument("--measure-frames", type=int, default=sc["A"]["measure_frames"])
    p_pose.add_argument("--max-measure-s", type=float, default=sc["A"]["max_measure_s"])

    def voice_args(p):
        p.add_argument("--vad-threads", type=int, default=1)
        p.add_argument("--voice", choices=["vad", "full"], default=sc["B"]["voice"],
                       help="vad = VAD 만, full = VAD + Vosk 명령 + ECAPA 화자 (configs/voice.toml)")
        p.add_argument("--wav", type=Path, default=config.repo_path(c["wav"]), help="voice=vad 일 때 입력 wav")
        p.add_argument("--voice-set", type=Path, default=None,
                       help="voice=full 일 때 음성 세트 폴더 (기본: configs/voice.toml [pipeline] dataset)")
        p.add_argument("--bucket-s", type=float, default=None, help="시간 구간별 요약 (장시간용)")

    p_cc = sub.add_parser("concurrent", help="시나리오 B/C: 포즈 + 음성 동시")
    common_args(p_cc)
    voice_args(p_cc)
    p_cc.add_argument("--scenario", default="B")
    p_cc.add_argument("--pose-threads", type=int, required=True)
    p_cc.add_argument("--fps-cap", type=float, default=0.0, help="포즈 FPS 상한 (0 = 최대 속도)")
    p_cc.add_argument("--duration", type=float, default=sc["B"]["duration_s"])
    p_cc.add_argument("--choice-reason", default="")

    p_v = sub.add_parser("voice", help="시나리오 V: 음성 명령 체인 단독 (포즈 없음)")
    p_v.add_argument("--out", type=Path, required=True)
    p_v.add_argument("--cooldown", action="store_true")
    p_v.add_argument("--cooler", default="unknown", choices=COOLERS + ["unknown"])
    p_v.add_argument("--duration", type=float, default=sc["V"]["duration_s"])
    voice_args(p_v)

    p_mx = sub.add_parser("matrix", help="configs/scenarios.toml 매트릭스 실행")
    p_mx.add_argument("which", type=_steps, help="실행할 시나리오: A, V, B, C 조합 (예: AVB) 또는 all(=AVBC)")
    p_mx.add_argument("--cooler", required=True, choices=COOLERS,
                      help="냉각 상태 (none = 쿨러 없음, active = 액티브 쿨러). env.json·결과·세션 폴더 이름에 기록")
    p_mx.add_argument("--session", type=Path, default=None, help="기존 세션 폴더에 이어서 (C 의 auto 선택은 같은 세션의 B 결과 사용)")
    p_mx.add_argument("--note", default="", help="쿨러/케이스/실내온도 등 메모 → env.json")
    p_mx.add_argument("--voice-set", type=Path, default=None, help="음성 세트 폴더 (예: data/voice/my_voice)")

    args = ap.parse_args(argv)
    cooldown = (c["cooldown_temp_c"], c["cooldown_timeout_s"]) if getattr(args, "cooldown", False) else None

    if args.cmd == "env":
        print(json.dumps(collect_env(args.note, cooler=args.cooler), indent=2, ensure_ascii=False, default=str))
    elif args.cmd == "pose":
        from .scenarios import single
        single.run(args.out, args.res, args.threads, args.frames, args.warmup, args.measure_frames,
                   args.max_measure_s, c["monitor_interval_s"], cooldown, args.cooler)
    elif args.cmd == "concurrent":
        from .scenarios import concurrent
        s = concurrent.run(args.out, args.scenario, args.res, args.pose_threads, args.vad_threads, args.duration,
                           args.frames, args.wav, args.warmup, c["warmup_chunks"], c["monitor_interval_s"],
                           cooldown, args.bucket_s, args.cooler, args.fps_cap, args.voice, args.voice_set)
        if args.choice_reason:
            s["choice_reason"] = args.choice_reason
            from .common.sysinfo import write_json
            write_json(args.out / "summary.json", s)
    elif args.cmd == "voice":
        from .scenarios import concurrent
        concurrent.run(args.out, "V", None, 0, args.vad_threads, args.duration, config.repo_path(c["frames"]),
                       args.wav, 0, c["warmup_chunks"], c["monitor_interval_s"], cooldown, args.bucket_s,
                       args.cooler, 0.0, "full", args.voice_set)
    elif args.cmd == "matrix":
        from .runner import run_matrix
        session = run_matrix(args.which, args.session, args.note, args.cooler, args.voice_set)
        print(f"\n다음: python bench/summarize.py {session.relative_to(config.REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
