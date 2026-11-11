"""configs/scenarios.toml 의 매트릭스를 실행. 조합마다 새 프로세스를 띄워 메모리/세션 상태를 분리한다."""
import json
import socket
import subprocess
import sys
import time
from pathlib import Path

from .common import config
from .common.sysinfo import collect_env, write_json


def new_session_dir(cooler: str) -> Path:
    return (config.REPO_ROOT / "results" /
            f"{time.strftime('%Y%m%d-%H%M')}_{socket.gethostname()}_python_cooler-{cooler}")


def write_env(session: Path, note: str, cooler: str) -> None:
    sc = config.load("scenarios")["common"]
    pose_tpl = config.load("pose")["model"]["path_template"]
    models = [config.repo_path(pose_tpl.format(res=r)) for r in (640, 416, 320)]
    models.append(config.repo_path(config.load("vad")["model"]["path"]))
    env = collect_env(note, [m for m in models if m.exists()],
                      [config.repo_path(sc["frames"]), config.repo_path(sc["wav"])], cooler)
    write_json(session / "env.json", env)
    if env.get("throttled") and not env["throttled"].endswith("0x0"):
        print(f"WARNING: 측정 시작 전 {env['throttled']} — 전원/온도 문제를 먼저 확인하세요", flush=True)
    if env.get("cooler_warning"):
        print(f"WARNING: {env['cooler_warning']}", flush=True)


class Matrix:
    def __init__(self, session: Path, cooler: str, voice_set: Path | None = None):
        self.session, self.cooler, self.voice_set = session, cooler, voice_set
        self.sc = config.load("scenarios")

    def _child(self, args: list[str]) -> int:
        cmd = [sys.executable, "-m", "pdbench", *args, "--cooldown", "--cooler", self.cooler]
        if self.voice_set and args[0] in ("voice", "concurrent"):
            cmd += ["--voice-set", str(config.repo_path(self.voice_set))]
        print("$", " ".join(cmd[1:]), flush=True)
        return subprocess.run(cmd, cwd=config.REPO_ROOT / "python").returncode

    def run_A(self) -> None:
        a = self.sc["A"]
        for res in a["resolutions"]:
            for th in a["threads"]:
                self._child(["pose", "--res", str(res), "--threads", str(th),
                             "--out", str(self.session / f"A_r{res}_t{th}"),
                             "--measure-frames", str(a["measure_frames"]), "--max-measure-s", str(a["max_measure_s"])])

    def run_V(self) -> None:
        v = self.sc["V"]
        self._child(["voice", "--vad-threads", str(v["vad_threads"]), "--duration", str(v["duration_s"]),
                     "--out", str(self.session / "V_voice")])

    def run_B(self) -> None:
        b = self.sc["B"]
        for res in b["resolutions"]:
            for th in b["pose_threads"]:
                self._child(["concurrent", "--scenario", "B", "--res", str(res), "--pose-threads", str(th),
                             "--vad-threads", str(b["vad_threads"]), "--duration", str(b["duration_s"]),
                             "--voice", b["voice"], "--out", str(self.session / f"B_r{res}_t{th}")])
        # 포즈 FPS 상한: 같은 해상도/스레드의 상한 없는 B 와 CPU·온도 비교
        bc = self.sc["Bcap"]
        for cap in bc["fps_caps"]:
            for res in bc["resolutions"]:
                for th in bc["pose_threads"]:
                    self._child(["concurrent", "--scenario", "B", "--res", str(res), "--pose-threads", str(th),
                                 "--vad-threads", str(b["vad_threads"]), "--duration", str(b["duration_s"]),
                                 "--voice", b["voice"], "--fps-cap", str(cap),
                                 "--out", str(self.session / f"B_r{res}_t{th}_cap{cap:g}")])

    def choose_for_C(self) -> tuple[int, int, str]:
        c = self.sc["C"]
        if c["resolution"] != "auto":
            return int(c["resolution"]), int(c["pose_threads"]), "config"
        min_fps = config.load("criteria")["draft"]["pose_min_fps"]
        cands = []
        for s in self.session.glob("B_*/summary.json"):
            p = json.loads(s.read_text(encoding="utf-8"))["pose"]
            if p.get("fps_cap"):
                continue
            if p.get("fps") and p["fps"] >= min_fps:
                cands.append((p["res"], p["fps"], p["threads"]))
        if not cands:
            print("WARNING: B 결과 중 FPS 기준을 넘는 조합이 없어 fallback 설정으로 C 실행", flush=True)
            return int(c["fallback_resolution"]), int(c["fallback_pose_threads"]), "fallback"
        res, _, th = max(cands)
        return res, th, f"auto: B 에서 {min_fps} FPS 이상인 가장 높은 해상도"

    def run_C(self) -> None:
        c = self.sc["C"]
        res, th, why = self.choose_for_C()
        print(f"[C] res={res} pose_threads={th} ({why}), {c['duration_s'] / 60:.0f} min", flush=True)
        self._child(["concurrent", "--scenario", "C", "--res", str(res), "--pose-threads", str(th),
                     "--vad-threads", str(c["vad_threads"]), "--duration", str(c["duration_s"]),
                     "--voice", self.sc["B"]["voice"], "--fps-cap", str(c.get("fps_cap", 0)),
                     "--out", str(self.session / f"C_r{res}_t{th}"), "--bucket-s", "300", "--choice-reason", why])


def run_matrix(which: str, session: Path | None, note: str, cooler: str, voice_set: Path | None = None) -> Path:
    session = session or new_session_dir(cooler)
    session.mkdir(parents=True, exist_ok=True)
    env_path = session / "env.json"
    if env_path.exists():
        prev = json.loads(env_path.read_text(encoding="utf-8")).get("cooler")
        if prev != cooler:
            raise SystemExit(f"세션 {session.name} 은 cooler={prev} 로 측정됨 — 같은 세션에 cooler={cooler} 결과를 섞을 수 없음")
    else:
        write_env(session, note, cooler)
    print(f"session: {session}  (cooler={cooler})", flush=True)
    m = Matrix(session, cooler, voice_set)
    steps = {"A": m.run_A, "V": m.run_V, "B": m.run_B, "C": m.run_C}
    for s in which:
        t = time.monotonic()
        steps[s]()
        print(f"== {s} done in {(time.monotonic() - t) / 60:.1f} min", flush=True)
    return session
