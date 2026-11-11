"""측정 환경 기록, 외부 모니터 실행, 시스템 지표 요약."""
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

from .config import REPO_ROOT

# vcgencmd get_throttled 현재 상태 비트 (16~19 는 부팅 후 누적 sticky 비트)
THROTTLE_BITS = {0: "under-voltage", 1: "arm-freq-capped", 2: "throttled", 3: "soft-temp-limit"}


def _run(cmd: list[str], timeout: float = 10) -> str | None:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return (r.stdout + r.stderr).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _read(path: str) -> str | None:
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def temp_c() -> float | None:
    v = _read("/sys/class/thermal/thermal_zone0/temp")
    return int(v) / 1000.0 if v else None


def sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def peak_rss_mb() -> float | None:
    """프로세스 최대 RSS (Linux: VmHWM). 다른 OS 에서는 근사값."""
    status = _read("/proc/self/status")
    if status:
        for line in status.splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) / 1024.0
    try:
        import psutil
        mi = psutil.Process().memory_info()
        return getattr(mi, "peak_wset", mi.rss) / 2**20
    except Exception:
        return None


def wait_cooldown(target_c: float, timeout_s: float) -> dict:
    """CPU 온도가 target 아래로 내려갈 때까지 대기. Pi 가 아니면 즉시 반환."""
    t0 = time.monotonic()
    start = temp_c()
    cur = start
    while cur is not None and cur > target_c and time.monotonic() - t0 < timeout_s:
        time.sleep(5)
        cur = temp_c()
    return {"temp_before_cooldown_c": start, "temp_at_start_c": cur,
            "cooldown_wait_s": round(time.monotonic() - t0, 1)}


def collect_env(note: str = "", model_files: list[Path] | None = None, data_files: list[Path] | None = None,
                cooler: str = "unknown") -> dict:
    import numpy
    import onnxruntime

    env: dict = {"collected_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "cooler": cooler, "note": note}
    osr = _read("/etc/os-release") or ""
    env["os"] = next((l.split("=", 1)[1].strip('"') for l in osr.splitlines() if l.startswith("PRETTY_NAME=")),
                     platform.platform())
    env["debian_version"] = _read("/etc/debian_version")
    env["kernel"] = platform.release()
    env["machine"] = platform.machine()
    env["board"] = (_read("/proc/device-tree/model") or "").rstrip("\x00") or None
    env["firmware"] = _run(["vcgencmd", "version"])
    env["bootloader"] = _run(["vcgencmd", "bootloader_version"])
    env["cpu_count"] = os.cpu_count()
    env["governor"] = _read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    env["arm_freq_max_mhz"] = (int(v) / 1000 if (v := _read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq")) else None)
    try:
        import psutil
        env["mem_total_mb"] = round(psutil.virtual_memory().total / 2**20)
        env["swap_total_mb"] = round(psutil.swap_memory().total / 2**20)
        env["psutil"] = psutil.__version__
    except ImportError:
        pass
    env["temp_c"] = temp_c()
    env["throttled"] = _run(["vcgencmd", "get_throttled"])
    fan = list(Path("/sys/devices/platform/cooling_fan").glob("hwmon/*/fan1_input"))
    env["fan_rpm"] = int(fan[0].read_text()) if fan else None
    env["active_cooler_detected"] = bool(fan)
    if cooler == "none" and fan:
        env["cooler_warning"] = "cooler=none 으로 지정했지만 팬이 감지됨"
    elif cooler == "active" and not fan and env["board"]:
        env["cooler_warning"] = "cooler=active 로 지정했지만 팬이 감지되지 않음"
    cams = _run(["rpicam-hello", "--list-cameras"], timeout=15)
    env["camera_list"] = cams
    env["camera_sensor"] = next((s for s in ("imx708_wide_noir", "imx708_wide", "imx708_noir", "imx708")
                                 if cams and s in cams), None)
    env["python"] = sys.version.split()[0]
    env["numpy"] = numpy.__version__
    env["numpy_path"] = numpy.__file__
    try:
        import cv2
        env["opencv"] = cv2.__version__
    except ImportError:
        env["opencv"] = None
    env["onnxruntime"] = onnxruntime.__version__
    env["ort_providers"] = onnxruntime.get_available_providers()
    env["git_commit"] = _run(["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"])
    # 측정 결과물(results/)은 코드 변경이 아니므로 dirty 판정에서 제외. 변경된 파일 목록도 남김
    try:  # _run 은 출력을 strip 해서 porcelain 첫 줄의 앞 공백(상태 칸)이 사라지므로 직접 호출
        porcelain = subprocess.run(["git", "-C", str(REPO_ROOT), "status", "--porcelain"],
                                   capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        porcelain = ""
    changed = [ln for ln in porcelain.splitlines() if ln.strip() and not ln[3:].startswith("results/")]
    env["git_dirty"] = bool(changed)
    env["git_changed"] = changed[:20]
    env["models"] = {str(p.relative_to(REPO_ROOT)): sha256(p) for p in (model_files or [])}
    env["data"] = {}
    for p in data_files or []:
        meta = p.with_suffix(".json")
        env["data"][str(p.relative_to(REPO_ROOT))] = (json.loads(meta.read_text()) if meta.exists() and p.suffix == ".rgb"
                                                     else {"sha256": sha256(p)})
    return env


class Monitor:
    """tools/monitor.py 를 자식 프로세스로 띄워 현재 프로세스를 샘플링."""

    def __init__(self, out_csv: Path, interval_s: float = 1.0, pid: int | None = None):
        self.out_csv = out_csv
        self.proc = subprocess.Popen(
            [sys.executable, str(REPO_ROOT / "tools" / "monitor.py"),
             "--pid", str(pid or os.getpid()), "--out", str(out_csv), "--interval", str(interval_s)])

    def stop(self) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()


def _f(v: str) -> float | None:
    return float(v) if v not in ("", None) else None


def summarize_system(csv_path: Path) -> dict:
    """system.csv → 요약. 스로틀은 '현재 상태' 비트(0~3)가 한 번이라도 켜졌는지로 판정."""
    if not csv_path.exists():
        return {}
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}

    def col(name):
        return [v for r in rows if (v := _f(r.get(name))) is not None]

    out: dict = {"samples": len(rows)}
    for name, key in (("cpu_total_pct", "cpu_total"), ("proc_cpu_pct", "proc_cpu"), ("temp_c", "temp_c"),
                      ("arm_freq_mhz", "arm_freq_mhz"), ("rss_mb", "rss_mb"), ("sys_mem_used_mb", "sys_mem_used_mb"),
                      ("swap_used_mb", "swap_used_mb"), ("fan_rpm", "fan_rpm"), ("ext5v_v", "ext5v_v")):
        vals = col(name)
        if vals:
            out[f"{key}_mean"] = sum(vals) / len(vals)
            out[f"{key}_max"] = max(vals)
            out[f"{key}_min"] = min(vals)
    thr = [int(r["throttled_hex"], 16) for r in rows if r["throttled_hex"]]
    if thr:
        now_bits = 0
        for v in thr:
            now_bits |= v & 0xF
        out["throttle_now_bits_or"] = hex(now_bits)
        out["throttle_events"] = [name for bit, name in THROTTLE_BITS.items() if now_bits >> bit & 1]
        out["throttle_samples"] = sum(1 for v in thr if v & 0xF)
        out["throttled_start_hex"] = hex(thr[0])
        out["throttled_last_hex"] = hex(thr[-1])
        out["throttled_values"] = sorted({hex(v) for v in thr})
    return out


def run_flags(system: dict, cool: dict) -> list[str]:
    """측정 결과를 그대로 믿으면 안 되는 상황 표시 (configs/criteria.toml [flags])."""
    from . import config
    max_temp = config.load("criteria")["flags"]["max_temp_c"]
    start_target = config.load("scenarios")["common"]["cooldown_temp_c"]
    flags = []
    t = system.get("temp_c_max")
    if t is not None and t > max_temp:
        flags.append(f"온도 {t:.1f}°C > {max_temp:.0f}°C")
    if "under-voltage" in system.get("throttle_events", []):
        v = system.get("ext5v_v_min")
        flags.append("저전압 감지(under-voltage)" + (f", 입력 최저 {v:.2f} V" if v else "") + " — 전원 영향 가능성")
    nonzero = [v for v in system.get("throttled_values", []) if v != "0x0"]
    if nonzero:
        carried = system.get("throttled_start_hex") != "0x0" and set(nonzero) == {system["throttled_start_hex"]}
        flags.append(f"throttled={'/'.join(nonzero)}" + (" (측정 전부터 남아 있던 값)" if carried else ""))
    ts = cool.get("temp_at_start_c")
    if ts is not None and ts > start_target:
        flags.append(f"시작 온도 {ts:.1f}°C (>{start_target:.0f}°C, 대기 시간 초과)")
    return flags


def write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
