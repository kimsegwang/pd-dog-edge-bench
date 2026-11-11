"""언어 무관 시스템 모니터. 대상 PID 를 주기적으로 샘플링해 CSV 로 기록한다.

Python/Rust 벤치마크 모두 이 스크립트를 별도 프로세스로 띄워 같은 정의로 지표를 잰다.

    python tools/monitor.py --pid 1234 --out results/x/system.csv --interval 1.0

대상 프로세스가 끝나거나 SIGTERM/SIGINT 를 받으면 종료한다.
라즈베리파이가 아닌 환경(노트북)에서는 온도/클럭/스로틀 칸이 비어 있다.
"""
import argparse
import csv
import glob
import signal
import subprocess
import time
from pathlib import Path

import psutil

COLUMNS = [
    "t_s", "wall_time", "cpu_total_pct", "cpu_per_core_pct", "proc_cpu_pct", "proc_threads",
    "rss_mb", "sys_mem_used_mb", "sys_mem_avail_mb", "swap_used_mb",
    "temp_c", "arm_freq_mhz", "arm_freq_req_mhz", "throttled_hex", "fan_rpm", "ext5v_v",
]
# arm_freq_mhz     : 실제 ARM 클럭 (vcgencmd measure_clock arm). 펌웨어 스로틀이 반영됨
# arm_freq_req_mhz : OS(cpufreq)가 요청한 클럭 (sysfs). 펌웨어 스로틀 중에도 2400 으로 보일 수 있음
# ext5v_v          : Pi 5 PMIC 가 잰 5V 입력 전압 (vcgencmd pmic_read_adc EXT5V_V). 저전압 진단용


def read_first(pattern: str) -> str | None:
    for path in sorted(glob.glob(pattern)):
        try:
            return Path(path).read_text().strip()
        except OSError:
            pass
    return None


def temp_c() -> float | None:
    v = read_first("/sys/class/thermal/thermal_zone0/temp")
    return int(v) / 1000.0 if v else None


def arm_freq_req_mhz() -> float | None:
    v = read_first("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq")
    return int(v) / 1000.0 if v else None


def _vcgencmd(*args: str) -> str | None:
    try:
        return subprocess.run(["vcgencmd", *args], capture_output=True, text=True, timeout=2).stdout
    except (OSError, subprocess.SubprocessError):
        return None


def arm_freq_mhz() -> float | None:
    out = _vcgencmd("measure_clock", "arm")          # frequency(0)=2400000000
    try:
        return int(out.strip().split("=")[-1]) / 1e6 if out and "=" in out else None
    except ValueError:
        return None


def ext5v_v() -> float | None:
    out = _vcgencmd("pmic_read_adc", "EXT5V_V")      # EXT5V_V volt(24)=5.12345000V
    try:
        return round(float(out.strip().split("=")[-1].rstrip("V")), 3) if out and "=" in out else None
    except ValueError:
        return None


def fan_rpm() -> int | None:
    v = read_first("/sys/devices/platform/cooling_fan/hwmon/*/fan1_input")
    return int(v) if v else None


def throttled_hex() -> str | None:
    out = _vcgencmd("get_throttled")
    return out.strip().split("=")[-1] if out and "=" in out else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--interval", type=float, default=1.0)
    args = ap.parse_args()

    stop = False

    def _stop(*_):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    proc = psutil.Process(args.pid)
    proc.cpu_percent(None)
    psutil.cpu_percent(None, percpu=True)
    t0 = time.monotonic()
    next_t = t0 + args.interval
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        while not stop:
            time.sleep(max(0.0, next_t - time.monotonic()))
            next_t += args.interval
            try:
                p_cpu = proc.cpu_percent(None)
                rss = proc.memory_info().rss / 2**20
                nthreads = proc.num_threads()
            except psutil.NoSuchProcess:
                break
            cores = psutil.cpu_percent(None, percpu=True)
            vm, sw = psutil.virtual_memory(), psutil.swap_memory()
            row = [
                f"{time.monotonic() - t0:.2f}", f"{time.time():.3f}",
                f"{sum(cores) / len(cores):.1f}", ";".join(f"{c:.0f}" for c in cores),
                f"{p_cpu:.1f}", nthreads,
                f"{rss:.1f}", f"{(vm.total - vm.available) / 2**20:.0f}", f"{vm.available / 2**20:.0f}",
                f"{sw.used / 2**20:.0f}",
                temp_c(), arm_freq_mhz(), arm_freq_req_mhz(), throttled_hex(), fan_rpm(), ext5v_v(),
            ]
            w.writerow(["" if v is None else v for v in row])
            f.flush()


if __name__ == "__main__":
    main()
