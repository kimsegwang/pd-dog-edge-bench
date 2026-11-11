"""측정 세션 폴더 → 마크다운 결과표 + 통과 기준 판정 (REPORT.md).

    python bench/summarize.py results/20261112-1430_pd-dog_python_cooler-none

Pi 에서도 노트북에서도 실행 가능 (표준 라이브러리 + numpy).
"""
import argparse
import csv
import json
import sys
import tomllib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PASS, FAIL, PARTIAL, NA = "✅ 통과", "❌ 미통과", "🟡 부분 측정", "— 측정 없음"
UV = "⚡"


# ---------------------------------------------------------------- 공통 도우미
def f(v, nd=1, unit=""):
    return "–" if v is None else f"{v:.{nd}f}{unit}"


def g(d, *keys):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def uv(r: dict) -> bool:
    """측정 중 저전압(throttled 비트 0)이 한 번이라도 감지됐는지."""
    return "under-voltage" in (g(r, "system", "throttle_events") or [])


def fps_cell(r: dict) -> str:
    return f"**{f(g(r, 'pose', 'fps'), 2)}**" + (f" {UV}" if uv(r) else "")


def flag_cell(r: dict) -> str:
    return "⚠ " + "; ".join(r["flags"]) if r.get("flags") else "–"


def busy_cores(r: dict) -> int:
    """포즈 스레드 + (음성이 있으면) VAD 1 — 동시에 바쁜 코어 수의 근사."""
    return (g(r, "pose", "threads") or 0) + (1 if r["scenario"] in "BCV" else 0)


def load_runs(session: Path) -> dict[str, list[dict]]:
    runs: dict[str, list[dict]] = {"A": [], "V": [], "B": [], "C": []}
    for s in sorted(session.glob("*/summary.json")):
        d = json.loads(s.read_text(encoding="utf-8"))
        d["_dir"] = s.parent
        runs.setdefault(d["scenario"], []).append(d)
    for k in runs:
        runs[k].sort(key=lambda d: (-(g(d, "pose", "res") or 0), g(d, "pose", "threads") or 0,
                                    g(d, "pose", "fps_cap") or 0))
    return runs


def measured_mask(run_dir: Path, n: int) -> np.ndarray | None:
    """측정 구간에서 처리한 프레임. 새 결과는 kpts.npz, 이전 결과는 frames.csv 의 frame_idx 로 복원."""
    fc = run_dir / "frames.csv"
    if not fc.exists():
        return None
    m = np.zeros(n, bool)
    with open(fc, newline="") as fh:
        for row in csv.DictReader(fh):
            m[int(row["frame_idx"])] = True
    return m


def load_kpts(run_dir: Path) -> tuple[np.ndarray, np.ndarray | None] | None:
    if (run_dir / "kpts.npz").exists():
        z = np.load(run_dir / "kpts.npz")
        return z["kpts"], z["measured"]
    if (run_dir / "kpts.npy").exists():
        k = np.load(run_dir / "kpts.npy")
        return k, measured_mask(run_dir, len(k))
    return None


def kpt_diff(run: dict, ref: dict | None, vis_thr: float):
    """기준(640) 대비 키포인트 평균 거리(px), 검출 일치율, 비교 프레임 수 — 두 실행이 모두 처리한 프레임만 비교."""
    if ref is None or run is ref:
        return None, None, None
    a, b = load_kpts(run["_dir"]), load_kpts(ref["_dir"])
    if a is None or b is None:
        return None, None, None
    (ka, ma), (kb, mb) = a, b
    common = np.ones(len(ka), bool)
    if ma is not None and mb is not None:
        common = ma & mb
    da, db = ~np.isnan(ka[:, 0, 0]) & common, ~np.isnan(kb[:, 0, 0]) & common
    agree = float((da == db)[common].mean()) if common.any() else None
    both = da & db
    if not both.any():
        return None, agree, int(common.sum())
    va, vb = ka[both], kb[both]
    m = (va[..., 2] >= vis_thr) & (vb[..., 2] >= vis_thr)
    dist = np.linalg.norm(va[..., :2] - vb[..., :2], axis=-1)[m]
    return (float(dist.mean()) if dist.size else None), agree, int(common.sum())


# ---------------------------------------------------------------- 섹션
def env_section(env: dict) -> list[str]:
    if not env:
        return ["_env.json 없음_", ""]
    rows = [
        ("보드", env.get("board")), ("OS", env.get("os")), ("커널", env.get("kernel")),
        ("Python / numpy / OpenCV", f"{env.get('python')} / {env.get('numpy')} / {env.get('opencv')}"),
        ("onnxruntime", f"{env.get('onnxruntime')} {env.get('ort_providers')}"),
        ("CPU 거버너 / 최대 클럭", f"{env.get('governor')} / {env.get('arm_freq_max_mhz')} MHz"),
        ("메모리 / 스왑", f"{env.get('mem_total_mb')} MB / {env.get('swap_total_mb')} MB"),
        ("쿨러", f"감지={env.get('active_cooler_detected')}, 팬 {env.get('fan_rpm')} RPM"),
        ("카메라", env.get("camera_sensor") or "미확인"),
        ("시작 시 온도 / throttled", f"{env.get('temp_c')} °C / {env.get('throttled')}"),
        ("git", f"{env.get('git_commit')}{' (dirty)' if env.get('git_dirty') else ''}"),
        ("냉각 (cooler)", f"{env.get('cooler')}" + (f" ⚠ {env['cooler_warning']}" if env.get("cooler_warning") else "")),
        ("메모", env.get("note") or "–"),
    ]
    return ["| 항목 | 값 |", "|---|---|"] + [f"| {k} | {v} |" for k, v in rows] + [""]


def power_section(runs: dict) -> list[str]:
    allr = [r for k in "AVBC" for r in runs[k]]
    bad = [r for r in allr if uv(r)]
    if not bad:
        if any(g(r, "system", "throttle_events") is not None for r in allr):
            return ["측정 중 저전압 감지 없음.", ""]
        return []
    out = [f"{UV} 표시 = 측정 중 **저전압(under-voltage)** 이 감지된 실행 → **저전압 영향 가능성 있음** (전원 어댑터·케이블 점검 후 재측정 필요).", ""]
    a4 = [r for r in runs["A"] if g(r, "pose", "threads") == 4]
    if a4 and all(uv(r) for r in a4):
        out.append("- **A 4스레드 결과는 저전압 영향 가능성 있음** (A 4스레드 전부에서 감지).")
    a_low = [r for r in runs["A"] if (g(r, "pose", "threads") or 0) < 4]
    if a_low and not any(uv(r) for r in a_low):
        out.append("- A 1~3스레드에서는 감지되지 않음.")
    b_uv = [r for r in runs["B"] + runs["C"] if uv(r)]
    if b_uv:
        out.append(f"- B/C(포즈 + 음성 동시)에서도 {len(b_uv)}/{len(runs['B']) + len(runs['C'])}개 실행에서 감지: "
                   + ", ".join(sorted({f'포즈 {g(r, "pose", "threads")}스레드+음성' for r in b_uv}))
                   + " → 4코어가 모두 바쁠 때 전류가 늘며 발생하는 것으로 보임.")
    vmin = [g(r, "system", "ext5v_v_min") for r in bad if g(r, "system", "ext5v_v_min")]
    if vmin:
        out.append(f"- 저전압 실행의 입력 전압(EXT5V) 최저 {min(vmin):.2f} V (정상 ≈ 5.0~5.2 V).")
    out.append("")
    out.append("| 실행 | 동시 바쁜 코어(근사) | 저전압 샘플 / 전체 | throttled 값 |")
    out.append("|---|---:|---:|---|")
    for r in bad:
        n_uv, n = _uv_samples(r["_dir"])
        out.append(f"| {r['_dir'].name} | {busy_cores(r)} | {n_uv} / {n} | {', '.join(g(r, 'system', 'throttled_values') or [])} |")
    return out + [""]


def _uv_samples(run_dir: Path) -> tuple[str, str]:
    p = run_dir / "system.csv"
    if not p.exists():
        return "–", "–"
    with open(p, newline="") as fh:
        vals = [int(r["throttled_hex"], 16) for r in csv.DictReader(fh) if r.get("throttled_hex")]
    return str(sum(1 for v in vals if v & 1)), str(len(vals))


def section_A(runs: list[dict], crit: dict, vis_thr: float) -> list[str]:
    if not runs:
        return ["_시나리오 A 결과 없음_", ""]
    refs = {r["pose"]["threads"]: r for r in runs if r["pose"]["res"] == 640}
    out = [
        "입력: 녹화 프레임(640×480)을 최대 속도로 처리. 전체 지연 = 전처리 + 추론 + 후처리 (파일 읽기 제외).",
        "키포인트 차이·검출 일치율 = 같은 스레드 수의 640 결과 대비, **두 실행이 모두 처리한 프레임만** 비교 (FP32 640 = 기준).",
        f"{UV} = 측정 중 저전압 감지 (전원 영향 가능성).",
        "",
        "| 해상도 | 스레드 | FPS | 추론 p50 / p95 (ms) | 전체 p50 / p95 (ms) | 전처리 p50 (ms) | 최대 RSS (MB) | 프로세스 CPU 평균 (%) | 최고 온도 (°C) | 키포인트 차이 vs 640 (px) | 검출 일치율 (비교 프레임) | 단독 ≥5 FPS | 경고 |",
        "|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|:--:|---|",
    ]
    for r in runs:
        p, s = r["pose"], r.get("system", {})
        diff, agree, n_cmp = kpt_diff(r, refs.get(p["threads"]), vis_thr)
        ok = PASS if (p.get("fps") or 0) >= crit["pose_min_fps"] else FAIL
        agree_txt = "–" if p["res"] == 640 else (f"{agree * 100:.0f}% ({n_cmp})" if agree is not None else "–")
        out.append(
            f"| {p['res']} | {p['threads']} | {fps_cell(r)} | {f(g(p, 'infer_ms', 'p50'))} / {f(g(p, 'infer_ms', 'p95'))} "
            f"| {f(g(p, 'total_ms', 'p50'))} / {f(g(p, 'total_ms', 'p95'))} | {f(g(p, 'pre_ms', 'p50'))} "
            f"| {f(r.get('peak_rss_mb'), 0)} | {f(s.get('proc_cpu_mean'), 0)} | {f(s.get('temp_c_max'))} "
            f"| {f(diff, 1) if p['res'] != 640 else '기준'} | {agree_txt} | {ok} | {flag_cell(r)} |")
    best = max(runs, key=lambda r: r["pose"].get("fps") or 0)
    out += ["", f"가장 빠른 조합: **{best['pose']['res']} / {best['pose']['threads']}스레드 → {f(best['pose']['fps'], 2)} FPS**", ""]
    return out


def voice_cells(r: dict) -> str:
    vo = r.get("voice")
    if not vo:
        return (f"VAD 종료 {f(g(r, 'vad', 'end_decision_ms', 'p50'), 0)} / {f(g(r, 'vad', 'end_decision_ms', 'max'), 0)} "
                f"(VAD 만) | – | –")
    e = vo["e2e_cmd_ms"]
    return (f"{f(e.get('p50'), 0)} / {f(e.get('p95'), 0)} / **{f(e.get('max'), 0)}** "
            f"| {vo['cmd_accepted_correct']}/{vo['cmd_utts']} | {vo['false_accepts']}/{vo['neg_utts']}")


def section_V(runs: list[dict]) -> list[str]:
    if not runs:
        return ["_시나리오 V 결과 없음_", ""]
    out = ["포즈 없이 음성 명령 체인만 실행 → 각 단계 지연의 기준값.", ""]
    for r in runs:
        out += voice_detail(r)
    return out


def voice_detail(r: dict) -> list[str]:
    vo, v = r.get("voice"), r["vad"]
    if not vo:
        return []
    comps = [("발화 종료 → 판정 (명령 발화)", vo["e2e_cmd_ms"]), ("　VAD 종료 판정 (무음 대기 포함)", vo["vad_end_ms"]),
             ("　큐 대기 (speech 스레드 밀림)", vo["queue_wait_ms"]), ("　Vosk 최종 결과", vo["vosk_final_ms"]),
             ("　ECAPA (특징 + 임베딩)", vo["ecapa_ms"]), ("(참고) 발화 중 Vosk 스트리밍 처리 합계", vo["vosk_accept_total_ms"]),
             ("(참고) VAD 청크 추론", v["infer_ms"]), ("(참고) VAD lag", v["lag_ms"])]
    out = [f"**{r['_dir'].name}** — 데이터 `{vo['dataset']}`, 등록 {vo['n_enroll']}개 문장, 화자 기준 코사인 ≥ {vo['spk_threshold']}", "",
           "| 단계 | p50 (ms) | p95 (ms) | 최대 (ms) | n |", "|---|---:|---:|---:|---:|"]
    for name, d in comps:
        out.append(f"| {name} | {f(d.get('p50'))} | {f(d.get('p95'))} | {f(d.get('max'))} | {d.get('n', 0)} |")
    tgt, oth = vo["spk_score_target"], vo["spk_score_other"]
    out += ["",
            f"- 명령 발화 {vo['cmd_utts']}개 중 정확히 인식·수락 **{vo['cmd_accepted_correct']}**, 명령 인식만 맞음 {vo['cmd_correct']}",
            f"- 명령이 아니거나 다른 화자인 구간 {vo['neg_utts']}개 중 잘못 수락 **{vo['false_accepts']}**",
            f"- 구간 {vo['segments']}개 (라벨 매칭 {vo['matched']}, 라벨 없는 구간 {vo['spurious']}), 놓친 발화 {vo['missed_utts']}/{vo['expected_utts']}",
            f"- 화자 점수: 등록 화자 평균 {f(tgt.get('mean'), 2)} (최소 {f(tgt.get('min'), 2)}), 다른 화자 평균 {f(oth.get('mean'), 2)} (최대 {f(oth.get('max'), 2)})",
            f"- 모델 로드: Vosk {f(vo['vosk_load_ms'], 0)} ms, ECAPA {f(vo['ecapa_load_ms'], 0)} ms, 등록 {f(vo['enroll_ms'], 0)} ms · 최대 RSS {f(r.get('peak_rss_mb'), 0)} MB",
            ""]
    return out + per_command_section(vo) + critical_section(vo)


REASON_KO = {"no_match": "인식 안 됨", "word_conf": "신뢰도 미달", "not_exact": "다른 단어 섞임",
             "speaker": "화자 불일치", "short": "구간 짧음", "other": "기타"}


def pct(n: int, d: int) -> str:
    return f"{100 * n / d:.0f}%" if d else "–"


def per_command_section(vo: dict) -> list[str]:
    pc = vo.get("per_command")
    if not pc:
        return []
    neg = vo["neg_utts"]
    out = ["명령별 결과 — **인식률** = 그 명령 발화를 정확히 수락한 비율, **놓침** = 수락 못 함(이유), "
           f"**잘못 수락** = 그 명령으로 잘못 수락된 횟수 (다른 명령 발화에서 / 명령 아닌 말·다른 화자 {neg}개 구간에서)", "",
           "| 명령 | 말 | 규칙 (단어 신뢰도 / 화자 / 단독) | 발화 | 인식률 | 놓침 (이유) | 다른 명령으로 | 잘못 수락: 명령에서 / 명령 아닌 말에서 (비율) |",
           "|---|---|---|---:|---:|---|---:|---|"]
    for key, c in pc.items():
        tag = " 🚨" if c["critical"] else (" (후보)" if c["group"] else "")
        rules = c["rules"]
        rule_txt = f"{rules['min_word_conf']} / {rules['spk_threshold']} / {'예' if rules['exact'] else '–'}"
        reasons = ", ".join(f"{REASON_KO.get(k, k)} {v}" for k, v in c["missed_reasons"].items())
        out.append(f"| {key}{tag} | {c['phrase']} | {rule_txt} | {c['utts']} | {pct(c['correct'], c['utts'])} ({c['correct']}) "
                   f"| {c['missed']}{f' ({reasons})' if reasons else ''} | {c['confused']} "
                   f"| {c['false_from_other_cmd']} / **{c['false_from_noncmd']}** ({pct(c['false_from_noncmd'], neg)}) |")
    groups: dict[str, list[str]] = {}
    for key, c in pc.items():
        if c["group"]:
            groups.setdefault(c["group"], []).append(
                f"{c['phrase']} {c['correct']}/{c['utts']} (잘못 수락 {c['false_from_other_cmd'] + c['false_from_noncmd']})")
    for gname, items in groups.items():
        out += ["", f"- 같은 동작 후보 비교 (group `{gname}`): " + " vs ".join(items)]
    return out + ["", "🚨 = 위급 명령(잘못 수락 시 거짓 위급 알림) — 더 엄격한 규칙 적용", ""]


def critical_section(vo: dict) -> list[str]:
    pc = vo.get("per_command") or {}
    crit = [k for k, c in pc.items() if c["critical"]]
    if not crit:
        return []
    names = ", ".join(f"'{pc[k]['phrase']}'" for k in crit)
    fa = vo.get("critical_false_accepts", [])
    out = [f"**🚨 명령 아닌 말 → {names} 잘못 수락 (거짓 위급 알림): {len(fa)}건**" + (" — 없음 ✅" if not fa else " ❌"), ""]
    if fa:
        out += ["| 발화 | 화자 | 정답 | 인식 텍스트 | 단어 신뢰도 | 화자 점수 |", "|---|---|---|---|---:|---:|"]
        for d in fa:
            out.append(f"| {d.get('file') or '(라벨 없는 구간)'} | {d.get('speaker') or '–'} | {d.get('label_cmd') or '–'} "
                       f"| {d['text']} | {f(d['cmd_conf'], 2)} | {f(d['spk_score'], 2)} |")
        out.append("")
    rej = vo.get("critical_strict_rejects", [])
    if rej:
        # 진짜 위급 명령 발화(등록 화자 + 그 명령 라벨)인데 막힌 것 = 엄격함의 비용, 나머지 = 막아서 다행인 것
        lost = [d for d in rej if d.get("speaker") == "target" and d.get("label_cmd") == d["raw_cmd"]]
        blocked = len(rej) - len(lost)
        why = ", ".join(sorted({REASON_KO.get(d["reject"], d["reject"]) for d in lost}))
        out.append(f"- 거부된 {names} 후보 {len(rej)}건: 잘못된 후보 차단 {blocked}, 진짜 {names} 를 놓침 {len(lost)}"
                   + (f" ({why})" if lost else ""))
        out.append("")
    return out


def section_B(runs: list[dict], crit: dict) -> list[str]:
    base = [r for r in runs if not g(r, "pose", "fps_cap")]
    if not base:
        return ["_시나리오 B 결과 없음_", ""]
    out = [
        "포즈(최대 속도) + 음성 체인(wav 를 실시간 속도로 투입)을 한 프로세스에서 동시 실행.",
        "발화→판정 = 정답 발화 끝 → 명령 판정 (VAD 만 측정한 실행은 VAD 종료 판정까지). "
        f"VAD lag 32 ms 초과 = 실시간 처리 실패. {UV} = 저전압 감지.",
        "",
        "| 해상도 | 포즈 스레드 | 포즈 FPS | 1초 구간 FPS p5 | 포즈 전체 p95 (ms) | 발화→판정 p50 / p95 / 최대 (ms) | 명령 정답 | 잘못 수락 | VAD lag p95 / max (ms) | 전체 CPU 평균 (%) | 최대 RSS (MB) | 최고 온도 (°C) | 동시 ≥5 FPS | 경고 |",
        "|---:|---:|---:|---:|---:|---|---:|---:|---|---:|---:|---:|:--:|---|",
    ]
    for r in base:
        p, v, s = r["pose"], r["vad"], r.get("system", {})
        ok = PASS if (p.get("fps") or 0) >= crit["pose_min_fps"] else FAIL
        out.append(
            f"| {p['res']} | {p['threads']} | {fps_cell(r)} | {f(p.get('window_fps_p5'), 1)} | {f(g(p, 'total_ms', 'p95'))} "
            f"| {voice_cells(r)} | {f(g(v, 'lag_ms', 'p95'))} / {f(g(v, 'lag_ms', 'max'))} "
            f"| {f(s.get('cpu_total_mean'), 0)} | {f(r.get('peak_rss_mb'), 0)} | {f(s.get('temp_c_max'))} | {ok} | {flag_cell(r)} |")
    return out + [""]


def section_cap(runs: list[dict]) -> list[str]:
    capped = [r for r in runs if g(r, "pose", "fps_cap")]
    if not capped:
        return ["_FPS 상한 실행 없음_", ""]
    base = {(r["pose"]["res"], r["pose"]["threads"]): r for r in runs if not g(r, "pose", "fps_cap")}
    out = ["같은 해상도·스레드에서 포즈 FPS 상한을 건 실행과 상한 없는 실행(B)을 비교. 둘 다 50°C 이하에서 시작해 같은 시간 동안 실행.",
           "",
           "| 해상도 | 포즈 스레드 | 상한 | 포즈 FPS (상한 없음 → 상한) | 전체 CPU 평균 % | 프로세스 CPU 평균 % | 최고 온도 °C | 평균 온도 °C | 발화→판정 최대 (ms) |",
           "|---:|---:|---:|---|---|---|---|---|---|"]
    for r in capped:
        p = r["pose"]
        b = base.get((p["res"], p["threads"]))

        def pair(path, nd=0):
            new = g(r, *path)
            old = g(b, *path) if b else None
            if old is None or new is None:
                return f(new, nd)
            return f"{f(old, nd)} → **{f(new, nd)}** ({new - old:+.{nd}f})"
        out.append(f"| {p['res']} | {p['threads']} | {p['fps_cap']:g} | {pair(('pose', 'fps'), 2)} "
                   f"| {pair(('system', 'cpu_total_mean'))} | {pair(('system', 'proc_cpu_mean'))} "
                   f"| {pair(('system', 'temp_c_max'), 1)} | {pair(('system', 'temp_c_mean'), 1)} "
                   f"| {pair(('voice', 'e2e_cmd_ms', 'max'))} |")
    out += ["", "_실행 시간이 짧아(2분) 온도는 정상 상태에 도달하기 전 값 — 장시간 효과는 시나리오 C 에서 fps_cap 을 걸어 확인._", ""]
    return out


def section_C(runs: list[dict]) -> list[str]:
    if not runs:
        return ["_시나리오 C 미측정 (쿨러 장착 후 측정 예정)_", ""]
    out = []
    for r in runs:
        p = r["pose"]
        out += [f"설정: 해상도 {p['res']}, 포즈 스레드 {p['threads']}, 상한 {p.get('fps_cap') or '없음'}, "
                f"{r['duration_s'] / 60:.0f}분. 선택 근거: {r.get('choice_reason') or '수동 지정'}", "",
                "| 구간 시작 (분) | 포즈 FPS | 포즈 전체 p95 (ms) | 최고 온도 (°C) | 최저 클럭 (MHz) | 전체 CPU 평균 (%) | 최대 RSS (MB) | 스로틀 샘플 |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|"]
        sysb = {b["t0_s"]: b for b in r.get("system_buckets", [])}
        for b in p.get("buckets", []):
            sb = sysb.get(b["t0_s"], {})
            out.append(f"| {b['t0_s'] / 60:.0f} | {f(b['fps'], 2)} | {f(b['total_p95_ms'])} | {f(sb.get('temp_max_c'))} "
                       f"| {f(sb.get('freq_min_mhz'), 0)} | {f(sb.get('cpu_total_mean'), 0)} | {f(sb.get('rss_max_mb'), 0)} "
                       f"| {sb.get('throttle_samples', '–')} |")
        s = r.get("system", {})
        out += ["", f"전체: 포즈 {f(p['fps'], 2)} FPS, 최고 온도 {f(s.get('temp_c_max'))} °C, 최저 클럭 {f(s.get('arm_freq_mhz_min'), 0)} MHz, "
                f"스로틀 이벤트 {s.get('throttle_events', '측정 불가(vcgencmd 없음)')}, 최대 RSS {f(r.get('peak_rss_mb'), 0)} MB, "
                f"스왑 최대 {f(s.get('swap_used_mb_max'), 0)} MB", ""]
        out += voice_detail(r)
    return out


def verdict(runs: dict, crit: dict, ref: dict) -> list[str]:
    C = runs["C"][-1] if runs["C"] else None
    B = [r for r in runs["B"] if not g(r, "pose", "fps_cap")]
    V = runs["V"][-1] if runs["V"] else None
    rows = []
    best = max(B, key=lambda r: r["pose"]["fps"]) if B else None
    # 1) 포즈 FPS (동시 실행)
    src = C or best
    if src:
        fps = src["pose"]["fps"]
        tag = (f"C({C['pose']['res']}/{C['pose']['threads']}스레드, {C['duration_s'] / 60:.0f}분)" if C
               else f"B 최고 {best['pose']['res']}/{best['pose']['threads']}스레드")
        rows.append(("포즈 추정: 동시 실행 ≥ {:.0f} FPS".format(crit["pose_min_fps"]), "초안",
                     f"{tag} {fps:.2f} FPS" + (f" {UV}" if uv(src) else ""), PASS if fps >= crit["pose_min_fps"] else FAIL))
        ok_runs = [r for r in B if r["pose"]["fps"] >= crit["pose_min_fps"]]
        rows.append(("　└ 5 FPS 이상인 B 조합", "참고", ", ".join(f"{r['pose']['res']}/{r['pose']['threads']}" for r in ok_runs) or "없음",
                     f"{len(ok_runs)}/{len(B)}"))
    else:
        rows.append(("포즈 추정: 동시 실행 ≥ 5 FPS", "초안", "–", NA))
    # 2) 음성: 발화 종료 → 명령 판정
    budget = crit["voice_max_end_to_decision_s"] * 1000
    name = "음성: 발화 종료 → 명령 판정 ≤ {:.1f} s".format(crit["voice_max_end_to_decision_s"])
    full_B = [r for r in B if r.get("voice")]
    if C and C.get("voice") or full_B:
        worst = C if (C and C.get("voice")) else max(full_B, key=lambda r: r["voice"]["e2e_cmd_ms"].get("max") or 0)
        e = worst["voice"]["e2e_cmd_ms"]
        where = "C" if worst is C else f"B 중 가장 느린 {worst['pose']['res']}/{worst['pose']['threads']}스레드"
        rows.append((name, "초안", f"{where}: 명령 발화 {e.get('n', 0)}개 p95 {f(e.get('p95'), 0)} ms, 최대 {f(e.get('max'), 0)} ms",
                     PASS if (e.get("max") or 1e9) <= budget else FAIL))
        if best and best.get("voice") and best is not worst:
            eb = best["voice"]["e2e_cmd_ms"]
            rows.append(("　└ 포즈 최고속 조합에서", "참고", f"{best['pose']['res']}/{best['pose']['threads']}: p95 {f(eb.get('p95'), 0)} / 최대 {f(eb.get('max'), 0)} ms",
                         PASS if (eb.get("max") or 1e9) <= budget else FAIL))
    elif src and g(src, "vad", "end_decision_ms", "max") is not None:
        mx = src["vad"]["end_decision_ms"]["max"]
        rows.append((name, "초안", f"VAD 종료 판정 최대 {mx:.0f} ms → 화자·명령 인식에 남는 예산 {budget - mx:.0f} ms",
                     PARTIAL if mx < budget else FAIL))
    else:
        rows.append((name, "초안", "–", NA))
    if V and V.get("voice"):
        e = V["voice"]["e2e_cmd_ms"]
        rows.append(("　└ 음성 체인 단독(V)", "참고", f"p95 {f(e.get('p95'), 0)} / 최대 {f(e.get('max'), 0)} ms",
                     PASS if (e.get("max") or 1e9) <= budget else FAIL))
    acc_src = (C if C and C.get("voice") else None) or V or (full_B[0] if full_B else None)
    if acc_src and acc_src.get("voice"):
        vo = acc_src["voice"]
        rows.append(("음성 정확도 (명령 수락 / 잘못 수락)", "참고",
                     f"{vo['cmd_accepted_correct']}/{vo['cmd_utts']} / {vo['false_accepts']}/{vo['neg_utts']} — 데이터 `{vo['dataset']}`",
                     "합성 음성" if vo["dataset"].startswith("tts") else "–"))
        crit_names = [c["phrase"] for c in (vo.get("per_command") or {}).values() if c["critical"]]
        if crit_names:
            n = len(vo.get("critical_false_accepts", []))
            rows.append((f"🚨 명령 아닌 말 → {'/'.join(crit_names)} 잘못 수락 = 0", "참고",
                         f"{n}건 / 명령 아닌 구간 {vo['neg_utts']}개", PASS if n == 0 else FAIL))
    # 3) 장시간
    if C:
        s = C.get("system", {})
        minutes = C["duration_s"] / 60
        if "throttle_events" in s:
            ok = not s["throttle_events"] and minutes >= crit["longrun_min_minutes"]
            rows.append((f"장시간: {crit['longrun_min_minutes']}분 스로틀링 없음", "초안",
                         f"{minutes:.0f}분, 이벤트 {s['throttle_events'] or '없음'}, 최고 {f(s.get('temp_c_max'))} °C, 최저 클럭 {f(s.get('arm_freq_mhz_min'), 0)} MHz",
                         PASS if ok else FAIL))
        else:
            rows.append(("장시간: 스로틀링 없음", "초안", "vcgencmd 없음 (Pi 아님)", NA))
        rss = C.get("peak_rss_mb")
        rows.append((f"메모리 ≤ {crit['max_rss_mb'] / 1024:.0f} GB", "초안", f"최대 RSS {f(rss, 0)} MB, 스왑 최대 {f(s.get('swap_used_mb_max'), 0)} MB",
                     PASS if rss is not None and rss <= crit["max_rss_mb"] else FAIL))
        p = C["pose"]
        for nm, val, thr, le in [
            ("1초 구간 FPS p5 ≥ {}".format(ref["pose_window_fps_p5"]), p.get("window_fps_p5"), ref["pose_window_fps_p5"], False),
            ("포즈 전체 지연 p95 ≤ {} ms".format(ref["pose_total_latency_p95_ms"]), g(p, "total_ms", "p95"), ref["pose_total_latency_p95_ms"], True),
            ("최고 온도 ≤ {} °C".format(ref["max_temp_c"]), s.get("temp_c_max"), ref["max_temp_c"], True),
            ("전체 CPU 평균 ≤ {} %".format(ref["max_cpu_total_pct"]), s.get("cpu_total_mean"), ref["max_cpu_total_pct"], True),
            ("최대 RSS ≤ {} MB".format(ref["max_rss_mb"]), rss, ref["max_rss_mb"], True),
        ]:
            res = NA if val is None else (PASS if (val <= thr if le else val >= thr) else FAIL)
            rows.append((nm, "참고", f(val, 1), res))
    else:
        rows.append(("장시간: 30분 스로틀링 없음 / 메모리 ≤ 3 GB", "초안", "시나리오 C 미측정 (쿨러 장착 후)", NA))
        peak = max((r.get("peak_rss_mb") or 0 for k in "AVB" for r in runs[k]), default=0)
        if peak:
            rows.append(("　└ (참고) 지금까지 최대 RSS", "참고", f"{peak:.0f} MB", PASS if peak <= crit["max_rss_mb"] else FAIL))

    flagged = [r["_dir"].name for k in "AVBC" for r in runs[k] if r.get("flags")]
    uv_runs = [r["_dir"].name for k in "AVBC" for r in runs[k] if uv(r)]
    out = ["| 기준 | 구분 | 측정값 | 판정 |", "|---|---|---|:--:|"]
    out += [f"| {a} | {b} | {c} | {d} |" for a, b, c, d in rows]
    out += ["", "_구분: 초안 = 사용자 초안 기준(판정 근거), 참고 = 보조 정보. 음성 기준은 '명령 발화(등록 화자 + 명령어)' 전부가 1초 이내일 때 통과(최대값 기준)._", ""]
    if uv_runs:
        out += [f"{UV} **저전압 감지 실행 {len(uv_runs)}개 — 저전압 영향 가능성 있음** (아래 '전원 상태' 참고): {', '.join(uv_runs)}", ""]
    out += ([f"⚠ **경고 표시된 실행 {len(flagged)}개** (80°C 초과 / throttled≠0 / 시작 온도 초과 / 저전압) — 표의 '경고' 열 참고", ""]
            if flagged else ["경고 표시된 실행 없음 (모든 실행이 80°C 이하, throttled=0x0, 50°C 이하에서 시작)", ""])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("session", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    session = args.session if args.session.is_absolute() else (Path.cwd() / args.session)
    crit_all = tomllib.loads((ROOT / "configs" / "criteria.toml").read_text(encoding="utf-8"))
    vis_thr = tomllib.loads((ROOT / "configs" / "pose.toml").read_text(encoding="utf-8"))["model"]["kpt_conf_threshold"]
    env = json.loads((session / "env.json").read_text(encoding="utf-8")) if (session / "env.json").exists() else {}
    runs = load_runs(session)

    cooler = env.get("cooler", "unknown")
    cooler_txt = {"none": "**쿨러 없음** (cooler: none) — 쿨러 장착 후 결과와 따로 비교할 것",
                  "active": "액티브 쿨러 (cooler: active)"}.get(cooler, f"cooler: {cooler}")
    lines = [f"# 결과: Python + CPU 만으로 YOLO11n-pose + 음성 명령 ({session.name})", "", f"냉각 상태: {cooler_txt}", "",
             "## 판정 요약", ""] + verdict(runs, crit_all["draft"], crit_all["ref"])
    pw = power_section(runs)
    if pw:
        lines += ["## 전원 상태 (저전압)", ""] + pw
    lines += ["## 시나리오 A — 포즈 단독 (해상도 × 스레드)", ""] + section_A(runs["A"], crit_all["draft"], vis_thr)
    lines += ["## 시나리오 V — 음성 명령 체인 단독", ""] + section_V(runs["V"])
    lines += ["## 시나리오 B — 포즈 + 음성 동시 실행", ""] + section_B(runs["B"], crit_all["draft"])
    full_B = [r for r in runs["B"] if r.get("voice") and not g(r, "pose", "fps_cap")]
    if full_B:
        worst = max(full_B, key=lambda r: r["voice"]["e2e_cmd_ms"].get("max") or 0)
        lines += ["### B 중 발화→판정이 가장 느린 조합의 단계별 지연", ""] + voice_detail(worst)
    lines += ["## 포즈 FPS 상한 효과", ""] + section_cap(runs["B"])
    lines += ["## 시나리오 C — 장시간", ""] + section_C(runs["C"])
    lines += ["## 측정 환경", ""] + env_section(env)
    text = "\n".join(lines)
    out = args.out or session / "REPORT.md"
    out.write_text(text, encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    print(text)
    print(f"\n→ {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
