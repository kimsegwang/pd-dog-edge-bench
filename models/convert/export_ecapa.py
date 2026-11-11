"""[노트북 전용] SpeechBrain ECAPA-TDNN(spkrec-ecapa-voxceleb, Apache-2.0) → ONNX + 특징 추출 상수.

    python models/convert/export_ecapa.py

결과
  models/ecapa_tdnn_voxceleb_fp32.onnx   입력 feats (1, T, 80) float32 → 출력 embedding (1, 192)
  models/ecapa_fbank.npz                 특징 추출 상수 (멜 필터 행렬, 윈도우, n_fft, hop, top_db)
  models/ecapa_export.json               버전, sha256, 검증 결과

특징 추출(STFT·멜·log·정규화)은 ONNX 밖에서 numpy(그리고 이후 Rust)로 계산한다.
멜 필터 행렬은 SpeechBrain 에서 그대로 뽑아 저장하므로 두 언어가 같은 상수를 쓴다.
검증: numpy 특징 + ONNX 임베딩 vs SpeechBrain encode_batch (같은 wav) 의 코사인 유사도/최대 오차.
"""
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"
WORK = Path(__file__).resolve().parent / "_work"
OPSET = 17
SOURCE = "speechbrain/spkrec-ecapa-voxceleb"

sys.path.insert(0, str(ROOT / "python"))
from pdbench.models.ecapa import fbank_features  # noqa: E402  (런타임과 같은 numpy 구현으로 검증)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    import onnxruntime as ort
    import speechbrain
    import torch
    from speechbrain.inference.speaker import EncoderClassifier
    from speechbrain.utils.fetching import LocalStrategy

    WORK.mkdir(exist_ok=True)
    clf = EncoderClassifier.from_hparams(source=SOURCE, savedir=str(WORK / "spkrec-ecapa-voxceleb"),
                                         run_opts={"device": "cpu"}, local_strategy=LocalStrategy.COPY)
    clf.eval()
    feat = clf.mods.compute_features
    stft, fb = feat.compute_STFT, feat.compute_fbanks
    assert not feat.deltas and not feat.context

    # 1) 멜 필터 행렬: log 끄고 단위행렬(각 주파수 bin 하나씩)을 통과시켜 그대로 추출
    n_bins = stft.n_fft // 2 + 1
    fb.log_mel = False
    with torch.no_grad():
        mel = fb(torch.eye(n_bins).unsqueeze(0))[0].numpy().astype(np.float32)  # (n_bins, n_mels)
    fb.log_mel = True
    window = stft.window.numpy().astype(np.float32)
    consts = {
        "sample_rate": int(stft.sample_rate), "n_fft": int(stft.n_fft),
        "win_length": int(stft.win_length), "hop_length": int(stft.hop_length),
        "center": bool(stft.center), "pad_mode": str(stft.pad_mode),
        "top_db": float(fb.top_db), "amin": float(fb.amin), "multiplier": float(fb.multiplier),
        "db_multiplier": float(fb.db_multiplier),
    }
    assert consts["pad_mode"] == "constant" and consts["center"], consts
    np.savez(MODELS / "ecapa_fbank.npz", mel=mel, window=window,
             **{k: np.array(v) for k, v in consts.items() if not isinstance(v, str)})

    # 2) 임베딩 신경망만 ONNX 로 (입력: 정규화된 특징)
    class Embed(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, feats):
            return self.m(feats).reshape(1, -1)

    net = Embed(clf.mods.embedding_model).eval()
    dst = MODELS / "ecapa_tdnn_voxceleb_fp32.onnx"
    dummy = torch.randn(1, 200, mel.shape[1])
    torch.onnx.export(net, (dummy,), str(dst), input_names=["feats"], output_names=["embedding"],
                      dynamic_axes={"feats": {1: "frames"}}, opset_version=OPSET, dynamo=False)

    # 3) 검증: SpeechBrain 원본 vs (numpy 특징 + ONNX)
    sess = ort.InferenceSession(str(dst), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    checks = []
    wav_dir = ROOT / "data" / "voice" / "tts_heami" / "utts"
    wavs = sorted(wav_dir.glob("*.wav"))[:6] if wav_dir.exists() else []
    import wave
    signals = []
    for p in wavs:
        with wave.open(str(p)) as w:
            signals.append((p.name, np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768))
    signals.append(("noise_1.5s", (rng.standard_normal(24000) * 0.05).astype(np.float32)))
    for name, x in signals:
        with torch.no_grad():
            ref = clf.encode_batch(torch.from_numpy(x)[None]).reshape(-1).numpy()
        f = fbank_features(x, np.load(MODELS / "ecapa_fbank.npz"))
        emb = sess.run(None, {"feats": f[None]})[0].reshape(-1)
        cos = float(emb @ ref / (np.linalg.norm(emb) * np.linalg.norm(ref)))
        checks.append({"input": name, "frames": int(f.shape[0]), "cosine_vs_speechbrain": cos,
                       "max_abs_diff": float(np.abs(emb - ref).max())})
        print(f"  {name:28s} frames={f.shape[0]:4d} cos={cos:.6f} maxdiff={np.abs(emb - ref).max():.2e}")

    report = {
        "source": SOURCE, "license": "Apache-2.0", "opset": OPSET,
        "speechbrain": speechbrain.__version__, "torch": torch.__version__, "host": platform.platform(),
        "onnx": {"file": dst.name, "sha256": sha256(dst), "bytes": dst.stat().st_size,
                 "input": "feats (1, frames, 80) — sentence mean-normalized log-mel", "output": "embedding (1, 192)"},
        "fbank": {"file": "ecapa_fbank.npz", "sha256": sha256(MODELS / "ecapa_fbank.npz"), **consts,
                  "n_mels": int(mel.shape[1])},
        "verification": checks,
    }
    (MODELS / "ecapa_export.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    worst = min(c["cosine_vs_speechbrain"] for c in checks)
    print(f"wrote {dst.name}, ecapa_fbank.npz, ecapa_export.json  (worst cosine {worst:.6f})")
    if worst < 0.999:
        raise SystemExit("검증 실패: numpy 특징 + ONNX 결과가 SpeechBrain 과 다름")


if __name__ == "__main__":
    main()
