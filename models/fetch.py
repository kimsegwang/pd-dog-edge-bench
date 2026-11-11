"""Pi 에서 바로 받을 수 있는 모델(Silero VAD, Vosk 한국어 small)을 내려받고 sha256 을 검증한다.
노트북에서 변환해 옮기는 모델(YOLO11n-pose, ECAPA)은 있는지만 확인한다.

    python models/fetch.py
"""
import hashlib
import sys
import tomllib
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    manifest = tomllib.loads((ROOT / "models" / "manifest.toml").read_text(encoding="utf-8"))
    ok = True
    for name, m in manifest.items():
        if m.get("where") != "pi":
            continue
        dst = ROOT / m["file"]
        if not dst.exists():
            print(f"[{name}] downloading {m['url']}")
            urllib.request.urlretrieve(m["url"], dst)
        digest = sha256(dst)
        good = digest == m["sha256"]
        ok &= good
        print(f"[{name}] {dst.relative_to(ROOT)} sha256={digest[:16]}… {'OK' if good else 'MISMATCH'}")
        if good and m.get("extract_to") and not (ROOT / m["extract_to"]).exists():
            with zipfile.ZipFile(dst) as z:
                z.extractall(dst.parent)
            print(f"[{name}] extracted → {m['extract_to']}")
    for name, m in manifest.items():
        if m.get("where") == "laptop":
            for f in m["files"]:
                print(f"[{name}] {f}: {'present' if (ROOT / f).exists() else 'MISSING (노트북에서 변환 후 scp)'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
