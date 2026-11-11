"""공유 TOML 설정 로드와 레포 경로 처리."""
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def load(name: str) -> dict:
    with open(REPO_ROOT / "configs" / f"{name}.toml", "rb") as f:
        return tomllib.load(f)


def repo_path(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p
