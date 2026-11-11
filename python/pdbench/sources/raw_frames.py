"""무압축 RGB24 프레임 파일(.rgb + .json)을 순차로 읽는 소스. 끝에 닿으면 처음으로 되감는다.

전체를 메모리에 올리지 않고 프레임 하나 크기의 버퍼에 readinto 한다
→ 프로세스 RSS 가 입력 파일 크기에 좌우되지 않음 (반복 시에는 OS 페이지 캐시가 처리).
"""
import json
from pathlib import Path

import numpy as np


class RawFrameSource:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.meta = json.loads(self.path.with_suffix(".json").read_text())
        self.width, self.height, self.n_frames = self.meta["width"], self.meta["height"], self.meta["frames"]
        self.frame_bytes = self.width * self.height * 3
        self._f = open(self.path, "rb", buffering=0)
        self._buf = bytearray(self.frame_bytes)
        self._view = np.frombuffer(self._buf, np.uint8).reshape(self.height, self.width, 3)
        self.index = 0

    def read(self) -> tuple[int, np.ndarray]:
        """(프레임 번호, HxWx3 RGB uint8). 반환 배열은 다음 read() 에서 덮어써짐."""
        if self.index >= self.n_frames:
            self._f.seek(0)
            self.index = 0
        n = self._f.readinto(self._buf)
        if n != self.frame_bytes:
            raise IOError(f"short read at frame {self.index}: {n} bytes")
        idx = self.index
        self.index += 1
        return idx, self._view

    def close(self) -> None:
        self._f.close()
