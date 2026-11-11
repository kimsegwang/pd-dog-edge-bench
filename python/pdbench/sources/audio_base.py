"""오디오 청크 소스 공통 인터페이스. wav 파일 / 마이크가 같은 형태로 청크를 준다."""
from dataclasses import dataclass

import numpy as np


@dataclass
class AudioChunk:
    index: int
    samples: np.ndarray      # float32, [-1, 1], 길이 = chunk_samples
    start_sample: int        # 스트림 시작 기준 이 청크의 첫 샘플 번호 (반복 재생 시에도 계속 증가)
    arrival_ns: int          # 이 청크의 마지막 샘플이 '녹음 완료'된 시각 (perf_counter_ns 기준)


class AudioSource:
    sample_rate: int
    chunk_samples: int

    def __iter__(self):
        raise NotImplementedError

    def sample_time_ns(self, sample_index: int) -> int:
        """스트림 상의 샘플 번호가 녹음된 시각 (perf_counter_ns 기준)."""
        raise NotImplementedError

    def close(self) -> None:
        pass
