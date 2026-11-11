"""wav 파일을 실시간 속도로 흘려 넣는 오디오 소스 (마이크 대체).

청크 k 는 t0 + (k+1)·chunk_dur 시각에 '도착'한다. 처리 스레드가 늦으면 sleep 없이 바로 다음 청크를
주므로, (처리 완료 시각 - 도착 시각) = 실제 마이크였다면 쌓였을 지연(lag) 이 된다.
"""
import time
import wave
from pathlib import Path

import numpy as np

from .audio_base import AudioChunk, AudioSource


class RealtimeWavSource(AudioSource):
    def __init__(self, path: Path, chunk_samples: int = 512, expected_sr: int = 16000, loop: bool = True):
        with wave.open(str(path), "rb") as w:
            if (w.getnchannels(), w.getsampwidth(), w.getframerate()) != (1, 2, expected_sr):
                raise ValueError(f"{path}: need mono 16-bit {expected_sr}Hz (use data/tools/wav_to_16k.py)")
            pcm = np.frombuffer(w.readframes(w.getnframes()), np.int16)
        self.audio = (pcm.astype(np.float32) / 32768.0)
        self.sample_rate = expected_sr
        self.chunk_samples = chunk_samples
        self.loop = loop
        self.t0_ns: int | None = None

    def sample_time_ns(self, sample_index: int) -> int:
        return self.t0_ns + int(sample_index * 1e9 / self.sample_rate)

    def __iter__(self):
        n_per_pass = len(self.audio) // self.chunk_samples
        self.t0_ns = time.perf_counter_ns()
        k = 0
        while True:
            if not self.loop and k >= n_per_pass:
                return
            pos = (k % n_per_pass) * self.chunk_samples
            start_sample = k * self.chunk_samples
            arrival = self.sample_time_ns(start_sample + self.chunk_samples)
            delay = arrival - time.perf_counter_ns()
            if delay > 0:
                time.sleep(delay / 1e9)
            yield AudioChunk(k, self.audio[pos:pos + self.chunk_samples], start_sample, arrival)
            k += 1
