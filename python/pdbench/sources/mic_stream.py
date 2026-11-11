"""[자리만 잡아 둠] USB 마이크 입력 소스. 마이크 도착 후 구현.

계획: sounddevice.InputStream(samplerate=16000, channels=1, dtype='int16', blocksize=512) 콜백에서
AudioChunk 를 큐에 넣고, arrival_ns 는 콜백 시점의 perf_counter_ns 를 쓴다.
RealtimeWavSource 와 같은 인터페이스이므로 시나리오 코드는 바꾸지 않아도 된다.
"""
from .audio_base import AudioSource


class MicSource(AudioSource):
    def __init__(self, device=None, chunk_samples: int = 512, sample_rate: int = 16000):
        raise NotImplementedError("USB 마이크 입력은 아직 구현되지 않음 — wav 파일 입력(RealtimeWavSource)을 사용")
