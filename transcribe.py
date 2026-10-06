"""Распознавание речи локально через faster-whisper — без внешнего API и без ключа.

Модель скачивается при первом запуске (small ≈ 480 МБ) и кэшируется.
На CPU small распознаёт быстрее реального времени; medium точнее, но в 2-3 раза медленнее.
"""
import os
import time
from dataclasses import dataclass

import av
import numpy as np
from faster_whisper import WhisperModel

SAMPLE_RATE = 16000


def decode_audio(path: str) -> np.ndarray:
    """Любой аудио/видеофайл → моно 16 кГц float32, как ждёт Whisper.

    Свой декодер вместо встроенного в faster-whisper: тот вызывает av.open(metadata_errors=...),
    а в av 19 этого параметра больше нет, старый av под Python 3.12/Windows не собирается.
    """
    resampler = av.AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)
    chunks = []
    with av.open(path) as container:
        for frame in container.decode(audio=0):
            for out in resampler.resample(frame):
                chunks.append(out.to_ndarray().reshape(-1))
        for out in resampler.resample(None):  # дослать хвост из буфера ресемплера
            chunks.append(out.to_ndarray().reshape(-1))
    if not chunks:
        raise ValueError("В файле нет звуковой дорожки")
    return np.concatenate(chunks).astype(np.float32) / 32768.0

MODEL_SIZE = os.getenv("WHISPER_MODEL") or "small"

_model = None


def _get_model() -> WhisperModel:
    global _model
    if _model is None:
        # int8 на CPU — в разы быстрее float32 при почти той же точности
        _model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
    return _model


@dataclass
class Transcript:
    text: str  # «[мм:сс] Спикер N: …» — для выжимки и для файла с расшифровкой
    speakers: int
    duration_sec: float
    elapsed_sec: float


def _ts(sec: float) -> str:
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def transcribe(path: str) -> Transcript:
    """Текст + кто говорит. Подряд идущие фразы одного говорящего склеиваются в одну реплику."""
    import diarize  # тяжёлый импорт моделей — только когда реально нужен

    started = time.monotonic()
    samples = decode_audio(path)
    segments, info = _get_model().transcribe(
        samples,
        language="ru",
        vad_filter=True,  # пропускает тишину — быстрее и меньше «галлюцинаций» на паузах
        beam_size=5,
    )
    segments = [s for s in segments if s.text.strip()]
    turns = diarize.diarize(samples)

    # Номера говорящих из кластеризации произвольные (0, 2, 5…) — перенумеровываем по порядку появления.
    numbering: dict[int, int] = {}
    replicas: list[list] = []  # [start, номер, текст]
    for seg in segments:
        raw = diarize.speaker_for(seg.start, seg.end, turns)
        num = numbering.setdefault(raw, len(numbering) + 1) if raw is not None else 0
        if replicas and replicas[-1][1] == num:
            replicas[-1][2] += " " + seg.text.strip()
        else:
            replicas.append([seg.start, num, seg.text.strip()])

    lines = [f"[{_ts(st)}] {'Спикер ' + str(n) if n else 'Неизвестный'}: {t}" for st, n, t in replicas]
    return Transcript(
        text="\n".join(lines),
        speakers=len(numbering),
        duration_sec=info.duration,
        elapsed_sec=time.monotonic() - started,
    )
