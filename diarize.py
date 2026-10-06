"""Диаризация — кто когда говорит. sherpa-onnx: сегментация pyannote 3.0 + эмбеддинги голоса 3D-Speaker,
всё в ONNX на CPU, без PyTorch и без токена HuggingFace. Модели лежат в models/ (см. README, как скачать).

Whisper даёт текст по сегментам, но не знает, чей это голос. Здесь каждому сегменту Whisper
приписывается говорящий с наибольшим пересечением по времени.
"""
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sherpa_onnx

MODELS = Path(__file__).parent / "models"
SEG_MODEL = MODELS / "sherpa-onnx-pyannote-segmentation-3-0" / "model.onnx"
EMB_MODEL = MODELS / "embedding.onnx"
# Порог кластеризации: меньше — больше разных говорящих; калибруется на тестовой записи.
THRESHOLD = float(os.getenv("DIARIZATION_THRESHOLD") or 0.9)

_sd = None


@dataclass
class Turn:
    start: float
    end: float
    speaker: int


def _get() -> sherpa_onnx.OfflineSpeakerDiarization:
    global _sd
    if _sd is None:
        config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
            segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
                pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(model=str(SEG_MODEL)),
            ),
            embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(EMB_MODEL)),
            clustering=sherpa_onnx.FastClusteringConfig(num_clusters=-1, threshold=THRESHOLD),
            min_duration_on=0.3,
            min_duration_off=0.5,
        )
        if not config.validate():
            raise RuntimeError(f"Модели диаризации не найдены в {MODELS}")
        _sd = sherpa_onnx.OfflineSpeakerDiarization(config)
    return _sd


def diarize(samples: np.ndarray) -> list[Turn]:
    result = _get().process(samples).sort_by_start_time()
    return [Turn(r.start, r.end, r.speaker) for r in result]


def speaker_for(start: float, end: float, turns: list[Turn]) -> int | None:
    """Говорящий с наибольшим пересечением с отрезком [start, end]."""
    best, best_overlap = None, 0.0
    for t in turns:
        overlap = min(end, t.end) - max(start, t.start)
        if overlap > best_overlap:
            best, best_overlap = t.speaker, overlap
    return best
