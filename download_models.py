"""Скачивает модели диаризации в models/ (≈46 МБ). Модель Whisper скачивается сама при первом запуске."""
import tarfile
import urllib.request
from pathlib import Path

MODELS = Path(__file__).parent / "models"
BASE = "https://github.com/k2-fsa/sherpa-onnx/releases/download"
SEG = f"{BASE}/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
EMB = f"{BASE}/speaker-recongition-models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"


def main():
    MODELS.mkdir(exist_ok=True)
    if not (MODELS / "sherpa-onnx-pyannote-segmentation-3-0" / "model.onnx").exists():
        archive = MODELS / "seg.tar.bz2"
        print("Скачиваю сегментацию (pyannote 3.0)…")
        urllib.request.urlretrieve(SEG, archive)
        with tarfile.open(archive) as tar:
            tar.extractall(MODELS)
        archive.unlink()
    if not (MODELS / "embedding.onnx").exists():
        print("Скачиваю модель голосовых эмбеддингов (3D-Speaker)…")
        urllib.request.urlretrieve(EMB, MODELS / "embedding.onnx")
    print("Готово:", MODELS)


if __name__ == "__main__":
    main()
