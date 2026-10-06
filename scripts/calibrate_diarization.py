"""Подбор порога диаризации на тестовой записи: правильный ответ — 3 говорящих
(Ольга, Дмитрий, Светлана). Печатает для каждого порога число найденных говорящих
и разметку реплик, чтобы сверить со сценарием в make_test_meeting.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import diarize  # noqa: E402
from transcribe import decode_audio  # noqa: E402

samples = decode_audio(str(Path(__file__).parent / "test_meeting.mp3"))
for th in (0.3, 0.4, 0.5, 0.6, 0.7):
    diarize.THRESHOLD = th
    diarize._sd = None
    turns = diarize.diarize(samples)
    seq = " ".join(str(t.speaker) for t in turns)
    print(f"порог {th}: говорящих {len({t.speaker for t in turns})}, отрезков {len(turns)} | {seq}")
