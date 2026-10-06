"""Выжимка из файла без Telegram — для больших записей (Telegram-боту доступны файлы только до 20 МБ).

    python cli.py запись.mp3
Рядом сохраняет запись.transcript.txt и запись.summary.txt
"""
import json
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

from summarize import summarize, to_telegram_html  # noqa: E402
from transcribe import transcribe  # noqa: E402


def main():
    if len(sys.argv) < 2:
        sys.exit("Использование: python cli.py <файл записи>")
    path = Path(sys.argv[1])
    print(f"Распознаю {path.name}…")
    tr = transcribe(str(path))
    print(f"  {tr.duration_sec:.0f} с записи распознано за {tr.elapsed_sec:.0f} с, говорящих: {tr.speakers}")

    print("Готовлю выжимку…")
    summary = summarize(tr.text)
    plain = re.sub(r"<[^>]+>", "", to_telegram_html(summary))

    path.with_suffix(".transcript.txt").write_text(tr.text, encoding="utf-8")
    path.with_suffix(".summary.txt").write_text(plain, encoding="utf-8")
    print()
    print(plain)
    if "--json" in sys.argv:
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
