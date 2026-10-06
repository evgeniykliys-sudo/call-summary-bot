import asyncio
import logging
import os
import tempfile
from pathlib import Path

from aiogram import Bot, Dispatcher, F
from aiogram.filters import CommandStart
from aiogram.types import BufferedInputFile, Message
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

from summarize import summarize, to_telegram_html  # noqa: E402
from transcribe import transcribe  # noqa: E402

BOT_TOKEN = os.getenv("BOT_TOKEN")
# Кому разрешено пользоваться ботом (через запятую). Пусто — всем. Ключ Claude платный, поэтому лучше ограничить.
ALLOWED = {int(x) for x in (os.getenv("ALLOWED_USER_IDS") or "").replace(" ", "").split(",") if x}
# Bot API отдаёт ботам файлы только до 20 МБ — больше скачать нельзя в принципе.
MAX_FILE_MB = 20

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()
logging.basicConfig(level=logging.INFO)

# Распознавание грузит CPU целиком — обрабатываем записи по одной.
_job_lock = asyncio.Lock()

HELP = (
    "Пришлите запись созвона или планёрки — голосовое, аудио или видео (до 20 МБ).\n\n"
    "Верну:\n"
    "• выжимку: о чём договорились, кто что делает и к какому сроку, открытые вопросы;\n"
    "• полную расшифровку с разметкой, кто говорит, — файлом.\n\n"
    "Обработка — примерно минута на минуту записи. Запись длиннее? Сожмите в mp3 64 кбит/с "
    "(час разговора ≈ 28 МБ — можно разделить на две части)."
)


@dp.message(CommandStart())
async def start(message: Message):
    await message.answer(HELP)


@dp.message(F.voice | F.audio | F.video | F.video_note | F.document)
async def handle_record(message: Message):
    if ALLOWED and message.from_user.id not in ALLOWED:
        return

    media = message.voice or message.audio or message.video or message.video_note or message.document
    if message.document and not (message.document.mime_type or "").startswith(("audio/", "video/")):
        await message.answer("Это не похоже на аудио или видео. Пришлите запись созвона.")
        return
    if media.file_size and media.file_size > MAX_FILE_MB * 1024 * 1024:
        await message.answer(
            f"Файл {media.file_size / 1024 / 1024:.0f} МБ — Telegram не даёт ботам скачивать файлы больше "
            f"{MAX_FILE_MB} МБ. Сожмите запись в mp3 64 кбит/с или разделите на части."
        )
        return

    status = await message.answer("⏳ В очереди…" if _job_lock.locked() else "⏳ Скачиваю запись…")
    async with _job_lock:
        suffix = Path(getattr(media, "file_name", None) or "record.ogg").suffix or ".ogg"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / f"record{suffix}"
            try:
                await bot.download(media, destination=path)
                await status.edit_text("🎧 Распознаю речь и кто говорит… (≈ минута на минуту записи)")
                tr = await asyncio.to_thread(transcribe, str(path))
                if not tr.text.strip():
                    await status.edit_text("В записи не нашлось речи.")
                    return
                await status.edit_text("🧠 Готовлю выжимку…")
                summary = await asyncio.to_thread(summarize, tr.text)
            except Exception:
                logging.exception("Ошибка обработки записи")
                await status.edit_text("Не получилось обработать запись. Попробуйте другой формат (mp3, m4a, ogg).")
                return

    await status.delete()
    text = to_telegram_html(summary)
    d = int(tr.duration_sec)
    footer = f"\n\n<i>Запись {d // 60}:{d % 60:02d} · говорящих: {tr.speakers}</i>"
    await message.answer(text + footer, parse_mode="HTML")
    await message.answer_document(
        BufferedInputFile(tr.text.encode("utf-8"), filename="расшифровка.txt"),
        caption="Полная расшифровка",
    )


async def main():
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
