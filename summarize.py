"""Выжимка созвона через Claude: структурированный ответ (tool use), а не свободный текст —
чтобы решения, задачи и сроки всегда приходили в одном и том же виде.
"""
import html
import os

from anthropic import Anthropic

MODEL = os.getenv("SUMMARY_MODEL") or "claude-sonnet-5"

_client = None


def _get_client() -> Anthropic:
    global _client
    if _client is None:
        _client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    return _client


SYSTEM_PROMPT = """Ты делаешь выжимку рабочего созвона по автоматической расшифровке речи.

Особенности расшифровки:
- в ней есть ошибки распознавания (искажённые слова, «БОТО» вместо «бота» и т.п.) — восстанавливай смысл по контексту;
- реплики размечены автоматически: «Спикер 1», «Спикер 2»… Разметка в целом надёжна, но бывают ошибки
  (короткие реплики похожих голосов иногда приписаны не тому). «Неизвестный» — голос не определён;
- имён в разметке нет. Сопоставь спикера с именем только по явному признаку: к нему обратились по имени, и он
  ответил («Дмитрий, что по срокам?» → следующая реплика Спикера 2 → Спикер 2 = Дмитрий), он назвал себя,
  или о нём сказали «X подготовит». Если имя установить нельзя — оставляй «Спикер N», не угадывай;
- исполнитель задачи — тот, кто сказал «я сделаю / подключу / подготовлю», или кому задачу прямо назначили.
  Если непонятно — «не указан».

Правила:
- решение — только то, о чём договорились (а не любое предложение). Если предложение явно отклонено или отложено — это не решение;
- задача — конкретное действие с исполнителем; срок — как прозвучал («до пятницы», «до 20 октября»), если не прозвучал — «не указан»;
- открытый вопрос — то, что подняли, но не решили;
- без воды и пересказа светской части разговора; по-русски.

Всегда вызывай функцию save_summary."""

TOOL = {
    "name": "save_summary",
    "description": "Сохранить структурированную выжимку созвона",
    "input_schema": {
        "type": "object",
        "properties": {
            "topic": {"type": "string", "description": "Тема созвона одной фразой"},
            "participants": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Участники: «Спикер 2 — Дмитрий» если имя установлено, иначе «Спикер 1»",
            },
            "key_points": {"type": "array", "items": {"type": "string"}, "description": "3-6 главных тезисов"},
            "decisions": {"type": "array", "items": {"type": "string"}, "description": "О чём договорились"},
            "tasks": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "task": {"type": "string"},
                        "owner": {"type": "string", "description": "Исполнитель или «не указан»"},
                        "deadline": {"type": "string", "description": "Срок как прозвучал или «не указан»"},
                    },
                    "required": ["task", "owner", "deadline"],
                },
            },
            "open_questions": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["topic", "participants", "key_points", "decisions", "tasks", "open_questions"],
    },
}


def summarize(transcript: str) -> dict:
    response = _get_client().messages.create(
        model=MODEL,
        max_tokens=2000,
        system=SYSTEM_PROMPT,
        tools=[TOOL],
        tool_choice={"type": "tool", "name": "save_summary"},
        messages=[{"role": "user", "content": f"Расшифровка созвона:\n\n{transcript}"}],
    )
    for block in response.content:
        if block.type == "tool_use":
            return block.input
    raise ValueError("Claude не вернул выжимку")


def to_telegram_html(s: dict) -> str:
    e = html.escape
    parts = [f"<b>📝 {e(s['topic'])}</b>"]
    if s.get("participants"):
        parts.append("<b>👥 Участники:</b> " + ", ".join(e(p) for p in s["participants"]))
    if s["key_points"]:
        parts.append("<b>Главное</b>\n" + "\n".join(f"• {e(p)}" for p in s["key_points"]))
    if s["decisions"]:
        parts.append("<b>✅ Решения</b>\n" + "\n".join(f"{i}. {e(d)}" for i, d in enumerate(s["decisions"], 1)))
    if s["tasks"]:
        lines = [f"• {e(t['task'])} — <b>{e(t['owner'])}</b>, срок: {e(t['deadline'])}" for t in s["tasks"]]
        parts.append("<b>📌 Задачи</b>\n" + "\n".join(lines))
    if s["open_questions"]:
        parts.append("<b>❓ Открытые вопросы</b>\n" + "\n".join(f"• {e(q)}" for q in s["open_questions"]))
    return "\n\n".join(parts)
