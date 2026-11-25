"""Зрение для моделей, которые не видят картинки и не читают PDF.

Если основная модель (например Groq qwen) без зрения, фото и кадры видео описывает отдельная
модель со зрением (VISION_MODEL — например бесплатный Gemini Flash-Lite). Коломбина получает
подробное описание и честную пометку «видишь через описание». PDF без поддержки файлов
читается локально (pypdf).
"""

from __future__ import annotations

import io
import logging

from ..llm.base import DocumentPart, ImagePart, LLMError, LLMProvider, LLMRequest, Message, SystemBlock, TextPart
from .model import Percept

log = logging.getLogger(__name__)

DESCRIBE = (
    "Ты — глаза собеседницы, которая сама не видит изображение. Опиши его по-русски точно и живо, 3–7 предложений: "
    "что и кто изображён, детали, цвета и свет, время суток и погода, животные и их поведение, текст на изображении "
    "(дословно), настроение кадра. Если это скриншот или мем — перескажи содержание и смысл. Не выдумывай того, "
    "чего не видно; если что-то неразборчиво — так и скажи."
)
DESCRIBE_FRAMES = (
    "Это кадры одного видео по порядку. Опиши по-русски, что происходит в видео: место, люди и животные, действия, "
    "как меняется сцена от кадра к кадру, текст в кадре. 3–8 предложений. Не выдумывай то, чего нет на кадрах."
)


def pdf_to_text(data: bytes, limit: int = 15000) -> tuple[str, int, int]:
    """Текст PDF: (текст, прочитано_страниц, всего_страниц)."""
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    out, pages = [], 0
    for page in reader.pages:
        try:
            out.append(page.extract_text() or "")
        except Exception:
            out.append("")
        pages += 1
        if sum(len(x) for x in out) > limit:
            break
    return "\n".join(out)[:limit], pages, len(reader.pages)


class VisionAdapter:
    def __init__(self, main_vision: bool, main_pdf: bool, describer: LLMProvider | None):
        self.main_vision = main_vision
        self.main_pdf = main_pdf
        self.describer = describer

    async def _describe(self, images: list[ImagePart], frames: bool) -> str | None:
        if not self.describer:
            return None
        try:
            r = await self.describer.complete(LLMRequest(
                system=[SystemBlock(DESCRIBE_FRAMES if frames else DESCRIBE)],
                messages=[Message(role="user", parts=[*images[:6], TextPart("Опиши.")])],
                tier="main", max_tokens=1500, effort="low", purpose="vision",
            ))
            return r.text.strip() or None
        except (LLMError, Exception) as e:  # noqa: BLE001
            log.warning("vision describe failed: %s", e)
            return None

    async def adapt(self, percepts: list[Percept]) -> list[Percept]:
        for p in percepts:
            images = [x for x in p.parts if isinstance(x, ImagePart)]
            docs = [x for x in p.parts if isinstance(x, DocumentPart)]
            if images and not self.main_vision:
                desc = await self._describe(images, frames=len(images) > 1)
                p.parts = [x for x in p.parts if not isinstance(x, ImagePart)]
                what = "кадры видео" if len(images) > 1 else "изображение"
                if desc:
                    p.text = (f"[{what} — описание, через которое ты это видишь]: {desc}" + (f"\n{p.text}" if p.text else ""))
                    p.channel = "described"
                    p.meta["vision_description"] = desc
                else:
                    p.channel, p.delivered = "metadata", False
                    p.note += " — рассмотреть не получилось"
            if docs and not self.main_pdf:
                p.parts = [x for x in p.parts if not isinstance(x, DocumentPart)]
                texts = []
                for d in docs:
                    try:
                        text, read, total = pdf_to_text(d.data)
                    except Exception:
                        text, read, total = "", 0, 0
                    if text.strip():
                        extra = f" (прочитано {read} из {total} стр.)" if read < total else ""
                        texts.append(f"[текст PDF «{d.title or 'документ'}»{extra}]:\n{text}")
                    else:
                        texts.append(f"[PDF «{d.title or 'документ'}» — текста в нём не нашлось, возможно это сканы]")
                p.text = "\n\n".join(texts) + (f"\n\n{p.text}" if p.text else "")
        return percepts
