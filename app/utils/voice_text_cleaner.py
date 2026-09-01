"""
Defensive text cleanup applied to LLM output before it is sent to TTS.

The system prompt instructs the model to never produce markdown, bullet
points, emojis, etc. Models mostly comply, but a belt-and-braces cleanup step
protects the voice experience if the model slips (e.g. under long context, or
if asked to "list" something despite instructions).

This is intentionally conservative: it strips formatting *markup*, not the
underlying words, so meaning is preserved for both the transcript event sent
to the client and the audio synthesized from it.
"""

import re


_MARKDOWN_BOLD_ITALIC = re.compile(r"(\*\*\*|\*\*|\*|___|__|_)")
_MARKDOWN_HEADER = re.compile(r"^\s{0,3}#{1,6}\s+", re.MULTILINE)
_MARKDOWN_BULLET = re.compile(r"^\s*[-*+]\s+", re.MULTILINE)
_MARKDOWN_NUMBERED = re.compile(r"^\s*\d+[.)]\s+", re.MULTILINE)
_MARKDOWN_CODE_FENCE = re.compile(r"```.*?```", re.DOTALL)
_MARKDOWN_INLINE_CODE = re.compile(r"`([^`]*)`")
_EMOJI = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\U00002600-\U000027BF"
    "\U0001F1E6-\U0001F1FF"
    "]+",
    flags=re.UNICODE,
)
_MULTI_SPACE = re.compile(r"[ \t]{2,}")
_MULTI_NEWLINE = re.compile(r"\n{2,}")


def clean_for_voice(text: str) -> str:
    """Strip markdown, emojis, and other visual-only formatting from LLM text."""
    if not text:
        return text

    cleaned = _MARKDOWN_CODE_FENCE.sub(" ", text)
    cleaned = _MARKDOWN_INLINE_CODE.sub(r"\1", cleaned)
    cleaned = _MARKDOWN_HEADER.sub("", cleaned)
    cleaned = _MARKDOWN_BULLET.sub("", cleaned)
    cleaned = _MARKDOWN_NUMBERED.sub("", cleaned)
    cleaned = _MARKDOWN_BOLD_ITALIC.sub("", cleaned)
    cleaned = _EMOJI.sub("", cleaned)
    cleaned = cleaned.replace("`", "")
    cleaned = _MULTI_NEWLINE.sub(". ", cleaned)
    cleaned = cleaned.replace("\n", " ")
    cleaned = _MULTI_SPACE.sub(" ", cleaned)
    return cleaned.strip()
