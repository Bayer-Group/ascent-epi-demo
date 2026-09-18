"""Which model answers which kind of question.

This is a domain decision -- that criteria extraction is worth an Opus call and
sanity checking is not -- so it sits with the domain rather than with the
application settings.

The shape is a plain nested dict because that is what ``CriteriaProcessor`` and
``QuestionAnsweringSystem`` already take.
"""

from __future__ import annotations

_CLAUDE_OPUS = {"type": "claude_sonnet", "model": "us.anthropic.claude-opus-4-6-v1"}

# Keys are task names the AI layer looks up by string; each must resolve to a
# registered assistant type.
assistant_config: dict[str, dict[str, str]] = {
    "text_to_criteria": dict(_CLAUDE_OPUS),
    "mask": dict(_CLAUDE_OPUS),
    "text2sql": dict(_CLAUDE_OPUS),
    "question_answering": dict(_CLAUDE_OPUS),
}
