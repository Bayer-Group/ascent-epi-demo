"""User-analytics logging for medical-coder requests.

Ported from the legacy ``api/core/user_analytics.py``. No-ops when Postgres is
unconfigured. The request payload is typed loosely (``Any``) because the request
schema lives outside this module.
"""

from __future__ import annotations

import logging
from typing import Any

from ascent_medical_coder.db.models import MCSLog
from ascent_medical_coder.db.session import session_factory

logger = logging.getLogger(__name__)


async def log_request(
    user_id: str,
    payload: Any,
    exec_time: float,
    domain_id: str | None,
    lts_used: bool,
) -> None:
    """Create a database entry for a medical-coder request.

    If Postgres is not configured this function does nothing.
    """
    factory = session_factory()
    if factory is None:
        return

    if payload.vocabulary is None:
        vocabulary = None
    elif isinstance(payload.vocabulary, list):
        vocabulary = ",".join(payload.vocabulary)
    else:
        vocabulary = None

    mcs_log = MCSLog(
        user_id=user_id,
        query=payload.query,
        top_k=payload.top_k,
        standard_concept=payload.standard_concept,
        domain_id=domain_id,
        vocabulary=vocabulary,
        llm_filter=bool(payload.llm_filter),
        lts_used=lts_used,
        execution_time_s=exec_time,
    )

    async with factory() as session:
        session.add(mcs_log)
        await session.commit()
