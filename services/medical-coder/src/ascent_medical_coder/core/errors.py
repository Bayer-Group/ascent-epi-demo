"""Error notification (AWS SNS) and request profiling (AWS S3) helpers.

Ported from the legacy ``api/core/error_email.py`` and the pyinstrument
profiling middleware in ``api/main.py``.

AWS is the decoupling seam here: SNS publishing and S3 profile uploads are the
only cloud-coupled pieces in this module. Both are gated (SNS on a deployed
release version, profiling on ``PROFILE_REQUESTS``) and lazily initialized so
local / unconfigured runs never touch AWS.
"""

from __future__ import annotations

import datetime
import logging
import sys
import traceback
from collections.abc import Awaitable, Callable
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

import boto3
from fastapi import Request, Response
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

from ascent_medical_coder.core.settings import get_settings

if TYPE_CHECKING:
    from fastapi import FastAPI

logger = logging.getLogger(__name__)

_sns_client = None
_sns_topic_arn: str | None = None


def _sns() -> tuple[object, str] | None:
    """Lazily build the SNS client + topic ARN. Returns ``None`` if not deployed."""
    global _sns_client, _sns_topic_arn
    if not get_settings().service.is_deployed:
        return None
    if _sns_client is None:
        aws_region = boto3.session.Session().region_name
        account_id = boto3.client("sts").get_caller_identity()["Account"]
        _sns_client = boto3.client("sns", region_name=aws_region)
        _sns_topic_arn = f"arn:aws:sns:{aws_region}:{account_id}:medcod-backend-error"
    return _sns_client, _sns_topic_arn


def send_error_notification(exc: Exception, info: str = "") -> None:
    """Publish an error stack trace to SNS (only on deployed release versions)."""
    target = _sns()
    if target is None:
        return
    sns_client, topic_arn = target

    release_version = get_settings().service.release_version
    _exc_type, _exc_value, exc_traceback = sys.exc_info()
    traceback_details = traceback.extract_tb(exc_traceback)
    filename, line, _func, _text = traceback_details[-1]
    error_msg = (
        f"An error occured on the medical coder v2 {release_version} \n{info!s} "
        f"\n{exc!s} (File: {filename}, Line: {line}) \n \n{traceback.format_exc()}"
    )
    sns_client.publish(TopicArn=topic_arn, Message=error_msg)


def register_error_handlers(app: FastAPI) -> None:
    """Register the 500 handler that notifies via SNS and returns a JSON body."""

    @app.exception_handler(500)
    async def internal_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        try:
            send_error_notification(exc)
        except Exception:
            # The notifier (STS/SNS) failing must not replace the JSON 500 body.
            logger.exception("error notification failed")
        return JSONResponse(status_code=500, content=jsonable_encoder({"code": 500, "reason": str(exc)}))


def add_profiling_middleware(app: FastAPI) -> None:
    """Install the pyinstrument profiling middleware when ``PROFILE_REQUESTS`` is set.

    Each request is profiled and rendered as Speedscope JSON. Set
    ``PROFILE_S3_BUCKET`` to upload it; otherwise it is written under
    ``/tmp`` in the container.
    """
    settings = get_settings()
    if not settings.service.profile_requests:
        return

    from pyinstrument import Profiler
    from pyinstrument.renderers.speedscope import SpeedscopeRenderer

    s3_bucket = settings.service.profile_s3_bucket
    s3_prefix = settings.service.profile_s3_prefix
    s3_client = boto3.client("s3") if s3_bucket else None

    @app.middleware("http")
    async def profile_request(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        # Sample the full stack every 1ms across all middlewares.
        with Profiler(interval=0.001, async_mode="enabled") as profiler:
            response = await call_next(request)

        ts = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d_%H-%M-%S")
        name = f"{request.url.path}_{ts}.json".strip("/")
        body = profiler.output(renderer=SpeedscopeRenderer())

        try:
            if s3_client is not None:
                s3_client.put_object(Body=body, Bucket=s3_bucket, Key=str(PurePosixPath(s3_prefix) / name))
            else:
                target = Path("/tmp") / s3_prefix / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(body)
        except Exception as e:
            logger.error(e)

        return response
