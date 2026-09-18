import asyncio
import inspect
import linecache
import logging
import re
import sys
import traceback
from contextlib import redirect_stdout
from io import StringIO
from typing import Any

import boto3
from fastapi import Request
from starlette.websockets import WebSocket

from ascent_http.settings import settings as settings
from ascent_http.util.correlation_id import get_correlation_id_short
from ascent_http.util.version import get_version_info

logger = logging.getLogger(__name__)

EMAIL_NOTIFICATIONS_ACTIVE = settings.RELEASE_VERSION in ["dev", "qa", "prod", "main"]  # only report errors in these systems

if EMAIL_NOTIFICATIONS_ACTIVE:
    AWS_REGION = boto3.Session().region_name

    sts_client = boto3.client("sts")
    account_id = sts_client.get_caller_identity()["Account"]

    sns_client = boto3.client("sns", region_name=AWS_REGION)

SNS_PUBLISH_TIMEOUT = 5.0

# Frame arguments are rendered into a traceback that is returned to the CALLER
# in the 500 body (main.py) and published to SNS. _create_user_connection takes
# `oauth_token` -- a live Snowflake-scoped OBO bearer -- as a named argument and
# re-raises unclassified driver errors, so an unredacted dump handed a user's
# credential to anyone who could read the response or the notification.
_SENSITIVE_ARG = re.compile(
    r"token|secret|password|passwd|credential|api_key|apikey|private_key|authorization|auth|session_key",
    re.IGNORECASE,
)
_MAX_ARG_REPR = 200


def _render_arg(name: str, value: Any) -> str:
    """repr() an argument, redacting anything whose NAME looks sensitive.

    Name-based rather than value-based: a value-sniffing heuristic misses
    opaque tokens, and the parameter names here are stable.
    """
    if _SENSITIVE_ARG.search(name or ""):
        return "'<redacted>'"
    try:
        rendered = repr(value)
    except Exception:
        return "<unreprable>"
    if len(rendered) > _MAX_ARG_REPR:
        return f"{rendered[:_MAX_ARG_REPR]}...<truncated {len(rendered)} chars>"
    return rendered


# SUBSTRING match on the absolute filename, which is why the bare prefix is
# correct and must stay: it covers ascent_http, ascent_domain, ascent_platform
# and ascent_mcp at once. Do not "tidy" this into exact module names -- the
# 500 handler uses it to decide which frames get their arguments redacted,
# and a package that falls out of the list starts dumping oauth_token and
# private_key_pem_b64 into responses. tests/seams/test_traceback_redaction.py
# guards the behaviour but cannot see a package that was never listed.
PROJECT_MODULES = ("ascent",)


def build_enhanced_traceback_from_exc(exc: Exception) -> str:
    """
    Builds an enhanced traceback for a given exception.

    This function takes an exception object and generates a detailed traceback
    containing the original traceback as well as additional project-specific
    enhancements. The enhanced traceback includes detailed information about
    the arguments passed to each frame, excluding frames from third-party
    libraries or non-project modules. It also attempts to include relevant source
    code snippets for better debugging.

    Parameters:
        exc (Exception): The exception object for which the traceback needs
            to be built.

    Returns:
        str: A string representation of the combined original and enhanced
            tracebacks.

    Raises:
        None
    """

    # First, get the original traceback
    original_tb = StringIO()
    traceback.print_exception(type(exc), exc, exc.__traceback__, file=original_tb)
    original_traceback_str = original_tb.getvalue()

    # Now build the enhanced traceback with arguments
    tb = exc.__traceback__
    lines = [f"\n{'=' * 80}"]
    lines.append("ENHANCED TRACEBACK (Project code only, with arguments and source):")
    lines.append("=" * 80)
    lines.append(f"{type(exc).__name__}: {exc}\n")

    for frame, lineno in traceback.walk_tb(tb):
        filename = frame.f_code.co_filename

        # SKIP Starlette, FastAPI, stdlib frames
        if not any(module_name_ in filename for module_name_ in PROJECT_MODULES):
            continue  # skip non-project code

        try:
            arg_info = inspect.getargvalues(frame)
            args_strs = []
            for arg_name in arg_info.args:
                args_strs.append(f"{arg_name}={_render_arg(arg_name, arg_info.locals.get(arg_name))}")
            if arg_info.varargs:
                val = arg_info.locals.get(arg_info.varargs)
                args_strs.append(f"*{arg_info.varargs}={_render_arg(arg_info.varargs, val)}")
            if arg_info.keywords:
                val = arg_info.locals.get(arg_info.keywords)
                args_strs.append(f"**{arg_info.keywords}={_render_arg(arg_info.keywords, val)}")
            arg_str = ", ".join(args_strs)
        except Exception as e:
            arg_str = f"<could not get args: {e}>"

        # Get the actual source code line
        try:
            source_line = linecache.getline(filename, lineno).strip()
            code_snippet = f"\n    Code: {source_line}" if source_line else ""
        except Exception:
            code_snippet = ""

        lines.append(f'File "{filename}", line {lineno}, in {frame.f_code.co_name}({arg_str}){code_snippet}')

    enhanced_str = "\n".join(lines)

    # Combine original and enhanced
    return f"{original_traceback_str}\n{enhanced_str}"


async def on_error(exception: Exception, context: Request | WebSocket | None = None) -> None:
    try:
        version_info = get_version_info()
        exc_type, exc_value, exc_traceback = sys.exc_info()

        # Format the traceback to get a list of stack trace entries
        traceback_details = traceback.extract_tb(exc_traceback)

        # Get the last entry in the traceback which would be the point where the exception was raised
        filename, line, func, text = traceback_details[-1]

        # Format error message
        with StringIO() as buffer, redirect_stdout(buffer):
            url = getattr(context, "url", None)
            if url is not None:
                flow = get_flow(url._url)
            else:
                flow = ""
            print(f"An error has occurred on the ASCENT backend {flow}")
            print(f"Version Info: {version_info.model_dump_json(indent=4)}")
            if context:
                method = "WS" if isinstance(context, WebSocket) else context.method
                url = getattr(context, "url", None)
                path = getattr(url, "path", None)
                print(f"Request: {method} {path}")
                print(f"Correlation ID: {get_correlation_id_short()}")
                trace_id = getattr(context.state, "trace_id", "0" * 32)
                print(f"Amzn Trace ID hex: {trace_id}")
                print(
                    f"Amzn Trace ID URL; login to AWS account {version_info.deployment}: https://us-east-1.console.aws.amazon.com/cloudwatch/home?region=us-east-1#xray:traces/{trace_id}"
                )
                print(f"HTTP Request Body:\n{getattr(context.state, 'body', '')}")
            print(f"Exception: {exception!r}")
            print(f"File: {filename}, Line: {line}")

            # Print full traceback (includes both original and enhanced)
            enhanced_tb_str = build_enhanced_traceback_from_exc(exception)
            print(f"\n{enhanced_tb_str}")

            error_msg = buffer.getvalue().strip()

        logger.warning("Send error notification mail")
        await _send_email(error_msg)

    except Exception as exception:
        logger.exception(f"Error while sending error message, {exception}")


async def _send_email(message: str) -> None:
    if EMAIL_NOTIFICATIONS_ACTIVE:
        logger.info("Send error notification mail")
        topic = f"arn:aws:sns:{AWS_REGION}:{account_id}:ascent-epi-demo-error"
        try:
            # botocore is synchronous. Publishing inline blocked the event loop
            # for a full SNS round trip -- and this runs from the global
            # exception handler, so a burst of 500s serialised every worker
            # through SNS at the worst possible moment. Bounded and offloaded:
            # failing to send a notification must never delay the response.
            await asyncio.wait_for(
                asyncio.to_thread(sns_client.publish, TopicArn=topic, Message=message),
                timeout=SNS_PUBLISH_TIMEOUT,
            )
        except asyncio.TimeoutError:
            logger.warning("SNS publish timed out after %ss; dropping notification", SNS_PUBLISH_TIMEOUT)
        except Exception:
            logger.warning("SNS publish failed; dropping notification", exc_info=True)
    else:
        logger.info("Cannot send error notification mail")
        logger.debug(message)


def get_flow(route: str) -> str:
    if route.__contains__("omop"):
        return "- OMOP flow"
    if route.__contains__("non-omop"):
        return "- NON OMOP flow"
    return ""
