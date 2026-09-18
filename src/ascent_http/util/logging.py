import atexit
import logging
import os
import queue
import sys
from datetime import datetime, timezone
from logging.handlers import QueueHandler, QueueListener

from opentelemetry.trace import get_current_span
from pythonjsonlogger.json import JsonFormatter

from ascent_http.settings import settings
from ascent_http.util.correlation_id import get_correlation_id_short

logger = logging.getLogger(__name__)

# The listener thread that owns the real (blocking) handlers. Module-level so a
# re-run of setup_logging() replaces it instead of leaking a second thread.
_LOG_LISTENER: QueueListener | None = None


def stop_log_listener() -> None:
    """Stop the logging listener thread, tolerating a second call.

    ``QueueListener.stop()`` raises if the thread is already gone, and we can
    reach it twice — once from ``setup_logging`` re-running, once from atexit.
    """
    global _LOG_LISTENER
    listener, _LOG_LISTENER = _LOG_LISTENER, None
    if listener is not None and getattr(listener, "_thread", None) is not None:
        listener.stop()


def get_trace_id() -> str:
    span = get_current_span()
    if span and span.get_span_context().trace_id != 0:
        return format(span.get_span_context().trace_id, "032x")
    return "-"


class CustomJsonFormatter(JsonFormatter):
    def add_fields(self, log_record, record, message_dict):
        super(CustomJsonFormatter, self).add_fields(log_record, record, message_dict)
        log_record["timestamp"] = datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
        log_record["level"] = record.levelname
        log_record["cid"] = get_correlation_id_short()
        log_record["traceID"] = get_trace_id()
        log_record["logger"] = record.name
        log_record["lineno"] = record.lineno


class CorrelationIdFormatter(logging.Formatter):
    def format(self, record):
        # Add the correlation ID to the log record; default to "no-correlation-id"
        record.correlation_id = get_correlation_id_short()
        return super().format(record)


def setup_logging():
    # Unregister all handlers, taking full control
    for name in logging.root.manager.loggerDict:
        logger_ = logging.getLogger(name)
        for handler in logger_.handlers:
            logger_.removeHandler(handler)

    # Setup formatter
    if settings.USE_STRUCTURED_LOGGING:
        formatter = CustomJsonFormatter()
    else:
        formatter = CorrelationIdFormatter("%(asctime)s [%(levelname)s] - [%(correlation_id)s] %(name)s (%(filename)s:%(lineno)d) - %(message)s")

    # Setup handlers -- force UTF-8 to avoid cp1252 crashes on Windows
    # when log messages contain Unicode characters (e.g. emoji)
    utf8_stderr = open(sys.stderr.fileno(), mode="w", encoding="utf-8", closefd=False)
    console_handler = logging.StreamHandler(utf8_stderr)
    console_handler.setFormatter(formatter)

    # Create the directory rather than assuming it. In a checkout it exists
    # because logs/.gitignore is tracked, which made this look safe -- but the
    # container image excludes logs/ (a developer's local logs have no business
    # being baked into an image), so /app/logs was absent and every hypercorn
    # worker died on FileHandler with FileNotFoundError before serving anything.
    logdir = "logs"
    os.makedirs(logdir, exist_ok=True)
    logfile = os.path.join(logdir, f"ascent_{datetime.now().strftime('%Y%m%d%H%M%S_%f')}.log")
    file_handler = logging.FileHandler(logfile, encoding="utf-8")
    file_handler.setFormatter(formatter)

    # Update global config.
    #
    # Emit through a QueueHandler rather than writing to stderr inline. A
    # StreamHandler write is a synchronous syscall on whichever thread logs —
    # for us, the event loop. If the container's log pipe backs up (a slow or
    # stalled awslogs driver), that write blocks, and with it every coroutine on
    # the worker, including in-flight MCP `initialize` handshakes. The queue is
    # unbounded and its put() never blocks; a listener thread does the actual
    # I/O, so a stalled pipe costs memory instead of liveness.
    sink_handlers = [console_handler, file_handler] if settings.RELEASE_VERSION == "local" else [console_handler]

    global _LOG_LISTENER
    stop_log_listener()

    log_queue: queue.Queue = queue.Queue(-1)
    _LOG_LISTENER = QueueListener(log_queue, *sink_handlers, respect_handler_level=True)
    _LOG_LISTENER.start()
    atexit.register(stop_log_listener)

    handlers = [QueueHandler(log_queue)]
    logging.basicConfig(level=logging.INFO, handlers=handlers, force=True)

    # Apply to all important loggers
    for logger_name in [
        "ascent_http",
        "ascent_domain.omop",
        "botocore",
        "snowflake.connector",
        "sentence_transformers.SentenceTransformer",
        "ascent_platform.external.ascent_client",
        "sqlalchemy",
        "uvicorn.access",
        "hypercorn.access",
        "apscheduler",
    ]:
        logger_ = logging.getLogger(logger_name)
        logger_.setLevel(settings.LOG_LEVEL if logger_name.startswith("ascent") else logging.WARNING)
        logger_.handlers = handlers
        logger_.propagate = False
