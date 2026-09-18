import asyncio
import logging
from functools import wraps

logger = logging.getLogger(__name__)


def retry_on_exception(max_retries: int = 3, retry_exceptions: tuple = (Exception,)):
    """
    Decorator that retries an async function when specified exceptions are raised.
    """

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            attempt = 0
            exception = None
            while attempt < max_retries:
                try:
                    return await func(*args, **kwargs)
                except retry_exceptions as e:
                    logger.warning(f"Error in {func.__name__}: {e}, retrying... ({attempt + 1}/{max_retries})")
                    attempt += 1
                    exception = e
                    await asyncio.sleep(attempt)  # Backoff multiplier
            if exception is not None:
                raise exception
            raise RuntimeError("Unknown error raised in retry_on_exception: no exception captured after retries.")

        return wrapper

    return decorator
