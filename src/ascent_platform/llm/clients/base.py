import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Type, Union

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class BaseLLMClient(ABC):
    """
    An abstract base class for Large Language Model (LLM) clients.

    This class provides a common interface for sending prompts to LLMs and
    handles cross-cutting concerns such as retries with exponential backoff,
    logging, and fallback to a backup model.
    """

    def __init__(
        self,
        *,
        retries: int = 3,
        initial_delay: float = 5.0,
        backoff_factor: float = 2.0,
        backup_model: Optional["BaseLLMClient"] = None,
    ) -> None:
        self.retries = retries
        self.initial_delay = initial_delay
        self.backoff_factor = backoff_factor
        self.backup_model = backup_model

    async def _retry(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        delay = self.initial_delay
        for attempt in range(1, self.retries + 1):
            logger.info("Attempt %d/%d ", attempt, self.retries)
            attempt_start = time.time()
            try:
                result = await func(*args, **kwargs)
                elapsed_attempt = time.time() - attempt_start
                logger.info(
                    "Attempt %d/%d succeeded in %.3f seconds.",
                    attempt,
                    self.retries,
                    elapsed_attempt,
                )
                return result
            except Exception as e:
                elapsed_attempt = time.time() - attempt_start
                logger.warning(
                    "Attempt %d/%d failed in %.3f seconds: %s",
                    attempt,
                    self.retries,
                    elapsed_attempt,
                    e,
                    exc_info=True,
                )
                if attempt == self.retries:
                    raise
                # asyncio.sleep, not time.sleep: this used to run inside a
                # default-executor thread, so a backoff of up to 15s held a
                # worker doing nothing. Now it yields the loop instead.
                await asyncio.sleep(delay)
                delay *= self.backoff_factor

    async def send(
        self,
        prompt: str,
        *,
        schema: Optional[Type[BaseModel]] = None,
        **kwargs: Any,
    ) -> Union[str, Dict[str, Any]]:
        """
        Sends a prompt to the LLM, with retry and fallback logic.
        """
        start = time.time()
        try:
            return await self._retry(self._send, prompt=prompt, schema=schema, **kwargs)
        except Exception as main_exception:
            logger.error("Primary model failed after all retries: %s", main_exception)
            if self.backup_model:
                logger.info("Falling back to the backup model.")
                return await self.backup_model.send(prompt, schema=schema, **kwargs)
            else:
                raise main_exception
        finally:
            elapsed = time.time() - start
            logger.info(
                "Total execution time for prompt: %.3f seconds", elapsed
            )

    @abstractmethod
    async def _send(
        self,
        prompt: str,
        *,
        schema: Optional[Type[BaseModel]] = None,
        **kwargs: Any,
    ) -> Union[str, Dict[str, Any], None]:
        """
        Provider-specific implementation for sending a prompt.
        """
        ...