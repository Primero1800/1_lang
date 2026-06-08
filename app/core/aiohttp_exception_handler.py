import time
from functools import wraps
from typing import Any, Callable, TypeVar

from aiohttp import ClientConnectionError, ClientResponseError

from app.common.exceptions import ConnectionException
from app.common.logging import logger

T = TypeVar("T", bound=Callable[..., Any])


def external_request_exception_handler(
    fallback: Any = None,
    raise_exception: Any = ConnectionException,
    is_raise: bool = True,
) -> Callable[[T], T]:
    """Decorator to handle exceptions from external HTTP requests"""

    def decorator(func: Any) -> Any:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.time()
            try:
                return await func(*args, **kwargs)
            except raise_exception as exc:
                logger.warning(f"External response: {exc.args[0]}")
                if is_raise:
                    raise
                return fallback
            except (ClientResponseError, ClientConnectionError) as exc:
                logger.error(f"[{func.__name__}] HTTP error: {exc!r}")
                if is_raise:
                    raise raise_exception(f"Client error: {exc}") from exc
                return fallback
            except Exception as exc:
                logger.error(f"[{func.__name__}] Unexpected error: {exc!r}")
                if is_raise:
                    raise raise_exception(f"Unexpected error: {exc}") from exc
                return fallback
            finally:
                duration = time.time() - start
                if duration > 1:
                    logger.warning(f"[{func.__name__}] Slow request: {duration:.3f}s")

        return wrapper  # type: ignore

    return decorator  # type: ignore
