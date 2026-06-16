import asyncio
from functools import wraps

from asyncpg import (  # type: ignore
    CheckViolationError,
    ForeignKeyViolationError,
    UniqueViolationError,
)
from sqlalchemy.exc import IntegrityError

from app.common.exceptions import IntegrityDataException
from app.common.logging import logger


def sqlalchemy_exception_handler(func):
    """Wrap an async repository method to translate known IntegrityErrors into IntegrityDataException

    :param:
        func: the async repository method to wrap

    :returns:
        wrapper: the wrapped coroutine function
    """

    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except IntegrityError as exc:
            orig = getattr(exc, "orig", None)

            sqlstate = None
            if orig is not None:
                sqlstate = getattr(orig, "pgcode", None) or getattr(
                    orig, "sqlstate", None
                )

            known_states = {
                UniqueViolationError.sqlstate,
                ForeignKeyViolationError.sqlstate,
                CheckViolationError.sqlstate,
            }

            if sqlstate in known_states or isinstance(
                orig,
                (UniqueViolationError, ForeignKeyViolationError, CheckViolationError),
            ):
                logger.warning(f"DB Integrity Error: {exc}")
                raise IntegrityDataException(detail=str(exc)) from exc
            logger.error(
                f"Unexpected DB Integrity Error (sqlstate={sqlstate}, orig={type(orig)}): {exc}"
            )
            raise

    return wrapper


def repository_error_handler(cls):
    """Class decorator that wraps all async methods with sqlalchemy_exception_handler

    :param:
        cls: the repository class to decorate

    :returns:
        cls: the same class with all async methods wrapped
    """
    for name, method in list(vars(cls).items()):
        if name.startswith("__"):
            continue
        if asyncio.iscoroutinefunction(method):
            setattr(cls, name, sqlalchemy_exception_handler(method))
    return cls
