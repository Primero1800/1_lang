from sqlalchemy.ext.asyncio import AsyncSession


class BaseRepository:
    """Base repository for creating repositories for all database instances"""

    _session: AsyncSession

    def __init__(self, session: AsyncSession) -> None:
        """Initialize the repository with a database session

        :param:
            session: the async SQLAlchemy session

        :returns:
            None
        """
        self._session = session
