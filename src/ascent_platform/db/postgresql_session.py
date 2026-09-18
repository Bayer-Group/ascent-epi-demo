from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import declarative_base

from ascent_platform.config.runtime import get_runtime_settings

url_template = "postgresql+asyncpg://{user}:{password}@{host}:{port}/{db}"

_DB_FIELDS = ("DB_APP_USR", "DB_APP_PWD", "DB_HOST", "DB_PORT", "DB_NAME")


def _database_url() -> str:
    """The Postgres URL, or a clear error naming what is missing.

    These five used to be required fields on ascent.Settings, so an unset one
    failed validation by name. On the merged Settings they are optional -- the
    platform has to stay importable without a database password -- and without
    this check the None values were formatted straight into the URL, where
    SQLAlchemy failed with ``invalid literal for int() with base 10: 'None'``
    from inside its URL parser. That happens at import, before
    ``require_app_config()`` can run, so the clear message has to live here.
    """
    settings = get_runtime_settings()
    missing = [name for name in _DB_FIELDS if getattr(settings, name, None) is None]
    if missing:
        raise RuntimeError(
            "Cannot build the Postgres connection URL; missing "
            + ", ".join(missing)
            + ". Set them in the environment or .env."
        )
    return url_template.format(
        user=settings.DB_APP_USR,
        password=settings.DB_APP_PWD,
        host=settings.DB_HOST,
        port=settings.DB_PORT,
        db=settings.DB_NAME,
    )


database_url = _database_url()

# Create an async engine instance
engine = create_async_engine(database_url)

# Create a custom session class that can be used for async ORM operations
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

Base = declarative_base()


# Dependency
async def get_db():
    async with AsyncSessionLocal() as session:
        yield session
