import logging
from pathlib import Path

import alembic
import alembic.command
from alembic.config import Config as AlembicSettings

from ascent_platform.config.paths import PROJECT_ROOT
from ascent_platform.db.postgresql_session import database_url

# Was Path(__file__).parent x4. Counting directory hops to find alembic.ini
# breaks silently the moment the file moves -- and this runs at container
# start, before anything else, so a wrong root means migrations quietly
# target the wrong config. PROJECT_ROOT is the anchor.
ALEMBIC_INI = PROJECT_ROOT / "alembic.ini"

logger = logging.getLogger(__name__)


def migrate_database():
    database_url_alembic = database_url.replace("%", "%%")

    alembic_settings = AlembicSettings(ALEMBIC_INI)
    alembic_settings.set_main_option("sqlalchemy.url", database_url_alembic)

    alembic.command.upgrade(alembic_settings, "head")
