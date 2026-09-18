from ascent_domain.data_queries.cohort import (
    delete_abandoned_cohort_descriptors,
    delete_orphaned_cohort_tables,
    flag_orphaned_cohort_descriptors,
)
from ascent_platform.db.postgresql_session import AsyncSessionLocal
from ascent_platform.warehouse.session import get_db


async def garbage_collect_cohorts():
    async with get_db("ASCENT") as ascent_db, AsyncSessionLocal() as app_db:
        await delete_abandoned_cohort_descriptors(app_db)
        await flag_orphaned_cohort_descriptors(ascent_db, app_db)
        await delete_orphaned_cohort_tables(ascent_db, app_db)
