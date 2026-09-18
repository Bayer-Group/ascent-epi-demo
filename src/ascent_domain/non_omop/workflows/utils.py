import logging

import sqlparse

from ascent_domain.non_omop.schemas.reformat_sql import ReformatSQL

logger = logging.getLogger(__name__)

def reformat_sql(sql: str) -> ReformatSQL:
    """
    Reformats the SQL query into a properly formatted, human-readable, and executable format.
    """
    try:
        formatted_sql = sqlparse.format(sql, reindent=True, keyword_case='upper')
        return ReformatSQL(sql=formatted_sql, success=True)
    except Exception as e:
        logger.error(f"Error in reformat_sql: {e}")
        return ReformatSQL(sql=sql, success=False)

