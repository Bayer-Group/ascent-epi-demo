import logging
import re

logger = logging.getLogger(__name__)


async def str_difference_llm(original_query, corrected_query, assistant):
    """Quick difference analysis between two strings."""
    prompt = f"""You are given in input two SQL queries, one is the original query, the other one its correction.
    Can you briefly explain what are the changes in the corrected query?
    Original query: {original_query} \n
    Corrected query: {corrected_query} \n
    """

    assistant.add_message(role="user", message=prompt)
    answer = await assistant.get_response()

    logger.info(answer)


def lowercase_placeholder_values(sql_text: str) -> str:
    """
    Lowercase values in placeholders of the form [entity_type@value].

    Args:
        sql_text: SQL query text containing placeholders

    Returns:
        Modified SQL text with lowercased values in placeholders
    """
    pattern = r"\[([a-zA-Z_]+)@([a-zA-Z0-9_/\-\(\)\'\\ ]+)\]"

    def lowercase_match(match):
        entity_type = match.group(1)
        value = match.group(2).lower()
        return f"[{entity_type}@{value}]"

    return re.sub(pattern, lowercase_match, sql_text)
