def get_prompt_sql_to_question(final_sql_query):
    prompt_question_creation = f"""
    Create a question from the SQL query below. 
    The question should be a natural language question that can be answered by the SQL query. 
    The question should be in English and should be detailed. Just return the question with no explanation. 
    The queries are in Snowflake SQL format.
    The SQL query is: {final_sql_query}
    """
    return prompt_question_creation
