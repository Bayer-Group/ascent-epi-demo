def get_prompt_questions_comparison(original_question, question_rebuilt, cohort_used: bool = False) -> str:
    prompt_question_comparison = f"""Compare semantically the original question and the question created from the SQL query. 
    The queries are in Snowflake SQL format.
    If they are semantically equivalent, return similar: true, and return explanation for the similarity.
    If they are not semantically equivalent and the sql needs to be changed to answer to original question, return similar: false, and return explanation for the differences.
    If no, include what are the critical differences and what should be taken into account for fixing the SQL query.
    In the explanation, clearly state that the results are from comparing the generated question from the SQL query and the original question.
    Note that when working with ages, dates it is not required to have the very exact and strict matching.
    The original question is: {original_question}. The question created from the SQL query is: {question_rebuilt}.
    """

    if cohort_used:
        prompt_question_comparison += "Also, Note that all the results are based on a specific cohort of patients defined earlier."
    return prompt_question_comparison
