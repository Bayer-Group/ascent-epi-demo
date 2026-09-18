from ascent_platform.config.runtime import Settings

settings = Settings()
from ascent_platform.llm.factory import create_assistant


class SQLExplainer:
    explanation_prompt_base = """
    You are an AI assistant tasked with interpreting SQL queries and explain to non technical users what the query does. Maintain Simplicity and Clarity: Use straightforward language, avoiding technical terms unless absolutely necessary. Where technical terms are used, provide a brief, understandable definition. The goal is to make the explanation as accessible as possible.

    Follow step by step the instructions:

    1. Explain Purpose and Process: For each component identified, explain in plain language what it does and how it contributes to the overall operation of the query. This includes describing how data is selected, filtered, combined, or sorted.

    2. Summarize Expected Outcome: Explain what the query will retrieve from the database. This could involve explaining the expected results.

    3. If you notice that the query is wrong, stay factual and explain the error in the query that could be both semantic or syntactic. Do not correct the query or explain what the query is supposed to do.

    Be succinct and clear in your explanation.
    """

    def __init__(self, assistant_type="gpt", model_name=None):
        self.assistant_type = assistant_type
        self.model_name = model_name
        self.assistant = create_assistant(assistant_type=self.assistant_type, model_name=self.model_name)

    def _build_prompt(self, input_question=None):
        if input_question:
            prompt = (self.explanation_prompt_base +
                      f"This was the input question for your reference: {input_question}")
        else:
            prompt = self.explanation_prompt_base
        return prompt

    async def get_explanation(self, sql_query: str, input_question: str = None):
        prompt = self._build_prompt(input_question)
        full_prompt = f"{prompt}\n\nPlease explain this query:\n{sql_query}"
        sql_explanation = await self.assistant.get_response(full_prompt)
        return sql_explanation



