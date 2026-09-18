# coding=utf-8
__author__ = "Angelo Ziletti"
__maintainer__ = "Angelo Ziletti"
__date__ = "24/11/23"

import logging
import re

import pandas as pd
from duckdb import Error as WarehouseError

from ascent_domain.omop.data.processing.utils_processing import str_difference_llm
from ascent_domain.omop.models.inference.rag import RAGProcessor
from ascent_platform.warehouse.session import execute_query

logger = logging.getLogger(__name__)


class RWDRequest:
    def __init__(
        self,
        user_input,
        query_filled=None,
        query_template=None,
        error_type: str = None,
        error_message: str = None,
        retrieved_data=None,
        answer=None,
        rag=None,
    ):
        # Define all allowed attributes
        self.user_input = user_input
        self.query_filled = query_filled
        self.query_template = query_template
        self.error_type = error_type
        self.error_message = error_message
        self.query_expert = None
        self.explorer_concepts = None
        self.preferred_coding_system = None
        self.med_sql_processor = None
        self.rag = rag
        self.rag_params = self._get_rag_params(rag) if rag else None
        self.retrieved_data = retrieved_data
        self.answer = answer
        self.sql_executed = False
        self.sql_executed_self_healing_attempts = 0
        self.query_df_retrieved_rag = None
        self.initial_prompt = None
        self.rag_prompt_add_on = None
        self.rag_top_similarity = 0.0
        self.question_masked = None
        self.verification_results = None
        self.verdict = None
        self.recommendations = None
        self.sql_query_results = None

    @property
    def rag(self):
        return self._rag

    @rag.setter
    def rag(self, value: RAGProcessor):
        self._rag = value
        self.rag_params = self._get_rag_params(value)
        logger.debug(f"Rag set with params: {self.rag_params}")

    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove the rag from the state for allowing pickling
        if "rag" in state:
            del state["rag"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Set rag_agent to None, it will be reinitialized if needed
        self.rag = None

    @staticmethod
    def _get_rag_params(rag):
        if rag is None:
            return None
        params = {
            "assistant_type": type(rag).__name__,
        }
        # Add other parameters as needed
        for attr in ["model_name", "max_tokens", "temperature", "top_p", "region_name"]:
            if hasattr(rag, attr):
                params[attr] = getattr(rag, attr)
        return params

    def reinitialize_rag(self, new_rag):
        self.rag = new_rag

    async def get_answer(self, assistant, max_lines=100, max_lines_sql=3000):
        if self.retrieved_data is not None and not self.retrieved_data.empty:
            prompt = f"""
                    This is the data retrieved from our database:\n {self.retrieved_data[:max_lines].to_markdown()}\n which is sufficient to answer the question "{self.user_input}".\n
                    This is the SQL used (placeholders are filled later):\n {self.query_template[:max_lines_sql]} \n
                    
                    - Please provide a concise answer to the following question: {self.user_input}
                    - Assume all provided data is relevant and necessary for the response.
                    - Assume that all filtering requested in the question are correctly applied, unless you see evidence 
                    against this in the SQL used
                    - If the question refers to a distribution, please only include summary statistics in your answer.
                    - Please refrain from offering data comparisons, conducting trend analysis, or attempting to create plots or visualizations in your response.
                    - Please specify if the response contains an approximation rather than the precise result.
                    - Please include all relevant data in your answer.
                    """
            answer = await assistant.get_response(prompt)
            logger.info(f"Getting answer for '{self.user_input}'...")
            self.answer = answer
        else:
            message = f"No data returned to answer '{self.user_input}'."
            logger.info(message)
            self.answer = message

    async def run_query(self, db=None, max_retries=1, reset_conversation=True):
        """
        Runs a SQL query against a database and returns the results as a DataFrame.

        This method attempts to execute the provided SQL query up to a maximum number of retries.
        If the query fails due to a SQL error, it can use ai to attempt to correct the query and retry.

        Parameters:
        - query_filled (str): The SQL query to be executed.
        - query_template (str): The template SQL query.
        - db (DatabaseError or Connection): The database connection to execute the query on.
        - assistant (Assistant): Ai assistant object.
        - max_retries (int): The maximum number of retries for executing the query (default is 5).
        - websocket (WebSocket): Websocket connection object.
        - reset_conversation (bool): Whether to reset the conversation state of the assistant before
          running the query (default is True).

        Returns:
        - pd.DataFrame: A DataFrame containing the results of the query if successful, None otherwise.

        Raises:
        - DatabaseError: If a database error occurs that cannot be resolved after max_retries.
        - Exception: If an unexpected error occurs during query execution.
        """
        if reset_conversation:
            self.rag.default_assistant.reset_conversation()

        if self.query_filled is None or self.query_template is None:
            logger.exception("Please provide both query_filled and query_template")
            return None

        first_exception_message = None
        attempts = 0
        success = False

        for attempt in range(max_retries):
            attempts += 1
            try:
                results, columns = await execute_query(db, self.query_filled)
                out_df = pd.DataFrame(data=results, columns=columns)
                self.sql_executed = True
                self.sql_executed_self_healing_attempts = attempt
                self.retrieved_data = out_df
                success = True
                break
            except WarehouseError as db_ex:
                if first_exception_message is None:
                    first_exception_message = str(db_ex)

                # Check for timeout errors - don't retry these
                if "timeout" in str(db_ex).lower():
                    logger.warning("Timeout error detected - NOT retrying (query too complex for warehouse)")
                    logger.warning(f"Failed query: {self.query_template}")
                    break

                new_query_template = await self.handle_db_exception(db_ex, self.query_template, attempt, max_retries, self.rag.default_assistant)
                if new_query_template != self.query_template:
                    await str_difference_llm(
                        original_query=self.query_template, corrected_query=new_query_template, assistant=self.rag.default_assistant
                    )

                    self.query_template = new_query_template
                    self.query_filled = await self.med_sql_processor.post_process_sql_query(
                        sql_text=self.query_template,
                        preferred_coding_system=self.preferred_coding_system,
                        rag_agent=self.rag,
                        db_session=db,
                    )
                continue
            except Exception as e:
                logger.exception(e)
                if first_exception_message is None:  # Capture only the first exception message
                    first_exception_message = str(e)
                logger.exception("Error in SQL could not be resolved")
                break

        if success:
            pass
        else:
            await self.handle_max_retries_reached(first_exception_message, self.query_filled)

        return self.retrieved_data

    async def handle_db_exception(self, db_ex, query_template, attempt, max_retries, assistant):
        logger.warning(f"Error in SQL detected. Failed query: {query_template}")

        if assistant:
            message = f"Self-healing process in progress. Attempt: {attempt + 1}/{max_retries}"
            logger.info(message)
            query_template = await self.handle_invalid_sql(query_template, assistant, db_ex.args[0])
        else:
            logger.warning("AI assistant required for self-healing process. Continuing without.")
        return query_template

    @staticmethod
    async def handle_max_retries_reached(last_exception_message, sql_query):
        if last_exception_message:
            message = f"Max retries reached without successful SQL execution, error: {last_exception_message}; Failed sql query: {sql_query}"
        else:
            message = f"Max retries reached without successful SQL execution, Failed sql query: {sql_query}"

        logger.warning(message)

    async def handle_invalid_sql(self, query_template, assistant, error):
        prompt = f"""Generated SQL query: \n {query_template} \n
                Here is the error returned: ***{error}***.\n
                Analyze the error. Review the generated SQL. Think about the OMOP CDM schema. Fix and rewrite the SQL query.\n
                Please keep in mind that you are querying a Snowflake database. Avoid using functions that are not supported by Snowflake.\n
                For dates in Snowflake:
                - Use 'YYYY-MM-DD' format for date literals (e.g., '2000-01-01'). Snowflake automatically converts this format to dates.
                - Use TO_DATE() function only when:
                  * Working with dates in other formats (e.g., TO_DATE('01/15/2024', 'MM/DD/YYYY'))
                  * Converting string columns to dates
                  * Making data type conversion explicitly clear in code
                Please only return the corrected SQL query. Do not return any extraneous data or information.\n
                Return the SQL query ONLY within ```sql ``` code block.
                Please make sure that corrected query includes all necessary GROUP BY expressions if aggregate functions are used.\n
            """
        assistant.add_message(role="user", message=prompt)
        completed_prompt = await assistant.get_response()

        new_query = self.parse_sql_from_response(completed_prompt)
        logger.info(f"Error: {error}")
        logger.info("new_query")
        logger.info(new_query)
        logger.info("-------------")
        return new_query

    @staticmethod
    def parse_sql_from_response(resp=""):
        pattern1 = r"(?:Snowflake )?SQL query:\s*\n\n(.*?);"
        pattern2 = r"(?:```sql|```) ?\n(.*?)\n```"

        match1 = re.search(pattern1, resp, re.DOTALL)
        match2 = re.search(pattern2, resp, re.DOTALL)

        if match2:
            return match2.group(1)
        elif match1:
            return match1.group(1) + ";"
        else:
            logger.info("No SQL code found.")
            return None

    @classmethod
    def from_dict(cls, data):
        obj = cls(
            user_input=data["user_input"],
            query_filled=data["query_filled"],
            query_template=data["query_template"],
            retrieved_data=(pd.read_json(data["retrieved_data"]) if data["retrieved_data"] is not None else None),
            answer=data["answer"],
        )
        obj.sql_executed = data["sql_executed"]
        obj.sql_executed_self_healing_attempts = data["sql_executed_self_healing_attempts"]
        obj.rag = data["rag"]
        obj.query_df_retrieved_rag = data["query_df_retrieved_rag"]
        obj.prompt = data["prompt"]
        obj.rag_top_similarity = data["rag_top_similarity"]
        obj.question_masked = data["question_masked"]
        return obj

    def to_dict(self):
        return {
            "user_input": self.user_input,
            "query_filled": self.query_filled,
            "query_template": self.query_template,
            "retrieved_data": (self.retrieved_data.to_json(orient="records") if self.retrieved_data is not None else None),
            "answer": self.answer,
            "sql_executed": self.sql_executed,
            "sql_executed_self_healing_attempts": self.sql_executed_self_healing_attempts,
            "rag": self.rag,
            "query_df_retrieved_rag": (self.query_df_retrieved_rag.to_json(orient="records") if self.query_df_retrieved_rag is not None else None),
            "prompt": self.prompt,
            "rag_top_similarity": self.rag_top_similarity,
            "question_masked": self.question_masked,
        }
