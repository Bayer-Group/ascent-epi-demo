import json
import logging
import re

logger = logging.getLogger(__name__)


class SQLValidator:
    combined_prompt_template = """
    You are an expert in SQL and natural language processing, with deep knowledge of medical data analysis and the OMOP Common Data Model. Your task is to analyze the given SQL query, verify if it correctly addresses the original question, and provide a detailed breakdown of your analysis.

    Original Question/Criteria: ${question}

    SQL Query:
    ${query_with_placeholders}

    Please perform the following steps:

    1. Analyze the SQL query:
       a. List the key components of this query (e.g., table selections, joins, where clauses, aggregations, etc.)
       b. For each component, write a clear and concise natural language statement that describes what that part of the query is doing.
       c. Identify and list any implicit assumptions made in the query.

        Do not include in your output "Implicit Assumptions" assumptions related to:
           - Data quality, completeness, or accuracy in the OMOP CDM
           (Do not include "Implicit Assumptions" like "The OMOP CDM's default behavior of storing condition occurrences is assumed to be sufficient for identifying patients with the specified conditions"
           or 'The query assumes that the condition_concept_id values provided in the placeholders [condition@cancer], [condition@hypertension], and [condition@anemia] are accurate and comprehensive for identifying the respective conditions.'
           or 'The drug_concept_id for warfarin is correctly specified in the placeholder [drug@warfarin])
           - Correct mapping of condition_concept_id or other concept IDs
           (Do not include "Implicit Assumptions" like "The condition_concept_id for atopic dermatitis is correctly specified in the placeholder [condition@atopic dermatitis]",
           or "The gender_concept_id '8507' correctly identifies male patients.")
           - Uniqueness of person_ids (no person shares its id with other persons)
            (Do not include "Implicit Assumptions" like "The OMOP CDM tables and columns are used as intended, with person_id being unique and correctly linking the PERSON and CONDITION_OCCURRENCE tables.")
           - LIMIT statements

    2. Verify the query against the original question:
       a. Identify the key requirements in the original question.
       b. For each requirement, determine if the SQL query satisfies it based on your analysis.
       c. Provide a verdict on whether the SQL query fully addresses the question, partially addresses it, or fails to address it.

    3. Provide an explanation of your verdict and any recommendations for improvement.

    You must return only a JSON as follows:

    {
      "Components": [
        "[Component 1]",
        "[Component 2]",
        ...
      ],
      "Statements": [
        "Statement describing Component 1",
        "Statement describing Component 2",
        ...
      ],
      "Implicit Assumptions": [
        "[Assumption 1]",
        "[Assumption 2]",
        ...
      ],
      "Requirements": [
        "[Requirement 1]",
        "[Requirement 2]",
        ...
      ],
      "Satisfaction Check": [
        {
          "Requirement": "[Requirement 1]",
          "Satisfied": true/false,
          "Explanation": "[Brief explanation]"
        },
        {
          "Requirement": "[Requirement 2]",
          "Satisfied": true/false,
          "Explanation": "[Brief explanation]"
        },
        ...
      ],
      "Verdict": "Fully Addresses/Partially Addresses/Fails to Address",
      "Explanation": "[Detailed explanation of the verdict]",
      "Recommendations": [
        "[Recommendation 1]",
        "[Recommendation 2]",
        ...
      ]
    }

    Do not include "Requirement 1" text in your answer, but the actual requirement text.
    Ensure that your statements and assumptions are easy to understand for someone who might not be familiar with SQL syntax or the OMOP CDM, but accurately represent the logic, operations, and implicit decisions in the query.
    """

    def __init__(self, query_with_placeholders, question):
        self.query_analysis_response = None
        self.query_with_placeholders = query_with_placeholders
        self.question = question
        # self.prompt_verify = self._fill_prompt()
        self.combined_prompt = self._fill_combined_prompt()
        self.query_components = None
        self.query_statements = None
        self.query_implicit_assumptions = None
        self.query_analysis_response_dict = {}
        self.verification_result = None
        self.analysis_verification_result = None

    def _fill_combined_prompt(self):
        # Escape single quotes in both query and question
        escaped_query = self.query_with_placeholders.replace("'", "\\'")
        escaped_question = self.question.replace("'", "\\'")

        # Replace both placeholders
        prompt = self.combined_prompt_template.replace("${query_with_placeholders}", escaped_query)
        prompt = prompt.replace("${question}", escaped_question)

        return prompt

    def _parse_response(self, response):
        try:
            parsed_json = json.loads(response)
        except json.JSONDecodeError:
            logger.error("Failed to parse JSON response")
            parsed_json = {}

        self.query_components = self._parse_section(parsed_json, "Components")
        self.query_statements = self._parse_section(parsed_json, "Statements")
        self.query_implicit_assumptions = self._parse_section(parsed_json, "Implicit Assumptions")

        self.query_analysis_response_dict = parsed_json

    @staticmethod
    def _parse_section(parsed_json, section_name):
        try:
            section_data = parsed_json.get(section_name, [])
            if not isinstance(section_data, list):
                raise ValueError(f"{section_name} is not a list")
            return section_data
        except Exception as e:
            logger.error(f"Failed to parse {section_name}: {e}")
            return []

    async def analyze_and_verify(self, assistant, original_prompt=None):
        logger.info("Performing combined SQL query analysis and verification...")
        try:
            if original_prompt:
                prompt = self.combined_prompt + ("\n\n ### The original prompt used for the model "
                                                 "to perform the prediction was:\n\n") + original_prompt
            else:
                prompt = self.combined_prompt

            response = await assistant.get_response(prompt)

            if response is None:
                logger.error("Received None response from assistant")
                self.analysis_verification_result = {}
            else:
                parsed_json = self.parse_json_from_response(response)
                if parsed_json is not None:
                    self.analysis_verification_result = parsed_json
                    logger.info("Combined SQL query analysis and verification completed.")
                else:
                    logger.error("Failed to parse response as JSON")
                    self.analysis_verification_result = {}
        except Exception as e:
            logger.error(f"Error during analysis and verification: {e}")
            self.analysis_verification_result = {}

        return self.analysis_verification_result

    @staticmethod
    def parse_json_from_response(resp=""):
        # First, try to find JSON within code blocks
        pattern = r"(?:```json|```) ?\n([\s\S]+?)\n```"
        match = re.search(pattern, resp)
        if match:
            json_str = match.group(1)
        else:
            # If no code blocks, assume the entire response is JSON
            json_str = resp

        # Try to parse the JSON
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            logger.info(f"Problematic JSON string: {json_str}")
            return None

    def get_parsed_results(self):
        if not self.analysis_verification_result:
            logger.error("No analysis and verification results available.")
            return None

        return {
            "analysis": {
                "components": self.analysis_verification_result.get("Components", []),
                "statements": self.analysis_verification_result.get("Statements", []),
                "implicit_assumptions": self.analysis_verification_result.get("Implicit Assumptions", [])
            },
            "verification": {
                "requirements": self.analysis_verification_result.get("Requirements", []),
                "satisfaction_check": self.analysis_verification_result.get("Satisfaction Check", []),
                "verdict": self.analysis_verification_result.get("Verdict", ""),
                "explanation": self.analysis_verification_result.get("Explanation", ""),
                "recommendations": self.analysis_verification_result.get("Recommendations", [])
            }
        }

    @staticmethod
    def set_default_results(rwd_request_pred):
        """Set default values for all result attributes of rwd_request_pred."""
        rwd_request_pred.verification_results = {}
        rwd_request_pred.query_components = []
        rwd_request_pred.query_statements = []
        rwd_request_pred.implicit_assumptions = []
        rwd_request_pred.requirements = []
        rwd_request_pred.satisfaction_check = []
        rwd_request_pred.verdict = ""
        rwd_request_pred.explanation = ""
        rwd_request_pred.recommendations = []

    # async def validate_with_feedback(self, verify_agent, med_sql_processor, sql_query_result=None,
    #                                  max_attempt_verify=3):
    async def validate_with_feedback(self, verify_agent, med_sql_processor, original_prompt=None,
                                     max_attempt_verify=3):
        logger.info("Starting SQL query verification...")

        attempt_verify = 0
        verdict = ""

        # if sql_query_result:
        #     original_prompt = sql_query_result["result"].initial_prompt + sql_query_result["result"].rag_prompt_add_on
        # else:
        #     original_prompt = None

        verification_results = []
        for attempt_verify in range(max_attempt_verify):
            # try:
            # Analyze and verify the current query
            _ = await self.analyze_and_verify(verify_agent, original_prompt=original_prompt)
            verdict = self.analysis_verification_result.get('Verdict', '').lower()
            verification_results.append(self.analysis_verification_result)

            if verdict == "fully addresses":
                logger.info("SQL query fully addresses user input")
                break

            # If the verdict is not "fully addresses", prepare for the next iteration
            if attempt_verify < max_attempt_verify - 1:
                logger.info("Detected an error in generated query. Triggering new prediction.")
                logger.info(f"User input: {self.question}")
                logger.info(f"Rationale: {self.analysis_verification_result.get('Explanation', '')}")

                error_message = (f"We detected an error in the following query: {self.query_with_placeholders}. "
                                 f"This is the rationale: "
                                 f"{self.analysis_verification_result.get('Explanation', '')}."
                                 f"Please provide a corrected SQL query."
                                 f"Neglect any previous questions in the conversation that might have confused you.")

                verify_agent.add_message("user", message=error_message)
                verify_assistant_answer = await verify_agent.get_response(str(verify_agent.conversation))

                query_template_pred = med_sql_processor.parse_sql_from_response(verify_assistant_answer)
                if not query_template_pred:
                    raise ValueError("Failed to parse SQL from response")

                logger.info("Complete template query generated")
                logger.debug(f"{query_template_pred}")

                self.query_with_placeholders = query_template_pred
                self.combined_prompt = self._fill_combined_prompt()

        if attempt_verify == max_attempt_verify - 1 and verdict != "fully addresses":
            logger.info(f"Maximum verification attempts reached. Final verdict: {verdict}")

        return self.query_with_placeholders, self.question, verdict, verification_results


async def verify_and_correct_query(query_template_pred, user_input, verify_agent, med_sql_processor,
                                   original_prompt=None, max_attempt_verify=3):
    original_query_template_pred = query_template_pred

    # First, check if we have a valid query to verify
    if query_template_pred is None:
        logger.error("No SQL query to verify: query_template_pred is None")
        return (
            None,  # query_template_pred
            user_input,
            "Verification Failed - No SQL Query",
            [{  # minimal verification results structure
                "Verdict": "Fails to Address",
                "Explanation": "No SQL query was generated to verify",
                "Components": [],
                "Statements": [],
                "Implicit Assumptions": [],
                "Requirements": [],
                "Satisfaction Check": [],
                "Recommendations": ["Generate a valid SQL query first"]
            }]
        )

    try:
        sql_validator = SQLValidator(query_with_placeholders=query_template_pred,
                                     question=user_input)

        query_template_pred, user_input, verdict, verification_results = await sql_validator.validate_with_feedback(
            verify_agent=verify_agent,
            med_sql_processor=med_sql_processor,
            max_attempt_verify=max_attempt_verify,
            original_prompt=original_prompt,
        )
        return query_template_pred, user_input, verdict, verification_results
    except Exception as e:
        logger.error(f"Error during query verification: {str(e)}. Keeping the original query")
        return (
            original_query_template_pred,
            user_input,
            "Verification Failed",
            [{  # minimal verification results structure
                "Verdict": "Verification Error",
                "Explanation": f"Error during verification: {str(e)}",
                "Components": [],
                "Statements": [],
                "Implicit Assumptions": [],
                "Requirements": [],
                "Satisfaction Check": [],
                "Recommendations": ["Review and fix verification process"]
            }]
        )
