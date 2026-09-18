import asyncio
import json
import logging
import time
from pathlib import Path
from typing import List

from langchain_core.tools import tool

from ascent_domain.omop.models.inference.qa import QuestionAnsweringSystem
from ascent_domain.omop.models.uncertainty.disambiguation import recursive_disambiguation
from ascent_domain.omop.schemas.constants import CodingType
from ascent_domain.omop.utils.conversions import DecimalEncoder
from ascent_platform.config.runtime import Settings

logger = logging.getLogger(__name__)

settings = Settings()

@tool
async def query_medical_data(original_question: str, database: str) -> str:
    """
    Query real-world-data (RWD) data in OMOP CDM format to answer epidemiological questions.

    Args:
        original_question: A clinical question about patient data, medical conditions, or treatments.
        database: Database name to query (e.g., "SYNTHETIC_EHR_OMOP", "SYNTHETIC_EHR_OMOP") - REQUIRED

    Returns:
        A string containing the answer based on the query results or an error message string.
    """
    if not database:
        error_msg = "Database parameter is required and cannot be empty"
        logger.error(error_msg)
        return json.dumps({
            "error": error_msg,
            "answer": error_msg,
            "query_template": "",
            "data": []
        })

    logger.info(f"Executing query: '{original_question}' on database: {database}")
    start_time = time.time()

    try:
        # Initialize the QA system with your configuration
        qa_system = await QuestionAnsweringSystem.initialize(get_config(database=database))

        disambiguation_results = await recursive_disambiguation(question=original_question, mode="qa")

        question = disambiguation_results["final_interpretation"]
        logger.info(f"Disambiguated question: {question}")

        # Generate query template
        template_result = await qa_system.generate_query_template(question)
        query_template = template_result["query_template"]

        # Post-process query
        query_filled = await qa_system.post_process_query(query_template)

        # Execute query
        df, rwd_request = await qa_system.execute_query(
            query_filled,
            query_template,
            question
        )

        # Generate answer
        answer = await qa_system.generate_answer(rwd_request, df)

        # Return the same structure as smolagents version
        result = {
            "answer": answer,
            "query_template": query_template,
            "data": df.head(n=100).to_dict(orient='records'),
        }

        duration = time.time() - start_time
        logger.info(f"Query completed in {duration:.2f} seconds: '{question}'")

        return json.dumps(result, cls=DecimalEncoder)

    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"Query failed after {duration:.2f} seconds: '{original_question}'. "
                     f"Error: {str(e)}", exc_info=True)
        return json.dumps({
            "error": f"Error processing query: {str(e)}",
            "answer": f"Error processing query: {str(e)}",
            "query_template": "",
            "data": []
        })


@tool
async def batch_query_medical_data(questions: List[str], database: str) -> str:
    """
    Execute multiple medical data queries in parallel and return all results.

    Args:
        questions: A list of clinical questions to query in parallel
        database: Database name to query (e.g., "SYNTHETIC_EHR_OMOP", "SYNTHETIC_EHR_OMOP") - REQUIRED

    Returns:
        A JSON string containing results for all queries, with each question mapped to its answer
    """
    if not database:
        error_msg = "Database parameter is required and cannot be empty"
        logger.error(error_msg)
        return json.dumps({
            "error": error_msg,
            "total_queries": len(questions),
            "successful_queries": 0,
            "failed_queries": len(questions),
            "results": [{"question": q, "error": error_msg, "status": "error"} for q in questions]
        })

    logger.info(f"Starting batch execution of {len(questions)} queries on database: {database}")
    batch_start_time = time.time()

    # Track progress for logging
    completed_queries = 0
    total_queries = len(questions)

    logger.info(f"Using asyncio.gather for parallel execution of {total_queries} queries")

    # Function to execute a single query with timing and error handling
    async def execute_single_query_with_logging(question):
        nonlocal completed_queries

        query_id = id(question) % 10000  # Generate a short ID for tracking
        logger.info(f"[Query {query_id}] Starting: '{question}'")

        try:
            start_time = time.time()
            # Call the single query function with database parameter
            result_json = await query_medical_data.ainvoke({"original_question": question, "database": database})
            result = json.loads(result_json)
            duration = time.time() - start_time

            completed_queries += 1
            progress = (completed_queries / total_queries) * 100

            logger.info(
                f"[Query {query_id}] Completed ({completed_queries}/{total_queries}, {progress:.1f}%) in {duration:.2f}s")

            return {
                "question": question,
                "result": result.get("answer", "No answer provided"),
                "query_template": result.get("query_template", ""),
                "duration_seconds": round(duration, 2),
                "status": "success"
            }

        except Exception as e:
            completed_queries += 1
            progress = (completed_queries / total_queries) * 100

            logger.error(
                f"[Query {query_id}] Failed ({completed_queries}/{total_queries}, {progress:.1f}%): {str(e)}",
                exc_info=True
            )
            return {
                "question": question,
                "error": str(e),
                "status": "error"
            }

    # Execute all queries in parallel using asyncio.gather
    results = await asyncio.gather(*[execute_single_query_with_logging(q) for q in questions], return_exceptions=True)

    # Process any exceptions that weren't caught
    processed_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            processed_results.append({
                "question": questions[i],
                "error": str(result),
                "status": "error"
            })
        else:
            processed_results.append(result)

    # Format the response (same as smolagents version)
    successful_queries = [r for r in processed_results if r["status"] == "success"]
    failed_queries = [r for r in processed_results if r["status"] == "error"]

    total_duration = time.time() - batch_start_time

    response = {
        "total_queries": len(questions),
        "successful_queries": len(successful_queries),
        "failed_queries": len(failed_queries),
        "total_duration_seconds": round(total_duration, 2),
        "results": processed_results
    }

    logger.info(f"Batch execution completed in {total_duration:.2f}s. "
                f"Success: {len(successful_queries)}, Failed: {len(failed_queries)}")

    if failed_queries:
        logger.warning(f"Failed queries: {[r['question'] for r in failed_queries]}")

    # Use the custom JSON encoder to handle Decimal types (same as smolagents)
    return json.dumps(response, indent=2, cls=DecimalEncoder)


def get_config(database: str):
    """Helper function to get configuration for the QA system."""
    if not database:
        raise ValueError("Database parameter is required and cannot be empty")

    # Determine base directory - use /app in Docker, otherwise navigate up from package location
    docker_base_dir = Path("/app")
    if docker_base_dir.exists() and (docker_base_dir / "querylib_20251216.db").exists():
        # We're in Docker container
        base_dir = docker_base_dir
        logger.info(f"Detected Docker environment, using base_dir: {base_dir}")
    else:
        # Local development - navigate up from package location
        base_dir = Path(__file__).resolve().parent.parent.parent.parent.parent.parent
        logger.info(f"Using local development base_dir: {base_dir}")

    querylib_file = base_dir / "querylib_20251216.db"
    log_directory = base_dir / "logs"
    log_directory.mkdir(parents=True, exist_ok=True)

    logger.info(f"Path to querylib file: {querylib_file}")
    logger.info(f"Using database: {database}")

    return {
        "querylib_file": querylib_file,
        "log_directory": log_directory,
        "assistant": {
            "type": "claude_sonnet",
            "model": "us.anthropic.claude-opus-4-6-v1"
        },
        "database": database,
        "database_schema": settings.SNOWFLAKE_DATABASE_SCHEMA,
        "preferred_coding_system": CodingType.STANDARD_CODING,
        "settings": settings
    }
