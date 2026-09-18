import asyncio
import json
import logging
import re

from langchain_core.tools import tool

logger = logging.getLogger(__name__)


database_metadata_dict = {
    # Coverage of the shipped synthetic datasets. Both describe the same
    # generated population and neither represents a country.
    "SYNTHETIC_EHR_OMOP": {
        "country": None,
        "total_patients": 1000,
        "population": None,
        "estimated_coverage": None,
    },
    "SYNTHETIC_CLAIMS": {
        "country": None,
        "total_patients": 1000,
        "population": None,
        "estimated_coverage": None,
    },
}


def get_database_metadata(database: str) -> dict | None:
    """
    Get population metadata for a specific database.

    Returns a dict with database_name, country, total_patients, population,
    estimated_coverage, and coverage_description. Returns None if database
    not found.
    """
    if database not in database_metadata_dict:
        logger.warning(f"Database {database} not found in metadata dict")
        return None
    meta = database_metadata_dict[database]
    return {
        "database_name": database,
        "country": meta["country"],
        "total_patients": meta["total_patients"],
        "population": meta["population"],
        "estimated_coverage": meta["estimated_coverage"],
        "coverage_description": _coverage_description(meta),
    }


def _coverage_description(meta: dict) -> str:
    """Describe a database's population coverage, or say that it has none.

    The shipped datasets are generated, not sampled: they cover no country and
    no real population, so ``country``/``population``/``estimated_coverage``
    are None. Formatting those straight into the f-string raised
    ``unsupported format string passed to NoneType.__format__`` and took down
    compare_rwd_with_literature *after* it had already done its literature
    search.
    """
    if meta["estimated_coverage"] is None or meta["population"] is None:
        return (
            f"{meta['total_patients']:,} synthetic patients. This dataset is generated, "
            "not sampled from a real population, so counts cannot be scaled to any country."
        )
    return f"{meta['estimated_coverage']:.2%} of {meta['country']} population ({meta['total_patients']:,} out of {meta['population']:,})"


def population_context_note(db_metadata: dict | None) -> str:
    """The "Database Population Context" block injected into agent prompts.

    Three call sites built this by hand with the same f-string, and each one
    crashed on the synthetic datasets, whose country/population/coverage are
    None. Building it in one place means a dataset without population
    metadata degrades to an honest note instead of a TypeError.
    """
    if not db_metadata:
        return ""

    header = (
        f"\n\n--- Database Population Context ---\n"
        f"Database: {db_metadata['database_name']}\n"
        f"Total patients in database: {db_metadata['total_patients']:,}\n"
    )
    if db_metadata["estimated_coverage"] is None or db_metadata["population"] is None:
        return header + (
            f"Coverage: {db_metadata['coverage_description']}\n"
            f"NOTE: This is synthetic data. It is not a sample of any real population, so raw "
            f"counts must NOT be extrapolated to national estimates and prevalence figures "
            f"derived from it say nothing about the real world.\n"
            f"--- End Database Context ---"
        )
    return header + (
        f"Country: {db_metadata['country']}\n"
        f"Country population: {db_metadata['population']:,}\n"
        f"Estimated population coverage: {db_metadata['coverage_description']}\n"
        f"NOTE: Any prevalence/percentage derived from this database reflects only "
        f"{db_metadata['estimated_coverage']:.2%} of the {db_metadata['country']} population.\n"
        f"--- End Database Context ---"
    )


@tool
async def get_patient_count_scaling(condition_name: str, previous_output: str, database: str) -> str:
    """
    Provides patient count scaling for a given disease by extracting patient count from previous output
    using an LLM, then extrapolating the results to the target population.

    Args:
        condition_name: Name of the condition or disease (e.g., "atopic dermatitis")
        previous_output: Raw text output from previous analysis steps (e.g., the complete output from
                        batch_query_medical_data). Pass the entire text output directly without
                        attempting to parse it as JSON or extract specific fields. The tool will
                        automatically extract the patient count from the text.
        database: Database name used for the query (e.g., "SYNTHETIC_EHR_OMOP", "SYNTHETIC_EHR_OMOP") - REQUIRED

    Returns:
        A JSON string with population context information, including database context and extrapolated estimates

    Example:
        # Correct usage:
        result = await batch_query_medical_data(questions=["How many patients have atopic dermatitis?"], database="SYNTHETIC_EHR_OMOP")
        context = await get_patient_count_scaling(condition_name="atopic dermatitis", previous_output=result, database="SYNTHETIC_EHR_OMOP")

        # Do NOT parse the result or extract fields before passing to this tool
    """
    if not database:
        error_msg = "Database parameter is required and cannot be empty"
        logger.error(error_msg)
        return json.dumps({"error": error_msg, "condition": condition_name, "database": "NOT_SPECIFIED", "patient_count": "Could not be extracted"})

    logger.info(f"Generating population context for: {condition_name} using database: {database}")

    response = None
    try:
        # Extract patient count from previous_output using LLM via the sync wrapper
        prompt = f"""
        Extract the total patient count from the following text. Return ONLY the number as an integer with no formatting or additional text.

        TEXT:
        {previous_output}

        TOTAL PATIENT COUNT:
        """

        # Use asyncio.to_thread to run the synchronous function in a thread pool
        response = await asyncio.to_thread(get_llm_response_sync, prompt, 0)

        # Clean the response and convert to integer
        count_str = response.strip().replace(",", "")

        # Try to extract just the digits if there's any other text
        match = re.search(r"\d+", count_str)
        if match:
            patient_count = int(match.group(0))
        else:
            try:
                patient_count = int(count_str)
            except ValueError:
                logger.error(f"Could not parse '{count_str}' as an integer")
                raise ValueError(f"LLM returned '{count_str}' which could not be parsed as a patient count")

        logger.info(f"LLM extracted patient count: {patient_count:,}")

        # Get database metadata for the selected database
        if database not in database_metadata_dict:
            error_msg = f"Database {database} not found in metadata. Available databases: {', '.join(database_metadata_dict.keys())}"
            logger.error(error_msg)
            return json.dumps(
                {
                    "error": error_msg,
                    "condition": condition_name,
                    "database": database,
                    "patient_count": patient_count,
                    "available_databases": list(database_metadata_dict.keys()),
                }
            )

        db_metadata = database_metadata_dict[database]
        db_total = db_metadata["total_patients"]
        db_coverage = db_metadata["estimated_coverage"]
        country_population = db_metadata["population"]
        country = db_metadata["country"]

        # Generated data is not a sample of anything, so there is no
        # population to extrapolate to. Say that, rather than dividing by None.
        if db_coverage is None or country is None:
            return json.dumps(
                {
                    "database_context": {
                        "database_name": database,
                        "total_patients": db_total,
                        "coverage_description": _coverage_description(db_metadata),
                    },
                    "condition_statistics": {
                        "condition_name": condition_name,
                        "patient_count_in_database": patient_count,
                    },
                    "summary": (
                        f"{patient_count:,} patients with {condition_name} out of {db_total:,} in "
                        f"{database}. This is synthetic data covering no real population, so the "
                        "count cannot be extrapolated to a country-level estimate."
                    ),
                },
                indent=2,
            )

        # Extrapolate to target population based on database coverage
        estimated_population_patients = int(patient_count / db_coverage)
        population_description = f"{country} population"

        # Prepare response
        context = {
            "database_context": {
                "database_name": database,
                "total_patients": db_total,
                "country": country,
                "estimated_coverage": f"{db_coverage:.2%} of {country} population",
            },
            "condition_statistics": {
                "condition_name": condition_name,
                "patient_count_in_database": patient_count,
                f"estimated_{country.lower()}_patients": estimated_population_patients,
            },
            "summary": f"Based on {patient_count:,} patients with {condition_name} in the {database} database "
            f"and our estimated coverage of {db_coverage:.2%} of the {population_description}, "
            f"we estimate approximately {estimated_population_patients:,} people in {country} "
            f"have this condition.",
        }

        return json.dumps(context, indent=2)

    except Exception as e:
        logger.error(f"Error generating population context: {str(e)}")
        return json.dumps(
            {
                "error": f"Failed to generate population context: {str(e)}",
                "condition": condition_name,
                "database": database,
                "patient_count": "Could not be extracted",
                "raw_llm_response": response if "response" in locals() else "No response",
            },
            indent=2,
        )


def get_llm_response_sync(prompt: str, temperature: float = 0, json_format: bool = False) -> str:
    """
    Synchronous wrapper for getting responses from the LLM.

    Args:
        prompt: The input prompt to send to the LLM
        temperature: Temperature setting for response generation (default: 0)
        json_format: Whether to request JSON formatted response (default: False)

    Returns:
        The text response from the LLM
    """
    from ascent_platform.llm.factory import create_assistant

    # Create a new event loop for this synchronous context
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        # Create assistant and get response asynchronously
        assistant = create_assistant(assistant_type="gpt")
        response = loop.run_until_complete(assistant.get_response(prompt=prompt, temperature=temperature, json_format=json_format))
        return response
    finally:
        # Clean up the event loop
        loop.close()
