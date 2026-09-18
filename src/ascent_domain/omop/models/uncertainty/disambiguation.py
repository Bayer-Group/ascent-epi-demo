import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from ascent_domain.omop.schemas.uncertainty_interpretations import DisambiguationResponse, ExploreMeaningsResponse
from ascent_platform.llm.clients.assistants import async_retry_fn
from ascent_platform.llm.factory import create_assistant

logger = logging.getLogger(__name__)


@dataclass
class QueryData:
    """Flexible data class to store any key-value pairs from query results."""

    data: Dict[str, Any] = field(default_factory=dict)

    def __getattr__(self, name):
        """Allow accessing dictionary keys as attributes."""
        if name in self.data:
            return self.data[name]
        raise AttributeError(f"'QueryData' object has no attribute '{name}'")


@dataclass
class QueryResult:
    answer: str
    query_template: str
    data: List[QueryData]


async def generate_question_interpretations(
    question: str,
    mode: str = "qa",
    num_interpretations: Optional[int] = None,
    assistant: Optional[Any] = None,
    assistant_type: str = "gemini",
    model_name: Optional[str] = None,
    temperature: float = 0.7,
) -> DisambiguationResponse:
    """
    Generate multiple interpretations of an ambiguous question

    Args:
        question: Original user question
        mode: Disambiguation mode - "qa", "criteria", or "index_date" (default: "qa")
        num_interpretations: If specified, generate exactly this many interpretations
        assistant: LLM assistant (if None, one will be created)
        assistant_type: Type of assistant to create if none provided
        model_name: Model name to use when creating assistant

    Returns:
        DisambiguationResponse: Complete disambiguation analysis including interpretations
    """
    # Create assistant if not provided
    if assistant is None:
        assistant = create_assistant(assistant_type=assistant_type, model_name=model_name, temperature=temperature)

    no_counts_instructions = """- Never return disambiguated text requesting patient counts.
    Thus, disambiguate to 
    "Count of unique female patients who have at least two distinct condition occurrence records for endometriosis, 
    where the condition start date for each of these two or more records is between January 1, 2010, and December 31, 
    2019, inclusive." 
    but rather to: 
    "Unique female patients who have at least two distinct condition occurrence records for endometriosis, 
    where the condition start date for each of these two or more records is between January 1, 2010, and December 31, 
    2019, inclusive. """

    index_date_instructions = """The disambiguated text must not contain any reference to patients or patient counts, "
                               "since the input text is an index event and defines a time point."
                               "Never return expression like "Patients with" or "Count of patients with"" or similar."""

    # Define mode-specific guidelines
    mode_specific_guidelines = {
        "qa": "",
        "qa_cohort": "If the question contains the text 'index date' leave it as is since it is an information which"
        " is explicitely available in the database",
        "criteria": no_counts_instructions,
        "index_date": index_date_instructions,
    }

    # Base disambiguation extension
    disambiguation_extension = """
    # Disambiguation for Epidemiological Text-to-SQL Questions or Inclusion/Exclusion Criteria or Index Date Criteria

    You are a disambiguation assistant for an epidemiological text-to-SQL system. Your role is to analyze input 
    text and identify potential ambiguities that could lead to different SQL interpretations 
    when querying EHR and Claims databases. 

    ## Your Task:
    1. Analyze the input text for ambiguities related to:
       - Patient counts (always refers to unique patients unless explicitly stated otherwise)
       - Temporal relationships (before/after, during, within timeframes)
       - Population definitions (inclusion/exclusion criteria)
       - Demographic specifications (age calculations, gender, race, ethnicity)
       - Measurement specifications (values, units, thresholds)
       - Relationship between entities (causal, temporal, coincidental)
       - For prevalence, we consider only period prevalence. 

    2. For each identified ambiguity:
       - Provide a clear explanation of the ambiguity
       - List all possible ways to resolve this specific ambiguity
       - Explain how each resolution affects the SQL query structure

    3. Calculate an overall ambiguity score from 0 to 1:
       - 0: Completely unambiguous text
       - 0.1-0.3: Slightly ambiguous (minor clarifications needed)
       - 0.4-0.6: Moderately ambiguous (multiple valid interpretations)
       - 0.7-0.9: Highly ambiguous (fundamentally different interpretations possible)
       - 1.0: Completely ambiguous (cannot determine intent)
       - Formula: Take the average of all ambiguity impact scores, then adjust based on:
         * Number of ambiguities (more ambiguities → higher score)
         * Interactions between ambiguities (compounding ambiguities → higher score)
    """

    # Base interpretation instructions that are common to both cases
    base_interpretation_instructions = """
       - Each interpretation must resolve every identified ambiguity for which there is a clear resolution
       - Each interpretation should be a complete, unambiguous text
       - Each interpretation should be standardized for the text-to-SQL task
       - Order the interpretations from more plausible (first) to less plausible (last)
       - Make each interpretation specific and detailed enough to generate a precise SQL query
       - For prevalence questions, ALWAYS specify both the time period AND the exact denominator population
    """

    # Add specific instructions based on whether num_interpretations is specified
    if num_interpretations is not None:
        interpretation_instructions = f"""
    4. Generate EXACTLY {num_interpretations} final interpretations that:
       - Cover the most meaningful combinations of the identified ambiguities
       {base_interpretation_instructions}
       - If the text is unambiguous and doesn't have {num_interpretations} possible interpretations, 
         simply return multiple times the same interpretation.
        """
    else:
        interpretation_instructions = f"""
    4. Generate a comprehensive set of final interpretations that:
       - Covers ALL meaningful combinations of the identified ambiguities
       {base_interpretation_instructions}
        """

    # Common guidelines with mode-specific placeholder
    common_guidelines = f"""
    ## Guidelines:
    - Focus on ambiguities specific to epidemiological research
    - When the text asks for "patient counts" or "how many patients", always interpret this as UNIQUE patients (distinct patient IDs), 
      unless the text explicitly says otherwise (e.g., "all visits", "all records"). Always use "unique patients" in your interpretations.
    - Do not include any concept_id or table names in your answer
    - Do not include ambiguities related to the medical conditions, as these are resolved separately.
    - Do not include dates or quantities when not specified in the original text
    - Do not change the text's request. For example, if asked for patients' counts, calculate only counts and not 
    ratios or not period prevalence
        e.g., questions like "How many patients have hypertension in between 2020-2023?" should only return counts, not denominators
    - Pay special attention to:
      * Temporal relationships and condition definitions        
      * Event ordering (first vs. any occurrence)
      * Time points and intervals (specific dates vs. relative timeframes)
      * Inclusion/exclusion criteria for patient cohorts
    - Standardize terminology in interpretations to match OMOP CDM concepts
    - If a text appears completely unambiguous, state so and explain why, but still provide one standardized interpretation
    - If multiple entities (drugs, conditions, etc.) or splits (e.g., 2, 7, 14, 30, 60 or 90 days) are mentioned, 
    include all entities in every interpretation.
         For example text like "How many patients died within 30, 60 or 90 days after their first diagnosis of ischemic stroke?"
         must include in the disambiguated text all splits (30, 60, or 90 days)
    - If no timeframe is defined, include the whole period available (e.g., at any time in the patient's history)
    - For age distribution calculations, if no age range is specified in the text, split the age buckets as follows:
      0-17; 18-34; 35-49; 50-64; 65-79; 80+.
      If an age range is specified (e.g., women with 18-55), split that range in 6 equal splits. Ensure the age groups are contiguous and cover the entire relevant age range
    - If the text is about reproductive age, consider the range 15-49 unless otherwise specified.  
    - When analyzing follow-up times or time-to-event distributions, use these standard intervals unless the text specifically requires different groupings:
        <30 days; 30-89 days; 90-179 days; 180-364 days; 1 year or more
    - If no observation period is specified in the input, do not include any constraints regarding it 
    - Never add calendar years that are not included in the original text (e.g., during calendar year 2023 or similar)   
    - Never formulate the text to get result for a single patient, i.e. avoid expressions like "For each patient"
{mode_specific_guidelines.get(mode, "")}    - When asked for medication adherence rates without a specified threshold, calculate the average adherence metric 
    (e.g., mean PDC) across the eligible population and do NOT introduce arbitrary thresholds or observation period requirements if not specified.
    - For text about observation periods before/after events, do NOT add constraints requiring observation periods to end exactly 
    on specific dates unless explicitly stated in the original text; only enforce the minimum duration requirements as specified.
    - Never mention any specific medical codes (e.g., ICD-10 L20.x) and do not mention also any coding ontology (e.g., ICD or SNOMED) 

    ## Example Ambiguities to Consider:
    - Does "patients with condition X and drug Y" mean patients who had both at any time, or specifically patients who had the drug while having the condition?
    - Does "first diagnosis" mean first ever in patient history, or first within a specific time period?
    - Does "patients over 65" mean age at the time of a specific event, or current age?
    - For prevalence calculations, what is the denominator population? For simple cases, please suggest a population at risk 
    (e.g., for prostate cancer all male, for breast cancer all females)
    - For "after diagnosis," does it mean immediately after or at any time after the diagnosis?
    - For measurements, are we looking at any recorded value or a specific threshold?

    ## Specific Guidelines for Prevalence Text:
    1. specify the time period (e.g., "during calendar year 2023", "over the entire database history")
    2. specify the denominator population:
       - Define enrollment criteria (e.g., "with any observation period", "with continuous enrollment for X days")
       - Define demographic criteria if any (e.g., "among females", "among patients over 18")
    3. specify how to handle patients with multiple observation periods
    4. specify whether to count patients with the condition at any time or only those with the condition during the specified period

    ## Specific Guidelines for Entities
    1. Please note that estradiol is a drug

    """

    # Combine all parts of the prompt
    analysis_instruction = "# Now analyze the following text and provide a structured disambiguation analysis"
    if num_interpretations is not None:
        analysis_instruction += f" with EXACTLY {num_interpretations} interpretations:"
    else:
        analysis_instruction += ":"

    combined_prompt = f"{disambiguation_extension}\n{interpretation_instructions}\n{common_guidelines}\n\n{analysis_instruction}\n\n{question}"

    # Get structured response using the enhanced schema
    response = await assistant.get_response(prompt=combined_prompt, response_schema=DisambiguationResponse, quiet=True)

    return response


@async_retry_fn(max_retries=3, base_delay=1.0, backoff_factor=2.0)
async def recursive_disambiguation(
    question: str,
    mode: str = "qa",  # Added mode parameter
    assistant_type: str = "gemini",
    model_name: Optional[str] = None,
    assistant: Optional[Any] = None,
    max_rounds: int = 3,
    ambiguity_threshold: float = 0.1,
    temperature: float = 1.0,
) -> Dict:
    """
    Recursively apply disambiguation until reaching max rounds or ambiguity threshold

    Args:
        question: Original user question
        mode: Disambiguation mode - "qa", "criteria", or "index_date" (default: "qa")
        assistant_type: Type of assistant to create if assistant is None (default: "gemini")
        model_name: Model name to use if assistant is None; None uses the configured default
        assistant: LLM assistant (if None, one will be created using assistant_type and model_name)
        max_rounds: Maximum number of disambiguation rounds
        ambiguity_threshold: Stop when ambiguity score falls below this threshold
        temperature: Temperature for LLM generation (default: 0.7)

    Returns:
        Dict: Complete analysis of disambiguation process including timing information
    """
    # Create assistant if not provided
    if assistant is None:
        logger.info(f"Disambiguation: creating {assistant_type} assistant with model {model_name} (temperature={temperature})")
        assistant = create_assistant(assistant_type=assistant_type, model_name=model_name, temperature=temperature)

    current_question = question
    rounds_results = []

    # Track overall start time
    overall_start_time = time.time()

    logger.info(f"Starting disambiguation for question: {question}")
    logger.debug(f"Max rounds: {max_rounds}, Ambiguity threshold: {ambiguity_threshold}")

    for round_num in range(1, max_rounds + 1):
        logger.debug(f"Round {round_num} - Disambiguating: {current_question}")

        # Track round start time
        round_start_time = time.time()

        try:
            # Apply disambiguation with the specified mode
            result = await generate_question_interpretations(question=current_question, mode=mode, assistant=assistant)

            # Calculate round duration
            round_duration = time.time() - round_start_time

            # Store round results with timing information
            rounds_results.append(
                {
                    "round": round_num,
                    "question": current_question,
                    "result": result,
                    "start_time": datetime.fromtimestamp(round_start_time).isoformat(),
                    "duration_seconds": round_duration,
                }
            )

            # Log round results safely
            try:
                logger.debug(f"Round {round_num} - Ambiguity score: {result.ambiguity_score}")
                logger.debug(f"Round {round_num} - Interpretations: {len(result.final_interpretations)}")
                logger.debug(f"Round {round_num} - Duration: {round_duration:.2f} seconds")
            except AttributeError:
                logger.warning(f"Round {round_num} - Could not access ambiguity score or interpretations")

            # Check stopping conditions safely
            try:
                # Log the actual values for debugging
                logger.debug(
                    f"Stop condition check: is_ambiguous={result.is_ambiguous}, score={result.ambiguity_score}, threshold={ambiguity_threshold}"
                )

                if result.ambiguity_score <= ambiguity_threshold:
                    logger.info(f"Stopping at round {round_num} - Ambiguity score {result.ambiguity_score} below threshold {ambiguity_threshold}")
                    break

                # Select first interpretation for next round
                current_question = result.final_interpretations[0]
                logger.debug(f"Selected for next round: {current_question}")
            except AttributeError:
                logger.warning(f"Round {round_num} - Error accessing result attributes, stopping disambiguation")
                break
        except Exception as e:
            logger.warning(f"Error in disambiguation round {round_num}: {str(e)}")
            break

    # Calculate total duration
    total_duration = time.time() - overall_start_time
    logger.info(f"Total duration: {total_duration:.1f}s")

    # If no rounds completed successfully, raise so @async_retry_fn can retry
    if not rounds_results:
        raise RuntimeError(f"Disambiguation failed: all {max_rounds} rounds produced errors for question: '{question[:100]}...'")

    try:
        initial_ambiguity = rounds_results[0]["result"].ambiguity_score
        final_ambiguity = rounds_results[-1]["result"].ambiguity_score
        ambiguity_reduction = initial_ambiguity - final_ambiguity
        percent_reduction = (ambiguity_reduction / initial_ambiguity * 100) if initial_ambiguity > 0 else 0
    except (AttributeError, IndexError, ZeroDivisionError) as e:
        logger.warning(f"Could not calculate ambiguity metrics: {str(e)}")
        initial_ambiguity = np.nan
        final_ambiguity = np.nan
        ambiguity_reduction = np.nan
        percent_reduction = np.nan

    # Get final interpretation safely
    try:
        final_interpretation = (
            rounds_results[-1]["result"].final_interpretations[0] if rounds_results[-1]["result"].final_interpretations else current_question
        )
    except (AttributeError, IndexError) as e:
        logger.warning(f"Could not get final interpretation from results: {str(e)}")
        # Use the last question we were working with
        final_interpretation = current_question

    return {
        "original_question": question,
        "rounds": rounds_results,
        "total_rounds": len(rounds_results),
        "initial_ambiguity": initial_ambiguity,
        "final_ambiguity": final_ambiguity,
        "ambiguity_reduction": ambiguity_reduction,
        "percent_reduction": percent_reduction,
        "final_interpretation": final_interpretation,
        "start_time": datetime.fromtimestamp(overall_start_time).isoformat(),
        "total_duration_seconds": total_duration,
    }


async def generate_question_interpretations_parallel(
    questions: Union[str, List[str]],
    num_interpretations: int,
    assistant_config: Optional[Dict[str, str]] = None,
    disambiguate: bool = True,
    interpretation_mode: str = "explore_meanings",
    temperature: float = 0.7,
) -> Tuple[List[str], Dict[int, List[int]]]:
    """
    Generate interpretations for questions in parallel and return flattened list with mapping.

    TWO-STAGE FRAMEWORK: "Detect, then Refine"

    This function supports two modes:

    1. **"explore_meanings" (Stage 1 - Uncertainty Estimation)**:
       - Simple, exploratory prompt without domain constraints
       - Generates N=10 diverse interpretations
       - Used for: H(I) calculation, AUROC evaluation, uncertainty detection
       - NOT user-facing

    2. **"guided_disambiguation" (Stage 2 - Actionable Disambiguation)**:
       - Detailed, domain-constrained prompt with clinical rules
       - Generates N=3-5 high-quality interpretations
       - Used for: User-facing disambiguation, production system
       - Clinically plausible and actionable

    Args:
        questions: Either a single question string or a list of question strings
        num_interpretations: Number of interpretations to generate per question
        assistant_config: Configuration for the assistant including type and model
        disambiguate: [DEPRECATED] Ignored - kept for backward compatibility
        interpretation_mode: "explore_meanings" (Stage 1) or "guided_disambiguation" (Stage 2)
                           Default: "explore_meanings" for diverse uncertainty estimation

    Returns:
        Tuple containing:
        - List of all interpretations
        - Dictionary mapping original question indices to interpretation indices
    """
    if assistant_config is None:
        assistant_config = {"type": "gemini"}

    # Ensure questions is a list, even if a single string is provided
    if isinstance(questions, str):
        logger.info(f"Single question provided as string, converting to list: '{questions[:100]}...'")
        questions = [questions]

    logger.info(f"Starting interpretation generation for {len(questions)} questions")
    logger.info(f"Requesting {num_interpretations} interpretations per question")
    logger.info(f"Interpretation Mode: {interpretation_mode.upper()}")
    logger.info(f"Creating {assistant_config['type']} assistant with model {assistant_config.get('model') or 'the configured default'}")

    if interpretation_mode == "explore_meanings":
        logger.info("STAGE 1: Explore Meanings - Raw uncertainty estimation for H(I) calculation")
    elif interpretation_mode == "guided_disambiguation":
        logger.info("STAGE 2: Guided Disambiguation - High-quality user-facing interpretations")
    else:
        logger.warning(f"Unknown interpretation_mode '{interpretation_mode}', defaulting to 'explore_meanings'")
        interpretation_mode = "explore_meanings"

    tasks = []
    for i, q in enumerate(questions):
        logger.info(f"Creating task for question {i + 1}/{len(questions)}: '{q[:100]}...'")

        if interpretation_mode == "explore_meanings":
            # STAGE 1: Explore Meanings - Simple exploratory prompt for uncertainty estimation
            tasks.append(
                generate_explore_meanings(
                    question=q,
                    num_interpretations=num_interpretations,
                    assistant_type=assistant_config["type"],
                    model_name=assistant_config.get("model"),
                    temperature=temperature,
                )
            )
        else:
            # STAGE 2: Guided Disambiguation - Detailed domain-constrained prompt
            tasks.append(
                generate_question_interpretations(
                    question=q,
                    mode="qa",
                    num_interpretations=num_interpretations,
                    assistant_type=assistant_config["type"],
                    model_name=assistant_config.get("model"),
                    temperature=temperature,
                )
            )

    logger.info(f"Launching {len(tasks)} parallel interpretation tasks")
    interpretation_results = await asyncio.gather(*tasks, return_exceptions=True)
    logger.info("All interpretation tasks completed")

    # Flatten all interpretations for processing
    all_interpretations = []
    question_interpretation_map = {}
    current_idx = 0

    for i, interp_result in enumerate(interpretation_results):
        if isinstance(interp_result, Exception):
            logger.error(f"Error generating interpretations for question {i + 1}: {str(interp_result)}")
            # Handle the exception case - add empty list for this question
            question_interpretation_map[i] = []
            continue

        # Handle different response types based on mode
        if interpretation_mode == "explore_meanings":
            # ExploreMeaningsResponse has .interpretations attribute
            interpretations = interp_result.interpretations
        else:
            # DisambiguationResponse has .final_interpretations attribute
            interpretations = interp_result.final_interpretations

        logger.info(f"Question {i + 1} generated {len(interpretations)} interpretations")

        question_interpretation_map[i] = list(range(current_idx, current_idx + len(interpretations)))
        all_interpretations.extend(interpretations)
        current_idx += len(interpretations)

    logger.info(f"Generated {len(all_interpretations)} total interpretations across {len(questions)} questions")

    # Log the mapping structure for debugging
    for q_idx, interp_indices in question_interpretation_map.items():
        logger.debug(f"Question {q_idx + 1} maps to interpretations: {interp_indices}")

    return all_interpretations, question_interpretation_map


async def disambiguate_criteria_dict(
    criteria_dict: Dict,
    assistant_type: str = "gemini",
    model_name: Optional[str] = None,
    assistant: Optional[Any] = None,
    max_rounds: int = 3,
    ambiguity_threshold: float = 0.1,
    max_concurrency: int = 25,  # Default concurrency limit
) -> Dict:
    """
    Apply recursive disambiguation to each element in the criteria dictionary in parallel.

    Args:
        criteria_dict: Dictionary containing include, exclude, and index_date criteria
        assistant_type: Type of assistant to use for disambiguation
        model_name: Model name to use for disambiguation
        assistant: LLM assistant (if None, one will be created)
        max_rounds: Maximum number of disambiguation rounds
        ambiguity_threshold: Stop when ambiguity score falls below this threshold
        max_concurrency: Maximum number of concurrent API requests

    Returns:
        Dict: A new criteria dictionary with disambiguated elements
    """
    start_time = time.time()

    # Create a semaphore to limit concurrency
    semaphore = asyncio.Semaphore(max_concurrency)

    # Create a shared assistant if not provided
    if assistant is None:
        logger.info(f"Creating {assistant_type} assistant with model {model_name} for disambiguation")
        assistant = create_assistant(assistant_type=assistant_type, model_name=model_name)

    # Define a helper function to disambiguate a single criterion with semaphore
    async def disambiguate_criterion_with_semaphore(criterion: str, mode: str) -> str:
        async with semaphore:
            logger.debug(f"Starting disambiguation for criterion: {criterion}")
            result = await recursive_disambiguation(
                question=criterion,
                mode=mode,
                assistant_type=assistant_type,
                model_name=model_name,
                assistant=assistant,
                max_rounds=max_rounds,
                ambiguity_threshold=ambiguity_threshold,
            )
            logger.debug(f"Completed disambiguation for criterion: {criterion}")
            return result["final_interpretation"]

    # Create all tasks
    tasks = []
    task_mapping = {}  # To keep track of which task belongs to which category and index

    # Add include criteria tasks
    for i, criterion in enumerate(criteria_dict.get("include", [])):
        task = disambiguate_criterion_with_semaphore(criterion, "criteria")
        tasks.append(task)
        task_mapping[len(tasks) - 1] = ("include", i)

    # Add exclude criteria tasks
    for i, criterion in enumerate(criteria_dict.get("exclude", [])):
        task = disambiguate_criterion_with_semaphore(criterion, "criteria")
        tasks.append(task)
        task_mapping[len(tasks) - 1] = ("exclude", i)

    # Add index_date criteria tasks
    for i, criterion in enumerate(criteria_dict.get("index_date", [])):
        task = disambiguate_criterion_with_semaphore(criterion, "index_date")
        tasks.append(task)
        task_mapping[len(tasks) - 1] = ("index_date", i)

    # Count total requests
    total_criteria = len(tasks)
    logger.info(f"Starting parallel disambiguation of {total_criteria} criteria with max concurrency {max_concurrency}")

    # Run all tasks concurrently
    all_results = await asyncio.gather(*tasks, return_exceptions=True)

    # Process results and build the disambiguated dictionary
    criteria_dict_disambiguated = {
        "include": [None] * len(criteria_dict.get("include", [])),
        "exclude": [None] * len(criteria_dict.get("exclude", [])),
        "index_date": [None] * len(criteria_dict.get("index_date", [])),
    }

    # Count successful and failed requests
    success_count = 0
    error_count = 0

    # Process the results
    for i, result in enumerate(all_results):
        category, index = task_mapping[i]

        if isinstance(result, Exception):
            logger.error(f"Error disambiguating {category}[{index}]: {str(result)}")
            # Use the original criterion if disambiguation failed
            criteria_dict_disambiguated[category][index] = criteria_dict[category][index]
            error_count += 1
        else:
            criteria_dict_disambiguated[category][index] = result
            success_count += 1

    # Calculate total duration
    total_duration = time.time() - start_time

    # Log summary statistics
    logger.info(f"Disambiguation completed in {total_duration:.2f} seconds")
    logger.info(f"Total criteria: {total_criteria}, Successful: {success_count}, Failed: {error_count}")

    return criteria_dict_disambiguated


async def generate_explore_meanings(
    question: str,
    num_interpretations: int = 10,
    assistant: Optional[Any] = None,
    assistant_type: str = "gemini",
    model_name: Optional[str] = None,
    temperature: float = 0.7,
) -> ExploreMeaningsResponse:
    """
    STAGE 1: Explore Meanings - Uncertainty Estimation Prompt

    Generate diverse semantic interpretations to measure aleatoric uncertainty H(I).
    This is the simple, exploratory prompt designed to capture the full breadth of
    potential ambiguity WITHOUT domain constraints or clinical plausibility filters.

    Use Case: Uncertainty detection and AUROC evaluation
    NOT for user-facing disambiguation (use generate_question_interpretations for that)

    Args:
        question: Original user question (possibly ambiguous)
        num_interpretations: Number of diverse interpretations to generate (default: 10)
        assistant: LLM assistant (if None, one will be created)
        assistant_type: Type of assistant to create if none provided
        model_name: Model name to use when creating assistant

    Returns:
        ExploreMeaningsResponse: Structured response with diverse interpretations
    """
    # Create assistant if not provided
    if assistant is None:
        assistant = create_assistant(assistant_type=assistant_type, model_name=model_name, temperature=temperature)

    prompt = f"""You are a **clinical data analyst** whose job is to translate a user's question into specific, executable queries for a large clinical database (like an EHR or claims database).
    Your task is to generate {num_interpretations} plausible and specific interpretations of the following question.

**User Question:**
"{question}"

## Instructions:
1.  Analyze the question for any potential ambiguities (in concepts, timeframes, cohorts, etc.) that need to be resolved to write a database query.
2.  Generate {num_interpretations} specific interpretations that a junior analyst could execute **against a single, unified database**.
3.  If the original question is ambiguous, your interpretations should explore different valid ways to query the database to answer it.
4.  If the original question is already a specific and clear query, it is perfectly acceptable for your generated interpretations to be simple paraphrases of the same core meaning. Your goal is to represent the space of plausible *database queries*, not to invent new research projects.
5.  Never mention any ontology or medical codes in the question (e.g., SNOMED, ICD10, LOINC). Focus on clinical concepts only.

## What NOT to Do:
- Do not invent new constraints or change the core logic if the original question is already clear.
- **Do not propose interpretations that would require external data, literature reviews, or data sources outside of a standard clinical database.**
- **Do not specify the data source (e.g., "national claims data," "systematic review," "hospital records"). Assume all queries run against the same database.**
- Do not number or label your interpretations.

---

**Your Task:**
Generate {num_interpretations} plausible interpretations of the original question. Return ONLY the interpretation text."""

    # Get structured response using Pydantic schema
    response = await assistant.get_response(prompt=prompt, response_schema=ExploreMeaningsResponse, quiet=True)

    # Ensure we have exactly num_interpretations
    if len(response.interpretations) < num_interpretations:
        logger.warning(f"Only got {len(response.interpretations)} interpretations, expected {num_interpretations}. Padding with original.")
        while len(response.interpretations) < num_interpretations:
            response.interpretations.append(question)
    elif len(response.interpretations) > num_interpretations:
        response.interpretations = response.interpretations[:num_interpretations]

    logger.info(f"[EXPLORE MEANINGS] Generated {len(response.interpretations)} diverse interpretations for uncertainty estimation")

    return response
