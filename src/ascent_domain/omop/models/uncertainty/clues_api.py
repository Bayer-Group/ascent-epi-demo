"""
CLUES Split API

Split API for answering questions with CLUES uncertainty decomposition.
Each step can be called independently, enabling backend integration with
user interaction between steps (e.g., modify codes after seeing entities).

Paper Reference:
    "Separating Input Ambiguity from Model Instability in Large Language Models:
     A Clinical Text-to-SQL Case Study" (LREC 2026)

Split Functions (call independently for backend integration):
    1. generate_question_interpretations()    → returns primary + alt interpretations
    2. generate_query_templates()          → returns all templates (1 + N×M)
    3. extract_medical_entities()          → returns unique entities from templates
    4. code_medical_entities()             → returns code mappings (SINGLE PASS)
       ← SPLIT POINT: User can modify code_mappings here
    5. fill_query_templates()              → fills templates with codes
    6. get_query_explanation()       → explanation for primary query
    7. execute_queries()             → executes all queries
    8. generate_all_answers()            → generates all answers
    9. compute_clues_metrics()             → computes H(I), H(R), H(R|I)
    10. generate_combined_answer()   → combined answer from all

Convenience Function (calls all steps):
    answer_question_with_clues()           → runs full pipeline

Usage:
    # Split API (for backend integration)
    interp = await generate_question_interpretations(question, assistant_config)
    templates = await generate_query_templates(qa_system, interp["all_interpretations"])
    entities = extract_medical_entities(templates["template_results"])
    codes = await code_medical_entities(qa_system, entities, coding_system)
    # ← User can modify `codes` here
    filled = await fill_query_templates(qa_system, templates["template_results"], codes)
    # ... continue with remaining steps

    # Convenience function (runs all steps)
    result = await answer_question_with_clues(question, qa_system, config, ...)
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from ascent_domain.omop.schemas.constants import CodingType

logger = logging.getLogger(__name__)


# =============================================================================
# STEP 1: Generate Interpretations
# =============================================================================


async def generate_question_interpretations(
    original_question: str,
    assistant_config: Optional[Dict[str, str]] = None,
    num_interpretations: int = 3,
    interpretation_mode: str = "explore_meanings",
    primary_interpretation: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Step 1: Generate primary + alternative interpretations.

    Args:
        original_question: The original user question
        assistant_config: Config for uncertainty assistant {"type": str, "model": str}.
            If None, defaults to {"type": "gemini"} and the configured model.
        num_interpretations: Number of alternative interpretations (N, default: 3)
        interpretation_mode: Mode for generating interpretations
            - "explore_meanings": More diverse for uncertainty estimation
            - "guided_disambiguation": Higher quality user-facing interpretations
        primary_interpretation: If provided, skip internal recursive_disambiguation
            and use this value directly. Avoids redundant LLM calls when the caller
            has already run disambiguation.

    Returns:
        {
            "primary_interpretation": str,
            "alt_interpretations": List[str],
            "all_interpretations": List[str],  # [primary, alt1, alt2, ...]
        }
    """
    from ascent_domain.omop.models.uncertainty.disambiguation import generate_question_interpretations_parallel, recursive_disambiguation

    if assistant_config is None:
        assistant_config = {"type": "gemini"}

    logger.info(f"Step 1: Generating interpretations (mode={interpretation_mode})...")

    # Get primary interpretation — skip disambiguation if already provided
    if primary_interpretation is not None:
        logger.info("Using pre-computed primary interpretation (skipping disambiguation)")
    else:
        disambiguated = await recursive_disambiguation(question=original_question)
        primary_interpretation = disambiguated["final_interpretation"]
    logger.info(f"Primary interpretation: {primary_interpretation}")

    # Generate alternative interpretations
    alt_interpretations, _ = await generate_question_interpretations_parallel(
        questions=original_question,
        num_interpretations=num_interpretations,
        assistant_config=assistant_config,
        interpretation_mode=interpretation_mode,
    )

    all_interpretations = [primary_interpretation] + list(alt_interpretations)
    logger.info(f"Generated {len(all_interpretations)} total interpretations")

    return {
        "primary_interpretation": primary_interpretation,
        "alt_interpretations": list(alt_interpretations),
        "all_interpretations": all_interpretations,
    }


# =============================================================================
# STEP 2: Generate Templates
# =============================================================================


async def generate_query_templates(
    qa_system: Any,
    all_interpretations: List[str],
    num_sql_samples: int = 3,
) -> Dict[str, Any]:
    """
    Step 2: Generate SQL templates for all interpretations.

    Implements 1 + N×M structure:
    - 1 template for primary interpretation
    - M templates for each of N alternative interpretations

    Args:
        qa_system: Initialized QuestionAnsweringSystem instance
        all_interpretations: List of [primary, alt1, alt2, ...]
        num_sql_samples: Number of SQL samples per alternative (M, default: 3)

    Returns:
        {
            "template_results": List[Dict],  # 1 + N×M templates
            "questions_executed": List[str],
            "result_to_interpretation_mapping": List[int],
        }
    """
    logger.info("Step 2: Generating SQL templates...")

    # Build question list: 1 primary + (N-1) × M alternatives
    questions_to_execute = [all_interpretations[0]]  # Primary
    result_to_interpretation_mapping = [0]

    for interp_idx, interpretation in enumerate(all_interpretations[1:], start=1):
        for _ in range(num_sql_samples):
            questions_to_execute.append(interpretation)
            result_to_interpretation_mapping.append(interp_idx)

    logger.info(f"Built {len(questions_to_execute)} questions to execute")

    # Generate all templates in parallel
    template_results = await qa_system.generate_query_templates(input_questions=questions_to_execute)
    logger.info(f"Generated {len(template_results)} templates")

    return {
        "template_results": template_results,
        "questions_executed": questions_to_execute,
        "result_to_interpretation_mapping": result_to_interpretation_mapping,
    }


# =============================================================================
# STEP 3: Extract Medical Entities
# =============================================================================


def extract_medical_entities(
    template_results: List[Dict],
) -> List[Tuple[str, str]]:
    """
    Step 3: Extract unique entities from all templates.

    This is the KEY EFFICIENCY: extracts ALL entities from ALL templates
    so they can be coded in a SINGLE PASS instead of multiple redundant calls.

    Args:
        template_results: List of template results from generate_query_templates()

    Returns:
        List of (entity_type, value) tuples (unique, lowercase)
    """
    from ascent_domain.omop.data.processing.sql_post_processor import MedicalSQLProcessor

    logger.info("Step 3: Extracting entities from all templates...")

    all_entities = set()
    for result in template_results:
        if result.get("query_template"):
            entities = MedicalSQLProcessor.extract_entities_from_sql(result["query_template"])
            all_entities.update(entities)

    unique_entities = list(all_entities)
    logger.info(f"Found {len(unique_entities)} unique entities")
    for entity_type, value in unique_entities:
        logger.debug(f"  - {entity_type}@{value}")

    return unique_entities


# =============================================================================
# STEP 4: Code Medical Entities (SPLIT POINT)
# =============================================================================


async def code_medical_entities(
    qa_system: Any,
    entities: List[Tuple[str, str]],
    preferred_coding_system: Optional[CodingType] = None,
) -> Dict[Tuple[str, str], str]:
    """
    Step 4: Code all entities (SINGLE PASS).

    ← SPLIT POINT: User can modify returned mappings before calling fill_query_templates()

    This codes all entities ONCE instead of multiple times across isolated workers.
    The returned mapping can be modified before the next step.

    Args:
        qa_system: Initialized QuestionAnsweringSystem instance
        entities: List of (entity_type, value) tuples from extract_medical_entities()
        preferred_coding_system: Coding system preference (optional)

    Returns:
        Dict mapping (entity_type, value) to comma-separated concept IDs
        Example: {('condition', 'diabetes'): '201820,4182210'}
    """
    logger.info("Step 4: Coding all entities (single pass)...")

    async with qa_system._db_connection() as db:
        code_mappings = await qa_system.med_sql_processor.batch_code_entities(
            entities=entities,
            preferred_coding_system=preferred_coding_system,
            db_session=db,
        )

    logger.info(f"Coded {len(code_mappings)} entities")
    for (entity_type, value), codes in code_mappings.items():
        num_codes = len(codes.split(",")) if codes else 0
        logger.debug(f"  - {entity_type}@{value}: {num_codes} concepts")

    return code_mappings


# =============================================================================
# STEP 5: Fill Templates
# =============================================================================


async def fill_query_templates(
    qa_system: Any,
    template_results: List[Dict],
    code_mappings: Dict[Tuple[str, str], str],
    preferred_coding_system: Optional[CodingType] = None,
) -> List[Dict]:
    """
    Step 5: Fill all templates with codes.

    Args:
        qa_system: Initialized QuestionAnsweringSystem instance
        template_results: List of template results from generate_query_templates()
        code_mappings: Dict from code_medical_entities() (possibly user-modified)
        preferred_coding_system: Coding system preference (optional)

    Returns:
        List of processed results with "query_filled" added
    """
    logger.info("Step 5: Filling templates with codes...")

    processed_results = []
    for result in template_results:
        if result.get("query_template"):
            try:
                filled = await qa_system.post_process_query(
                    result["query_template"],
                    custom_code_mappings=code_mappings,
                    preferred_coding_system=preferred_coding_system,
                )
                processed_results.append(
                    {
                        **result,
                        "query_filled": filled,
                    }
                )
            except Exception as e:
                logger.warning(f"Error filling template: {e}")
                processed_results.append(
                    {
                        **result,
                        "query_filled": None,
                        "error": str(e),
                    }
                )
        else:
            processed_results.append(
                {
                    **result,
                    "query_filled": None,
                    "error": result.get("error", "No template generated"),
                }
            )

    logger.info(f"Filled {len(processed_results)} templates")
    return processed_results


# =============================================================================
# STEP 6: Query Explanation
# =============================================================================


async def get_query_explanation(
    qa_system: Any,
    primary_template: str,
    primary_question: str,
) -> str:
    """
    Step 6: Get explanation for primary query.

    Args:
        qa_system: Initialized QuestionAnsweringSystem instance
        primary_template: The SQL template for the primary interpretation
        primary_question: The primary interpretation text

    Returns:
        Human-readable explanation of what the query does
    """
    logger.info("Step 6: Getting query explanation...")

    if not primary_template:
        return "No template available for explanation"

    try:
        explanation = await qa_system.get_query_explanation(primary_template, primary_question)
        return explanation
    except Exception as e:
        logger.warning(f"Could not get query explanation: {e}")
        return f"Error generating explanation: {e}"


# =============================================================================
# STEP 7: Execute Queries
# =============================================================================


async def execute_queries(
    qa_system: Any,
    processed_results: List[Dict],
    config: Dict[str, Any],
) -> List[Dict]:
    """
    Step 7: Execute all queries.

    Args:
        qa_system: Initialized QuestionAnsweringSystem instance
        processed_results: List from fill_query_templates()
        config: Full pipeline config

    Returns:
        List of execution results with query outputs
    """
    logger.info("Step 7: Executing queries...")

    execution_results = await qa_system.execute_queries(
        processed_results=processed_results, config=config, preferred_coding_system=config.get("preferred_coding_system")
    )

    logger.info(f"Executed {len(execution_results)} queries")
    return execution_results


# =============================================================================
# STEP 8: Generate Answers
# =============================================================================


async def generate_all_answers(
    qa_system: Any,
    execution_results: List[Dict],
) -> List[str]:
    """
    Step 8: Generate answers for all queries.

    Args:
        qa_system: Initialized QuestionAnsweringSystem instance
        execution_results: List from execute_queries()

    Returns:
        List of answer strings (main answer is first)
    """
    logger.info("Step 8: Generating answers...")

    all_answers = await qa_system.generate_answers_simple(execution_results)
    logger.info(f"Generated {len(all_answers)} answers")

    return all_answers


# =============================================================================
# STEP 9: Compute CLUES Metrics
# =============================================================================


async def compute_clues_metrics(
    original_question: str,
    primary_question: str,
    alt_interpretations: List[str],
    all_answers: List[str],
    assistant_config: Optional[Dict[str, str]] = None,
    num_sql_samples: int = 3,
) -> Dict[str, Any]:
    """
    Step 9: Compute CLUES uncertainty metrics.

    Computes:
    - H(I): Ambiguity - entropy of interpretation distribution
    - H(R): Diversity - entropy of answer distribution
    - H(R|I): Instability - conditional entropy (model variance)

    Answer structure: all_answers has 1 + len(alt_interpretations) × num_sql_samples items.
    The first answer is the primary (skipped), the rest are grouped by alternative interpretation.

    Args:
        original_question: The original user question
        primary_question: The primary interpretation
        alt_interpretations: List of alternative interpretations (excluding primary)
        all_answers: All answers — [primary, alt1_s1, ..., alt1_sM, alt2_s1, ..., altN_sM]
        assistant_config: Config for uncertainty assistant
        num_sql_samples: Number of SQL samples per interpretation (M)

    Returns:
        {
            "h_i": float,           # Ambiguity
            "h_r": float,           # Diversity
            "h_r_given_i": float,   # Instability
            "regime": str,          # e.g., "high_ambiguity", "high_instability"
            "recommended_action": str,
        }
    """
    from ascent_domain.omop.models.uncertainty.uncertainty_processing import (
        extract_clues_metrics,
        perform_uncertainty_analysis,
    )

    if assistant_config is None:
        assistant_config = {"type": "gemini"}

    logger.info("Step 9: Computing CLUES uncertainty metrics...")

    # Build mapping for alternatives (excluding primary)
    uncertainty_answers = all_answers[1:]  # Skip main answer
    uncertainty_mapping = []
    for interp_idx in range(len(alt_interpretations)):
        for _ in range(num_sql_samples):
            uncertainty_mapping.append(interp_idx)

    # Validate answer/mapping alignment
    expected_answers = len(alt_interpretations) * num_sql_samples
    if len(uncertainty_answers) != expected_answers:
        logger.error(
            f"Answer count mismatch: got {len(uncertainty_answers)} answers "
            f"(all_answers={len(all_answers)} - 1 primary) but expected "
            f"{expected_answers} ({len(alt_interpretations)} interpretations × {num_sql_samples} samples). "
            f"Matrix construction will be incorrect!"
        )
    if len(uncertainty_answers) != len(uncertainty_mapping):
        logger.error(
            f"Mapping length mismatch: {len(uncertainty_answers)} answers vs {len(uncertainty_mapping)} mapping entries. Matrix will be wrong!"
        )

    logger.info(
        f"CLUES matrix structure: {len(alt_interpretations)} interpretations, {len(uncertainty_answers)} answers, mapping={uncertainty_mapping}"
    )

    uncertainty_results = await perform_uncertainty_analysis(
        original_questions=original_question,
        disambiguated_questions=primary_question,
        all_interpretations=alt_interpretations,
        answers=uncertainty_answers,
        assistant_config=assistant_config,
        result_to_interpretation_mapping=uncertainty_mapping,
        uncertainty_folder=None,
        generate_combined_final_answer=False,
    )

    clues = extract_clues_metrics(uncertainty_results)

    logger.info(f"H(I)={clues['h_i']:.4f}, H(R)={clues['h_r']:.4f}, H(R|I)={clues['h_r_given_i']:.4f}")
    logger.info(f"Regime: {clues['regime']}")

    return clues


# =============================================================================
# STEP 10: Generate Combined Answer
# =============================================================================


async def generate_combined_answer(
    original_question: str,
    all_answers: List[str],
    all_interpretations: List[str],
    assistant_config: Optional[Dict[str, str]] = None,
    num_sql_samples: int = 3,
) -> str:
    """
    Step 10: Generate combined answer from all interpretations.

    Synthesizes all answers into a single comprehensive response.

    Args:
        original_question: The original user question
        all_answers: All answers from generate_all_answers() — 1 + (N-1)×M items
        all_interpretations: All interpretations from generate_question_interpretations() — N items
        assistant_config: Config for uncertainty assistant
        num_sql_samples: Number of SQL samples per alternative interpretation (M)

    Returns:
        Combined answer string
    """
    from ascent_domain.omop.models.uncertainty.uncertainty_processing import generate_final_answer
    from ascent_platform.llm.factory import create_assistant

    if assistant_config is None:
        assistant_config = {"type": "claude_sonnet", "model": "us.anthropic.claude-opus-4-6-v1"}

    logger.info("Step 10: Generating combined answer...")

    assistant = create_assistant(assistant_type=assistant_config["type"], model_name=assistant_config.get("model"))

    # Expand interpretations to match 1 + (N-1)×M answer structure
    # Primary interpretation has 1 answer, each alternative has num_sql_samples answers
    expanded_interpretations = [all_interpretations[0]]
    for interp in all_interpretations[1:]:
        expanded_interpretations.extend([interp] * num_sql_samples)

    combined_answer = await generate_final_answer(
        question=original_question,
        answers=all_answers,
        interpretations=expanded_interpretations,
        assistant=assistant,
    )

    return combined_answer


# =============================================================================
# CONVENIENCE FUNCTION (Calls All Steps)
# =============================================================================


async def answer_question_with_clues(
    original_question: str,
    qa_system: Any,
    assistant_config: Dict[str, str],
    config: Dict[str, Any],
    num_interpretations: int = 3,
    num_sql_samples: int = 3,
    custom_code_mappings: Optional[Dict[Tuple[str, str], str]] = None,
    generate_combined_answer: bool = True,
    interpretation_mode: str = "explore_meanings",
) -> Dict[str, Any]:
    """
    Convenience function that runs all CLUES steps.

    For backend integration with user interaction, use the individual
    split functions instead (generate_question_interpretations, etc.).

    Implements the paper's 1 + N×M structure:
    - 1 primary answer from the disambiguated question
    - N alternative interpretations × M SQL samples each for uncertainty estimation

    Args:
        original_question: The original user question
        qa_system: Initialized QuestionAnsweringSystem instance
        assistant_config: Config for uncertainty assistant {"type": str, "model": str}
        config: Full pipeline config (for SQL execution)
        num_interpretations: Number of alternative interpretations (N, default: 3)
        num_sql_samples: Number of SQL samples per interpretation (M, default: 3)
        custom_code_mappings: Optional dict mapping (entity_type, value) to code strings
                             Example: {('condition', 'diabetes'): '201820,4182210'}
        generate_combined_answer: If True, synthesize all answers (default: True)
        interpretation_mode: Mode for generating interpretations (default: "explore_meanings")

    Returns:
        dict with keys:
            - main_answer: Answer to the primary disambiguated question
            - combined_answer: Synthesized answer from all interpretations
            - primary_interpretation: The disambiguated question
            - interpretations: All interpretations [primary, alt1, alt2, ...]
            - all_answers: All answers (1 primary + N×M samples)
            - query_template: Primary query template (with placeholders)
            - query_filled: Primary filled query (with concept IDs)
            - query_explanation: Explanation of the primary query
            - extracted_entities: List of (entity_type, value) tuples found
            - code_mappings: Dict mapping entities to concept IDs
            - clues: CLUES metrics (h_i, h_r, h_r_given_i, regime, recommended_action)
    """
    total_samples = 1 + num_interpretations * num_sql_samples
    logger.info(f"=== CLUES Analysis: 1 + {num_interpretations}×{num_sql_samples} = {total_samples} executions ===")

    # Step 1: Generate interpretations
    interp_result = await generate_question_interpretations(
        original_question=original_question,
        assistant_config=assistant_config,
        num_interpretations=num_interpretations,
        interpretation_mode=interpretation_mode,
    )

    # Step 2: Generate templates
    template_result = await generate_query_templates(
        qa_system=qa_system,
        all_interpretations=interp_result["all_interpretations"],
        num_sql_samples=num_sql_samples,
    )

    # Step 3: Extract entities
    entities = extract_medical_entities(template_result["template_results"])

    # Step 4: Code entities
    code_mappings = await code_medical_entities(
        qa_system=qa_system,
        entities=entities,
        preferred_coding_system=config.get("preferred_coding_system"),
    )

    # Apply custom overrides
    if custom_code_mappings:
        normalized_custom = {(entity_type, value.lower()): codes for (entity_type, value), codes in custom_code_mappings.items()}
        code_mappings = {**code_mappings, **normalized_custom}
        logger.info(f"Applied {len(normalized_custom)} custom code overrides")

    # Step 5: Fill templates
    processed_results = await fill_query_templates(
        qa_system=qa_system,
        template_results=template_result["template_results"],
        code_mappings=code_mappings,
        preferred_coding_system=config.get("preferred_coding_system"),
    )

    # Step 6: Query explanation
    primary_template = template_result["template_results"][0].get("query_template", "") if template_result["template_results"] else ""
    query_explanation = await get_query_explanation(
        qa_system=qa_system,
        primary_template=primary_template,
        primary_question=interp_result["primary_interpretation"],
    )

    # Step 7: Execute queries
    execution_results = await execute_queries(
        qa_system=qa_system,
        processed_results=processed_results,
        config=config,
    )

    # Step 8: Generate answers
    all_answers = await generate_all_answers(
        qa_system=qa_system,
        execution_results=execution_results,
    )

    main_answer = all_answers[0] if all_answers else ""

    # Step 9: Compute CLUES metrics
    clues = await compute_clues_metrics(
        original_question=original_question,
        primary_question=interp_result["primary_interpretation"],
        alt_interpretations=interp_result["alt_interpretations"],
        all_answers=all_answers,
        assistant_config=assistant_config,
        num_sql_samples=num_sql_samples,
    )

    # Step 10: Combined answer
    combined_answer = main_answer
    if generate_combined_answer:
        combined_answer = await generate_combined_answer(
            original_question=original_question,
            all_answers=all_answers,
            all_interpretations=interp_result["all_interpretations"],
            assistant_config=assistant_config,
        )

    # Log final results
    logger.info("\n=== CLUES Results ===")
    logger.info(f"H(I)   - Ambiguity:   {clues['h_i']:.4f}")
    logger.info(f"H(R)   - Diversity:   {clues['h_r']:.4f}")
    logger.info(f"H(R|I) - Instability: {clues['h_r_given_i']:.4f}")
    logger.info(f"Regime: {clues['regime']}")
    logger.info(f"Action: {clues['recommended_action']}")

    return {
        # Answers
        "main_answer": main_answer,
        "combined_answer": combined_answer,
        # Interpretations
        "primary_interpretation": interp_result["primary_interpretation"],
        "interpretations": interp_result["all_interpretations"],
        "all_answers": all_answers,
        # Query details (primary only)
        "query_template": primary_template,
        "query_filled": processed_results[0].get("query_filled", "") if processed_results else "",
        "query_explanation": query_explanation,
        # Entity coding (for backend split)
        "extracted_entities": entities,
        "code_mappings": code_mappings,
        # CLUES metrics
        "clues": clues,
    }
