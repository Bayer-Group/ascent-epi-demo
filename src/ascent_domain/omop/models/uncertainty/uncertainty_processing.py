import logging
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ascent_domain.omop.models.uncertainty.uncertainty_decomposition import UncertaintyDecomposition
from ascent_domain.omop.models.uncertainty.uncertainty_visualization import plot_uncertainty_decomposition
from ascent_domain.omop.utils.conversions import convert_numpy_to_python
from ascent_platform.llm.factory import create_assistant

logger = logging.getLogger(__name__)


def extract_clues_metrics(uncertainty_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Extract CLUES uncertainty metrics from uncertainty analysis results.

    This is the high-level function for accessing decomposed uncertainty metrics
    with regime classification. Use this instead of manually parsing the results.

    Args:
        uncertainty_results: Output from perform_uncertainty_analysis()

    Returns:
        dict with keys:
            - h_i: float - Interpretation entropy (ambiguity)
            - h_r: float - Result entropy (diversity)
            - h_r_given_i: float - Conditional entropy (instability)
            - regime: str - One of: "confident", "ambiguity", "instability", "compound"
            - recommended_action: str - Human-readable action recommendation
            - final_answer: str - The generated answer

    Raises:
        ValueError: If uncertainty_results is empty or contains an error
    """
    if not uncertainty_results:
        raise ValueError("uncertainty_results is empty - perform_uncertainty_analysis must be called first")

    result = uncertainty_results[0]

    # Check if there was an error during processing
    if "error" in result:
        raise ValueError(f"Uncertainty analysis failed: {result['error']}")

    if "uncertainty_metrics" not in result:
        raise ValueError("uncertainty_metrics not found in result - analysis may have failed")

    uncertainty_metrics = result["uncertainty_metrics"]
    entropies = uncertainty_metrics["entropies"]
    uncertainty_result = uncertainty_metrics["uncertainty_result"]

    return {
        "h_i": entropies["H_I"],
        "h_r": entropies["H_R"],
        "h_r_given_i": entropies["H(R|I)"],
        "regime": uncertainty_result["regime"],
        "recommended_action": uncertainty_result["recommended_action"],
        "final_answer": result.get("final_answer", ""),
    }


async def analyze_uncertainty_decomposition(
        interpretations,
        results,
        mapping=None,
        out_folder=None,
        similarity_method="prompt",
        assistant_type="gemini",
        model_name="gemini-flash-latest",
        t=10.0,
        plot=False,
        max_concurrency: int = 25,
        interpretation_similarity_prompt: str = None,
        gamma: float = 1.0
):
    """
    Analyze uncertainty decomposition for interpretations and results

    Args:
        interpretations: List of interpretation texts
        results: List of result texts (can be placeholder strings in FAST MODE)
        mapping: List where mapping[i] = j means interpretation i produces result j (default: None, will be auto-generated)
        out_folder: Folder to save output files
        similarity_method: Method for computing similarities
        t: Heat kernel diffusion parameter (default 10.0, optimal for normalized Laplacian)
        plot: Whether to generate and save plots (default: False)
        max_concurrency: Maximum concurrent API calls
        interpretation_similarity_prompt: Optional custom prompt for interpretation similarity (W_II).
                                         If provided, used only for comparing interpretations, not results.
        gamma: Input kernel sharpening parameter for W_II matrix (default 1.0 = no sharpening).
               Values > 1 suppress off-diagonal noise while preserving diagonal (1.0).
               Recommended range: [1, 5, 10, 20]. Empirically γ=5 is optimal for clinical Text-to-SQL data.

    Returns:
        dict: Dictionary containing all uncertainty metrics and matrices

    Note: In FAST MODE, results can be placeholder strings. H(I) will still be computed
    correctly from interpretations, while H(R) and H(R|I) will be computed from placeholders
    (values will be meaningless but code won't crash).
    """

    if mapping is None:
        # Determine the minimum length between interpretations and results
        min_length = min(len(interpretations), len(results))

        if len(interpretations) != len(results):
            logger.warning(
                f"Interpretations ({len(interpretations)}) and results ({len(results)}) have different lengths. "
                f"Using only the first {min_length} items from each."
            )
            interpretations = interpretations[:min_length]
            results = results[:min_length]

        # Create 1:1 mapping (0->0, 1->1, etc.)
        mapping = list(range(min_length))

    # Initialize uncertainty decomposition with gamma parameter
    ud = UncertaintyDecomposition(t=t, similarity_method=similarity_method, gamma=gamma)

    # Compute uncertainty decomposition
    # Note: In FAST MODE, results are placeholder strings, but computation still runs
    # H(I) will be correct, H(R) and H(R|I) will be computed from placeholders
    uncertainty = await ud.compute_decomposed_uncertainty(
        interpretations=interpretations,
        results=results,  # Can be placeholder strings in FAST MODE
        mapping=mapping,
        assistant_type=assistant_type,
        model_name=model_name,
        t=t,
        max_concurrency=max_concurrency,
        interpretation_similarity_prompt=interpretation_similarity_prompt
    )

    # Log results
    logger.info("\n=== Uncertainty Decomposition Analysis ===")
    if gamma != 1.0:
        logger.info(f"Input Kernel Sharpening: γ={gamma}")
    logger.info(f"Interpretation Ambiguity (H_I): {uncertainty['entropies']['H_I']:.4f}")
    logger.info(f"Result Uncertainty (H_R): {uncertainty['entropies']['H_R']:.4f}")
    logger.info(f"Total Uncertainty (H_QIR): {uncertainty['entropies']['H_QIR']:.4f}")
    logger.info(f"Conditional Uncertainty H(R|I): {uncertainty['entropies']['H(R|I)']:.4f}")
    logger.info(f"Ambiguity Contribution: {uncertainty['contributions']['ambiguity']:.2%}")
    logger.info(f"Result Contribution: {uncertainty['contributions']['result']:.2%}")
    logger.info(f"Regime: {uncertainty['uncertainty_result'].regime.value}")

    # Log similarity matrices
    logger.debug("\n=== Interpretation Similarity Matrix (W_II) ===")
    w_ii = uncertainty['matrices']['W_II']
    for i in range(len(interpretations)):
        row = [f"{w_ii[i, j]:.4f}" for j in range(len(interpretations))]
        logger.debug(f"Interpretation {i + 1}: [{', '.join(row)}]")

    logger.debug("\n=== Result Similarity Matrix (W_RR) ===")
    w_rr = uncertainty['matrices']['W_RR']
    for i in range(len(results)):
        row = [f"{w_rr[i, j]:.4f}" for j in range(len(results))]
        logger.debug(f"Result {i + 1}: [{', '.join(row)}]")

    # Generate plots if requested
    if plot and out_folder:
        try:
            plot_uncertainty_decomposition(uncertainty, interpretations, results, out_folder)
        except Exception as e:
            logger.error(f"Error generating plots: {e}")

    return uncertainty


async def perform_uncertainty_analysis(
        original_questions: Union[List[str], str],
        disambiguated_questions: Union[List[str], str],
        all_interpretations: Union[List[str], str],
        answers: Union[List[str], str],  # Changed from results to answers
        assistant_config: Dict[str, str],
        question_interpretation_map: Optional[Dict[int, List[int]]] = None,  # Made optional with default
        result_to_interpretation_mapping: Optional[List[int]] = None,  # Maps each result to its interpretation
        uncertainty_folder: Optional[Path] = None,
        generate_combined_final_answer: bool = True,  # NEW: Flag to skip time-consuming LLM synthesis
) -> List[Dict[str, Any]]:
    """
    Perform uncertainty analysis on the results.

    Args:
        original_questions: Original user questions (string or list of strings)
        disambiguated_questions: Disambiguated questions (string or list of strings)
        all_interpretations: All interpretations generated (string or list of strings)
        answers: List of answers corresponding to interpretations (string or list of strings)
        assistant_config: Configuration for the assistant
        question_interpretation_map: Mapping from questions to their interpretations (defaults to single question mapping)
        result_to_interpretation_mapping: List mapping each result index to its interpretation index (defaults to 1:1)
        uncertainty_folder: Folder to save uncertainty results (can be None)
        generate_combined_final_answer: Whether to generate a combined final answer via LLM (time-consuming).
                                        If False, uses the longest valid answer as the final answer (much faster).

    Returns:
        List[Dict[str, Any]]: Uncertainty analysis results
    """
    # Convert string inputs to lists if necessary
    if isinstance(original_questions, str):
        original_questions = [original_questions]

    if isinstance(disambiguated_questions, str):
        disambiguated_questions = [disambiguated_questions]

    if isinstance(all_interpretations, str):
        all_interpretations = [all_interpretations]

    if isinstance(answers, str):
        answers = [answers]

    # Set default question_interpretation_map if not provided
    if question_interpretation_map is None:
        # Default: single question with all interpretations
        question_interpretation_map = {0: list(range(len(all_interpretations)))}

    # Define constants
    NO_ANSWER_PLACEHOLDER = "No answer generated"
    NO_VALID_ANSWERS_MESSAGE = "Unable to generate a valid answer to this question."

    uncertainty_results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for original_idx, original_question in enumerate(original_questions):
        try:
            # Get interpretations and answers for this original question
            interp_indices = question_interpretation_map[original_idx]
            question_interpretations = [all_interpretations[idx] for idx in interp_indices]

            # Handle two cases:
            # 1. SQL sampling: len(answers) > len(interpretations) with result_to_interpretation_mapping
            # 2. No sampling: len(answers) == len(interpretations) with 1:1 mapping
            if result_to_interpretation_mapping is not None and len(answers) > len(all_interpretations):
                # SQL sampling case: use ALL answers, they're already properly structured
                question_answers = answers
                question_mapping = result_to_interpretation_mapping
                logger.info(f"Using SQL sampling structure: {len(question_interpretations)} interpretations → {len(question_answers)} results")
            else:
                # Standard case: 1:1 mapping between interpretations and answers
                question_answers = [answers[idx] for idx in interp_indices]
                question_mapping = None  # Will default to 1:1 in analyze_uncertainty_decomposition
                logger.info(f"Using 1:1 structure: {len(question_interpretations)} interpretations = {len(question_answers)} results")

            # Handle cases where answer is None or empty
            processed_answers = []
            for answer in question_answers:
                if answer:
                    processed_answers.append(answer)
                else:
                    processed_answers.append(NO_ANSWER_PLACEHOLDER)

            logger.info(f"Question {original_idx + 1}: {original_question}")
            logger.info(f"Interpretations: {question_interpretations}")
            logger.info(f"Answers (first 3): {processed_answers[:3]}...")
            if question_mapping:
                logger.info(f"Mapping: {question_mapping}")

            # Use the uncertainty_folder directly (qa_pipeline already created question-specific folder)
            # Don't create another nested subfolder
            question_folder = uncertainty_folder

            # Perform uncertainty analysis
            # Note: t parameter omitted to use default (t=10.0) from analyze_uncertainty_decomposition
            uncertainty_analysis = await analyze_uncertainty_decomposition(
                interpretations=question_interpretations,
                results=processed_answers,
                mapping=question_mapping,
                out_folder=question_folder,
                similarity_method="prompt",
                assistant_type=assistant_config["type"],
                model_name=assistant_config["model"],
                # t parameter omitted - uses default t=10.0 from function signature
                plot=True
            )

            # Convert NumPy arrays in uncertainty_analysis to Python native types
            uncertainty_analysis = convert_numpy_to_python(uncertainty_analysis)

            # Store the results for this question
            question_result = {
                "original_question": original_question,
                "disambiguated_question": disambiguated_questions[original_idx] if original_idx < len(
                    disambiguated_questions) else original_question,
                "interpretations": question_interpretations,
                "answers": processed_answers,
                "uncertainty_metrics": uncertainty_analysis,
                "timestamp": timestamp,
                "question_index": original_idx + 1
            }

            # Generate final answer for this question
            if generate_combined_final_answer:
                # FULL MODE: Make LLM call to synthesize all interpretations (time-consuming)
                try:
                    # Create assistant for generating final answer
                    assistant = create_assistant(
                        assistant_type=assistant_config["type"],
                        model_name=assistant_config["model"]
                    )

                    # Generate final answer - now passing interpretations as well
                    final_answer = await generate_final_answer(
                        question=original_question,
                        answers=processed_answers,
                        interpretations=question_interpretations,
                        assistant=assistant
                    )

                    # Add final answer to the result
                    question_result["final_answer"] = final_answer

                    logger.info(f"Final answer generated for question {original_idx + 1}")
                    logger.info(f"Final answer: {final_answer}")

                except Exception as e:
                    logger.error(f"Error generating final answer for question {original_idx + 1}: {str(e)}")
                    # Use the longest answer as fallback
                    valid_answers = [a for a in processed_answers if a and a != NO_ANSWER_PLACEHOLDER]
                    if valid_answers:
                        question_result["final_answer"] = max(valid_answers, key=len)
                        logger.info(f"Using longest answer as fallback for question {original_idx + 1}")
                    else:
                        question_result["final_answer"] = NO_VALID_ANSWERS_MESSAGE
                        logger.info(f"No valid answers available for question {original_idx + 1}")
            else:
                # FAST MODE: Skip LLM synthesis, use first valid answer (primary prediction)
                logger.info("⚡ Skipping combined final answer generation (generate_combined_final_answer=False)")
                valid_answers = [a for a in processed_answers if a and a != NO_ANSWER_PLACEHOLDER]
                if valid_answers:
                    # Use first valid answer (corresponds to primary prediction)
                    question_result["final_answer"] = valid_answers[0]
                    logger.info(f"Using primary answer as final answer for question {original_idx + 1}")
                else:
                    question_result["final_answer"] = NO_VALID_ANSWERS_MESSAGE
                    logger.info(f"No valid answers available for question {original_idx + 1}")

            # Add the question result to the overall results
            uncertainty_results.append(question_result)

            logger.info(f"Uncertainty analysis completed for question {original_idx + 1}")

        except Exception as e:
            # Get detailed traceback
            tb_str = traceback.format_exc()
            logger.error(f"Error in uncertainty analysis for question {original_idx + 1}: {str(e)}\n{tb_str}")

            uncertainty_results.append({
                "original_question": original_question,
                "error": str(e),
                "timestamp": timestamp,
                "question_index": original_idx + 1
            })

    return uncertainty_results


async def generate_final_answer(
        question: str,
        answers: List[str],
        interpretations: List[str],
        assistant: Optional[Any] = None,
        assistant_type: str = "claude",
        model_name: str = "us.anthropic.claude-opus-4-6-v1",
) -> str:
    """
    Generate a final answer from multiple answers to the same epidemiological question.

    Args:
        question: The original question
        answers: List of answers from different interpretations
        interpretations: List of question interpretations corresponding to each answer
        assistant: LLM assistant (if None, one will be created)
        assistant_type: Type of assistant to create if none provided
        model_name: Model name to use when creating assistant

    Returns:
        str: Final consolidated answer
    """
    # Create assistant if not provided
    if assistant is None:
        assistant = create_assistant(
            assistant_type=assistant_type, model_name=model_name
        )

    # Filter out empty or malformed answers and their corresponding interpretations
    valid_pairs = [(a, i) for a, i in zip(answers, interpretations)
                   if a and len(a.strip()) > 20 and not a.endswith('\n')]

    # If no valid answers, return an appropriate message
    if not valid_pairs:
        return "Unable to generate a valid answer to this question based on available data."

    # If only one valid answer, return it directly
    if len(valid_pairs) == 1:
        return valid_pairs[0][0]

    # Group answers by interpretation (preserves insertion order)
    from collections import OrderedDict
    groups: OrderedDict[str, list] = OrderedDict()
    for answer, interp in valid_pairs:
        groups.setdefault(interp, []).append(answer)

    interp_list = list(groups.items())
    primary_interpretation, primary_answers = interp_list[0]
    primary_answer = primary_answers[0]

    # Build alternative interpretations text — all answers grouped by interpretation
    if len(interp_list) > 1:
        alt_parts = []
        for i, (interp, answers_group) in enumerate(interp_list[1:], start=1):
            alt_parts.append(f"[Interpretation {i}]: {interp}")
            if len(answers_group) == 1:
                alt_parts.append(f"[Answer]: {answers_group[0]}\n")
            else:
                for j, ans in enumerate(answers_group, 1):
                    alt_parts.append(f"  [SQL Variation {j}]: {ans}")
                alt_parts.append("")
        alt_interpretations_text = "\n".join(alt_parts)
    else:
        alt_interpretations_text = "None"

    # Prepare the prompt for the LLM to combine answers
    prompt = f"""
    You are an expert medical data analyst tasked with combining multiple answers to the same epidemiological question.
    These answers were generated by querying a database with potentially different SQL queries which arose from 
    the different interpretations. Your goal is to produce a single, accurate, and well-formatted answer that preserves ALL numerical information,
    and compares the results across interpretations when they differ.

    Original Question: {question}

    PRIMARY INTERPRETATION (most likely intended meaning):
    {primary_interpretation}

    PRIMARY ANSWER:
    {primary_answer}

    ALTERNATIVE INTERPRETATIONS AND ANSWERS:
    {alt_interpretations_text}

    Please provide a single, clear, and comprehensive answer that:
    1. Leads with the PRIMARY ANSWER as the main response to the question
    2. Preserves ALL numerical entities and information from ALL answers
    3. Clearly indicates which numbers correspond to which specific interpretation when they differ
    4. Is properly formatted as a complete response suitable for healthcare professionals
    5. Includes all relevant counts, percentages, and statistical information
    6. Is organized in a logical structure (you may use bullet points or tables if appropriate)
    7. When referring to the interpretations, report the actual interpretation. Never include referencing
    "Primary", "Alternative 1", "Alternative 2" etc without reporting the interpretation directly
    8. If alternative interpretations yield different results, present them as additional context after the primary answer,
    highlighting how each interpretation differs (e.g., first occurrence vs all occurrences)

    Do not omit any numerical information from any of the answers, as this data is important 
    for the user's understanding of the question. If different interpretations led to different numerical results, 
    explain this clearly in your answer. 
    If all results are the same, do not mention that there were different interpretations.
    All analysis are performed via SQL on the same database, so smaller number of patients arise from different question specifications.
    Thus, do not include expressions like "This analysis was performed on a smaller dataset." or "One study showed.."

    Do not include negative words like discrepancies etc which might make the user
    puzzled and confused.

    Your answer should be concise but complete, and should not mention that it was derived from multiple answers.
    Do not include the text "Final answer" in your response.

    Final answer:
    """

    # Get the combined answer from the LLM
    try:
        final_answer = await assistant.get_response(prompt=prompt, quiet=True)
        return final_answer

    except Exception as e:
        logger.error(f"Error generating final answer: {str(e)}")
        logger.error("Returning the longest answer...")
        # valid_answers never existed; the surviving answers live in valid_pairs.
        # As written this raised NameError, so an LLM failure here lost the answer
        # entirely instead of degrading to the longest one.
        return max((answer for answer, _ in valid_pairs), key=len)
