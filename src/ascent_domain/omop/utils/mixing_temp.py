import logging
import math
from typing import Optional

from ascent_domain.omop.models.inference.rag import RAGProcessor
from ascent_domain.omop.models.inference.rwd_request import RWDRequest

logger = logging.getLogger(__name__)


def prepare_rwd_request(user_input, query_filled_pred, query_template_pred, **kwargs):
    rwd_request = RWDRequest(
        user_input=user_input, query_filled=query_filled_pred, query_template=query_template_pred
    )
    logger.debug("RWDRequest created")

    # Set additional attributes
    for key, value in kwargs.items():
        logger.debug(f"Setting attribute: {key}")
        setattr(rwd_request, key, value)

    # Set RAG-specific attributes
    if kwargs.get("rag"):
        logger.debug("Setting RAG-specific attributes")
        rwd_request.rag = kwargs.get("rag")
        rwd_request.query_df_retrieved_rag = kwargs.get("df_recs_list_out")
        rwd_request.rag_top_similarity = kwargs["df_recs_list_out"]["Score"].max() if kwargs.get("df_recs_list_out") is not None else 0.0

    logger.debug("Finished prepare_rwd_request")
    return rwd_request


async def get_text_sql_template_for_rag(
        question_masked: str, rag: RAGProcessor, use_rag: Optional[bool] = True, rag_random: Optional[bool] = False, drop_first: Optional[bool] = False,
        question_type: Optional[str] = None
):
    # Zero-shot mode: skip RAG retrieval and return empty template
    if not use_rag:
        logger.info("🎯 ZERO-SHOT MODE: Skipping RAG retrieval (no examples provided)")
        import pandas as pd
        return "", pd.DataFrame()  # Empty template and empty dataframe

    # Standard RAG mode: retrieve examples
    params_dict = {
        "question_masked": question_masked,
        "top_k_screening": rag.config.top_k_screening,
        "top_k_prompt": rag.config.top_k_prompt,
        "sim_threshold": rag.config.sim_threshold,
        "question_type": question_type,
    }

    if rag_random is not None:
        params_dict["rag_random"] = rag_random
    if drop_first is not None:
        params_dict["drop_first"] = drop_first

    (text_sql_template, df_recs_list_out) = await rag.querylib.text_sql_template_for_rag(**params_dict)
    return text_sql_template, df_recs_list_out


def add_messages_to_assistant(messages: list, assistant=None):
    for message in messages:
        role = "user"
        assistant.add_message(role=role, message=message)


def replace_nan(obj):
    if isinstance(obj, float) and math.isnan(obj):
        return None
    elif isinstance(obj, list):
        return [replace_nan(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: replace_nan(v) for k, v in obj.items()}
    return obj


def prepare_prediction(user_input: str, prompt: str):
    new_prompt = prompt.replace("${question}", user_input.replace("'", "\\'"))
    return new_prompt

