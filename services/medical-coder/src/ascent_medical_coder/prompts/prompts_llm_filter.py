# ruff: noqa: RUF001  -- prompt text is copied verbatim (long lines / en-dash characters preserved)


async def get_prompt(medical_concept, context, include_reasons=False, custom_instructions=None):
    prompt = """
    You are an experienced medical coder whose main responsibility
    is to translate plain language descriptions of medical concepts
    into a list of possible codes corresponding to the medical entity (condition, procedure, etc.).

    Your response should be a JSON format with the following structure:

    {
        "include" : ["List[int] -- index of the codes to include"],
        "exclude": ["List[int] -- index of the codes codes to exclude"],
    """

    if include_reasons:
        prompt += """
        "reasons_include": ["List[str] -- reasons for including the codes always include the CONCEPT_CODE with each reason"],
        "reasons_exclude": ["List[str] -- reasons for excluding the codes always include the CONCEPT_CODE with each reason"],
        """

    prompt += """
    }

    Return only the index of the codes. Do not return the code or the description of the code.

    <example>
    For instance, if the condition is broken arm and the conditions
    are

    1 Broken left arm
    2 Broken left leg
    3 Broken right arm
    4 Broken right leg
    5 Cancer

    Then return:

    "include": [1, 3],
    "exclude": [2, 4, 5]
    """

    if include_reasons:
        prompt += """

    For describing reasons use the concept code or name, not the index. Describe more professionally and elaborate why the medical code is included or excluded.
    "reasons_include": ["concept_code_placeholder: broken left arm implies broken arm", "broken right arm implies broken arm"],
    "reasons_exclude": ["concept_code_placeholder: broken left leg is not a broken arm", "broken right leg is not a broken arm", "cancer is not a broken arm"]


    For each included or excluded code, provide a reason in the format "CONCEPT_CODE: reason", if CONCEPT_CODE is missing put N/A and mention the name.
        """

    prompt += """
    </example>

    ## DEFAULT FILTERING RULES
    Apply these universal exclusions unless the queried concept itself names them:

    1. EXCLUDE family-history codes (ICD-10 Z80–Z84 and equivalents) — these
       describe a relative's diagnosis, not the patient's.
    2. EXCLUDE external-cause / injury codes (ICD-10 V/W/X/Y chapters, ICD-9
       E-codes and 800–999) — these are mechanisms of injury, not diagnoses.
    3. EXCLUDE assessment-scale / score / questionnaire codes (e.g. NIHSS
       stroke scores, PHQ-9 depression questionnaires) when the query asks
       for a diagnosis — a score is not a diagnosis.
       NOTE: this rule does NOT apply to ICD-10 Z12.* "encounter for screening"
       codes. Those mark a patient encounter with a screening reason and are
       legitimate signals for screening, survivorship, and prevalent-disease
       cohorts. Leave Z12.* decisions to the per-call custom_instructions.

    Custom instructions below, when present, may add further exclusions or
    re-include codes that the defaults would drop.
    """

    if custom_instructions:
        prompt += f"""
    ## CUSTOM INSTRUCTIONS
    IMPORTANT — You MUST follow these additional instructions from the user when deciding which codes to include or exclude. These OVERRIDE the default filtering rules above where they conflict:
    {custom_instructions}
    """

    prompt += f"""
    Here are the codes to consider:

    {context}

    Return the indices of the codes that imply (include) and do not imply (exclude): {medical_concept}
        """

    return prompt
