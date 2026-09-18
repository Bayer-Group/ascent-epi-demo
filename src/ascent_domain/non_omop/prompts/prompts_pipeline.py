from typing import Dict, List

from ascent_domain.non_omop.schemas.cohort import CohortMetadata
from ascent_domain.non_omop.schemas.question_sanity_check import MedicalEntities
from ascent_domain.non_omop.schemas.question_to_analysis import QuestionToAnalysis


def get_prompt_sanity_check(question: str) -> str:
    """
    Generate a prompt for analyzing and validating a healthcare query.

    This function creates a prompt that instructs an LLM to perform a comprehensive
    analysis of a healthcare question through four stages: query sanity check,
    intent classification, medical entity recognition, and temporal analysis.

    Args:
        question (str): The healthcare question to be analyzed

    Returns:
        str: A formatted prompt for the LLM to use in analyzing the healthcare query
    """
    prompt_sanity_check = f"""
            # Healthcare Text-to-SQL Intent Analysis System Prompt

        You are a specialized healthcare query analyzer designed to prepare natural language questions for conversion to SQL. Your task is to analyze healthcare queries methodically through four critical analysis stages before any SQL is generated.

        ## System Capabilities

        For each user question about healthcare data, perform these four analyses:

        1. **Query Sanity Check** - Validate if the query is well-formed and answerable
        2. **Intent Classification** - Identify the primary analytical purpose
        3. **Medical Entity Recognition** - Extract and classify clinical concepts
        4. **Temporal Analysis** - Identify time-related expressions and requirements

        Provide your analysis in a structured JSON format that downstream SQL generation systems can use.

        ## Analysis Components

        ### 1. Query Sanity Check

        Evaluate if the query:
        - Is related to healthcare data analysis
        - Contains sufficient information to be answered
        - Has a clear analytical objective
        - Avoids ambiguous terms or contradictions
        - Is appropriate for SQL-based analysis

        If the query fails validation, provide specific feedback on what's missing or problematic.

        ### 2. Intent Classification

        Classify the query into one or more of these healthcare analytical intents:

        - **Descriptive Analytics**
          - Patient Demographics
          - Condition Prevalence
          - Resource Utilization
          - Medication Usage
          - Event Frequency
          - Clinical Metrics

        - **Comparative Analytics**
          - Treatment Comparison
          - Demographic Comparison
          - Temporal Comparison
          - Provider Comparison
          - Facility Comparison

        - **Predictive/Risk Analytics**
          - Outcome Prediction
          - Risk Factor Identification
          - Readmission Risk
          - Complication Likelihood
          - Resource Forecasting

        - **Operational Analytics**
          - Workflow Efficiency
          - Quality Metrics
          - Compliance Verification
          - Resource Allocation
          - Staff Productivity

        - **Financial Analytics**
          - Cost Analysis
          - Reimbursement Patterns
          - Claim Denial Analysis
          - Revenue Cycle Metrics
          - Charge Capture

        ### 3. Medical Entity Recognition

        Identify all healthcare entities mentioned in the query:

        - **Clinical Conditions**
          - Disease names and symptoms
          - Disease severity indicators

        - **Medications** — classify each into ONE of two buckets:

          **Specific drug** (`Drug`): a single chemical ingredient or brand name.
            Examples: metformin, atorvastatin, lisinopril, warfarin, apixaban,
                      Glucophage, Lipitor, Eliquis.

          **Drug class** (`Drug_classes`): a category, mechanism, or therapeutic group
            covering multiple ingredients.
            Examples: statins, ACE inhibitors, ARBs, anticoagulants, DOACs,
                      beta blockers, NSAIDs, SSRIs, biguanides, antihypertensives.

          **Decision rule** — classify as `Drug_classes` if ANY of these hold:
            - the term is plural ("statins", "anticoagulants")
            - the term ends in "inhibitors", "blockers", "antagonists", "agonists",
              "diuretics", "agents"
            - the term names a mechanism or therapeutic indication rather than a molecule
            - you would need to enumerate >1 ingredient to fully cover it

          Otherwise classify as `Drug`.

        - **Procedures**
          - Surgical and non-surgical procedures (CABG, hip replacement, dialysis, colonoscopy)

        - **Demographics**
          - Age groups or specific ages
          - Gender/sex specifications
          - Ethnic/racial groups if mentioned
          - Insurance or payment types

        - **Providers/Facilities**
          - Provider types (physician, nurse, etc.)
          - Facility types (inpatient, outpatient, ICU)
          - Specialties mentioned

        - **Clinical Measurements**
          - Lab tests and values
          - Vital signs
          - Clinical scores (APACHE, SOFA, etc.)
          - Relevant LOINC codes where applicable

        ### 4. Temporal Analysis

        Identify all time-related elements:

        - **Time Points**
          - Specific dates mentioned
          - Relative dates ("last week")
          - Clinical events as time anchors ("after surgery")

        - **Time Periods**
          - Explicit durations ("30 days")
          - Clinical periods ("length of stay")
          - Administrative periods ("fiscal year")

        - **Temporal Relationships**
          - Before/after relationships
          - During relationships
          - Temporal sequences

        - **Frequency Patterns**
          - Recurring events
          - Frequency specifications
          - Interval patterns

        ## Example Analysis

        When analyzing "What is the average length of stay for diabetic patients over 65 who were admitted through the emergency department in the last quarter?", produce a complete analysis covering all four components before any SQL generation would begin.

        Always return your full analysis, even if parts are inconclusive or have low confidence. If the query is completely outside healthcare or unsalvageable, explain why in the sanity check section rather than attempting the remaining analysis.

        Here's the question: {question}
    """

    return prompt_sanity_check


def get_prompt_cohort_builder(analytical_objective: str, original_query: str, medical_entities: MedicalEntities):
    prompt_cohort_builder = f"""
    Your goal is to create a cohort of patients to for a data analysis on a healthcare database the user has the following analytical objective:
    {analytical_objective}
    The user has provided the following question:
    {original_query}.
    The identified medical entities are the following:
    {medical_entities}
    List what are the inclusion, exclusion and index date for the cohort creation. Do not add redundant inclusion or exclusion criteria (eg. exclusion criteria that are the opposite of inclusion.
    Be precise when it come to temporal element like "first" or "last" clinical event.
    And provide the analytical steps necessary for the actual calculation of the answer and possible visualizations to be compelling in delivering the answer. Do not add any further consideration.
    """
    return prompt_cohort_builder


def get_prompt_sql_preparation(
    question_to_analysis: QuestionToAnalysis,
    medical_entities,
    tables_with_medical_coding_ontologies,
    m_schema_str,
    custom_instruction: str = "",
):
    prompt_sql_preparation = f"""
    Your goal is to prepare a data analyst to write SQL queries with Snowflake compatible SQL code. I am going to provide you with 1. Analysis plan 2. Medical entities 3. Database schema.
    
    You need to explain to your colleague given the analysis plan, the medical entities, the database schema:
    1. What are the tables and attributes to select to answer the question. Make sure you select ALL tables from the database schema that are needed to answer the question.
    2. What are the medical entities that require a placeholder for medical coding in the final SQL query.
    - **CRITICAL**: If the user's original question does not contain specific medical codes, you must instruct the SQL writer to use a placeholder in the format `[entity@entity_name@coding_system1,coding_system2,...]` combining ALL required coding systems for the same concept into ONE placeholder (e.g., `[condition@endometriosis@ICD10CM,ICD9CM]`).
    - **DO NOT** instruct the SQL writer to search for codes in lookup tables or make up codes. The placeholder is the only valid method.
    - **DO NOT** create separate placeholders for the same entity — always combine all coding systems into one comma-separated list.
    - List each medical entity that needs a placeholder, its domain (e.g., 'Drug', 'Condition'), and ALL target coding systems as a comma-separated list (e.g., 'NDC' or 'ICD10CM,ICD9CM').
    3. What are the steps to prepare the SQL code to answer the user question.
    
    CRITICAL: When identifying the target coding systems for a placeholder, you must carefully review the tables with medical coding ontologies metadata. If a column contains multiple coding systems (e.g., 'ICD-10, SNOMED'), combine them into ONE placeholder with comma-separated coding systems: `[condition@diabetes@ICD10CM,SNOMED]`. Never split a single concept across multiple placeholders.
    
    Here's the original question:
    {question_to_analysis.original_question}
    
    Here's the analysis plan:
    {question_to_analysis.analytical_steps}
    
    Here's the cohort definition:
    {question_to_analysis.cohort_definition}
    
    Here are the detected medical entities:
    {medical_entities}
    
    Here are the tables with medical coding ontologies:
    {tables_with_medical_coding_ontologies}
    
    Here's the database schema:
    {m_schema_str}
    
    {custom_instruction}
    """

    return prompt_sql_preparation


def get_prompt_sql_execution(sql_cohort_preparation_json: Dict, peek_into_db_for_relevant_tables_columns: Dict, feedback: str | None = None):
    """
    Prepare a prompt for SQL execution.
    """
    prompt_sql_execution = f"""
    Your goal is to write Snowflake SQL code. CRITICAL: You are an expert Snowflake SQL developer. All generated SQL functions and syntax MUST be 100% valid for Snowflake. Do not use functions from other dialects like PostgreSQL or T-SQL.
    
    Write clean clear code with CTEs and comments to explain the steps. 
    
    Snowflake Dialect Notes:
    To create a date from its parts (year, month, day), always use the DATE_FROM_PARTS() function. NEVER use MAKE_DATE().
    
    When using date-related functions like TO_DATE(), ensure the input column is explicitly cast to VARCHAR if its native type is a number (e.g., decimal::varchar).
    
    You will create several sequentially linked CTE SQL - start from CTE for index date and subsequent CTEs referring to the previous one. You will create one CTE for each inclusion and exclusion criteria so that sequentially builds the patients funnel. 
    
    Treat exclusion CTE like inclusions and manage the exclusion at the end with left join person_id null. DO NOT USE complicated SQL expressions like "WHERE EXISTS (SELECT 1.." or EXIST/NOT EXIST 
    
    If the last step is a COUNT, the final SQL should be SELECT COUNT(DISTINCT patient_id).
    If the last step is to create a cohort of patients, the final SQL will return distinct patient_id and index date combining the different CTE SQL.

    For prevalence / proportion / rate questions, the denominator must match the numerator's scope. If the condition is anatomically sex-specific (e.g. endometriosis, prostate cancer), restrict the denominator to that sex. If the question names a sub-population ("among diabetics"), use it as the denominator. Otherwise use all enrolled patients.

     Do not create physical tables temporary tables. Describe what the SQL code does not cover referring to the initial question, if necessary.
    
    I am going to provide you with
    1. Analysis goal and question
    2. Relevant database tables
    3. Sneak peek into the database values for each relevant tables and columns so you can adapt the SQL creation
    4. Medical entities for medical coding
    5. Key considerations for SQL creation
    6. Next steps for SQL creation
    
    Here's the analysis goal and question:
    Goal: {sql_cohort_preparation_json["goal"]}
    
    Question: {sql_cohort_preparation_json["original_question"]}
    
    Here are the relevant tables to build the SQL query:
    {sql_cohort_preparation_json["relevant_tables"]}
    
    Here you can see example of database values for each relevant tables and columns so you can adapt the SQL creation:
    {peek_into_db_for_relevant_tables_columns}
    
    Here you can find medical entities for the medical coding. DO NOT ADD any medical entity in the SQL beyond the ones listed below.
    Whenever medical coding in needed DO NOT make up codes. 
    
    If the user provides the medical codes in the question:
    - use the codes directly in the SQL and DO NOT add any placeholder in the SQL. 
    
    If the user does NOT provide medical codes in the question, you need to add medical coder placeholders.
    Use the format [entity@entity_name@coding_system1,coding_system2,...] combining ALL required coding systems for the same concept into ONE placeholder (e.g. [condition@endometriosis@ICD10CM,ICD9CM]).

    - entity is for example 'condition', 'procedure', 'drug', or 'drug_class'.
    - entity_name is the name of the medical entity (eg. endometriosis, hysterectomy, etc).
    - coding_system is a comma-separated list of ALL coding systems needed for this entity (eg. ICD10CM,ICD9CM or CPT4,ICD10PCS,ICD9Proc).
    
    CRITICAL: If the entity_name is an exact drug name use 'drug', if it's not a specific drug name use 'drug_class'

    CRITICAL: When identifying the target coding systems for a placeholder, carefully review the tables with medical coding ontologies metadata.
    If a column contains multiple coding systems (e.g. 'ICD-10, SNOMED'), combine them into ONE placeholder with comma-separated coding systems.
    Do NOT create separate placeholders for the same entity — always merge all coding systems into one.

    I will later replace them, so just add "IN ([condition@endometriosis@ICD10CM,ICD9CM])" in the WHERE clause.
    {sql_cohort_preparation_json["medical_coding"]}
    
    Follow the key considerations for SQL creation:
    {sql_cohort_preparation_json["key_considerations"]}
    
    Consider those next steps:
    {sql_cohort_preparation_json["next_steps"]}
    """

    if feedback:
        prompt_sql_execution += f"""
        Here is some feedback and more explanation. Please consider it when creating the new SQL:
        {feedback}
        """

    return prompt_sql_execution


def get_prompt_cohort_sql_execution(
    sql_cohort_preparation_json: Dict,
    peek_into_db_for_relevant_tables_columns: Dict,
    inclusion_criteria: List[str],
    exclusion_criteria: List[str],
    index_date: str,
    m_schema_string: str = "",
) -> str:
    """
    Prepare a prompt for generating cohort SQL that returns patient-level data.

    Unlike ``get_prompt_sql_execution`` (which may produce counts or analytics),
    this prompt **always** produces a ``SELECT DISTINCT`` returning a patient
    identifier column aliased as ``SUBJECT_ID`` and an index-date expression
    aliased as ``INDEX_DATE``.
    """
    inclusion_text = "\n".join(f"  - {c}" for c in inclusion_criteria) if inclusion_criteria else "  (none)"
    exclusion_text = "\n".join(f"  - {c}" for c in exclusion_criteria) if exclusion_criteria else "  (none)"
    index_date_text = index_date if index_date else "(not specified)"

    prompt = f"""
    Your goal is to write Snowflake SQL code that creates a patient cohort. CRITICAL: You are an expert Snowflake SQL developer. All generated SQL functions and syntax MUST be 100% valid for Snowflake. Do not use functions from other dialects like PostgreSQL or T-SQL.

    ## Output requirement

    The final SELECT statement MUST return exactly two columns:
      1. The patient identifier column aliased as SUBJECT_ID
      2. The index date expression aliased as INDEX_DATE

    Example final lines:
        SELECT DISTINCT c.patientid AS SUBJECT_ID, MIN(c.recordeddatetime) AS INDEX_DATE
        FROM condition c WHERE c.code_col IN ([condition@ischemic_stroke@ICD10CM,ICD9CM])
        GROUP BY c.patientid

    Do NOT return counts, aggregates, or any other analytical output.
    Do NOT create physical or temporary tables.

    ## SQL structure

    Write clean, clear code with CTEs and comments to explain the steps.

    - Start with a CTE that identifies the index date for each patient (MIN of the qualifying event date).
    - Add one CTE per inclusion criterion, each referencing the previous CTE.
    - Add one CTE per exclusion criterion, defining the excluded patient set.
    - Manage exclusions at the end using LEFT JOIN ... WHERE excluded_col IS NULL.
      DO NOT use WHERE EXISTS / NOT EXISTS or correlated subqueries.
    - The final SELECT must be SELECT DISTINCT <patient_col> AS SUBJECT_ID, <index_date_col> AS INDEX_DATE.

    ## Snowflake dialect notes

    To create a date from its parts use DATE_FROM_PARTS(). NEVER use MAKE_DATE().
    When using TO_DATE(), cast numeric inputs to VARCHAR first (e.g. col::VARCHAR).
    Date arithmetic: DATEADD('day', N, date_col) or date_col + N.

    ## Schema constraint

    The authoritative source of truth for which tables and columns exist is:
    (1) the full database schema below, and (2) the sample data values section further below.
    ONLY reference tables and columns that appear in one of these two sources.
    If a table does not appear in either source it does NOT exist — do not join to it.
    Use the EXACT column names shown. Never invent or rename columns.

    ## Full database schema

    {m_schema_string if m_schema_string else "(schema not available — rely on relevant tables below)"}

    ## Patient identifier

    Before writing any SQL, identify the patient identifier column from the schema above.
    Common names: patientid, patient_id, personid, member_id, enrollee_id, pat_id.
    Use the EXACT column name from the schema.

    ## Cohort definition

    Index date: {index_date_text}

    Inclusion criteria:
{inclusion_text}

    Exclusion criteria:
{exclusion_text}

    ## Analysis goal and question

    Goal: {sql_cohort_preparation_json["goal"]}
    Question: {sql_cohort_preparation_json["original_question"]}

    ## Relevant tables

    Cross-check each table against the full schema and sample data above before using it.
    {sql_cohort_preparation_json["relevant_tables"]}

    ## Sample data values for relevant tables

    {peek_into_db_for_relevant_tables_columns}

    ## Medical coding

    DO NOT add any medical entity beyond the ones listed below. DO NOT make up codes.
    DO NOT join to lookup/vocabulary tables that are not in the schema or sample data.

    If the user provides medical codes in the question, use them directly without placeholders.
    Otherwise add ONE placeholder per clinical concept in the format [entity@entity_name@coding_system1,coding_system2,...].

    - entity is for example 'condition', 'procedure', 'drug', 'drug_class', etc.
    - entity_name is the name of the medical entity (e.g. ischemic_stroke, alteplase).
    - coding_system is a comma-separated list of ALL coding systems needed (e.g. ICD10CM,ICD9CM).
    
    CRITICAL - if the entity_name is an exact drug name use 'drug', if it's not a specific drug name use 'drug_class'

    Combine all coding systems for the same concept into ONE placeholder. Never create separate placeholders for the same entity.
    Review the tables with medical coding ontologies to pick the correct coding systems.

    CRITICAL — placeholder syntax: always use `IN ([placeholder])` with parentheses around the
    placeholder. Example: `WHERE diag IN ([condition@ischemic_stroke@ICD10CM,ICD9CM])`.
    The placeholder will be replaced with a comma-separated list of quoted codes, so the
    parentheses are required to form a valid SQL IN clause.

    CRITICAL — placeholder column type: placeholders will be replaced with alphanumeric text codes
    (e.g. 'I63.9', 'J44.1'). Place them in WHERE clauses comparing against a VARCHAR / TEXT column,
    NOT a DECIMAL / NUMBER column. If the schema only exposes a numeric ID column and a mapping table
    is available in the schema or sample data, join through it. If no mapping table is visible,
    add a SQL comment noting the limitation and omit or approximate the filter.

    {sql_cohort_preparation_json["medical_coding"]}

    ## Key considerations

    {sql_cohort_preparation_json["key_considerations"]}

    {sql_cohort_preparation_json["next_steps"]}
    """
    return prompt


def get_prompt_sql_execution_with_wrong_placeholders(
    sql_cohort_preparation_json: Dict,
    wrong_placeholders: List[str],
    current_sql_with_placeholders: str,
    available_placeholders: List[str],
    cohort_metadata: CohortMetadata | None = None,
):
    """
    Prepare a prompt for SQL execution.
    """
    prompt_sql_execution = f"""
    You are an expert Snowflake SQL developer. All generated SQL functions and syntax MUST be 100% valid for Snowflake. Do not use functions from other dialects like PostgreSQL or T-SQL.

    Your goal to edit the Snowflake SQL code. CRITICAL: The query has placeholders like [entity@entity_name@coding_system1,coding_system2,...]. You will be provided with the wrong placeholders which you should remove.

    Write clean clear code with CTEs and comments to explain the steps. 

    Snowflake Dialect Notes:
    To create a date from its parts (year, month, day), always use the DATE_FROM_PARTS() function. NEVER use MAKE_DATE().

    When using date-related functions like TO_DATE(), ensure the input column is explicitly cast to VARCHAR if its native type is a number (e.g., decimal::varchar).

    You will create several sequentially linked CTE SQL - start from CTE for index date and subsequent CTEs referring to the previous one. You will create one CTE for each inclusion and exclusion criteria so that sequentially builds the patients funnel. 

    Treat exclusion CTE like inclusions and manage the exclusion at the end with left join person_id null. DO NOT USE complicated SQL expressions like "WHERE EXISTS (SELECT 1.." or EXIST/NOT EXIST 

    If the last step is a COUNT, the final SQL should be SELECT COUNT(DISTINCT patient_id).
    If the last step is to create a cohort of patients, the final SQL will return distinct patient_id and index date combining the different CTE SQL.

    Do not create physical tables temporary tables. Describe what the SQL code does not cover referring to the initial question, if necessary.

    I am going to provide you with
    1. Current Snowflake SQL code which you should edit
    2. The placeholders that are wrong and should be either removed or replaced
    3. Analysis goal and question
    4. Relevant database tables

    Here's the analysis goal and question:
    Goal: {sql_cohort_preparation_json["goal"]}

    Question: {sql_cohort_preparation_json["original_question"]}

    Here's the current SQL code which you should edit:
    {current_sql_with_placeholders}

    Here are the relevant tables to build the SQL query:
    {sql_cohort_preparation_json["relevant_tables"]}

    The following placeholders are wrong: {wrong_placeholders}:
    You need to remove them, by keeping the snowflake SQL syntax 100%. Make sure to remove all the placeholder which are wrong, but never remove or change those which are not wrong.

    You should return the Snowflake SQL which contains updated script.
    """

    if available_placeholders:
        prompt_sql_execution += f"""
        The following placeholders are available: {available_placeholders}
        Consider using them as replacement for wrong placeholders, so that we do not completely remove the wrong ones.
"""

    if cohort_metadata:
        prompt_sql_execution += cohort_prompt_addon(cohort_metadata=cohort_metadata)

    return prompt_sql_execution


def get_prompt_handle_invalid_sql(query: str, error: str, m_schema: str) -> str:
    """
    Generate a prompt for healing an invalid SQL query using LLM.

    This function creates a prompt that includes the original query, the error message,
    and the database schema to help the LLM understand and fix the SQL query.
    The prompt includes specific guidance about Snowflake SQL syntax and date handling.

    Args:
        query (str): The invalid SQL query that needs to be fixed
        error (str): The error message returned by the database
        m_schema (str): The database schema metadata to help with query correction

    Returns:
        str: A formatted prompt for the LLM to use in healing the SQL query
    """
    prompt = f"""Generated SQL query: \n {query} \n
                        Here is the error returned: ***{error}***.\n
                        Here's the database schema: {m_schema}\n
                        Analyze the error. Review the generated SQL. Fix and rewrite the SQL query.\n
                        Please keep in mind that you are querying a Snowflake database. Avoid using functions that are not supported by Snowflake.\n
                        For dates in Snowflake:
                        - Use 'YYYY-MM-DD' format for date literals (e.g., '2000-01-01'). Snowflake automatically converts this format to dates.
                        - Use TO_DATE() function only when:
                          * Working with dates in other formats (e.g., TO_DATE('01/15/2024', 'MM/DD/YYYY'))
                          * Converting string columns to dates
                          * Making data type conversion explicitly clear in code
                        Please only return the corrected SQL query. Do not return any extraneous data or information.\n
                        Please make sure that corrected query includes all necessary GROUP BY expressions if aggregate functions are used.\n
                    """

    return prompt


def cohort_prompt_addon(cohort_metadata: CohortMetadata) -> str:
    """
    prompt building strategy:
    1. only join based on `person_id` / subject_id.
    2. cohort_start_date AND cohort_end_date
    2. cohort_start_date <= DT
    3. DT <= cohort_end_date
    4. variables:
        4.1 index_date
        4.2 other variables

    assumed:
    1. cohort dates are not missing
    """
    cols_ordered = []
    variables = []
    rules_text = []
    rules_processing = []
    abbreviations = [
        "COHORT_TABLE – cohort table reference.",
        "TABLE – any table from CDM database which has PERSON_ID column, e.g. PERSON table.",
    ]

    ######################################################################################

    if "SUBJECT_ID" in cohort_metadata.columns:
        cols_ordered.append("SUBJECT_ID")

    if "INDEX_DATE" in cohort_metadata.columns:
        cols_ordered.append("INDEX_DATE")
        rules_text.append(
            (
                "COHORT_TABLE has INDEX_DATE column which contains reference start date for a participant.\n"
                "If derivation requires reference/index date then INDEX_DATE should be used in downstream derivations.\n"
                "Examples include variables definitions, like age which uses index date as a reference timepoint to derive age.\n"
            )
        )
        variables.append("INDEX_DATE")

    # abbreviations are same for all cases
    if ("COHORT_START_DATE" in cohort_metadata.columns) or ("COHORT_END_DATE" in cohort_metadata.columns):
        abbreviations.extend(
            [
                "TABLE_WITH_DATES – CDM tables which have date column (named `*_DATE`), e.g. DEATH, OBSERVATION, NOTE, PROCEDURE_OCCURRENCE, MEASUREMENT.",
                (
                    "TABLE_WITH_DATE_INTERVALS – CDM tables which have date/time interval columns (named as `*_START_DATE` and `*_END_DATE`), "
                    "e.g. CONDITION_OCCURRENCE, DRUG_EXPOSURE, VISIT_OCCURRENCE, VISIT_DETAIL, DEVICE_EXPOSURE."
                ),
                "ascent.ascent_cohorts.are_intervals_intersect(a_start DATE, a_end DATE, b_start DATE, b_end DATE) – Snowflake UDF function which compares 2 intervals and return TRUE if intervals intersect, FALSE otherwise.",
            ]
        )

    if ("COHORT_START_DATE" in cohort_metadata.columns) and ("COHORT_END_DATE" in cohort_metadata.columns):
        cols_ordered.append("COHORT_START_DATE")
        cols_ordered.append("COHORT_END_DATE")
        rules_text.append(
            "COHORT_TABLE has information on cohort start date and cohort end date in respective columns COHORT_START_DATE and COHORT_END_DATE."
        )
        rules_processing.append(
            (
                "When joining tables containing date columns with COHORT_TABLE – use the following additional condition in WHERE clause to select records based on date intervals:\n"
                "\n"
                "```sql\n"
                "JOIN question_cohort AS cohort\n"
                "<...>\n"
                "WHERE ascent.ascent_cohorts.are_intervals_intersect(<table-interval>, cohort.COHORT_START_DATE, cohort.COHORT_END_DATE)\n"
                "```,\n"
                "\n"
                "where <table-interval> is:\n"
                "- `<table>_start_date, <table>_end_date` for TABLE_WITH_DATE_INTERVALS,\n"
                "- `<table>_date, <table>_date` for TABLE_WITH_DATES."
                "\n"
            )
        )
    elif "COHORT_START_DATE" in cohort_metadata.columns:
        cols_ordered.append("COHORT_START_DATE")
        rules_text.append("COHORT_TABLE has information on cohort start date in column COHORT_START_DATE.")
        rules_processing.append(
            (
                "When joining tables containing date columns with COHORT_TABLE – use the following additional condition in WHERE clause to select records based on date intervals:\n"
                "\n"
                "```sql\n"
                "JOIN question_cohort AS cohort\n"
                "<...>\n"
                "WHERE ascent.ascent_cohorts.are_intervals_intersect(<table-interval>, cohort.COHORT_START_DATE, NULL)\n"
                "```,\n"
                "\n"
                "where <table-interval> is:\n"
                "- `<table>_start_date, <table>_end_date` for TABLE_WITH_DATE_INTERVALS,\n"
                "- `<table>_date, <table>_date` for TABLE_WITH_DATES."
                "\n"
            )
        )
    elif "COHORT_END_DATE" in cohort_metadata.columns:
        cols_ordered.append("COHORT_END_DATE")
        rules_text.append("COHORT_TABLE has information on cohort end date in column COHORT_END_DATE.")
        rules_processing.append(
            (
                "When joining tables containing date columns with COHORT_TABLE – use the following additional condition in WHERE clause to select records based on date intervals:\n"
                "\n"
                "```sql\n"
                "JOIN question_cohort AS cohort\n"
                "<...>\n"
                "WHERE ascent.ascent_cohorts.are_intervals_intersect(<table-interval>, NULL, cohort.COHORT_END_DATE)\n"
                "```,\n"
                "\n"
                "where <table-interval> is:\n"
                "- `<table>_start_date, <table>_end_date` for TABLE_WITH_DATE_INTERVALS,\n"
                "- `<table>_date, <table>_date` for TABLE_WITH_DATES."
                "\n"
            )
        )

    # add rest columns
    for col in cohort_metadata.columns:
        if col not in cols_ordered:
            cols_ordered.append(col.upper())
            variables.append(col.upper())

    if len(variables) > 0:
        rules_processing.append(
            (f"Following variables are available as columns in COHORT_TABLE for downstream processing in analysis: {', '.join(variables)}.")
        )

    rules_processing = (
        [
            (
                "In the beginning of the query use the following statement to pre-process (subset) the COHORT_TABLE once:\n"
                "\n"
                "```sql\n"
                "WITH question_cohort AS (\n"
                f"   SELECT {','.join(cols_ordered)}\n"
                f"   FROM {cohort_metadata.snowflake_table_ref}\n"
                ")\n"
                "```.\n"
            ),
            (
                "To join any TABLE with COHORT_TABLE use the following SQL-query syntax:\n"
                "\n"
                "```sql\n"
                "JOIN\n"
                "   question_cohort AS cohort\n"
                "ON TABLE.person_id = cohort.subject_id\n"
                "```.\n"
            ),
        ]
        + rules_processing
        + [
            "COHORT_TABLE should be joined only once, don't re-join already pre-processed tables again with COHORT_TABLE in the downstream processing."
        ]
    )

    analysis_section = "\n".join(rules_text)
    abbreviations_section = format_list_as_numbered(abbreviations)
    cohort_processing_rules_section = format_list_as_numbered(rules_processing)

    prompt_addon = f"""
# Population restrictions / Analysis scope

You are requested to limit the analysis for the data consisting of cohort participants only.
This is done by subsetting original database to the population of cohort participants which is defined in a separate Cohort table ({cohort_metadata.table_name}).
Whenever you need to use a table from CDM containing a participant column (PERSON_ID), you have to join this table with Cohort table to get the cohort participants subset.
Tables without PERSON_ID shouldn't be joined with cohort table, e.g. CONCEPT table.
{analysis_section}

Following definitions are introduced for data-processing rules section:

{abbreviations_section}

## Data-processing rules

The following SQL statements and data-processing rules should be followed:

{cohort_processing_rules_section}
"""
    return prompt_addon


def format_list_as_numbered(entries: list[str]) -> str:
    """
    process rules and construct text as numbered item list
    entries = ['rule 1\n line 2', 'rule 2', 'rule 3']

    1. rule 1
       line 2
    2. rule 2
    3. rule 3
    """
    items = []
    from textwrap import indent

    for idx, item in enumerate(entries):
        idx_text = f"{idx + 1}. "
        prepend_len = len(idx_text)
        item_text = idx_text + indent(item, " " * prepend_len).lstrip()
        items.append(item_text)
    return "\n".join(items)
