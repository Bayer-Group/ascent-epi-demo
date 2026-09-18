match_template = """Given a question, a reference answer and a hypothesis answer, determine if the hypothesis answer is correct.
If the answer contains numerical data, please consider the hypothesis answer correct if it is within 15% of the reference answer.
Do not provide any explanation.
Only return one word as an answer TRUE or FALSE.
Do not return any text before or after TRUE or FALSE.

Use the following format:

Question: Question here
Reference Answer: Reference answer here
Hypothesis Answer: Hypothesis answer here
Hypothesis Answer Correct: TRUE or FALSE

Question: {question}
Reference Answer: {reference_answer}
Hypothesis Answer: {hypothesis_answer}
Hypothesis Answer Correct: """

match_template_json = """Given a question, a reference answer and a hypothesis answer, determine if the hypothesis answer is correct.
If the answer contains numerical data, please consider the hypothesis answer correct if it is within 15% of the reference answer.
Do not provide any explanation.
Output result in JSON format containing two fields:
* explanation of your decision;
* one word answer true or false.

For the given inputs:

Question: {question}
Reference Answer: {reference_answer}
Hypothesis Answer: {hypothesis_answer}

Return JSON output as follows:

{{
    "explanation": "Reasoning for obtaining true or false result",
    "correct": "true or false"
}}
"""


entity_masking = """Given an input text, your task is to substitute the entities that fit into the following categories with their corresponding entity type labels:

CONDITION: This represents a clinical diagnosis or symptom documented in a patient's medical history.
MEASUREMENT: This includes various clinical tests, assessments, or instruments.
PROCEDURE: This refers to any intervention, surgical or non-surgical, that is performed on the patient for diagnostic or treatment purposes.
DRUG: This refers to any therapeutic or prophylactic substance prescribed to a patient, including prescription medications, over-the-counter drugs, and vaccines.
CODE: This refers to standardized medical codes, such as for example G71.038, N17.9, Z95.1, 92960
DRUG_CLASS: This refers to name of group of medications and other compounds that have similar chemical structures, the same mechanism of action, and/or are used to treat the similar diseases.

Please remember to only substitute entities that fall under the five categories: CONDITION, MEASUREMENT, PROCEDURE, DRUG, DRUG_CLASS. Always write entity type labels in capital letters.
If the text does not contain any of the entities specified, it must remain unchanged. 
During your substitution, do not substitute vocabulary names such as ICD-9, ICD10-CM, CPT4. Do not return the Input Text in your answer.
You must return ONLY the masked text, with NO explanations, introductions, or additional descriptive text.

Here are a few examples:

Input text: How many patients younger than 20 suffered from hypertension?
Masked text: How many patients younger than 20 suffered from CONDITION?

Input text: What is the adherence of Eylea?
Masked text: What is the adherence of DRUG?

Input text: How many patients are treated with Edoxaban and have atrial fibrillation in their disease history before initiating edoxaban?"
Masked text: How many patients are treated with DRUG and have CONDITION in their disease history before initiating DRUG?

Input text: What is the distribution of Alanine aminotransferase (ALT) and aspartate aminotransferase (AST)? Breakdowns by age bins <50, 50-55, >=55
Masked text: What is the distribution of MEASUREMENT and MEASUREMENT? Breakdowns by age bins <50, 50-55, >=55

Input text: How many females suffered from hypertension while taking venlafaxine?
Masked text: How many females suffered from CONDITION while taking DRUG?

Input text: Among the patients who had a Coronary Artery Bypass Grafting (CABG) surgery, as indicated by ICD-9-CM procedure codes (36.10 through 36.19) or ICD-10 code Z95.1, what proportion also had an Acute Kidney Injury (AKI) using ICD9 codes (584.0, 584.5, 584.6, 584.7, 584.8, 584.9, 586) and ICD10 code (N17.9)?
Masked text: Among the patients who had a Coronary Artery Bypass Grafting (CABG) surgery, as indicated by ICD-9-CM procedure codes (CODE through CODE) or ICD-10 code CODE, what proportion also had an Acute Kidney Injury (AKI) using ICD9 codes (CODE, CODE, CODE, CODE, CODE, CODE, CODE) and ICD10 code (CODE)?

Input text: What is the proportion of patients taking diuretics or calcium supplements?
Masked text: What is the proportion of patients taking DRUG_CLASS or DRUG_CLASS?

Input text: What is the proportion of patients with diabetes and hypertension who underwent coronary angioplasty with high creatinine level taking aspirin and antihistamines?
Masked text: What is the proportion of patients with CONDITION and CONDITION who underwent PROCEDURE with high MEASUREMENT taking DRUG and DRUG_CLASS?

Input text: {question}
Masked text:
"""

# hardcoded values
#- VISIT_CONCEPT_ID:
#'9202' for 'Outpatient Visit',
#'9201' for 'Inpatient Visit',
#'9203'    for 'Emergency Room Visit',
#'262' for 'Emergency room visit',
#'581478'     for 'Ambulance visit',
#'581458'    for 'Pharmacy Visit',
#'32036'    for 'Laboratory visit',
#'42898160'    for 'Non-hospital institution Visit',
#'581476'    for 'Home visit'
#- VISIT_DETAIL_CONCEPT_ID in VISIT_DETAIL table
#'32037' for 'Intensive care (ICU)'

prompt_gpt = """
# Introduction:
You are a data analyst for a pharmaceutical company. You help colleagues by answering questions about patients and diseases using real-world data like claims and electronic medical records. You are well-versed in the OHDSI world and the OMOP CDM.

Your task is to write SQL queries in the Snowflake dialect. The SQL you write should be syntactically correct.

# Instructions:

1. **Concept IDs:** Use the following IDs for the respective fields:
    - GENDER_CONCEPT_ID: '8507' for male, '8532' for female.
    - ETHNICITY_CONCEPT_ID: '38003563' for 'Hispanic or Latino', '38003564' for 'Not Hispanic or Latino'.
    -CONDITION_STATUS_CONCEPT_ID:
     '32890' for 'Admission diagnosis'
     '32891' for 'Cause of death'
     '32892' for 'Condition to be diagnosed by procedure'
     '32893' for 'Confirmed diagnosis'
     '32894' for 'Contributory cause of death'
     '32895' for 'Death diagnosis'
     '32896' for 'Discharge diagnosis'
     '32897' for 'Immediate cause of death'
     '32898' for 'Postoperative diagnosis'
     '32899' for 'Preliminary diagnosis'
     '32900' for 'Preoperative diagnosis'
     '32901' for 'Primary admission diagnosis'
     '32902' for 'Primary diagnosis'
     '32903' for 'Primary discharge diagnosis'
     '32904' for 'Primary referral diagnosis'


2. **Race Analysis:** For breakdown the analysis by 'Race', select CONCEPT_NAME joining PERSON and CONCEPT table on RACE_CONCEPT_ID = CONCEPT_ID (sql: CONCEPT_ID ON domain_id = 'Race' and standard_concept = 'S'

3. **Entity Extraction:**
    The entity types from different tables should be represented as follows:
    - MEASUREMENT table: [measurement@<name of the measurement>]
    - CONDITION_OCCURRENCE or CONDITION_ERA table: [condition@<name of the condition>]
    - PROCEDURE_OCCURRENCE table: [procedure@<name of the procedure>]
    - DRUG_EXPOSURE or DRUG_ERA table: [drug@<name of the drug>]
    - OBSERVATION table: [observation@<name of the observation>]
    - VISIT_OCCURRENCE table: [visit@<type of visit>]
    - VISIT_DETAIL table specific care settings: [visit@<type of care setting>]
    - When referring to a group of drugs that share a common mechanism of action or therapeutic use, known as a drug class, use the format: [drug_class@<name of the drug_class>]. This distinction is important for accurate data analysis.
    Do NOT extract any other entity beyond these.
    In case a drug is given with its weight or formulation, include the information in the entity, e.g. [drug@venlafaxine 75mg].
    Do not use the drug weight or formulation in other parts of the generated query.

    Please note that some concepts (e.g.,  'glioblastoma') might exist as both a [condition@glioblastoma] and an [observation@glioblastoma] with different concept IDs.
    Do not assume the concept IDs are the same across tables, and use the relevant placeholders when querying each table.

    When using visit types (like inpatient, emergency, outpatient), always use the [visit@<type>] format. Never hardcode visit_concept_ids in queries.
    This is valid for both VISIT_OCCURRENCE and VISIT_DETAIL table.
 
    Examples:
    - Incorrect: vo.visit_concept_id IN ('9201', '9203')
    - Correct: vo.visit_concept_id IN ([visit@Inpatient], [visit@Emergency])
    - Incorrect: vd.visit_detail_concept_id IN ('32037')
    - Correct: vd.visit_detail_concept_id IN ([visit@intensive care (ICU)]') 

4. **Drug class:**
    When querying for a group of drugs known as a drug class, ensure that the drug_concept_id is associated with the class rather than individual drugs. For example:
     ```
     WHERE
        de.drug_concept_id IN ([drug_class@<name of the drug_class>])
    ```
    Note: It is crucial to differentiate between individual drugs and drug classes in queries. Always use the [drug_class@<name of the drug_class>] format for drug classes.

5. **Concept Name:** Whenever you are returning an entity (drug, condition, procedure, measurement) concept id (e.g., drug_concept_id) join the concept table on concept_id to return the corresponding concept name. (e.g., join on concept_id = drug_concept_id).

6. **Geographical Analysis:** Use standard 2 letter codes for state, territory, or regional level analyses. For example, Texas as TX, California as CA, New York as NY.

7. **Tables:** Reminder about some OMOP CDM tables:
    The PROCEDURE_OCCURRENCE table contains records of activities or processes ordered or carried out by a healthcare Provider on the patient with a diagnostic or therapeutic purpose.
    The VISIT_OCCURRENCE table contains information about a patient’s encounters with the health care system.
    The OBSERVATION table captures clinical facts in the context of examination, questioning, or a procedure. Any data that cannot be represented by any other domains, such as social and lifestyle facts, medical history, family history, etc. are recorded here.
    The VISIT_DETAIL table provides additional details about specific encounters within a visit.
    - Type of visit detail (e.g., inpatient vs outpatient)
    - Care site where the encounter happened

8. **Date Filters:** If you need to filter by date, use the following date fields:
    - VISIT_OCCURRENCE: visit_start_date
    - VISIT_DETAIL: visit_detail_start_date
    - CONDITION_OCCURRENCE: condition_start_date
    - DRUG_EXPOSURE: drug_exposure_start_date
    - MEASUREMENT: measurement_date
    - OBSERVATION: observation_date
    - PROCEDURE_OCCURRENCE: procedure_date
    Use fields like visit_end_date, visit_detail_end_date, condition_end_date, drug_exposure_end_date only if you measure the duration of the event for each patient.
    To answer question related to the enrollment of a patient in a specific time frame, check whether the time frame is included in an observation period.
    When asked about first diagnosis or patients diagnosed for the first time, ensure the query checks for the absence of any prior diagnosis in the patient's entire history, not just within the specified time period.
    Use a subquery or NOT EXISTS clause to exclude patients with earlier diagnoses.

10. **Column Naming:** Every column name must start with a character and never with a number. For example, percentile_25 instead of 25th_Percentile. Instead of median, use percentile_50.

11. **Date Format:** When clarifying a date interval in your SQL queries, you are required to utilize the `TO_DATE` function along with the correct format 'YYYY-MM-DD'.
    The `TO_DATE` function is used in SQL to convert strings into dates. Here's an example of how to use it:
    TO_DATE('your_date_string', 'YYYY-MM-DD') Replace 'your_date_string' with the date you're inputting into the query.
    Any deviation from this date format will lead to errors or data misinterpretations, for example:
    TO_DATE('2015-01-01', 'YYYY-MM-DD') instead of '2015-01-01',
    TO_DATE('2020-12-31', 'YYYY-MM-DD') instead of '2020-12-31'.

    Make sure all dates in your SQL queries conform to this style and use the `TO_DATE` function when handling date information.

12. **Patient Count:** Use COUNT(DISTINCT person_id) when counting patients.

13. **Observation period:**  When implementing observation period requirements (e.g., "X days of observation before index date"):
    a. ALWAYS include TWO conditions in the observation period check:
       i. The observation period must START at least X days before the index date:
          op.observation_period_start_date <= DATEADD(day, -X, index_date)
       ii. The observation period must INCLUDE the index date:
          op.observation_period_end_date >= index_date

    b. NEVER use just DATEDIFF to check observation period requirements, as this doesn't ensure continuous enrollment through the index date.
       INCORRECT: DATEDIFF(day, op.observation_period_start_date, index_date) >= X
       CORRECT: op.observation_period_start_date <= DATEADD(day, -X, index_date) AND op.observation_period_end_date >= index_date
    
    c. For clarity, name CTEs that perform this check with terms like "cte_continuous_enrollment" or "cte_observation_check" to emphasize that continuous observation is being verified.

    d. Consider that patients may have multiple observation periods, so join appropriately.

    e. Always implement observation period requirements using this pattern:
    ```sql
    SELECT 
        patient.person_id,
        patient.index_date
    FROM 
        patient_cte patient
    JOIN 
        OBSERVATION_PERIOD op ON patient.person_id = op.person_id
    WHERE 
        op.observation_period_start_date <= DATEADD(day, -[required_days], patient.index_date)
        AND op.observation_period_end_date >= patient.index_date
    ```  
    
14. **Age Calculation:** When calculating a patient's age in relation to an event, such as a visit or condition onset, the age should be computed based on the year of the event in question not the current year.
    Use the year_of_birth field from the PERSON table and subtract it from the year of the event (visit_start_date, visit_detail_start_date, condition_start_date, etc.).
    For instance, if the task is to locate patients older than a certain age who have a certain condition, the age condition in the SQL query should refer to the year of the condition's start, like so: AND (YEAR(co.condition_start_date) - p.year_of_birth) > {desired_age}.

14b. **Age distribution**: When analyzing age distributions:

    1. If no age range is specified in the question, split as follows:
      ```sql
      CASE
        WHEN age < 18 THEN '0-17'
        WHEN age BETWEEN 18 AND 34 THEN '18-34'
        WHEN age BETWEEN 35 AND 49 THEN '35-49'
        WHEN age BETWEEN 50 AND 64 THEN '50-64'
        WHEN age BETWEEN 65 AND 79 THEN '65-79'
        WHEN age >= 80 THEN '80+'
        ELSE 'Unknown'
      END AS age_group
      
    2. If an age range is specified (e.g., women with 18-55), split that range in 6 equal splits.
    3. Ensure the age groups are contiguous and cover the entire relevant age range

14c. If the question is about reproductive age, consider the range 15-49 unless otherwise specified.  

14d. **Follow-up times breakdown**: When analyzing follow-up times or time-to-event distributions, use these standard intervals unless the question specifically requires different groupings:
    - <30 days (immediate/short-term)
    - 30-89 days (1-3 months)
    - 90-179 days (3-6 months)
    - 180-364 days (6-12 months)
    - 1 year or more (long-term)
    Include both categorical counts and statistical measures (minimum, median, mean, maximum, and quartiles) to provide a comprehensive view of the distribution.

15. **CTE Naming:** When creating Common Table Expressions (CTEs), always use a distinctive descriptive name.
    This is crucial to avoid conflicts with existing table names in the OMOP CDM or other temporary objects.
    Always ensure your CTE names are unique, descriptive, and clearly indicate their purpose.
    Never use existing table names from the OMOP CDM as CTE names.

16. **SQL Writing:** To write the SQL, use the following format:
    Question: the input question you must create a SQL for
    Database tables and columns: list all tables that are relevant to the query. If you write a WITH clause in SQL, make sure you will select all attributes needed in the WHERE clause of the main query
    If you need the value of concept_id, don't provide it. Instead, add a squared bracket like [entity@<name of the concept>] and always use the IN SQL operator to prepare for the concept ids list.
    For example: condition_concept_id IN ([disease@hypertension])
    If <name of the concept> in [entity@<name of the concept>] is an acronym, resolve and standardize it (eg. AF to Atrial Fibrillation, AKI to Acute Kidney Injury, LVEF to Left Ventricular Ejection Fraction).
    Do not include unnecessary or incorrect conditions, such as 'WHERE concept_name IN...'.
    Do not include statements like "WHERE c.concept_name = 'hypertension'" or  "WHERE concept.concept_name IN ('hypertension', 'anemia')"
    When using aggregate functions (like COUNT) along with non-aggregated columns, ensure that you either group by the non-aggregated columns or use a subquery to calculate constant values separately. This is particularly important when dealing with cross joins or cartesian products.
    Provide a brief explanation of the logic and structure of your query, including how you handle the index date, any aggregations, and joins.

17. **Query Structure:** General query structure: plan how you want to structure the general query structure (e.g., group by, nesting, multiple joins, set operations, etc.)
    Make sure to structure the query accordingly. "or" means UNION "and" means INNER JOIN of the clusters

18. **SQL Return:** Return the SQL query ONLY within ```sql ``` code block.

19. **Query Checking:** Before returning the Snowflake SQL query, check if it contains all the relevant SQL WHERE clauses with concept_id+[entity@<name of the concept>] you identified.

20. **Additional Guideline for GROUP BY Expressions**:
    When using GROUP BY, every column in the SELECT statement that is not an aggregate must be included in the GROUP BY clause.
    If you are selecting specific columns such as CO.CONDITION_START_DATE, ensure that these columns are listed in the GROUP BY clause to avoid SQL errors.
    Alternatively, use aggregate functions on the selected columns that are not intended to be part of the grouping criteria.
    Verify all relevant `WHERE` clauses with `concept_id IN ([entity@<name of the concept>])` are present.
    **Double-check the logic for calculating the index date and ensure it's consistent throughout the query.**
    **Confirm that all CTEs that select both `person_id` and a date use `GROUP BY person_id` to avoid duplicate patient counts.**
    **Test your query logic with simplified data to ensure it produces the expected results.**

21. **Handling Measurements:** For queries involving measurements, verify whether the measurement possesses a real value. This should be done by adding after the statements for the concept IDs "m.measurement_concept_id IN ([measurement@<name of the measurement>])" the Statement "AND ( m.value_as_number IS NOT NULL OR m.value_as_concept_id IS NOT NULL)".

22. **Distribution of Measurements:** For queries concerning the distribution of measured values, the measured values should be grouped according to the names of the units.

23.**Date relationships:** When dealing with date relationships, always use the `DATEDIFF` function with `DAY` parameter. Please convert all periods to days: 1 week = 7 days, 1 month = 30 days, 1 year = 365 days.

24. **Drug Dosage:** If a question is about the dosage of a particular drug, then the patient should be grouped by the drug name. The column "quantity" of the drug_exposure table should be not used. Here an example: "WITH first_dosage AS (SELECT fd.person_id, c.concept_name as first_dosage FROM first_drug fd JOIN CONCEPT c ON fd.first_drug_concept_id = c.concept_id) SELECT fdn.first_dosage, COUNT(DISTINCT fdn.person_id) as patient_count FROM first_dosage fd"

25. **Events and visit:** If a question refers to an entity that occurs during or is the reason (e.g.admission) for a specific visit, include that the start date of the visit and the start date of the entity match. The following example refers to a condition: "vo.visit_start_date = co.condition_start_date"

26. When dealing with temporal criteria (e.g., "diagnoses between 2020-2023") here is how you must handle index dates:
    a. Unless otherwise specified, the index date should be the first occurrence of the relevant event WITHIN the specified time period, not the first occurrence ever.
    b. When calculating observation periods relative to an index date (e.g., "365 days before index date"), use the index date as defined above.
    c. Do not automatically exclude patients who had relevant events before the specified time period unless explicitly instructed to do so.

27. When the prompt is ambiguous about temporal relationships:
    a. Default to using the first occurrence within the specified time period as the index date.
    b. Do not exclude patients based on events outside the specified time period unless explicitly instructed.
    c. Document your assumptions about temporal relationships in comments within the SQL.
    
28. **Split by Code:** If a question contains a request to split the results by code, the concept code must be specified in the result in addition to the concept name.

29. **Group drugs by ingredient:** If a question contains a requirement to group drugs by ingredient, join the DRUG_STRENGTH table to drug_concept_id and identify the ingredient using the ingredient_concept_id column. Look for the name of the ingredient by joining the concept_id table to get the corresponding concept name. (e.g. join on concept_id = ingredient_concept_id).

30. **Condition Status**: Only use the CONDITION_STATUS_CONCEPT_ID if a question explicitly refers to the condition status.

31. **[IMPORTANT] Avoiding Table Name Conflicts:** When writing SQL queries, especially when using Common Table Expressions (CTEs), never use names that match existing tables in the OMOP CDM. Always descriptive names. 
This is crucial to prevent errors like "invalid Recursive CTE" that can occur when a CTE name conflicts with an existing table name.

32. **Simple Demographic Queries:** For questions that only ask about basic patient demographics (age, gender, race, etc.) without any clinical conditions or events, write direct queries using only the PERSON table and relevant concept tables. Do not include clinical tables like DRUG_EXPOSURE, CONDITION_OCCURRENCE, etc. unless specifically required by the question. 
When calculating current age for demographic queries (with no specific event date): Use: YEAR(CURRENT_DATE()) - p.year_of_birth AS age
Examples are: "How many children < 18 in the database?"

33. **Mortality Reporting:**
When calculating deaths within specific timeframes after an event date (e.g., 30/60/90 days):
a. Do NOT restrict deaths to only those occurring within cohort period boundaries or observation periods.
b. Ony filter for deaths that occur on or after the event date, regardless of cohort end dates or observation period end dates.
c. Avoid using functions like `are_intervals_intersect()` that would limit deaths to the cohort observation period, 
as this leads to underreporting of mortality.
       
# This is the question you need to provide the SQL for:
# Question:
${question}
"""

cohort_creation_prompt = """
Given an epidemiological question, generate structured inclusion/exclusion criteria and index date definition. Follow these rules:

1. Extract key population characteristics from the question (age, conditions, treatments, time periods)
2. For prevalence calculations, include criteria for both numerator and denominator populations
3. For outcome studies, clearly separate baseline characteristics from outcomes
4. For medication adherence, specify measurement period and definition
5. For demographic/distribution analyses, focus on base population definition
6. For temporal trend analyses, specify time windows clearly

7. Format output as:
   Inclusion Criteria:
   - List each required characteristic
   - Include age restrictions
   - Include diagnosis/treatment requirements
   - Include any time period constraints

   Exclusion Criteria:
   - List any explicit or implied exclusions
   - Include standard data quality exclusions if relevant

   Index Date:
   - Define the date that marks the start of follow-up
   - Usually first occurrence of main condition/treatment
   - For prevalence questions, use date of first diagnosis

Do not include additional time constraints that are not explicitly mentioned in the question.
Keep criteria clear, specific and aligned with standard epidemiological study design principles. 
Include only criteria that can be operationalized with typical healthcare data on OMOP CDM. 
Do not include any requirements regarding observation periods or presence in the database (e.g., Active in database during study period) unless explicitely specified in the question.

Only return the inclusion/exclusion criteria and index date text, nothing more.

Examples:

1. Simple count question:
Question: "How many patients > 17 yo have atopic dermatitis"

Output:
Inclusion Criteria:
- Age > 17 years
- At least one diagnosis of atopic dermatitis

Index Date: 
- Date of first atopic dermatitis diagnosis

2. Outcome study question:
Question: "What is the percentage of people who have dysphagia after ischemic stroke"

Output:
Inclusion Criteria:
- Patients with diagnosis of ischemic stroke
- Patients with dysphagia after ischemic stroke

Exclusion Criteria:
- History of dysphagia before stroke

Index Date:
- Date of first ischemic stroke diagnosis


3. Prevalence question:
Question: "Calculate prevalence of atopic dermatitis based on database population > 18 yo"

Output:
Inclusion Criteria:
- Age > 18 years
- At least one diagnosis of atopic dermatitis
- Active in database during study period

Index Date:
- Date of first atopic dermatitis diagnosis

4. Medication adherence question:
Question: "What is the adherence rate for heparin?"

Output:
Inclusion Criteria:
- At least one prescription for heparin

Index Date:
- Date of first heparin prescription

5. Demographic distribution question:
Question: "What is the distribution breakouts per gender and age of persons with ischemic stroke?"
Output:

Inclusion Criteria:
- Diagnosis of ischemic stroke

Index Date:
- Date of first ischemic stroke diagnosis

6. Comorbidity analysis:
Question: "Which are the comorbidities for patients with type 2 diabetes?"
Output:
Inclusion Criteria:
- Diagnosis of type 2 diabetes
Index Date:
- Date of first type 2 diabetes diagnosis

7. Temporal trend analysis:
Question: "What is the annual growth of patients getting electric Cardioversion in acute Afib patients in years 2000-2022?"
Output:
Inclusion Criteria:
- Diagnosis of acute atrial fibrillation
- Underwent electric cardioversion procedure
- Event occurred between 2000-2022
Index Date:
- Date of first cardioversion procedure

Combined condition/medication analysis:
8. Question: "What is the average age of patients with hypertension taking venlafaxine?"
Output:
Inclusion Criteria:
- Diagnosis of hypertension
- At least one prescription for venlafaxine
Index Date:
- Date when both conditions are met (hypertension diagnosis and venlafaxine prescription)

Here is the question: ${question}.
"""

suggest_drug_condition_prompt_t1 = """
You are a data analyst for a pharmaceutical company. You help colleagues by answering questions about patients and diseases using real-world data like claims and electronic medical records.
Here is a question: ${question}. Based on this question, we are extracting a cohort of patients and calculating the so-called Table 1, a table that presents descriptive statistics of baseline characteristics of the study population.
Use your medical knowledge to provide a list of related diseases and active ingredients which are related to the input question.
Include also morbidities that are often found together with the input diseases or are a risk factor, and active ingredients to treat them. Only return active ingredient, and not drugs, or drug classes.
They will be used to calculate their distribution in the patient cohort.
Please only return a JSON object with the following structure: {"diseases": [disease1, disease2, ...] ; "active_ingredients": [activeingredient1, activeingredient2,...]}
"""

question_filter_prompt_with_rationale = """
We have a system that answers epidemiological questions based on data by generating text-to-SQL translation via a large language model (LLM) and retrieval augmented generation (RAG).
Based on our RAG library, these are the questions that we can answer with certainty: {query_library_questions}
We want to filter out questions that cannot be answered based on our RAG library.
In your evaluation, please also take into account the generalization ability of current LLMs: questions that are very similar - yet not exactly the same - from the ones in the RAG library could be answered correctly.
This is the input question: {input_question}

Return a JSON with three keys:
"Rationale": Explain the rational of your decision here
"Answerable" : True or False
"Suggested_questions" : If the question cannot be answered with reasonable certainty based on the query library, return a comma-separated Python-style list of possible questions that could be answered. Return [] otherwise.
"""

question_filter_prompt_no_rationale = """
We have a system that answers epidemiological questions based on data by generating text-to-SQL translation via a large language model (LLM) and retrieval augmented generation (RAG).
Based on our RAG library, these are the questions that we can answer with certainty: {query_library_questions}
We want to filter out questions that cannot be answered based on our RAG library.
In your evaluation, please also take into account the generalization ability of current LLMs: questions that are very similar - yet not exactly the same - from the ones in the RAG library could be answered correctly.
This is the input question: {input_question}

Return
If the question can be answered, return []
If the question cannot be answered with reasonable certainty based on the query library, return a comma-separated Python-style list of possible questions that could be answered.
"""

question_filter_prompt_no_rationale_no_questions = """Return True if the question is about human healthcare and False otherwise.

Question: "How many patients with atopic dermatitis?"
Answer: True

Question: "What is the weather in Berlin?"
Answer: False

Question: "How many dogs are depressed?"
Answer: False

Question: "Plot the top 5 drugs for people having atrial fibrillation"
Answer: True

Question: {input_question}
Answer:
"""

question_filter_suggest_questions = """
We have a system that answers epidemiological questions regarding patient counts, prevalence, proportions, follow-up periods or follow up periods, or top conditions/drug/measurements. Plotting is also in scope.
If an input question is not asking about those, please modify it accordingly. Do not answer the question, and do not ask for additional details.

Here are different Categories:
1. **Questions not related to Patient data:** If a question goes beyond the scope, suggest rephrasing it into a simpler question.
Input: "What is the weather in Berlin?"
Explanation:"This question is not realted to Patient Data"
Suggestion: ""

2. **Questions that are beyond scope:** If a question is out of scope, suggest to reformulate it as a simpler question.
Input: "Is apixaban more effective than warfarin?"
Explanation:"This question currently exceeds the analytical possibilities"
Suggestion: "How many patients are taking apixaban?"

3. **Question that are related to the label of a Drug:** If a question refers to the label of a drug, e.g. for which age group it is approved, rephrase it so that you ask whether there are patients in this age group who use this drug.
Input: "Is rivaroxaban approved for atrial fibrillation for patients over 65?"
Explanation:"This question refers to information on the drug's label that is not included in the patient data. "
Suggestion: "How many patients over 65 are taking rivaroxaban?"

4. **Question with unclear starting date for follow up time:** If no starting point is defined for the calculation of the follow-up time, add this information to the proposal.
Input: "What is the follow up time of Patient that take vericiguat?"
Explanation:"This Question does not define a clear starting point"
Suggestion: "What is the follow-up time for patients after their first prescription of vericiguat?"

5.**Question with Geographical details:** If the question contains a geographical reference, e.g. Germany, USA, Texas, rephrase the question without the geographical reference.
Input:"How many patients in US are diagnosed with adenomyosis?"
Explanation:"If you are interested in patient data for a specific country, please select a suitable database"
Suggestion:"How many patients are diagnosed with adenomyosis?"

5.**Questions with Codes:** If a question contains codes, such as ICD codes or SNOMED codes, rephrase the question and insert the concept name of the code instead of the code itself. No code should be in the suggestion.
Input:"How many patients have the ICD code E85.82 and use the drug diflunisal?"
Explanation:"Instead of using the codes directly, please enter the name of the drug, disease, procedure or measurement. This allows our tool to suggest codes that you may not have thought of and that could be relevant. You can filter the codes at any time afterwards."
Suggestion:"How many patients have the ICD code Wild-type transthyretin-related (ATTR) amyloidosis and use the drug diflunisal?"
Question: {input_question}
Answer: Explanation, Suggestions
"""

drug_class_keep = """
    Given an input text, your task is to check if text include entity that fit into DRUG_CLASS category.
    If found, look for a complete list of active ingredients that are part of DRUG_CLASS and provide after DRUG_CLASS a comma separated list with those those names in brackets.
    If there is several DRUG_CLASS entity mentioned do not join list of ingredients together, keep 'and' and 'or' in the sentence structure.
    If DRUG_CLASS entity is given in a single form look for complete list of its active ingredients anyway.

    DRUG_CLASS: This refers to name of group of medications and other compounds that have similar chemical structures, the same mechanism of action, and/or are used to treat the similar diseases.

    Return only text of modified question, no explanation of procedure. If no DRUG_CLASS detected return unchanged question.

    Here are a few examples:
    Input text: How many patients with depression takes selective serotonin reuptake inhibitors?
    Output text: How many patients with depression takes selective serotonin reuptake inhibitors (fluoxetine, sertraline, paroxetine, citalopram, escitalopram, fluvoxamine, dapoxetine)?

    Input text: What is the proportion of male patients taking anticoagulants or aspirin?
    Output text: What is the proportion of male patients taking anticoagulants (heparin, warfarin, rivaroxaban, dabigatran, apixaban, edoxaban, enoxaparin, fondaparinux) or aspirin?

    Input text: What is the number of female patients above 30 taking anticoagulant and Qlaira?
    Output text: What is the number of female patients above 30 taking anticoagulant (heparin, warfarin, rivaroxaban, dabigatran, apixaban, edoxaban, enoxaparin, fondaparinux) and Qlaira?

    Input text: ${question}
    Output text:
"""

personalized_questions_prompt = """
We have a system that - starting from natural language questions - is accessing real world database of Electronic Health Records and claims, and answers some questions.

I would like to generate questions that can be potentially answered starting from a possibly complex analysis of real-world data (electronic health records and administrative claims).

The user is a {roles} from the therapeutic area {therapeutic_area}.
Their last questions are {last_questions}

[!IMPORTANT] Avoid recommending the exact same questions already asked by the user.
[!IMPORTANT] Return 20 questions based on the last user's question only.
[!IMPORTANT] Return the questions in order of most likely to be able to be answered with claims or EHR data that specifically uses the OMOP common data model.
[!IMPORTANT] Return in the following format: {{"questions": [{{"question": "question1"}},{{"question": "question2"}}, ...]}} and NO additional explanation
"""

claim_extraction_prompt = """You are tasked with analyzing text including multiple inclusion and exclusion criteria for selecting a population of interest. Your objective is to:
1. Extract individual criteria within the text and assign them to the correct type
2. Extract the index date definition
3. Identify any implicit criteria contained within the index date definition and add them as separate inclusion/exclusion criteria as first criteria
e.g., if the index date is "First prescription of rivaroxaban", and no inclusion criteria mentions "rivaroxaban", add an implicit inclusion criterion like: "Patient must have received at least one prescription of rivaroxaban"
4. Resolve any references to "index date" by explicitly incorporating the index date definition. If dates are appearing in the inclusion criteria, please include them in the index date if applicable.
5. If no index date is provided, please infer the index date from the context and provide it in output
6. Please return the inclusion/exclusion criteria and index date in square brackets (python list-style) even if there is only one element.

Present your results in a structured JSON format:
{
    "include": [assertion1, assertion3, ...],
    "exclude": [assertion2, ...],
    "index_date": [definition of index date]
}

# Example 1:
input_text = "Inclusion criteria: women with endometriosis \n Exclusion criteria: previous histerectomy"
output = {
    "include": ["women with endometriosis"],
    "exclude": ["patient must not have had a previous histerectomy before the endometriosis diagnosis"],
    "index_date": ["first diagnosis of endometriosis"]
}

# Example 2:
input_text = "Inclusion criteria: Women having at least two diagnoses of endometriosis (between 2010 and 2019); 
Exclusion criteria: "Women with diagnosis of menopause prior to index date"
output = {
    "include": ["Women having at least two diagnoses of endometriosis (between 2010 and 2019)"],
    "exclude": ["Women with diagnosis of menopause prior to the first diagnosis of endometriosis"],
    "index_date": ["First diagnosis of endometriosis"]
}

# Example 3:          
input_text = "Inclusion criteria: - Observation Period 365 days before index
            - Men have least two medical claims with diagnosis codes for prostate cancer between 2020 to 2023 
            Index date:
            - First diagnosis of prostate cancer
            "
output = {
    "include": ["Patient must have continuous observation period of 365 days before the first diagnosis of prostate cancer occorring between 2020 and 2023",
    "Men must have at least two medical claims with diagnosis codes for prostate cancer between 2020 to 2023"],
    "index_date": ["First diagnosis of prostate cancer between 2020-2023"]
}



Please use double quotes, not single quotes.

For each assertion:
- Make it self-contained and independent from other assertions
- Do not include additional criteria that are not mentioned in the input or split criteria adding gender-related criteria
(e.g., "women with endometriosis" must not be split into two criteria "women" and "diagnosed with endometriosis", but kept together)
- Explicitly state specific entities (drugs, genes, diseases, etc.)
- Replace references to "index date" with the actual definition

Here is the text:
"""

criteria_to_sql_prompt_add_on = """
24. **Query Structure and Optimization:**
    a. Use one Common Table Expression (CTE) for each criterion.
    b. Include all criteria (and all relevant entities) in your query, following the order given in the input.
    c. Do not combine multiple criteria in one CTE, even if they are of the same type.
    d. Use appropriate JOINs between tables (e.g., INNER JOIN, LEFT JOIN) based on the criteria.
    e. Avoid using correlated subqueries in WHERE clauses, as these can lead to performance issues and errors like "Unsupported subquery type cannot be evaluated."
    f. Use CTEs to break down complex queries into manageable parts.
    g. When checking for the existence of records, prefer EXISTS clauses over correlated subqueries.
    h. Use JOINs where appropriate instead of subqueries in the WHERE clause.
    i. For inclusion criteria that require multiple occurrences (e.g., "at least two diagnoses"), use window functions like ROW_NUMBER() or COUNT() in a CTE, then filter in the main query.
    l. Include relevant dates (e.g., condition_start_date, procedure_date) in CTEs along with person_id.
    m. Pre-calculate aggregations (e.g., MIN, MAX, COUNT) within CTEs when possible.
    n. Prefer joining pre-calculated CTEs over multiple joins to large tables like condition_occurrence in the final query.
    o. In every CTE and subquery, always GROUP BY person_id when selecting both person_id and dates (e.g., condition_start_date, visit_start_date) to avoid double counting patients and thus wrong results. Never use DISTINCT in these cases.
        Correct:
        SELECT person_id, MIN(visit_start_date) AS first_visit_date
        FROM visit_occurrence
        GROUP BY person_id

        Incorrect:
        SELECT DISTINCT person_id, visit_start_date
        FROM visit_occurrence

25. **Inclusion and Exclusion Criteria:**
    The returned patients must satisfy ALL inclusion criteria (combined with AND logic), and they must NOT satisfy ANY exclusion criteria.
    a. Always combine all inclusion criteria with AND logic, regardless of their medical meaning.
       Example: "Inclusion: patients with breast cancer; patients with prostate cancer" should still be combined with an AND.
    b. Never use OR logic or UNION for inclusion criteria, even if the conditions seem mutually exclusive (e.g., gender-specific conditions).
    c. For inclusion criteria, use EXISTS when appropriate, especially for complex conditions.
       Example: WHERE EXISTS (SELECT 1 FROM table WHERE conditions)
    d. For exclusion criteria, prefer LEFT JOIN with NULL check over NOT EXISTS when possible.
       Example: LEFT JOIN excluded_condition ec ON patient.id = ec.patient_id WHERE ec.patient_id IS NULL
    e. Use INNER JOIN for inclusion criteria and LEFT JOIN for exclusion criteria in the final query structure.

26. **Query Formatting and Best Practices:**
    a. Do not use LIMIT in your query.
    b. Ensure the final SELECT statement includes a COUNT(DISTINCT person_id) and any other necessary columns or aggregations.
    c. Comment each CTE to explain which criterion it addresses.
    d. For comparisons involving multiple values:
       - Use 'IN' for inclusion: column_name IN (value1, value2, ...)
       - Use 'NOT IN' for exclusion: column_name NOT IN (value1, value2, ...)
       - Do not use '=' or '!=' with multiple values in parentheses.
    e. Always use table aliases when joining tables or referencing columns from multiple tables.
    f. Fully qualify all column names with their table alias, especially in JOIN conditions and WHERE clauses.
       Example: SELECT a.column1, b.column2 FROM table1 a JOIN table2 b ON a.id = b.id WHERE a.date > b.date
    g. Be particularly careful with commonly used column names like 'person_id', 'condition_start_date', etc., always qualifying them with the appropriate table alias.
    h. Do not use table names as CTE names. Always prefix CTE names with a descriptive word or abbreviation to avoid conflicts with existing table names.
    i. Always use the TO_DATE function when specifying dates in queries. This ensures proper date interpretation regardless of Snowflake's session settings. For example:
       TO_DATE('1900-01-01', 'YYYY-MM-DD') instead of '1900-01-01'
       TO_DATE('2020-12-31', 'YYYY-MM-DD') instead of '2020-12-31'

27. **CTE Naming Convention:**
    When creating Common Table Expressions (CTEs), use a distinctive prefix such as 'cte_' followed by a descriptive name.
    This helps avoid conflicts with existing table names in the OMOP CDM or other temporary objects.
    For example, use 'cte_patient_observation_period' instead of 'OBSERVATION_PERIOD'.
    Always ensure your CTE names are unique, descriptive, and clearly indicate their purpose.

28. **Index Date Calculation**:
    a. Calculate the index date using pre-computed dates from CTEs rather than accessing original tables in the final SELECT.
    b. Use COALESCE to handle potential NULL values when determining the index date from multiple conditions.
    c. Calculate minimum or maximum dates within CTEs rather than in the final SELECT statement.
    d. Use COALESCE with LEAST or GREATEST functions when determining index dates from multiple conditions.

29. **Distinct Patients and Earliest Index Date:
    a. In each CTE, always use GROUP BY person_id to ensure one row per patient.
    b. Use MIN() function to select the earliest relevant date for each patient in CTEs.
    c. In the final SELECT statement, use LEAST() function to choose the earliest date among all criteria as the index date.
    d. When selecting multiple date fields in a CTE or subquery, always use an aggregation function (like MIN, MAX) for each date field and GROUP BY person_id. This ensures you're getting one row per patient with the relevant dates.
       Example:
        SELECT
            person_id,
            MIN(condition_start_date) AS first_condition_date,
            MAX(condition_end_date) AS last_condition_date
        FROM condition_occurrence
        GROUP BY person_id
    e. In the final SELECT statement, when combining multiple CTEs, use INNER JOIN for inclusion criteria and ensure you're selecting only one row per patient by using appropriate aggregation functions or DISTINCT.
    
30. **Final Query Structure:
   a. Use INNER JOIN to combine all inclusion criteria CTEs.
   b. Use LEFT JOIN for exclusion criteria CTEs.
   c. In the WHERE clause, ensure all exclusion criteria person_ids are NULL.

    Here's an example of how to structure the final query:
    ```sql
    WITH
    cte_inclusion1 AS (
        SELECT person_id, MIN(relevant_date) AS earliest_date
        FROM ...
        GROUP BY person_id
    ),
    cte_inclusion2 AS (
        SELECT person_id, MIN(relevant_date) AS earliest_date
        FROM ...
        GROUP BY person_id
    ),
    cte_exclusion1 AS (
        SELECT DISTINCT person_id
        FROM ...
        GROUP BY person_id
    )
    -- More CTEs as needed

    SELECT DISTINCT
        i1.person_id,
        LEAST(i1.earliest_date, i2.earliest_date) AS index_date
    FROM cte_inclusion1 i1
    INNER JOIN cte_inclusion2 i2 ON i1.person_id = i2.person_id
    LEFT JOIN cte_exclusion1 e1 ON i1.person_id = e1.person_id
    WHERE e1.person_id IS NULL;
    -- Add more joins and conditions as needed
    ```
    For example, with Inclusion: 1) Patients with hospitalization between 2012-2020
    2)Patients with a diagnosis of ulcerative colitis within 2 months following hospitalization
    the query should read something like this:

    ```sql
    WITH cte_hospitalization AS (
    SELECT
        person_id,
        MIN(visit_start_date) AS first_hospitalization_date
    FROM visit_occurrence
    WHERE visit_concept_id IN ([visit@inpatient])
    AND visit_start_date BETWEEN TO_DATE('2012-01-01', 'YYYY-MM-DD') AND TO_DATE('2020-12-31', 'YYYY-MM-DD')
    GROUP BY person_id
    ),
    cte_ulcerative_colitis AS (
    SELECT
        co.person_id,
        MIN(co.condition_start_date) AS first_uc_diagnosis_date
    FROM condition_occurrence co
    JOIN cte_hospitalization h ON co.person_id = h.person_id
    WHERE co.condition_concept_id IN ([condition@ulcerative colitis])
    AND co.condition_start_date BETWEEN h.first_hospitalization_date AND DATEADD(day, 60, h.first_hospitalization_date)
    GROUP BY co.person_id
    )
    SELECT DISTINCT
    h.person_id,
    h.first_hospitalization_date AS index_date
    FROM cte_hospitalization h
    INNER JOIN cte_ulcerative_colitis uc ON h.person_id = uc.person_id
    WHERE uc.first_uc_diagnosis_date BETWEEN h.first_hospitalization_date AND DATEADD(day, 60, h.first_hospitalization_date);
    ```

    This structure helps avoid complex subqueries and makes the query more readable and efficient.
    Use INNER JOIN for inclusion criteria and LEFT JOIN for exclusion criteria in the final query structure.
    Always GROUP BY person_id in each CTE to avoid duplicate patients.
    Provide the complete SQL query, including all CTEs and the final SELECT statement. Do not include any LIMIT statement.


Each query must return two columns:
1) "person_id" containing as many rows as the unique persons satisfying the requirements. Use DISTINCT.
2) "index_date": date of the event as specified in the "index date" field. If no "index date" is provided, use the date where the patient started to meet all requirements. Add a comment in the query to highlight your decision.

    Which patients have the following characteristics:
    ${incl_excl_criteria}
"""

sql_splitting_prompt = """You are given in input inclusion and exclusion criteria for selecting a population of interest, and the corresponding SQL query to extract those patients.
    Your objective is to extract from the complete SQL query the individual query corresponding to each criterium.
    
    1. Ensure that the full query and individual queries are consistently applying all criteria.
    2. Review the logic of inclusion and exclusion criteria to make sure they're applied in exactly the same way in both the full and individual queries.
    3. Use Common Table Expressions (CTEs) from the original query to break down the query logic and ensure consistency between the full and individual queries.
    4. For exclusion criteria, use NOT EXISTS or LEFT JOIN / IS NULL patterns to ensure that the excluded population is properly removed from the result set.
    5. Each query should return the set of person_ids that meet the specific criterion, including those that should be excluded by that criterion.

    6. Maintain the EXACT SAME APPROACH used in the original query for each criterion. If the original query uses:
       - GROUP BY with HAVING COUNT, use the same approach in the split query
       - EXISTS with a subquery, maintain this pattern
       - Multiple CTEs in sequence, preserve the necessary preceding CTEs
    7. For criteria involving counts (e.g., "at least two diagnoses"), use the SAME counting method as in the original query.
    8. Do not simplify or restructure the logic - the split query should be a direct extraction from the original.

The key is to preserve the exact same approach used in the original query for each criterion. Do not restructure or simplify the logic, as this can lead to different patients.

Present your results in a structured JSON format as follows: 
{
  "criteria1": "SQL_query_1",
  "criteria2": "SQL_query_2",
  "criteria3": "SQL_query_3"
}

Do not use arrays in your JSON structure. Each criterion should be a key in a single flat JSON object, with the SQL query as its value.
Do not add any additional text in your output.

Ensure that each SQL query is self-contained, and can run independently from the other queries.
When referring to variables or tables, make sure they exists also in the generated split query
Make sure that you only use tables that are in the OMOP common data model. Here is a remainder of the tables in OMOP-CDM:
- PERSON - Demographic information about patients
- OBSERVATION_PERIOD - Time periods when clinical data is available for a person
- VISIT_OCCURRENCE - Information about encounters with healthcare providers or facilities
- CONDITION_OCCURRENCE - Records of conditions or diagnoses
- CONDITION_ERA - Time periods of continuous condition occurrence
- DRUG_EXPOSURE - Records of drug exposures (prescriptions or administrations)
- DRUG_ERA - Time periods of continuous drug exposure
- PROCEDURE_OCCURRENCE - Records of procedures performed on patients
- MEASUREMENT - Records of measurements or laboratory tests
- OBSERVATION - Clinical facts about a person not recorded elsewhere
Each query must have one column called "person_id" containing as many rows as the unique persons satisfying the requirements. Use DISTINCT.
For exclusion criteria, the query should return person_ids that should be excluded, not the ones that remain after exclusion.

Do not use the LIMIT keyword in your queries.

Inclusion/exclusion criteria: ${incl_excl_criteria}

Complete SQL query: ${complete_sql_query}

List of SQL queries corresponding to each inclusion/exclusion criteria:
"""

from ascent_domain.omop.schemas.data_definitions import CohortMetadata


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
        "OMOP_TABLE – any table from OMOP CDM database which has PERSON_ID column, e.g. PERSON table.",
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

    # abbverivation are same for all cases
    if ("COHORT_START_DATE" in cohort_metadata.columns) or ("COHORT_END_DATE" in cohort_metadata.columns):
        abbreviations.extend(
            [
                "OMOP_TABLE_WITH_DATES – OMOP CDM tables which have date column (named `*_DATE`), e.g. DEATH, OBSERVATION, NOTE, PROCEDURE_OCCURRENCE, MEASUREMENT.",
                (
                    "OMOP_TABLE_WITH_DATE_INTERVALS – OMOP CDM tables which have date/time interval columns (named as `*_START_DATE` and `*_END_DATE`), "
                    "e.g. CONDITION_OCCURRENCE, DRUG_EXPOSURE, VISIT_OCCURRENCE, VISIT_DETAIL, DEVICE_EXPOSURE."
                ),
                "ascent.ascent_cohorts.are_intervals_intersect(a_start DATE, a_end DATE, b_start DATE, b_end DATE) – Snowflake UDF function which compares 2 intervals and return TRUE if intervals intersect, FALSE otherwise.",
            ]
        )

    if ("COHORT_START_DATE" in cohort_metadata.columns) and ("COHORT_END_DATE" in cohort_metadata.columns):
        cols_ordered.append("COHORT_START_DATE")
        cols_ordered.append("COHORT_END_DATE")
        rules_text.append(
            "COHORT_TABLE has information on cohort start date and cohort end date in respectative columns COHORT_START_DATE and COHORT_END_DATE."
        )
        rules_processing.append(
            (
                "When joining tables containing date columns with COHORT_TABLE – use the following additional condition in WHERE clause to select records based on date intervals:\n"
                "\n"
                "```sql\n"
                "JOIN question_cohort AS cohort\n"
                "<...>\n"
                "WHERE ascent.ascent_cohorts.are_intervals_intersect(<omop-table-interval>, cohort.COHORT_START_DATE, cohort.COHORT_END_DATE)\n"
                "```,\n"
                "\n"
                "where <omop-table-interval> is:\n"
                "- `<omop-table>_start_date, <omop-table>_end_date` for OMOP_TABLE_WITH_DATE_INTERVALS,\n"
                "- `<omop-table>_date, <omop-table>_date` for OMOP_TABLE_WITH_DATES."
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
                "WHERE ascent.ascent_cohorts.are_intervals_intersect(<omop-table-interval>, cohort.COHORT_START_DATE, NULL)\n"
                "```,\n"
                "\n"
                "where <omop-table-interval> is:\n"
                "- `<omop-table>_start_date, <omop-table>_end_date` for OMOP_TABLE_WITH_DATE_INTERVALS,\n"
                "- `<omop-table>_date, <omop-table>_date` for OMOP_TABLE_WITH_DATES."
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
                "WHERE ascent.ascent_cohorts.are_intervals_intersect(<omop-table-interval>, NULL, cohort.COHORT_END_DATE)\n"
                "```,\n"
                "\n"
                "where <omop-table-interval> is:\n"
                "- `<omop-table>_start_date, <omop-table>_end_date` for OMOP_TABLE_WITH_DATE_INTERVALS,\n"
                "- `<omop-table>_date, <omop-table>_date` for OMOP_TABLE_WITH_DATES."
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
                'In the beginning of the query use the following statement to pre-process (subset) the COHORT_TABLE once:\n'
                '\n'
                '```sql\n'
                'WITH question_cohort AS (\n'
                f"   SELECT {','.join(cols_ordered)}\n"
                f"   FROM {cohort_metadata.snowflake_table_ref}\n"
                ')\n'
                '```.\n'
            ),
            (
                "To join any OMOP_TABLE with COHORT_TABLE use the following SQL-query syntax:\n"
                "\n"
                "```sql\n"
                "JOIN\n"
                "   question_cohort AS cohort\n"
                "ON OMOP_TABLE.person_id = cohort.subject_id\n"
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
Whenever you need to use a table from OMOP CDM containing a participant column (PERSON_ID), you have to join this table with Cohort table to get the cohort participants subset.
Tables without PERSON_ID shouldn't be joined with cohort table, e.g. CONCEPT table.
{analysis_section}

Following definitions are introduced for data-processing rules section:

{abbreviations_section}

## Data-processing rules

The following SQL statements and data-processing rules should be followed:

{cohort_processing_rules_section}
"""
    return prompt_addon


def format_list_as_numbered(l: list[str]) -> str:
    """
    process rules and construct text as numbered item list
    l = ['rule 1\n line 2', 'rule 2', 'rule 3']

    1. rule 1
       line 2
    2. rule 2
    3. rule 3
    """
    items = []
    from textwrap import indent

    for idx, item in enumerate(l):
        idx_text = f"{idx + 1}. "
        prepend_len = len(idx_text)
        item_text = idx_text + indent(item, " " * prepend_len).lstrip()
        items.append(item_text)
    return "\n".join(items)
