---
name: cohort-generation
description: Create patient cohorts by orchestrating text-to-criteria parsing, SQL template generation, medical code resolution, and cohort persistence. Use when the user asks to create, build, or generate a patient cohort from inclusion/exclusion criteria.
---

# Cohort Generation

Create patient cohorts by orchestrating the ascent-mcp-v1 tools in sequence. This workflow gives you full control over each step — parse criteria, generate SQL template, resolve medical codes, persist, and verify.

---

## Read first: the data is synthetic, and its codes collide with real ones

Every database in this deployment is synthetically generated. The coding systems are
invented for the demo — they are **not** ICD-10/ICD-9, NDC, CPT, RxNorm, SNOMED or LOINC.
Confirm per column with `get_ontology_info`, which names the coding systems that column
accepts; the M-schema column notes say the same.

**The code strings look like real ICD-10 codes and mean something different.** In this
data `Q40.0` and `R01.0` are *Essential (primary) hypertension* and `Q41.4` is *Type 2
diabetes mellitus with diabetic neuropathy*; in real ICD-10 those are congenital pyloric
stenosis, a benign cardiac murmur, and small-intestine atresia. The collision is expected.

So: judge a resolved codelist by its `CONCEPT_NAME`, never by the code string; never
replace resolved codes with real-world codes you know (they match zero rows and yield an
empty cohort); and never name a public vocabulary in a placeholder, which resolves to an
empty list. A cohort that silently comes back empty is usually one of these three.

---

## When to Use

Trigger this workflow when the user asks things like:
- "Create a cohort of patients with diabetes"
- "Build me a patient cohort from these criteria"
- "I want a cohort of women aged 16-45 with endometriosis"
- "Generate a cohort for my study"

---

## Workflow

### Step 1: Parse Criteria

If the user provides free-form text (not already structured criteria), call `cohort_text_to_criteria` with the user's description.

The tool returns disambiguated criteria along with per-criterion disambiguation details (original text, preferred interpretation, and alternative interpretations).

**Present the result to the user:**
- Show the **preferred (disambiguated) criteria** as the recommended interpretation
- For each criterion where the original differs from the preferred, briefly note what was clarified (e.g., "diabetes" → "Type 2 diabetes mellitus, coded with the database's own diagnosis vocabulary")
- If alternative interpretations are available, mention them
- Ask the user via AskUserQuestion:
  - "Use preferred interpretation" (recommended)
  - "Let me pick alternatives for specific criteria"
  - "Use original (un-disambiguated) criteria"
- If the user wants alternatives, show them per criterion and let the user pick
- If the user picks "original", use the `original` field from `disambiguation_details`

**If the user already provides structured criteria** (explicit include/exclude lists), skip this step.

### Step 2: Select Database

If the user hasn't already specified a database, help them pick one.

Call `get_omop_databases` AND `get_non_omop_databases` to get all databases.

**Present ALL databases as a formatted list** (not via AskUserQuestion — the list is too long). Print each database on one line with: name, type (OMOP/non-OMOP), latest schema. Then ask the user to type which one they want.

**Pick the schema:** Default to the **latest** schema (highest date suffix). Show alternatives and let the user confirm.

**Determine database type:** If the database name ends with `_OMOP` → OMOP flow. Otherwise → non-OMOP flow.

**The database type determines the coding system — don't ask the user a second time:**

| Database type | Coding system | Codes are | `coding_system` | `use_concept_ids` |
|---|---|---|---|---|
| OMOP (name ends `_OMOP`) | Standard | integer `CONCEPT_ID` | `"Standard"` | `True` |
| non-OMOP | Source | quoted `CONCEPT_CODE` | n/a | `False` |

Carry that through Steps 3 and 4 unchanged. The only exception is the user explicitly
asking for source codes on an OMOP database → `coding_system="Source"` with
`use_concept_ids=False`.

### Step 3: Generate SQL Template

**OMOP:** Call `cohort_generate_sql_omop` with:
- `criteria_include`, `criteria_exclude` from Step 1
- `database_name`, `database_schema` from Step 2
- `coding_system`: as derived in Step 2 — `"Standard"` unless the user asked for source codes

**Non-OMOP:** Call `cohort_generate_sql_non_omop` with:
- `criteria_include`, `criteria_exclude` from Step 1
- `database_name`, `database_schema` from Step 2
- `criteria_index_date` if available

**For both:** Show the user:
- The SQL template with placeholders highlighted
- The medical entities that will be coded
- The explanation (OMOP) or criteria summary

Ask the user to confirm the template looks correct before resolving codes.

### Step 4: Resolve Medical Codes

Call `resolve_placeholders` with the SQL template from Step 3:

`use_concept_ids` follows the coding system derived in Step 2 — it is not a fresh decision:

**OMOP** → `resolve_placeholders(sql_with_placeholders=<template>, database=<database_name>, use_concept_ids=True)`
**Non-OMOP** → `resolve_placeholders(sql_with_placeholders=<template>, database=<database_name>, use_concept_ids=False)`

Only if the user asked for source codes on an OMOP database: `use_concept_ids=False`.

A mismatch is silent — integers against a source-code column, or quoted strings against
`*_concept_id`, both resolve cleanly and then match zero patients. An empty cohort is this
mismatch more often than it is a real absence of patients.

Review the resolution summary:
- If any entity returned 0 codes, investigate with `lookup_medical_codes` for that entity
- Show the user a summary of how many codes were resolved per entity
- Confirm with the user before executing

### Step 5: Persist Cohort

Call `cohort_persist` with:
- `query_filled`: the resolved SQL from Step 4
- `database_name`: from Step 2
- `database_schema`: from Step 2 (required for non-OMOP)
- `criteria_include`, `criteria_exclude`, `criteria_index_date`: from Step 1 (for audit)
- `query_template`: the template from Step 3 (for attrition computation)
- `coding_system`: from Step 3 (OMOP only)
- `explanation`: from Step 3 (OMOP only)

Present the result:
- Cohort ID
- Patient count
- Table name
- Database

### Step 6: Quality Verification (optional)

After cohort creation, offer the user verification options using AskUserQuestion:
- "Generate attrition table" — shows patient funnel per criterion
- "Preview cohort data" — runs `execute_sql` with LIMIT 10 on the cohort table
- "Done" — end the workflow

**Attrition:** Call `generate_cohort_attrition(cohort_id)` to compute the patient funnel.
Present the funnel as a table showing how many patients remain after each criterion.

**Consistency check (mandatory after attrition):** Compare the final step count from the attrition funnel against the persisted cohort's `patient_count`. If they differ by more than 1%:
1. Flag the discrepancy to the user with both numbers
2. Investigate the original query and the split query/ies and regenerate the attrition if needed until they align within 1%. This ensures the cohort definition is consistent and reliable.

---

## Tool Reference

| Step | Tool | Key Parameters |
|------|------|----------------|
| 1 | `cohort_text_to_criteria` | `input_text`, `disambiguate` |
| 2 | `get_omop_databases` / `get_non_omop_databases` | — |
| 3 (OMOP) | `cohort_generate_sql_omop` | `criteria_include`, `criteria_exclude`, `database_name`, `coding_system` |
| 3 (non-OMOP) | `cohort_generate_sql_non_omop` | `criteria_include`, `criteria_exclude`, `database_name`, `database_schema` |
| 4 | `resolve_placeholders` | `sql_with_placeholders`, `database`, `use_concept_ids` |
| 5 | `cohort_persist` | `query_filled`, `database_name`, criteria fields, `query_template` |
| 6 | `generate_cohort_attrition` | `cohort_id` |

**Additional tools for fine-grained control:**
- `lookup_medical_codes` — Look up codes for a single entity if resolve_placeholders returns 0 codes
- `execute_sql` — Preview cohort table contents or run custom queries
- `list_cohorts` — Discover existing cohorts
- `get_cohort` — Inspect an existing cohort's definition and metadata

---

## Important Notes

- **Always confirm criteria with the user** before generating SQL. Misunderstood criteria waste compute.
- **Always show the SQL template** before resolving codes. Let the user validate the logic.
- **Non-OMOP databases require `database_schema`** — use `get_non_omop_databases` to find valid schemas.
- **The database type picks the coding system, not the user.** `_OMOP` → Standard / integer `CONCEPT_ID` / `use_concept_ids=True`; non-OMOP → Source / quoted `CONCEPT_CODE` / `use_concept_ids=False`. Only switch an OMOP database to Source when the user explicitly asks.
- **If `resolve_placeholders` returns 0 codes** for an entity, try `lookup_medical_codes` with different parameters.
- **The `cohort_persist` tool returns metadata only** — patient IDs stay in the warehouse.
- **Store `query_template` in `cohort_persist`** — it's needed for attrition computation later.
