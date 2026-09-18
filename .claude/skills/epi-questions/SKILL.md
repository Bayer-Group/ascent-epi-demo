---
name: epi-questions
description: Answer epidemiological questions on OMOP and non-OMOP databases by chaining disambiguation, SQL generation, medical coding, and execution tools step by step. Use when the user asks population-level medical questions like prevalence, incidence, patient counts, or cohort characteristics. Automatically routes to the correct pipeline based on database type.
---

# Epidemiological Question Answering (OMOP & Non-OMOP)

Answer population-level medical research questions by orchestrating the ascent-mcp-v1 tools in sequence. This workflow gives you full control over each step — disambiguate, generate SQL, resolve codes, execute, and present results.

---

## Read first: the data is synthetic, and its codes collide with real ones

Every database in this deployment is synthetically generated. The coding systems are
invented for the demo — they are **not** ICD-10/ICD-9, NDC, CPT, RxNorm, SNOMED or LOINC.
`get_ontology_info` names the ones each column accepts, and the M-schema column notes say
the same; take them from there rather than from memory.

**The code strings look like real ICD-10 codes and mean something different.** In this
data `Q40.0` and `R01.0` are *Essential (primary) hypertension*, and `Q41.4` is *Type 2
diabetes mellitus with diabetic neuropathy*. In real ICD-10 those three are congenital
pyloric stenosis, a benign cardiac murmur, and small-intestine atresia. The collision is
expected. It is not a bug, and it is not a sign that the resolver misfired.

**The OMOP `CONCEPT_ID`s are synthetic too**, so a real-world id such as `320128` for
essential hypertension matches nothing here. Resolve every id through the `concept` table
and read the column notes, which say the same. This applies to the Standard coding path in
Step 2.

So:

- **Judge a resolved codelist by its `CONCEPT_NAME`, never by the code string.** If
  `resolve_placeholders` returns `Q40.0` for hypertension and the concept name reads
  "Essential (primary) hypertension", the resolution is correct. Don't re-investigate it,
  don't warn the user it looks wrong, and don't reach for `lookup_medical_codes` to
  double-check a resolution whose names already fit the concept.
- **Never replace resolved codes with codes you know.** `WHERE DX_CD IN ('I10', 'I11.9')`
  matches zero rows, because nothing in the data carries those codes. Substituting a
  real-world-correct codelist is the single most likely way to make this pipeline return
  a confident, well-formatted **0**.
- **Never name a public vocabulary in a placeholder.** `[condition@stroke@ICD10CM]`
  resolves to an empty list. Take the vocabularies from `get_ontology_info`.
- **There is no external source of truth for these codes.** Don't reconcile them against
  a public code browser or your own memory — `get_ontology_info`, `resolve_placeholders`
  and `lookup_medical_codes` are the only authorities.

Because the data is synthetic, counts will not match published epidemiology. Present
`compare_rwd_with_literature` output as a demonstration of the comparison mechanic, not
as validation of the number.

---

## When to Use

Trigger this workflow when the user asks questions like:
- "How many patients have diabetes?"
- "What is the prevalence of heart failure?"
- "Count patients on metformin who also have CKD"
- "What percentage of stroke patients are over 65?"

---

## Workflow

### Step 1: Disambiguate the Question

Call `disambiguate_epidemiological_question` with the user's question.

Present the result to the user using AskUserQuestion:
- Show the **primary interpretation** as the recommended option
- Show **alternative interpretations** as other choices
- Let the user pick or provide their own refinement

**Do NOT skip this step.** Ambiguous questions produce unreliable results.

### Step 1b: Select Database and Schema

If the user hasn't already specified a database, ask them to pick one.

**Pick the database:** Call `get_omop_databases` AND `get_non_omop_databases` to get all databases the user has access to.

**Present ALL databases as a formatted list** (not via AskUserQuestion — the list is too long for interactive choices). Print each database on one line with: name, type (OMOP/non-OMOP), latest schema, and a brief description. Then ask the user to type which one they want.

Example format:
```
Available databases:
- SYNTHETIC_EHR_OMOP (OMOP, CDM) — synthetic OMOP CDM
- SYNTHETIC_CLAIMS (non-OMOP, DATA_202601) — synthetic source layer
...
```

The shipped datasets are synthetic and not tied to any country, so a geographic filter cannot be satisfied; say so rather than guessing.

**Pick the schema:** The response already includes `database_schemas` — a list of available schemas for each database. Default to the **latest** schema (highest date suffix like `CDM` or `DATA_202601`). Show alternatives and let the user confirm.

**Determine database type:** If the database name ends with `_OMOP` → OMOP flow. Otherwise → non-OMOP flow.

---

### Step 2: Derive the Coding System from the Database Type

**Do not ask the user.** The database choice in Step 1b already determines the coding
system. Carry it through every later step without a second question:

| Database type | Coding system | Codes are | `coding_system` | `use_concept_ids` | Columns |
|---|---|---|---|---|---|
| OMOP (name ends `_OMOP`) | Standard | integer `CONCEPT_ID` | `"Standard"` | `True` | `condition_concept_id`, `drug_concept_id`, … |
| non-OMOP | Source | quoted `CONCEPT_CODE` | n/a | `False` | the table's native code column |

State the derived choice in one line when you present the plan — "OMOP database, so
standard concept IDs" — so the user can redirect, but don't block on a picker.

**The only exception:** the user explicitly asks for source codes on an OMOP database
(e.g. "use the ICD-style codes", "query the source values"). Then use
`coding_system="Source"` with `use_concept_ids=False`, which targets
`condition_source_concept_id` / `drug_source_concept_id` instead. Non-OMOP has no
equivalent exception — it only ever carries source codes in its native vocabularies.

Which vocabularies a database actually holds comes from `get_ontology_info`, never from
what is common in public data.

### Step 3: Generate SQL Template

**OMOP:** Call `generate_epidemiological_sql_with_entities_omop` with:
- The confirmed interpretation as `question`
- The target database as `database_name`
- `coding_system` as derived in Step 2 — `"Standard"` for OMOP, unless the user asked for source codes

The tool generates the SQL template with correct columns (Standard → `*_concept_id`, Source → `*_source_concept_id`).

**Non-OMOP:** Tell the user "Generating SQL template — this involves entity extraction, question analysis, and SQL preparation from the database schema, which may take a minute..." then call `generate_epidemiological_sql_non_omop` with:
- The confirmed interpretation as `question`
- The database as `database_name`
- The schema as `database_schema`

**For both:** Review the returned template and entities. Show the user:
- The query explanation / what the SQL does
- The medical entities that will be coded

Please have a look at the resulting query and analyze if it fulfills the initial question, or needs improvement.
If the SQL looks wrong or the user wants changes, you can modify the template yourself or regenerate with a rephrased question.

**Check every column the template references actually exists.** A person-level key often
lives on a master table only, so a generated join can reference it on a clinical table and
fail with a binder error. Join the table that carries the key rather than counting the id
you already have — the M-schema column notes say which is which, and counting a site-local
id overcounts patients. Note that table names resolve in **uppercase**.

### Step 4: Resolve Medical Codes

`use_concept_ids` follows the coding system derived in Step 2 — it is not a fresh decision:

**OMOP** → `resolve_placeholders(sql_with_placeholders=<sql>, database=<database_name>, use_concept_ids=True)` — returns integer `CONCEPT_ID`s
**Non-OMOP** → `resolve_placeholders(sql_with_placeholders=<sql>, database=<database_name>, use_concept_ids=False)` — returns quoted `CONCEPT_CODE`s

Only if the user asked for source codes on an OMOP database: `use_concept_ids=False`.

A mismatch here is silent. `use_concept_ids=True` against a non-OMOP code column fills the
query with integers that the column never contains, and `use_concept_ids=False` against
`*_concept_id` fills it with quoted strings — both return **0** with no error. If a count
comes back as 0, check this before anything else.

Review the resolution summary. Check each entity's `code_count` and the concept *names* in
its sample:

- **`code_count` is 0** — the resolution failed. Most often the placeholder named a public
  vocabulary; re-check `get_ontology_info` and retry. This is the only case that warrants
  `lookup_medical_codes`.
- **Names fit the concept** — the resolution is good. Proceed, even when the code strings
  read like unrelated real-world ICD-10 codes. See the guardrail at the top of this skill.
- **Names don't fit the concept** — a genuine miss (wrong organ, family-history or
  external-cause concepts). Narrow it by hand in SQL rather than re-running the resolver.

Don't use `lookup_medical_codes` to sanity-check a resolution that already looks right by
name; it costs a turn and tempts you into "correcting" codes that were never wrong.

### Step 5: Execute the Query

Call `execute_sql` with:
- `database`: the database name
- `schema`: the appropriate CDM schema (e.g. "CDM") — `get_omop_databases` / `get_non_omop_databases` report each database's schema
- `query`: the filled SQL from Step 4

### Step 6: Present Results

Read the query results and present them to the user in natural language. Include:
- The direct answer to their question
- The number as context (e.g. "out of X total patients in the database")
- Any caveats about the interpretation or coding

### Step 7: Compare with Literature (optional)

After presenting results, call `compare_rwd_with_literature` to validate the finding against published evidence:
- `original_question`: the disambiguated question from Step 1
- `rwd_answer`: your verbalized answer from Step 6 (e.g. "There are 45,231 patients with atopic dermatitis")
- `database_name`: the OMOP database used

This produces a narrative comparing the RWD result with published prevalence estimates, including patient-count scaling to national estimates and citation markers.

**Always offer this step** using `AskUserQuestion` (not plain text) — present interactive choices for next actions:
- "Compare with literature" — run Step 7
- "Generate HTML report" — run Step 8
- "Done" — end the workflow

### Step 8: Generate HTML Report

After the full workflow is complete (including literature comparison if done), generate a comprehensive HTML report that documents the entire analysis. Write the file to the project root as `epi_report_<topic>_<date>.html`.

The report should include:
1. **Header** — Title, date, database used, schema
2. **Original question** — What the user asked
3. **Disambiguation** — The interpretation chosen and alternatives considered
4. **SQL Template** — The generated SQL with placeholders (in a code block)
5. **Medical Codes** — ALL concept IDs/codes used (full list, not just samples), grouped by entity. Include CONCEPT_ID, CONCEPT_CODE, CONCEPT_NAME, and VOCABULARY_ID for each code. This is critical for reproducibility.
6. **Final SQL** — The complete executable query (in a code block)
7. **Results** — Query output in a formatted table
8. **Interpretation** — Natural language answer with caveats
9. **Literature Comparison** (if done) — The full narrative with ALL citations as clickable hyperlinks. Include the complete publication list with: title, authors, year, journal, and a clickable URL/DOI link for each publication.
10. **Methodology** — Coding system used, database description, any notes on limitations

Use clean, modern HTML with inline CSS (no external dependencies). Make it print-friendly and readable.

**Always offer this step** via `AskUserQuestion` after Step 6 (combined with literature option).

---

## Tool Reference

| Step | Tool | Key Parameters |
|------|------|----------------|
| 1 | `disambiguate_epidemiological_question` | `question`, `num_interpretations` |
| 1b | `get_omop_databases` / `get_non_omop_databases` | — |
| 2 | — (derived, no tool call) | OMOP → Standard / `concept_id`; non-OMOP → Source / `concept_code` |
| 3 (OMOP) | `generate_epidemiological_sql_with_entities_omop` | `question`, `database_name`, `coding_system` |
| 2-3 (non-OMOP) | `generate_epidemiological_sql_non_omop` | `question`, `database_name`, `database_schema` |
| 4 | `resolve_placeholders` | `sql_with_placeholders`, `database`, `use_concept_ids` |
| 5 | `execute_sql` | `database`, `schema`, `query` |
| 7 | `compare_rwd_with_literature` | `original_question`, `rwd_answer`, `database_name` |

**Additional tools for fine-grained control:**
- `lookup_medical_codes` — Look up codes for a single medical concept with filters (domain, vocabulary, LLM filter)
- `list_placeholder_concepts` — Extract placeholder entities from SQL without resolving them
- `get_database_schema` — Understand the database structure before writing SQL

---

## Important Notes

- **Always disambiguate first.** "Diabetes" could mean Type 1, Type 2, gestational, or all forms.
- **Always show the user the interpretation and entities** before executing. Let them confirm.
- **Database names ending in `_OMOP`** are OMOP databases. Use `get_database_schema` if unsure about the schema name.
- **If `resolve_placeholders` returns 0 codes** for an entity, try `lookup_medical_codes` with different parameters (broader vocabulary, different encoder).
- **Resolved codes that read like unrelated real-world ICD-10 codes are still correct.** The data is synthetic — trust `CONCEPT_NAME`, and never swap in codes you know from real coding systems, because they match nothing. See the guardrail at the top of this skill.
- **SQL results are capped at 100 rows.** For large result sets, adjust the query (add aggregation, LIMIT, etc.).
