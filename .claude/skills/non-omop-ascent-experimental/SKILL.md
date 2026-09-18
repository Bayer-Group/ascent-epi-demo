---
name: non-omop-ascent-experimental
description: Answer medical research questions against a non-OMOP healthcare database by composing the Ascent Experimental MCP tools (schema discovery, placeholder-driven SQL authoring, SQL execution). Use whenever the `ascent-experimental` server is connected and the user asks anything that requires exploring a healthcare database or running SQL against claims/EHR data — counts, prevalence, time-series, drug exposure, condition cohorts, demographic breakdowns, table inspection, custom analytics.
---

# Answering non-OMOP medical questions with Ascent Experimental MCP

The `ascent-experimental` MCP server exposes nine tools over a non-OMOP healthcare database. They aren't a pipeline — they're primitives. This skill describes **how to compose them** to answer any natural-language medical question.

**Canonical pattern: write SQL with `[entity@name@vocab]` placeholders, resolve them in one shot, execute.** Don't reach for `lookup_medical_codes` to drive the answer — that's an inspection / debugging tool, not the coding step.

---

## Tool inventory

**Discovery (always cheap, always start here):**
- `get_database_schema(database, schema, tables=None)` — full M-schema text. Read this before authoring any SQL.
- `get_ontology_info(database, schema)` — which columns hold which coding systems. Tells you which vocabularies to put in placeholders. **Read it; do not assume the vocabularies you know from public data are present.**
- `list_tables(database, schema)` — bare table list when M-schema is too noisy.
- `describe_table(database, schema, table_name)` — column names, types, nullability.
- `get_sample_data(database, schema, table_name, limit)` — peek at real values (≤20 rows). Use to confirm column meaning when names are ambiguous (`DOS` = date of service? `DX1` = primary diagnosis?).

**Placeholder-driven coding (the main path):**
- `resolve_placeholders(sql_with_placeholders, database, codes_with_dots)` — replaces every `[entity@name@vocab1,vocab2]` in the SQL with the actual code list. **This is how you put codes in your SQL.** One call resolves every placeholder in the SQL string. `database` is required: it selects the vocabulary and the dot convention.
- `list_placeholder_concepts(sql)` — extracts placeholders from SQL so you can verify what will be resolved.

**Inspection / debugging only:**
- `lookup_medical_codes(query, …)` — direct medical-coder query. **Do not use this to feed codes into SQL.** It's for two things only: (a) the user explicitly asks "what codes does X have?", or (b) `resolve_placeholders` returned an implausible count for a placeholder and you want to inspect that one concept in isolation.

**Execution:**
- `execute_sql(database, schema, query)` — run SQL (Snowflake dialect, see below). Returns up to 100 rows; aggregate inside SQL for larger results.

---

## How the tools fit together

The canonical flow:

```
question
  │
  ├─→ [discovery]  get_database_schema  →  get_ontology_info
  │                       ↓                       ↓
  │              tables you'll touch       vocabularies per column
  │
  ├─→ [authoring] write SQL with [entity@name@vocab] placeholders
  │                       ↓
  │              (optional) list_placeholder_concepts to verify
  │                       ↓
  │              resolve_placeholders  →  SQL with codes filled in
  │                       ↓
  │              inspect resolution counts; if any look wrong → debug with lookup_medical_codes
  │
  └─→ [execute]   execute_sql           →     answer
```

---

## Sub-flows by question shape

### A. Database exploration ("What's in this database?")
`list_tables` → `describe_table` and/or `get_sample_data` for tables of interest. No placeholders, no SQL execution required.

### B. "What codes does X have?" / "Look up codes for X"
This is the **only** scenario where `lookup_medical_codes` is the primary tool. The user wants codes, not an answer — just call it and return the list.

### C. Single-concept patient count ("How many patients with X?")
1. `get_database_schema` to find the diagnosis/drug/procedure tables.
2. `get_ontology_info` to confirm which vocabularies the relevant column accepts.
3. Write SQL with a single placeholder, e.g.:
   ```sql
   SELECT COUNT(DISTINCT pm.MEMBER_ID)
   FROM DX d
   JOIN PATIENT_MASTER pm ON pm.MRN = d.MRN
   WHERE d.DX_CD IN ([condition@endometriosis@DXCODE,DXCODE9])
   ```
4. `resolve_placeholders(sql, database=<database>, codes_with_dots=True)` — confirm the convention by sampling the column.
5. Sanity-check the resolution count.
6. `execute_sql` on the resolved SQL.

### D. Multi-concept cohort ("Patients with X who took Y before diagnosis")
1. Schema + ontology lookup as above.
2. Author SQL with **one placeholder per concept** in their respective `IN (…)` clauses. Use a CTE per concept and a temporal join on event dates. Declare an explicit `index_date` CTE for "before / after / within N days" logic.
3. `resolve_placeholders` — one call fills all placeholders.
4. Inspect per-placeholder counts; if any is zero or implausible, see [debugging](#debugging-a-bad-resolution).
5. Execute.

### E. Prevalence / proportion / rate ("What fraction of population has X?")
Same as C/D, but the **denominator** matters:
- Sex-specific condition (endometriosis, prostate cancer, BPH, ovarian/uterine cancer, …) → restrict denominator to that sex.
- Age-bounded condition (pediatric-only, geriatric-only) → restrict denominator to the age range.
- "Among <subpopulation>" → use the subpopulation as the denominator.
- Otherwise → all enrolled patients in the period.

The numerator codes already encode anatomy/sex; the denominator must match or the rate is mechanically wrong.

### F. Time series / breakdowns ("By year", "By age group", "By gender")
Build the breakdown dimensions as their own CTE (e.g. a `study_years` generator + an `age_group` CASE expression), CROSS JOIN against the patient set, GROUP BY the dimensions. Aggregate inside SQL because `execute_sql` caps at 100 rows. Coding is still placeholder-driven.

### G. Distribution / demographics ("Distribution of patients with X by gender × age")
Same as F. The cohort CTE narrows to patients with the condition (placeholder-driven); the grouping CTE adds dimension columns; the final SELECT does the patient count per cell.

---

## Discovery rules

- **Always read the schema first** unless the user has named both the tables and the columns. Picking tables from memory leads to silent miscounts when a database records diagnoses across more than one table.
- **Find the person-level key before counting anyone.** A table's id column is not automatically a patient key — a registration or encounter id can repeat per person, so counting it inflates every patient count silently. The M-schema column notes name the person key and say which tables carry it; read them and follow them. Never `COUNT(*)`, and never count an id the notes describe as site-local or per-encounter. Where the key lives on a master table only, join it rather than falling back to the id you already have.
- **Confirm date columns** (`DOS`, `ADMDT`, `EVENT_DATE`, …) when the question has temporal logic. Names vary between tables in the same DB.
- **Check all diagnosis positions** in claims data. Where a table records several diagnoses per encounter, restricting to the primary one is rarely what the user wants; in the shipped schema see `DX.PRIMARY_FL` and `DX.DX_SEQ`.

---

## Placeholder rules

Placeholder format: `[entity@name@vocab1,vocab2]`.

- **`entity`** — one of `condition`, `drug`, `drug_class`, `procedure`, `measurement`, `observation`.
  - `drug` for a specific molecule/formulation; `drug_class` for a therapeutic class.
- **`name`** — the concept in plain English (e.g. `endometriosis`, `pseudoephedrine`, `colonoscopy`).
- **`vocab`** — comma-separated list of coding systems the target column accepts. Derive this from `get_ontology_info`; do not guess.
  - The coding systems are database-specific and are **not** public ICD/NDC/CPT.
    `get_ontology_info` names the ones each column accepts; the M-schema column notes say
    the same. Take them from there — a vocabulary named from memory resolves to nothing.
  - Naming `ICD10CM`, `NDC`, `CPT4`, `RxNorm` or `LOINC` here resolves to an empty code
    list, because no concept in the shipped data carries those vocabularies.
  - **The code strings collide with real ICD-10 strings and mean something different.**
    Here `Q40.0` and `R01.0` are *Essential (primary) hypertension* and `Q41.4` is *Type 2
    diabetes mellitus with diabetic neuropathy*; in real ICD-10 those are congenital
    pyloric stenosis, a benign cardiac murmur, and small-intestine atresia. Judge a
    resolution by its `CONCEPT_NAME`, never by the code string, and never hand-write a
    real-world codelist — `IN ('I10', 'E11.9')` matches zero rows.

Rules:
- **One placeholder per medical concept.** Don't compose multiple concepts inside one placeholder.
- **Place placeholders inside `IN (…)` clauses.** Anywhere else they're meaningless. `WHERE DX_CD IN ([condition@stroke@DXCODE,DXCODE9])` is right; `WHERE DX_CD = [condition@stroke@DXCODE,DXCODE9]` is wrong.
- **Same concept in multiple `IN (…)` clauses must use identical placeholder text.** `resolve_placeholders` replaces by exact string match.
- **`codes_with_dots`** — sample the column and match what it stores. The shipped claims data stores **dotted** codes (`Q10.0`), so it wants `True`; a column holding `E119` wants `False`. Getting this wrong matches nothing.
- **Use `list_placeholder_concepts(sql)`** before the first `resolve_placeholders` call on a long SQL to confirm the regex parsed every placeholder you wrote.

---

## Debugging a bad resolution

After `resolve_placeholders` returns, inspect each `{placeholder, code_count, sample}` entry:

- **`code_count` is 0** — the medical-coder returned nothing. Most common cause: wrong vocabulary. Re-check `get_ontology_info` and adjust the placeholder. Second most common: concept name too narrow ("type 2 diabetes without complications" instead of "type 2 diabetes").
- **`code_count` looks too high** (thousands for a specific condition) — over-broad concept name or cross-organ contamination. Inspect with `lookup_medical_codes(query=<same name>, vocabulary=[…], with_reasoning=true)` to see what's being kept and why.
- **`code_count` looks too low** (handful of codes for a major condition) — under-coverage. Same `lookup_medical_codes` debug call as above; check whether expected clinical variants of the concept are missing by reading the returned `CONCEPT_NAME`s. Don't check against real-world code families — the synthetic vocabulary doesn't mirror them.
- **Sample codes look wrong** (different organ, family-history, external-cause) — note them, then hand-prune in SQL by adding an explicit `AND code NOT IN (…)` after the placeholder is resolved. Don't fight the medical coder by re-iterating; its LLM filter has a noise floor on restrictive instructions.

The `lookup_medical_codes` calls here are diagnostic only — you're not feeding their output back into SQL. The SQL keeps its placeholder; you adjust the placeholder string or post-filter the resolved SQL.

---

## SQL authoring rules

- **CTE per criterion.** One CTE per inclusion / exclusion / temporal step. Final SELECT at the end. Easier to audit, easier to fix.
- **Count the person-level key for patient counts.** The M-schema names it; join whatever table carries it. Never `COUNT(*)`, and never count a site-local or per-encounter id.
- **`UNION` across diagnosis tables.** When patients can be diagnosed in multiple tables (outpatient, inpatient, confinement), `UNION` the patient sets first, then dedupe. Place the same placeholder in each `IN` clause.
- **Explicit `index_date` for temporal questions.** `MIN(event_date) AS index_date` per patient, then join exposures with `<` / `>` / `BETWEEN` against it.
- **Aggregate inside SQL** when the answer is a breakdown — `execute_sql` caps at 100 rows.
- **Snowflake dialect only.** Write Snowflake SQL whatever the backing engine is: statements are transpiled per-statement at the cursor. `DATE_FROM_PARTS()` not `MAKE_DATE()`. Cast to VARCHAR before `TO_DATE()` on numeric columns. `QUALIFY ROW_NUMBER() OVER (...)` for top-N-per-group.
- **Author the full SQL with placeholders first, resolve once.** Don't resolve partial SQL and then paste codes around.

---

## Execution rules

- **Read the error.** Type cast / "object does not exist" / permission errors all need different fixes; don't blindly retry.
- **`execute_sql` returns ≤100 rows.** If the answer needs more, restructure to aggregate inside SQL.
- **Don't mutate.** These tools are intended for reads; don't `INSERT` / `UPDATE` / `DROP` / `CREATE` even if the platform allows it.

---

## Reporting

When you have an answer, tell the user:
- The result (number, table, or breakdown).
- Tables touched.
- Code count per placeholder (from the `resolve_placeholders` resolutions output).
- Any deviations from defaults (e.g. "denominator restricted to female because endometriosis is female-only").
- Any concerns (e.g. "the resolved code list looks narrower than expected").

The SQL and the resolutions output together are the audit trail; the prose is just navigation.

---

## Quick checklist

```
[ ] Question restated; ambiguities named
[ ] database / schema confirmed (ask if not given)
[ ] get_database_schema (and get_ontology_info to pick placeholder vocabularies)
[ ] SQL drafted with [entity@name@vocab] placeholders, one per concept
[ ] (optional) list_placeholder_concepts to confirm parse
[ ] resolve_placeholders called once over the full SQL
[ ] Per-placeholder resolution counts inspected; debugged with lookup_medical_codes only if a count is implausible
[ ] SQL: CTE-per-step, patient count on the person-level key, explicit index_date if temporal
[ ] Denominator scope matches numerator (sex, age, subpopulation)
[ ] All diagnosis positions considered (`DX.PRIMARY_FL` / `DX.DX_SEQ`), not just the primary
[ ] execute_sql; error read carefully
[ ] Report: answer + tables + placeholder counts + caveats
```

---

## When NOT to use this skill

- The question is OMOP-shaped, or you want guided step-by-step disambiguation + literature comparison → use the `epi-questions` skill, which routes through `ascent-mcp-v1` and handles both OMOP and non-OMOP at a higher level of abstraction.
