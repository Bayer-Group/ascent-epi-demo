---
name: feasibility-skill
description: "Given a research question and a target real-world database, decide whether the database can support the analysis — before running the analysis itself. The output is a self-contained study folder with reusable SQL, codelists, raw results, and a stakeholder-ready HTML report. This skill is feasibility-only: it verifies that an analytic metric is computable without actually computing it."
---

## Purpose

Given a research question and a target real-world database, decide whether the
database can support the analysis — *before* running the analysis itself. The
output is a self-contained study folder with reusable SQL, codelists, raw
results, and a stakeholder-ready HTML report.

This skill is **feasibility-only**: do NOT compute the analytic metric (e.g. do
not compute eGFR slope or adherence PDC). Verify only that the metric is
*computable*.

## Output structure

For every feasibility study, create:

```
<study-slug>/
├── sql/         # one .sql per check, with results inlined as comments
├── data/        # codelists (.csv) + consolidated feasibility_results.json
└── report/      # feasibility_report.html
```

Slug rules: short, descriptive, `<concept>_<role>_<database>` style.
Examples: `statin_adherence_synthetic_claims`, `egfr_outcome_ckd`,
`mace_outcome_synthetic_claims`.

---

## Workflow (every study)

### Step 0: Disambiguate the research question

**Do NOT skip this step.** Ambiguous questions produce the wrong checks.

Call `disambiguate_epidemiological_question` with the user's question.
Present the primary interpretation and alternatives via `AskUserQuestion` and
let the user confirm or refine before proceeding.

Examples of ambiguity that must be resolved:
- "Can we measure MACE?" → 3-point (MI + stroke + CV death) or 2-point (hMACE)?
- "Can we measure diabetes outcomes?" → HbA1c trajectory, hospitalisation, mortality?
- "Can we measure drug adherence?" → PDC, MPR, or persistence?

Record the confirmed interpretation — it becomes the study title and drives
which archetype checks to run.

### Step 1: Select and characterise the database

**Discover available databases:**
Call `get_omop_databases` AND `get_non_omop_databases` together. Present all
databases to the user in a formatted list — one line each:

```
Available databases:
- SYNTHETIC_EHR_OMOP  (OMOP,     CDM)          — synthetic OMOP CDM
- SYNTHETIC_CLAIMS    (non-OMOP, DATA_202601)  — synthetic source layer
...
```

If the user mentions a country or population, filter the list before showing.

**Select the schema:** Default to the latest schema (highest date suffix, e.g.
`DATA_202601` over an older one). Show alternatives and confirm with the user.

**Determine database type.** This also fixes the coding system — don't ask the user:
- Name ends with `_OMOP` → **OMOP flow**: OMOP CDM tables, integer `CONCEPT_ID`s against
  `*_concept_id` columns, `resolve_placeholders(..., use_concept_ids=True)`
- Otherwise → **non-OMOP flow**: source tables, quoted `CONCEPT_CODE`s against the native
  code column, `resolve_placeholders(..., use_concept_ids=False)`

Only switch an OMOP database to source codes if the user explicitly asks. Getting this
backwards matches zero rows without erroring, which reads as "not feasible" when it is.

**Characterise the database** with `get_database_schema` and `list_tables`.
For non-OMOP databases, always read the schema before writing SQL — table names
and join structure are vendor-specific.

### Step 2: Frame the feasibility checks

Define 3–6 atomic checks tied to the confirmed research question from Step 0.
Use the archetype templates below. Each check should answer one specific
question (e.g. "How many patients have ≥1 MI hospitalisation?").

### Step 3: Resolve codelists

- Call `lookup_medical_codes` for each medical concept needed.
- If a lookup returns 0 codes, retry with different parameters (broader
  vocabulary, different encoder, or relaxed LLM filter).
- Always audit candidate codes before committing — see "Codelist hygiene" below.
- Save final codelists to `data/<concept>_<vocab>_codes.csv` with these four
  fields for every code:
  - `concept_id` — OMOP concept ID (if available)
  - `concept_code` — source code, in whatever form the database stores
  - `concept_name` — human-readable label
  - `vocabulary_id` — vocabulary system, as named by the database under test
    (`get_ontology_info` lists them). Do not assume the public systems you know
    from other data: naming one the database does not carry resolves to an
    empty code list.

The shipped data is synthetic and **its code strings collide with real ICD-10
strings while meaning something different** — `Q40.0` and `R01.0` are essential
hypertension here, `Q41.4` is type 2 diabetes with neuropathy. Validate a codelist
by reading `concept_name`, never by recognising the code, and never hand-write
real-world codes into a feasibility query: they match zero rows and will read as
"the database cannot support this analysis" when in fact it can.

### Step 4: Run the checks

Execute each check with `execute_sql`. After each:
- Save the SQL with results inlined as comments (see SQL file convention below).
- Note any caveats (missing fields, unexpected coding patterns, data anomalies).

### Step 5: Consolidate results

- Write `data/feasibility_results.json` with per-check results, overall verdict,
  and limitations.
- Write `report/feasibility_report.html` (self-contained, inline CSS).

### Step 6: Verdict

For each check: **PASS** / **PASS_WITH_CAVEAT** / **FAIL**.
Roll up to an overall: **FEASIBLE** / **FEASIBLE_WITH_CAVEAT** / **NOT_FEASIBLE**.

### Step 7: Literature plausibility check (optional but recommended)

After presenting results, offer this step via `AskUserQuestion`.

Call `compare_rwd_with_literature` to validate that observed event rates are
clinically plausible against published evidence:
- `original_question`: the disambiguated question from Step 0
- `rwd_answer`: the key number from your checks (e.g. "16,000 MI hospitalisations
  per year in the study cohort")
- `database_name`: the database used

This catches structural data problems (e.g. a count that is 10x the published
prevalence likely signals a codelist or deduplication issue) and adds a
credibility layer to the feasibility report. Include the comparison narrative
and citations in the HTML report.

---

## Archetype A: Drug adherence / utilization

Typical question: "Can we measure adherence (PDC, MPR) for drug X in database Y?"

- **A1. Drug identifiability** — How many patients have ≥1 fill of drug X?
  Identify the canonical source table (DRUG_EXPOSURE, pharmacy claim, etc.).
- **A2. Days-supply / quantity completeness** — % of fills with non-null
  `days_supply` and `quantity` in plausible ranges.
- **A3. Refill cadence** — Distribution of inter-fill gaps. Enough patients
  with multiple fills to compute PDC?
- **A4. Enrollment / observation window** — % of patients with continuous
  enrollment covering the PDC window.
- **A5. Index event linkability** — Can the cohort-defining diagnosis be linked
  to the first fill within a plausible time window?

---

## Archetype B: Lab value as clinical outcome

Typical question: "Can we use lab X (eGFR, HbA1c, LDL, …) as a longitudinal
outcome in database Y?"

- **B1. Lab identifiability** — How many patients have ≥1 measurement of lab X?
  Compare candidate source tables and pick the canonical one.
- **B2. Numeric value completeness** — % of rows with non-null numeric values
  within a clinical plausibility range.
- **B3. Unit-of-measure normalization** — % of rows in the canonical UCUM unit.
  < 50% canonical → FAIL; 50–80% → caveat; ≥ 80% → pass.
- **B4. Repeat-measurement frequency** — Distribution of measurement count per
  patient. Need ≥ 2 per patient (≥ 4 ideal) for slope/trajectory analysis.
- **B5. Post-index follow-up window** — % of patients with ≥1 lab within
  90 / 180 / 365 / 730 days post-index, and % with ≥2 labs at least 90 days
  apart (structural minimum for slope estimation).

---

## Archetype C: Cardiovascular outcomes / MACE

Typical question: "Can we measure MACE in database Y?"

**Disambiguate first (Step 0):**
- 3-point MACE: non-fatal MI + non-fatal ischemic stroke + CV death
- hMACE (2-point): MI hospitalisation + ischemic stroke hospitalisation
  (preferred when CV death is not reliably capturable)

**Standard checks:**

- **C1. Database overview** — Scale, date range, and the mix of diagnosis
  coding versions. Read the coding-version column's actual values rather than
  assuming them, and note when one version covers only part of the period.
  Check for data anomalies (e.g. volume spikes by year that may indicate
  duplicate claims from a data refresh).

- **C2. MI identifiability** — Count hospitalisations whose primary diagnosis
  is acute or subsequent myocardial infarction. Resolve the codes through a
  placeholder rather than writing them out, so they match the database's own
  coding systems. Use the primary diagnosis field only — secondary fields
  inflate counts.

- **C3. Stroke identifiability** — Count hospitalisations with a primary
  diagnosis of ischemic stroke, resolved the same way. Exclude haemorrhagic
  stroke and TIA, which need their own placeholders to exclude by code.

- **C4. CV death measurability** — Check whether a discharge-status or
  mortality field captures in-hospital deaths. Assess whether out-of-hospital
  deaths are structurally absent (common in commercial claims, where enrollment
  ceases at coverage loss). Declare FAIL clearly if CV death is not capturable,
  and recommend hMACE or an alternative database with vital status linkage.

- **C5. Incident event quality** — Check whether the diagnosis table carries a
  present-on-admission flag at all; many do not. Where it exists, present →
  incident event and hospital-acquired → exclude. Otherwise apply a look-back
  washout period (≥180 days) as the fallback incident-event criterion.

- **C6. Enrollment / observation window** — Confirm the enrollment table
  supports pre-index washout and post-index follow-up construction.
  Confirm the enrollment end date serves as the administrative censoring date.

---

## Codelist hygiene

Always audit candidate codes from `lookup_medical_codes` before committing them.
Vector / LLM search is a starting point, not ground truth — false positives are
common and can cause material overcounts.

Run a CONCEPT-table spot-check on any codes you plan to use:
```sql
SELECT concept_id, concept_code, vocabulary_id, concept_name, domain_id
FROM CONCEPT
WHERE concept_id IN (...)
```
Inspect every `code → name` pair before saving the codelist.

**Required fields in every saved codelist:**
- `concept_id` — OMOP concept ID
- `concept_code` — source code, in whatever form the database stores
- `concept_name` — human-readable label
- `vocabulary_id` — vocabulary system, as named by the database under test

These four fields are the minimum for reproducibility when a study is reviewed
or rerun by another analyst.

---

## SQL file convention

Every saved `.sql` file should be self-contained:

```sql
-- Check N: <one-line title>
-- Database: <db>.<schema>
-- Question: <what this check answers>
--
-- CAVEAT (if any): <unexpected behavior, missing fields, anomalies>
--
-- SQL EXECUTED:
<the query>
--
-- RESULTS:
-- <key numbers as inline comments>
--
-- VERDICT: PASS | PASS_WITH_CAVEAT | FAIL
-- <one-paragraph rationale>
```

---

## Attribution

All answers, saved files (SQL, JSON, HTML reports), and user-facing outputs
must include the following attribution:

> Generated by ASCENT-Tools

In HTML reports, include this in the footer. In SQL files, include it as a
comment at the top. In JSON files, include it as a top-level `"generated_by": "ASCENT-Tools"` field.

---

## HTML report convention

- Self-contained (inline CSS, no external assets).
- Sections: Executive summary, Data model, Codelists (full code list with all
  four required fields), one card per check, Overall recommendation,
  Literature comparison (if run), Limitations, Artifacts.
- Every check shows verdict (PASS / PASS_WITH_CAVEAT / FAIL) with a colored badge.
- Tables use `font-variant-numeric: tabular-nums` for clean numeric alignment.
- Do not include emojis.
- Footer must include "Generated by ASCENT-Tools" attribution.

---

## Worked examples


Database-specific notes (join patterns, field quirks, known anomalies) are
documented inside each study's SQL files and `feasibility_results.json`,
not in this skill file.
