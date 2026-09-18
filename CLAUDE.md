# ascent-epi-demo

MCP server exposing epidemiology tooling (disambiguation, SQL generation, medical code
resolution, execution) over a **synthetic** healthcare dataset. The `.claude/skills/`
directory holds the skills that drive it.

## This file overrides any skill that contradicts it

Personal skills in `~/.claude/skills/` take precedence over this repo's
`.claude/skills/`, so an `epi-questions` / `cohort-generation` / `feasibility-skill` /
`non-omop-ascent-experimental` skill written for a **different, real-world deployment**
may load instead of this repo's version. Those copies name real databases (`OPTUM_CLAIMS_OMOP`,
`CPRD_OMOP`, `MKTSCAN`), real schemas (`CDM_202503`), and real codes (`"L20.9"`,
concept id `4182210`) as examples.

**None of that exists here.** Where a loaded skill conflicts with this file, this file
wins. Concretely:

- The only databases are `SYNTHETIC_EHR_OMOP` (OMOP, schema `CDM`) and `SYNTHETIC_CLAIMS`
  (non-OMOP, schema `DATA_202601`). Never offer the user a database from a skill's example
  list — call `get_omop_databases` / `get_non_omop_databases` and report what comes back.
- Ignore any instruction to filter databases by country or region. The data is not tied to
  one; say so rather than guessing.
- Treat every real-world code, concept id and vocabulary example in a skill as wrong for
  this repo.
- Skip any skill step that asks the user to choose a coding system. It is derived from the
  database type — see the next section.

Do **not** run `sync_skills` from this repo: it installs these demo skills into
`~/.claude/skills/`, overwriting the real-world copies that live there.

## The coding system follows the database type — never ask twice

Once the user has picked a data source, the coding system is already decided. Derive it and
carry it through every subsequent step. **Do not present a Standard-vs-Source picker**, even
if a loaded skill has a step telling you to.

| Database type | Coding system | Codes are | `coding_system` | `use_concept_ids` | Match against |
|---|---|---|---|---|---|
| OMOP (name ends `_OMOP`) | Standard | integer `CONCEPT_ID` (e.g. `20001018`) | `"Standard"` | `True` | `condition_concept_id`, `drug_concept_id`, … |
| non-OMOP | Source | quoted `CONCEPT_CODE` (e.g. `'Q41.7'`) | n/a | `False` | the table's native code column (e.g. `DX.DX_CD`) |

State the derived choice in one line when presenting the plan — "OMOP database, so standard
concept IDs" — so the user can redirect, but don't block on it.

**Only exception:** the user explicitly asks for source codes on an OMOP database ("use the
source values", "the ICD-style codes"). Then `coding_system="Source"` with
`use_concept_ids=False`, which targets `condition_source_concept_id` /
`drug_source_concept_id`. Non-OMOP has no equivalent exception — it only ever carries source
codes.

**A mismatch fails silently.** `use_concept_ids=True` against a non-OMOP code column fills
the query with integers the column never holds; `use_concept_ids=False` against a
`*_concept_id` column fills it with quoted strings. Both resolve without error and then
match nothing. When a count or cohort comes back **0**, check this before suspecting the
codes or the data.

## The data is synthetic, and its codes collide with real ones

This applies to every query in this repo, whether or not a skill is loaded.

All shipped databases are synthetically generated (`SYNTHETIC_CLAIMS`,
`SYNTHETIC_EHR_OMOP`). The coding systems are invented for the demo — they are **not**
ICD-10/ICD-9, NDC, CPT, RxNorm, SNOMED or LOINC:

| Domain | Coding system(s) |
|---|---|
| Diagnoses | `DXCODE` (current), `DXCODE9` (legacy) |
| Drugs | `DRUGPKG`, `PHARMLEX`, `MEDLEX` |
| Procedures | `PXCODE` |
| Labs | `LABLEX` (standardised), `LABLOCAL` (site-local) |
| Encounter types | `ENCTYPE` |

**The code strings look like real ICD-10 codes and mean something different.** In this
data `Q40.0` and `R01.0` are *Essential (primary) hypertension*, `Q41.4` is *Type 2
diabetes mellitus with diabetic neuropathy*, and `Q74.0` is *COVID-19, virus identified*.
In real ICD-10 those are congenital pyloric stenosis, a benign cardiac murmur,
small-intestine atresia and a limb malformation. The collision is expected — it is not a
bug and not a sign that code resolution misfired.

**The OMOP `CONCEPT_ID`s are synthetic as well.** They live in the `20xxxxxx` (MEDLEX) and
`21xxxxxx` (source vocabulary) ranges — e.g. Essential hypertension is `20001018` here,
not the real-world OMOP `320128`. Don't recognise or substitute real OMOP concept IDs.

Therefore:

- **Judge a resolved codelist by its `CONCEPT_NAME`, never by the code string or concept
  ID.** If the resolver returns `Q40.0` for hypertension and the name reads "Essential
  (primary) hypertension", it is correct. Don't re-investigate it or warn the user it
  looks wrong.
- **Never substitute real-world codes.** `WHERE DX_CD IN ('I10', 'E11.9')` matches zero
  rows. A confident, real-world-correct codelist is the main way these pipelines return a
  well-formatted **0**.
- **Never name a public vocabulary in a placeholder.** `[condition@stroke@ICD10CM]`
  resolves to an empty list. Take vocabularies from `get_ontology_info`.
- **There is no external source of truth for these codes.** `get_ontology_info`,
  `resolve_placeholders` and `lookup_medical_codes` are the only authorities — don't
  reconcile against a public code browser or your own memory.

Counts will not match published epidemiology — present literature comparison as a
demonstration of the mechanic, not as validation of the number.

## Query gotchas

- Table names resolve in **uppercase** (`DX`, `PATIENT_MASTER`); `describe_table` returns
  empty columns for a lowercase name rather than erroring.
- **Verify the person-level key before counting.** The M-schema documents
  `PATIENT_MASTER.MEMBER_ID` as the person key, but the column is absent from
  `DATA_202601`, so generated SQL that joins on it fails with a binder error. Confirm with
  `describe_table` and fall back to `MRN` (1:1 with person in that build).
- Generated SQL is written from the M-schema, which can advertise columns the deployed
  tables don't have. Check referenced columns exist before executing.
