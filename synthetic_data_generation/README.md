# Synthetic Data Generation

Generates a fictitious but clinically coherent dataset for demoing a real-world-evidence (RWE) tool:
an OMOP CDM 5.4 database, a deliberately messy non-OMOP source EMR over the *same* patients, and a
ground-truth file of provable answers.

The clinical content is invented. No licensed terminology is redistributed: the vocabularies are
named `MEDLEX`, `PHARMLEX`, `LABLEX`, `DXCODE`, `DXCODE9`, `PXCODE`, `DRUGPKG`, `LABLOCAL` and
`ENCTYPE`, and their code formats, naming grammar and class names are all original. Drug and
analyte names use ordinary generic clinical terms (not trademarks), which are free to use.
Package codes use the `99999` labeler prefix, which no manufacturer holds.

There are two deliberate exceptions, both structural.

A small set of OMOP concept identifiers is reproduced verbatim — `8532` for female gender, and the
equivalents for race, ethnicity, units and type concepts — because a `person` row is only
interpretable if `gender_concept_id = 8532` resolves the way every OHDSI tool expects. These are
identifiers for interoperability, not terminology content.

Twenty concepts additionally carry their **real** OMOP concept ids *and their real standard names*,
tagged `vocabulary_id = 'SNOMED'`: six observation concepts (`Tobacco smoking status`, `Alcohol
intake`, …), twelve observation values (`Never smoker`, `Current every day smoker`, …) and two
death causes. They are in `catalog/standard_concepts.json` under `observation_concepts`,
`observation_values` and `death_causes`. Their `concept_code`s are invented (`OBS-SMK`,
`VAL-SMK-NEVER`), and no SNOMED codes, hierarchy or mappings are included — but the names are
genuine SNOMED descriptions, so it is not accurate to say that nothing from a licensed vocabulary
is reproduced. If your use needs that to be true, rename these to the synthetic `MEDLEX`/`OBSLEX`
pattern with ids from the generator's own blocks and regenerate; nothing else depends on them.

```bash
uv run synthetic_data_generation/generate.py                      # 10,000 patients, seed 42
uv run synthetic_data_generation/generate.py --patients 1000      # faster iteration
uv run synthetic_data_generation/generate.py --messiness nasty    # heavier data defects
uv run synthetic_data_generation/generate.py --postgres           # load into docker-compose Postgres
uv run synthetic_data_generation/generate.py --stages concepts    # rebuild vocabulary only
```

## Outputs

| File | Contents |
|---|---|
| `data/CONCEPT.csv.zip` | Vocabulary in the Athena CSV layout an OMOP loader expects — loads **unchanged**. |
| `data/CONCEPT_RELATIONSHIP.csv.zip` | Source-to-standard `Maps to` / `Mapped from` |
| `data/CONCEPT_ANCESTOR.csv.zip` | Hierarchy for descendant expansion |
| `data/omop_synthetic_<size>.db` | OMOP CDM 5.4 (SQLite) |
| `data/source_synthetic_<size>.db` | Messy non-OMOP source EMR (SQLite) |
| `data/ground_truth_<size>.json` | Cohort counts with attrition ladders, prevalence, lab stats |
| `data/KNOWN_QUIRKS_<size>.md` | Every deliberate defect, with counts — see [Known quirks](#known-quirks) |

OMOP tables written: `person`, `observation_period`, `visit_occurrence`, `condition_occurrence`,
`drug_exposure`, `measurement`, `procedure_occurrence`, `observation`, `death`, `condition_era`,
`drug_era`, `payer_plan_period`, `cost`, `fact_relationship`, `provider`, `care_site`, `location`,
`cdm_source`, plus the four vocabulary tables. Source EMR tables: `PATIENT_MASTER`, `FACILITY`,
`ENCOUNTER`, `DX`, `MED_ORDERS`, `LAB_RESULTS`, `PROCEDURES`, `SOCIAL_HX`, `COVERAGE`.

Patient files are stamped with the cohort size (`_1k`, `_10k`, `_2.5k`, `_100pt`), so datasets of
different scales sit side by side instead of overwriting each other. The vocabulary ZIPs are not
stamped: they do not depend on patient count, and loaders expect the fixed filename.

## Design

**One catalog, two renderings.** `catalog/*.json` is the single hand-authored source of truth.
Both the vocabulary and the patient data derive from it, so the concept table and the EMR cannot
drift apart. Events are generated once into a canonical event log, then rendered twice — clean
into OMOP, messy into the source EMR.

**Clinical coherence.** Patients get a latent state (age, sex, comorbidities) sampled from
age/sex-conditional prevalence with a shared frailty term and a dependency DAG, so disease
clusters the way it does in real populations. Care pathways then emit visits, drug refill chains
and lab panels whose *values are conditioned on the patient's actual disease state*:

```
HbA1c    post-onset T2DM     7.1 %          disease-free   6.1 %
eGFR     post-onset CKD     40 mL/min       disease-free  80 mL/min
troponin within 30d of MI   5.93 ng/mL      disease-free  0.02 ng/mL
```

Prevalence targets are calibrated to published US epidemiology (CDC/NHANES/SEER/AHA figures cited
per condition in `catalog/conditions.json`), so a literature-comparison step has a known expected answer.
A per-condition intercept is binary-searched at sampling time so realized marginals match the
targets rather than merely aiming at them.

Recorded prevalence sits **below** the population target by design — around 70-90% capture — because
a database only contains disease that was coded while the patient was enrolled.

**Fictitious but hierarchical codes.** Since no real code system is used, hierarchy is designed in.
Each disease family owns a `DXCODE` prefix block (all diabetes under `Q41`, heart failure under
`Q50`), so 3-character prefix recall returns genuine clinical siblings. A retrieval layer
recognises `DXCODE`/`PXCODE` for this reason.

**Facility-driven messiness.** The source EMR's defects are keyed to five facilities, each with its
own EHR vendor, date format (`2015-03-04`, `03/04/2015`, `04-Mar-2015`, epoch ints), lab dialect
(`GLUA1C` vs `LAB-100011`), specialty-driven case mix and null-rate profile — because that is what
real multi-site extracts look like. Diagnoses also switch code system at 2015-10-01, forcing the
tool to reason across a vocabulary transition. Counts differ from OMOP only by the known delta
recorded in `ground_truth.json`.

**Site-local MRNs, person-level member IDs.** A patient registered at three facilities has three
`PATIENT_MASTER` rows under three MRNs, and a handful of them disagree about the date of birth.
`MEMBER_ID` is the person-level key and the intended join path; it appears in OMOP as
`payer_plan_period.family_source_value`. Linking on the MRN under-counts people, which is the
point of the exercise.

**Nothing is stationary.** `catalog/calendar.json` makes the whole file a function of calendar
time, because a tool tested on a stationary extract fails silently on a real one:

| Force | Effect |
|---|---|
| Volume anchors | EHR onboarding growth, the April 2020 collapse, deferred-care rebound |
| Claims run-out | The last 120 days are progressively thinner, not a flat cliff |
| Visit mix anchors | Telehealth goes 0.4% (2019) → 32% (May 2020) → a plateau far above baseline |
| Seasonality profiles | Per condition (winter respiratory, spring allergy, summer injury) and per visit type |
| Month length | February really does see ~9% fewer encounters than January |
| Vocabulary windows | No code before its system was issued; no COVID-19 before 2020-03-01 |
| Cutover tail | 2015-10-01, with stragglers still coding the retired system for 240 days |

Seasonality multipliers are normalised to mean 1.0, so a profile only changes *when* events
happen, never how many. They move both the monthly total and the mix inside it.

**Coverage is a spell, not a window.** Enrolment is built backwards from the date it last lapsed,
which for most of the cohort is the extract date, because an extract's population is whoever the
payer covers *now*. Spells break and resume on January boundaries, and a patient is invisible in
the gap.

## Known quirks

`KNOWN_QUIRKS_<size>.md` is generated from the finished database on every run and is the contract
between this dataset and anything that profiles it. It has three parts:

**An injected error budget.** Rows that are genuinely wrong, planted at the rates in
`catalog/calendar.json`: duplicate encounters, orphaned visit references, records outside the
observation period, off-label prescribing, transposed date digits, labs pinned to a physiologic
extreme. A data-quality check that reports *zero* of these is broken; one that reports these
counts is correct. `--messiness clean` suppresses the source-side defects.

**Structural quirks that must not be "fixed."** Unmapped lab codes, four date formats, three
labels for two code systems, under-coded obesity, treatment gaps, coverage gaps. Each is what the
corresponding real-world artifact looks like, each is reconcilable, and the ground-truth file gives
the answer it reconciles to. Removing them removes the only parts of the dataset that resemble a
real extract.

**Coverage limitations.** What this catalog cannot represent — no staging or biomarkers, no notes
or imaging, one geography, and a cause-of-death mix that is an artifact of the catalog rather than
an estimate.

## Validation

`validate.py` runs two tiers, and the distinction matters:

**Invariants fail the build.** Nothing before birth or after death, nothing outside the
observation period, no lab value outside its analyte's physiologic range, no concept id referenced
but missing from `concept`, a `concept_ancestor` self-row for every standard concept, no code used
before its vocabulary existed, and the injected defects present at *exactly* their documented
counts — not zero, and not more.

**Distributional checks print a note.** A small cohort can miss a target shape by chance, so
prevalence monotonicity in age, seasonal arrival, annual periodicity in encounter counts, the
telehealth curve and the age pyramid warn rather than fail. Each carries a power estimate: the
seasonality tests report `untestable at this cohort size` instead of a spurious pass or failure
when the cohort cannot resolve a shallow profile, and pool related conditions before giving up.

## Ground truth

`ground_truth.json` is computed by querying the **finished** database, never from input parameters —
otherwise it would record intent rather than fact. Each demo cohort carries a step-by-step attrition
ladder that a cohort-building tool can be diffed against:

```
How many adults with type 2 diabetes are on metformin and still have an HbA1c above 8%?
    937  Any type 2 diabetes diagnosis
    877  At least 2 diagnoses >=30 days apart
    877  Age 18 or older at index
    333  At least 365 days of prior observation
    234  Metformin exposure after index
    198  HbA1c > 8% within 90 days of exposure
```

## Modules

| File | Role |
|---|---|
| `catalog/` | Hand-authored clinical content (conditions, drugs, labs, procedures, visits, cohorts) |
| `models.py` | Frozen dataclasses defining the catalog contract |
| `catalog.py` | Loading and referential validation |
| `vocabulary.py` | Catalog to concept / relationship / ancestor tables |
| `expansion.py` | Sibling codes for search density (disable with `--no-expand`) |
| `patients.py` | Vectorized latent-state sampling, prevalence calibration, Gompertz mortality |
| `obstetrics.py` | Pregnancy episodes, delivery encounters, mother-infant linkage |
| `pathways.py` | Longitudinal event emission, calendar tables, coverage spells |
| `omop_writer.py` | OMOP CDM 5.4 output |
| `source_writer.py` | Messy non-OMOP EMR output |
| `ground_truth.py` | Provable answers, and the known-quirks manifest, computed from the finished database |
| `validate.py` | Invariants, distributional checks and the realism report |
| `generate.py` | CLI |

## Reproducibility

A single `--seed` drives everything, split into independent child streams so changing `--patients`
does not perturb the vocabulary. Identical seeds produce byte-identical databases.

## Adding to the catalog

Edit the JSON, then run `--stages concepts` to check it loads. Validation enforces that every drug's
indication names a real condition, every lab distribution keys off a state the simulator can
produce, every unit has a concept id, codes are unique, and the comorbidity graph is acyclic.
