# Known data-quality quirks

Dataset: 1,000 patients, seed 42, window 2010-01-01 to 2026-01-31, messiness `realistic`, OMOP CDM 5.4.
Catalog hash `0dca3f0b48292f5d`. Regenerating with the same seed and catalog reproduces every number below exactly.

Everything on this page is deliberate. Nothing here is a bug report, and a pipeline
that "corrects" the items in section 2 is removing the only parts of the dataset
that resemble a real extract.

## 1. Injected error budget (genuinely wrong rows)

These rows are wrong. They are planted at the rates in `catalog/calendar.json` so that a
data-quality check has something to find: a checker that reports zero of these is broken,
and one that reports these counts is correct.

| Defect | Rows | Configured rate | Where it lands | How to find it |
|---|---:|---:|---|---|
| `duplicate_encounter` | 99 | 0.004 | OMOP `visit_occurrence` | same person, type and dates under a second `visit_occurrence_id` |
| `implausible_but_real_lab` | 212 | 0.0008 | OMOP `measurement` | value pinned to the analyte's physiologic min or max (still valid, still alarming) |
| `off_label_prescribing` | 356 | 0.015 | OMOP `drug_exposure` | exposure in a patient with no diagnosis the drug treats |
| `orphan_visit_reference` | 52 | 0.001 | OMOP `condition_occurrence` | `visit_occurrence_id IS NULL` |
| `records_outside_observation_period` | 108 | 0.005 | OMOP `measurement` | measurement_date after every `observation_period` of that person |
| `transposed_date_digits` | 209 | 0.0006 | source EMR date columns | year with its last two digits swapped, e.g. 2019 keyed as 2091 |

Set `--messiness clean` to suppress the source-side defects; the OMOP-side budget is
part of the event log and is always present.

## 2. Structural quirks - do NOT "fix" these

Each of these is what the corresponding real-world artifact looks like. They are
reconcilable, not random, and the ground-truth file gives the answer they reconcile to.

### 2.1 Site-local MRNs and record linkage
2,327 `PATIENT_MASTER` rows describe 1,000 people under 2,327 distinct MRNs; 668 people are registered at more than one facility.
MRNs are site-local. `MEMBER_ID` is the person-level key and is the intended join path
(`PATIENT_MASTER.MEMBER_ID` = OMOP `payer_plan_period.family_source_value`).

### 2.2 Unmapped and site-local lab codes
22.76% of `LAB_RESULTS` rows have a NULL `STD_LAB_CD`, and sites
running a local dictionary emit their own mnemonics in `LOCAL_CD` (`GLUA1C`, `SGPT`, `K`).
Vital signs are the worst affected. Rows with the most unmapped results:

| Test | Unmapped rows |
|---|---:|
| Systolic blood pressure | 4,890 |
| Diastolic blood pressure | 4,825 |
| Body mass index (BMI) | 3,552 |
| Creatinine | 3,018 |
| Cholesterol in LDL | 2,944 |
| Glucose | 2,684 |
| Cholesterol in HDL | 2,681 |
| Triglyceride | 2,507 |

Recover them by name and local code, not by standard code alone.

### 2.3 Four date formats across five facilities, including epoch seconds

| Facility | Format | Specialty |
|---|---|---|
| LSR | `%d-%b-%Y` | endocrine_renal |
| NGH | `%Y-%m-%d` | tertiary_cardiac |
| PAC | `epoch` | ambulatory_general |
| RBM | `%m/%d/%Y` | primary_care |
| SVO | `%Y-%m-%d` | oncology |

`epoch` is seconds since 1970-01-01 and is negative for births before then.

### 2.4 Three labels for two code systems
`DX_CD_TYPE` takes the values `D10` (9,909), `DX10` (30,260), `DX9` (10,680).
`DX10` and `D10` are the same system under two labels, so `GROUP BY DX_CD_TYPE` over-counts
the number of systems. Codes cross over at 2015-10-01 with a
240-day straggler tail at rate 0.06.
Units are also inconsistently cased per site (`mg/dL`, `MG/DL`, `mg/dl`).

### 2.5 Under-coded obesity
668 patients have a recorded BMI of 30 or above; only
352 of them carry an obesity diagnosis. 316 (47.3%) are obese by measurement and silent in the
claim. A claims-only obesity prevalence is supposed to under-count here.

### 2.6 Treatment gaps
Diagnosed patients are not all treated, by design:

| Condition | Diagnosed | Untreated | % untreated |
|---|---:|---:|---:|
| Cardiomyopathy | 3 | 3 | 100.0% |
| Transient ischemic attack | 15 | 15 | 100.0% |
| Atopic dermatitis | 54 | 51 | 94.4% |
| Endometriosis | 48 | 39 | 81.2% |
| Tobacco use disorder | 163 | 126 | 77.3% |
| COVID-19 | 137 | 92 | 67.2% |
| Malignant neoplasm of lung | 3 | 2 | 66.7% |
| Sepsis | 8 | 5 | 62.5% |
| Benign prostatic hyperplasia | 87 | 44 | 50.6% |
| Cirrhosis of liver | 2 | 1 | 50.0% |

### 2.7 Coverage gaps
153 of 1,000 patients have more than one
`observation_period`; the 196 gaps have a median of 69 days.
A patient is invisible during a gap. Do not merge the spells into one continuous window.

### 2.8 Ordered but unresulted tests
6,261 of 226,923 measurements (2.8%) have a
NULL `value_as_number` and `RESULT_STATUS = 'ORDERED'` in the source extract. They are
orders that never came back. NULL is the correct value: a potassium of 0 is not a missing
result, and treating one as the other is how a plausibility check learns the wrong lesson.

## 3. Coverage limitations

- **Cause of death is concentrated in this catalog.** 17 of 161 deaths (10.6%) are attributed to a cancer, against the ~20% seen in national
  statistics, because the competing-risk model can only assign causes the catalog carries.
  Treat the cause-of-death mix as an artifact of the catalog, not as an estimate.
- **No cancer staging, histology, biomarkers or genomics.** Oncology cohorts can be built by
  diagnosis and drug only.
- **No clinical notes, imaging, or waveform data.** The `SOCIAL_HX` table carries the
  structured facts (smoking, alcohol, family history) that would otherwise live in a note.
- **One geography and one currency.** All costs are USD and all facilities sit in the same
  synthetic region.
- **Every code is fictitious.** Vocabularies, code formats and concept names are invented,
  so nothing here can be cross-walked to a real terminology.

## 4. Invariants the data must satisfy

`validate.py` fails the build if any of these breaks; they are the counterpart to
section 1. Anything the file gets wrong that is not listed in section 1 or 2 is a bug.

- No condition, drug, measurement, procedure or coverage window dated before the
  patient's date of birth.
- No clinical record dated after the patient's death.
- No visit or diagnosis outside the patient's observation period.
- No lab value outside its analyte's physiologic range (unresulted tests are NULL).
- No concept id referenced by a clinical table that is missing from `concept`.
- Every standard concept has a self-row in `concept_ancestor`, and every `Is a`
  relationship appears in the closure.
- No code used before the date its vocabulary was issued, and no record of a disease,
  drug or procedure before it existed - COVID-19 not before
  2020-03-01, and no drug before its market date.
- Out-of-period and orphaned rows appear at exactly the counts in section 1 - not zero,
  and not more.

These are checked too, but they are distributional: a small cohort can miss the target
shape by chance, so they print a note rather than failing the build.

- Recorded prevalence rises with age for every age-related condition.
- Seasonal disease arrives in its season.
- The telehealth share of encounters tracks the anchored curve through 2020.
- The age distribution matches the reference pyramid weighted by utilisation.

