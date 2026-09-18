# EpiTrap — public benchmark release (dataset + grading rubric)

**156 questions** on a non-OMOP EHR schema and **153** on
OMOP CDM. Each item probes exactly one recognized epidemiological / RWE reasoning
error on a **goal-only prompt**: the trap is never stated in the prompt and must be
*discovered*, not merely executed.

This release includes the **full grading rubric**, so the benchmark is reusable and
scoreable end-to-end by anyone.

EpiTrap accompanies **ASCENT** — see [Citation](#citation).

## Files
| File | What |
|---|---|
| `epitrap-non-omop.jsonl` | 156 questions, non-OMOP EHR binding (Optum EHR). One JSON object per line. |
| `epitrap-omop.jsonl` | 153 questions, standalone OMOP CDM binding. One JSON object per line. |
| `epitrap.html` | Human-readable browser view of **both** sets (one tab each), rendered from exactly the same records: full-text search, category filter, per-question trap, grading logic and answer key. Self-contained single file — open it directly, no server or network needed. |

Load either `.jsonl` with `pandas.read_json(path, lines=True)` or
`datasets.load_dataset('json', data_files=path)`. The `.jsonl` files are the data of
record; the HTML adds nothing and withholds nothing.

## Composition

| Set | Questions | Category |
|---|---|---|
| non-OMOP | 156 | `count` 68 · `table` 38 · `proportion` 29 · `summary-statistic` 17 · `rate` 4 |
| OMOP | 153 | `count` 67 · `table` 38 · `proportion` 28 · `summary-statistic` 16 · `rate` 4 |

The OMOP set is the same benchmark rebound to OMOP CDM (153 vs
156). Questions whose trap is intact but whose *binding* language is
source-specific were re-expressed in OMOP terms (RxNorm `drug_concept_id`,
`drug_exposure_start_date`/`end_date` intervals, `observation_period`,
`unit_concept_id`, …) — the epidemiological trap and the PASS/FAIL structure are
preserved. 3 questions are **dropped** from the OMOP set because the trap
*dissolves* on OMOP rather than merely changing shape:

- **`CALW3-refdate`** — Requires a per-patient calendar birthday (month+day); OMOP person carries only year_of_birth (month/day empty) — the birthday reference date is uncomputable, so this feasibility trap is not expressible on OMOP.
- **`DQ1-feasibility`** — Trap is recognizing blood pressure may be absent in a claims-lite schema and declaring infeasibility; on an EHR-sourced OMOP CDM systolic/diastolic BP are present in measurement, so the feasibility-of-absence trap no longer holds.
- **`F2-enroll`** — Trap is discovering that the source lacks an observation_period table and stores enrollment as YYYYMM integers; OMOP has a first-class observation_period table, so the trap dissolves.

## Per-question schema
| Field | Meaning |
|---|---|
| `id` | Stable question id (e.g. `N8`, `L3-059`). |
| `trap_topic` | Short slug for the topic of the trap (e.g. `washout-boundary`). Free-text, may be empty. |
| `category` | The shape of the result the prompt asks for — see below. |
| `prompt` | **The goal-only question the system is given.** The trap is NOT stated here. |
| `what_it_tests` | The trap: which error is probed, and how a naive one-shot query commits it. |
| `grading_logic` | How a correct answer is distinguished from a trapped one. |
| `approaches` | The answer key: `[{name, description, verdict}]`, `verdict ∈ {PASS, FAIL}`. |

Every question has at least one `PASS` and (almost always) at least one `FAIL`
approach. One item (`L3-056`) is evaluated on two dataset scenarios and therefore
carries a scenario-conditional verdict string
(`"PASS (FIRES) / UNKNOWN (CLEAN)"`); parse `verdict` by its `PASS`/`FAIL` prefix
rather than assuming an exact two-value enum.

### `category` — the shape of the required result

| Value | Meaning |
|---|---|
| `count` | One number: how many patients, events, or records. |
| `proportion` | One dimensionless ratio over a person/event denominator — prevalence, percentage, fraction, risk, incidence proportion. |
| `rate` | One figure per unit of person-time (e.g. per 1,000 person-years). |
| `summary-statistic` | One mean, median, percentile, or total of a per-unit quantity (days, visits, lab values, person-time). |
| `table` | More than one row or figure — strata, a time series, a ranking, a matrix, per-patient values, or several figures reported together. |

A stratified result is `table` whatever statistic fills its cells: the axis is the
**shape of the answer**, not the epidemiological measure. So "prevalence of depression
by age group and sex" is `table`, while "point prevalence of COPD on January 1, 2023" is
`proportion`. `proportion` and `rate` are kept apart in the strict epidemiological sense
— a prevalence is a proportion, not a rate — and only four questions ask for a
person-time rate.

`category` is **descriptive metadata; grading does not depend on it.** A verdict comes
from `prompt` + `what_it_tests` + `grading_logic` + `approaches` alone. It is published
because it is a useful way to slice the benchmark, and because it is checkable: every
value follows from its own prompt, so you can re-derive the whole column yourself.

## How grading works (path-agnostic)

**There is no gold SQL.** Many different queries can be correct, so an item is graded
on *reasoning*, not string-match: given the question and the known trap, does the
produced query apply the required guardrail? `approaches` enumerates the defensible
PASS interpretations and the trap-committing FAIL ones.

## How to grade (reproducing our judge)

We score with an LLM judge that receives, per question, exactly the rubric fields in
this release, then the system's executed SQL. The rubric block is assembled as:

```
## Question: <id>

**Prompt:** <prompt>

**What it tests (the trap):** <what_it_tests>

**Grading logic — the SQL must satisfy this:** <grading_logic>

**SQL approaches → verdict** (which query strategies pass vs commit the trap):
  - <description> → <verdict>
  ...
```

followed by the candidate's SQL, and these label definitions:

- **PASS** — the SQL avoids the trap: it implements the inclusion/exclusion, time
  window, denominator, or unit handling required by the grading logic.
- **FAIL** — the system did not deliver a correct answer. Either (a) the SQL commits
  the trap (wrong denominator, no washout, no unit conversion, …), or (b) it failed
  through its own doing — SQL that repeatedly errored, wrong tables/columns, wrong
  codes, gave up, or ended mid-debugging with no valid final query.
- **UNKNOWN** — reserve ONLY for failures that are not the system's fault: the run was
  killed by infrastructure (server timeout, transport error, harness crash, empty
  trajectory). If the system ran and simply did not succeed, that is FAIL, not UNKNOWN.

`category` is deliberately absent from that block: it is a browsing facet, not a grading
criterion, and it was re-judged after our runs were scored.

We ask the judge for a grade plus the specific SQL clause as evidence, which keeps
verdicts human-auditable. In our own runs, two independent judges from different model
families graded every cell and all disagreements were adjudicated against the executed
SQL — see the paper for the protocol and inter-judge agreement.

## What this release does not contain

**No run artifacts:** no executed SQL, no tool trajectories, no system answers, and no
result values of any kind. Those carry true counts on the underlying licensed EHR data,
which we cannot publish. The rubric itself is free of data-derived values — grading is
reasoning-based, so no reference answer is needed to score an item.

## Citation

EpiTrap was built for and is released with **ASCENT**. Please cite the paper if you use
this dataset or its grading rubric:

> Angelo Ziletti, Leonardo D'Ambrosi, Melanie Tuchardt, Tim Kondziella. ASCENT: An Agentic System over the Model Context Protocol for Real-World Clinical Data Analysis. Bayer AG, Germany, August 2026.

The paper is currently under review, so no BibTeX key is published yet.

See the paper for the methodology, the judge protocol, and results.
