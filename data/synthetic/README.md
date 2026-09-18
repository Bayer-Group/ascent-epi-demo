# Synthetic demo data

The generated data needed by the local stack: no hosted warehouse or licensed
patient dataset is required. Model-backed workflows still need an external LLM
API key. None of this data derives from real patients.

```
omop_synthetic_1k.db.zip    OMOP CDM 5.4, 1,000 patients   (SQLite)
source_synthetic_1k.db.zip  non-OMOP source layer, same patients
ground_truth_1k.json        expected answers for the above
vocabulary/                 CONCEPT, CONCEPT_ANCESTOR, CONCEPT_RELATIONSHIP
```

Both databases describe the **same 1,000 people** — the OMOP one standardised,
the source one as a messy upstream feed. That is the point: it exercises both
pipelines against a population whose true answers are known.

## Where this comes from

Everything here is output of [`synthetic_data_generation/`](../../synthetic_data_generation/),
which is committed alongside it. The data is not a fixture someone once made and
lost the recipe for:

```bash
uv run synthetic_data_generation/generate.py --patients 1000 --seed 42
```

Same seed, same data: the two SQLite databases, the ground truth and the quirks
file come out byte-identical across runs. The vocabulary ZIPs are the one
exception — their *contents* match, but the archive embeds a timestamp, so the
`.zip` bytes differ. Compare the CSVs inside, not the archives.

The generator writes `KNOWN_QUIRKS_<size>.md` and the ground truth from the
*finished* database rather than from its own intentions, so the documented
defect counts are measured, not asserted.

Regenerating replaces the committed files — see the generator's README for the
zip-and-place step, since the stack reads `.db.zip` while the generator emits
plain `.db`.

## Scale

`1k` is the committed default. The larger `10k` archives are not committed,
and this repository does not supply a default download URL. Generate them
locally for a self-contained path, or use a mirror supplied by a maintainer.
Run the commands below from the repository root.

### Generate 10k locally

Install Python 3.12 dependencies with [uv](https://docs.astral.sh/uv/) and ensure
that the `zip` command is available:

```bash
uv sync --frozen --extra dev
uv run --no-sync python synthetic_data_generation/generate.py --patients 10000 --seed 42
zip -j data/synthetic/omop_synthetic_10k.db.zip data/omop_synthetic_10k.db
zip -j data/synthetic/source_synthetic_10k.db.zip data/source_synthetic_10k.db
cp data/ground_truth_10k.json data/synthetic/
cp data/KNOWN_QUIRKS_10k.md data/synthetic/
DATA_SCALE=10k docker compose up --build
```

This uses the default catalog/seed that corresponds to the shipped vocabulary.
For custom vocabulary or generator changes, follow the
[generator guide](../../synthetic_data_generation/README.md) and synchronize the
vocabulary files/index as well; do not mix unrelated data and code lists.

### Fetch from a supplied mirror

If a maintainer gives you a base URL, set `ASCENT_DATA_BASE_URL` in your shell.
The URL must serve `omop_synthetic_10k.db.zip`, `source_synthetic_10k.db.zip`, and
`ground_truth_10k.json` at those exact names:

```bash
python3 scripts/fetch_synthetic_data.py 10k \
  --base-url "${ASCENT_DATA_BASE_URL:?Set this to a maintainer-provided base URL}"
DATA_SCALE=10k docker compose up --build
```

The fetch script reads exported environment variables, not the root `.env`
file. It verifies the 10k downloads against the committed `SHA256SUMS` entries
and skips matching files. A newly generated archive may differ from a published
archive; do not bypass a checksum mismatch to accept an unknown download.

Scale is **one choice, not three**. `ground_truth_1k.json` holds counts over
the 1k population, so pairing it with the 10k databases makes every assertion
fail for no real reason. Move them together — the fetch script pulls the
matching ground truth with the databases for exactly this reason.

Built warehouses are keyed by scale (`SYNTHETIC_EHR_OMOP.10k.duckdb`), so the
two can coexist and switching back does not rebuild. That naming is not
cosmetic: without it, moving from 1k to 10k found the existing file, logged
"already built", and served 1,000 patients under a 10k banner.

The generator computes matching ground truth and runs its validation stage by
default. Check its report for a newly generated dataset rather than relying on
validation figures from an older build.

## The vocabularies are deliberately fake

Nine synthetic vocabularies — `MEDLEX`, `DXCODE`, `DXCODE9`, `PHARMLEX`,
`DRUGPKG`, `LABLEX`, `LABLOCAL`, `PXCODE`, `ENCTYPE`. Each one self-declares
`"Synthetic - not a real vocabulary"` in the `vocabulary` table.

The codes are **scrambled against reality on purpose**:

| Synthetic | Means here | Real ICD-10 |
|---|---|---|
| `Q48.0`–`Q48.9` | Atrial fibrillation | Congenital lung anomalies |
| `Q74.1` | COVID-19 with pneumonia | Congenital limb malformation |

So do not relabel `DXCODE` as `ICD10CM` to make prompts work unchanged. It
would make the stack emit authentic-looking ICD-10 that is medically wrong.
The vocabulary set is configuration; point it at these names.

This is also why the vocabulary can ship at all — real SNOMED and CPT4 cannot
be redistributed in a public repository.

## Two things that will silently give you wrong numbers

**1. Dates in the source layer are not one format.** `DX.DX_DT` holds four:

```
YYYY-MM-DD    19,104        DD-Mon-YYYY   20,020
MM/DD/YYYY     8,784        unix epoch ints   2,941
```

A naive `TRY_CAST(DX_DT AS DATE)` nulls **62.4 %** of rows (31,745 of 50,849)
and raises nothing. Counts come out ~⅗ short and everything looks fine. Parse
all four formats explicitly, and assert the row count survives.

`meta.messiness` in the ground truth is `"realistic"`. The mess is the feature.

**2. DuckDB requires explicit date casts.** `VARCHAR >= DATE` is a binder error
rather than an implicit coercion. That one is a gift — it fails loudly.

## The vocabulary CSVs duplicate the OMOP database

`CONCEPT`, `CONCEPT_ANCESTOR` and `CONCEPT_RELATIONSHIP` are byte-identical to
tables inside `omop_synthetic_1k.db`. They are kept separately so the medical
coder can seed its index from 127 KB of CSV instead of opening an 88 MB patient
database it has no business reading.

Of the three, `CONCEPT_RELATIONSHIP` is the load-bearing one: its 2,635
`Maps to` rows are the source→standard mapping (`DXCODE` → `MEDLEX`). Without
it, code lookup resolves nothing. `CONCEPT_ANCESTOR` matters less here than in
real OMOP — the hierarchy is shallow (2,619 self-rows, 2,522 at one level,
3,355 deeper, to a maximum depth of 4), so descendant expansion rarely changes
a count.

## Ground truth

`ground_truth_1k.json` is the acceptance harness, not documentation:

```
cohorts       12   full attrition ladders with expected_n
prevalence    50   rates with 95% CIs, by sex and age
incidence     50   per 1,000 person-years
treatment     35   drug uptake among the diagnosed
labs          39   distributions with reference ranges
```

Spot-checked against the database directly — the T2DM cohort's first attrition
step expects 109 patients, and `condition_occurrence` yields exactly 109.
