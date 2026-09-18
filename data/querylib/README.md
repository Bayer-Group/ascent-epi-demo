# Query library

The retrieval corpus behind OMOP question answering: 221 worked examples that
are embedded and retrieved as few-shot context when the model writes SQL.

Accompanies *Generating Patient Cohorts from Electronic Health
Records Using Two-Step Retrieval-Augmented Text-to-SQL Generation* (ECAI 2025,
arXiv:2502.21107) — `Bayer-Group/epi-cohort-text2sql-ecai2025`. This copy
tracks the same content and may carry a few additional questions.

## What is here, and what is not

Questions and SQL **templates** only.

| Kept | Why |
|---|---|
| `QUESTION`, `QUESTION_MASKED` | the question and its entity-masked form |
| `DISAMBIGUATED_QUESTION`, `..._MASKED` | what the retriever actually embeds and prompts with |
| `QUERY_SNOWFLAKE_WITH_PLACEHOLDERS` | the template, entities left as `[condition@name@vocab]` |
| `QUESTION_TYPE` | splits `COHORT_GENERATOR` (108) from `QA` (113) |

Removed: `QUERY_SNOWFLAKE_RUNNABLE` above all — the executable form, which
inlines resolved code lists (208 of 221 rows carried one). Also the study
metadata columns (`TITLE`, `INCLUSION_CRITERIA`, `EXCLUSION_CRITERIA`,
`INDEX_DATE_DESCRIPTION`) and the unused paraphrases.

No results, no answers, no resolved codes, no patient counts.

## Both disambiguated columns are required

The obvious minimal set is question + masked question + template. It does not
work: the library's own `metadata` table records which columns to use, and it
overrides what the calling code configures. It names
`DISAMBIGUATED_QUESTION` and `DISAMBIGUATED_QUESTION_MASKED`, so dropping them
loads cleanly and then fails at the first retrieval with
`KeyError: 'DISAMBIGUATED_QUESTION'`.

The stored embeddings were computed from the masked disambiguated column, so
they must stay together.

## Embeddings

Shipped, rather than computed at build time, so the library works without
downloading an embedding model. They are derived from text in this file.
