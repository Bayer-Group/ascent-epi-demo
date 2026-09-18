# Claude Code skills

Skills bundled with this demo. Each is a directory containing a `SKILL.md`
that tells a Claude Code client how to drive the MCP tools this server exposes.

| Skill | What it does |
|---|---|
| `epi-questions` | Answers population-level questions (prevalence, incidence, counts) by chaining disambiguation, SQL generation, medical coding and execution. |
| `cohort-generation` | Builds a patient cohort from inclusion/exclusion criteria, then persists it. |
| `non-omop-ascent-experimental` | Explores and queries the non-OMOP source tables directly, for questions the OMOP path does not cover. |
| `feasibility-skill` | Decides whether a database can support an analysis before running it, and writes up the answer. |

## Using them

The `sync_skills` MCP tool installs these into a client's skills directory, so
a connected Claude Code picks them up without anyone cloning this repository.
Running Claude Code *from* this repository also picks them up directly, since
they live in the project's `.claude/skills/`.

`src/ascent_mcp/bundled_skills/` overrides this directory when non-empty. It
ships empty; see its README.

## Editing them

Edit `SKILL.md` in place and re-run `sync_skills` on the client. Nothing is
published anywhere else — a skill is only ever read from this repository or
from the bundle directory above.
