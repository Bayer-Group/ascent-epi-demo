"""sync_skills must actually serve the skills this repo ships.

The failure this guards is silent: the tool used to read a directory that a
CI step populated at deploy time. Without that step it returned
``{"skills": [], "commands": []}`` and reported success -- indistinguishable
from a server that legitimately has no skills. Nothing failed, nothing
logged, and the tool looked fine in every smoke test.

So these assert content, not shape.
"""

import re

import pytest

from ascent_mcp.tools_v1_skills import (
    _REPO_SKILLS_DIR,
    _collect_files,
    _has_content,
    _source_dirs,
    sync_skills,
)


async def test_returns_the_skills_in_the_repository():
    result = await sync_skills()

    shipped = {p.name for p in _REPO_SKILLS_DIR.iterdir() if p.is_dir()}
    assert shipped, "the repository ships no skills — this test is vacuous"
    assert set(result["managed_skills"]) == shipped


async def test_every_skill_carries_its_content():
    """A path with empty content installs a file Claude Code cannot read."""
    result = await sync_skills()

    assert result["skills"], "no skills served"
    for entry in result["skills"]:
        assert entry["content"].strip(), f"{entry['path']} is empty"


async def test_every_skill_is_a_discoverable_SKILL_md():
    """Claude Code discovers `<skill-name>/SKILL.md`; anything else is inert."""
    result = await sync_skills()

    roots = {e["path"].split("/", 1)[0] for e in result["skills"]}
    have_skill_md = {e["path"].split("/", 1)[0] for e in result["skills"]
                     if e["path"].endswith("/SKILL.md")}
    assert roots == have_skill_md, f"missing SKILL.md: {sorted(roots - have_skill_md)}"


async def test_every_skill_declares_name_and_description():
    """Both are required frontmatter; without them the skill never triggers."""
    result = await sync_skills()

    for entry in result["skills"]:
        if not entry["path"].endswith("/SKILL.md"):
            continue
        head = entry["content"].lstrip()
        assert head.startswith("---"), f"{entry['path']} has no frontmatter"
        frontmatter = head.split("---", 2)[1]
        assert "name:" in frontmatter, f"{entry['path']} declares no name"
        assert "description:" in frontmatter, f"{entry['path']} declares no description"


@pytest.mark.parametrize("scope,expected", [
    ("user", "~/.claude/skills"),
    ("project", ".claude/skills"),
])
async def test_scope_selects_the_install_root(scope, expected):
    result = await sync_skills(scope=scope)

    assert result["skills_target_base"] == expected


async def test_an_empty_bundle_directory_does_not_shadow_the_repository():
    """The regression itself: bundle dirs hold only a README and .gitignore,
    both filtered out, so "exists" must not be mistaken for "populated"."""
    from ascent_mcp.tools_v1_skills import _BUNDLE_SKILLS_DIR

    assert _BUNDLE_SKILLS_DIR.is_dir()
    assert not _has_content(_BUNDLE_SKILLS_DIR)
    assert _source_dirs()[0] == _REPO_SKILLS_DIR


async def test_a_populated_bundle_overrides_the_repository(tmp_path, monkeypatch):
    """The override path stays usable for a distributor shipping its own set."""
    import ascent_mcp.tools_v1_skills as mod

    (tmp_path / "custom-skill").mkdir()
    (tmp_path / "custom-skill" / "SKILL.md").write_text("---\nname: custom-skill\n---\n")
    monkeypatch.setattr(mod, "_BUNDLE_SKILLS_DIR", tmp_path)

    assert mod._source_dirs()[0] == tmp_path
    assert [e["path"] for e in _collect_files(tmp_path)] == ["custom-skill/SKILL.md"]


# ---------------------------------------------------------------------------
# The skills are instructions for driving the tools, so they can drift out of
# step with them. A skill naming a tool or an argument that no longer exists
# sends the model down a path that fails at call time, and nothing in the
# build notices -- the skills are markdown.
# ---------------------------------------------------------------------------


def _tool_params() -> dict[str, set[str]]:
    """{tool_name: {parameter names}} across every MCP server, from source."""
    from tests.contracts.capture_contracts import mcp_contract

    out: dict[str, set[str]] = {}
    for key, spec in mcp_contract().items():
        name = key.split("::", 1)[1]
        out.setdefault(name, set()).update(p["name"] for p in spec["params"])
    return out


def _skill_text() -> list[tuple[str, str]]:
    return [(p.parent.name, p.read_text()) for p in sorted(_REPO_SKILLS_DIR.rglob("SKILL.md"))]


def test_skills_only_reference_tools_that_exist():
    known = set(_tool_params())
    assert known, "no tools discovered — this test is vacuous"

    # Only inline-code call syntax counts, so prose mentioning a word does
    # not trip this.
    unknown = []
    for skill, text in _skill_text():
        for candidate in re.findall(r"`([a-z][a-z0-9_]{3,})\s*\(", text):
            if candidate.endswith("_tool") or candidate in known:
                continue
            if candidate in _NON_TOOL_CALLABLES:
                continue
            unknown.append(f"{skill}: {candidate}()")
    assert not unknown, f"skills call tools that do not exist: {sorted(set(unknown))}"


def test_skills_only_pass_arguments_the_tools_accept():
    """The failure mode this catches: `cohort_add_covariates(covariates_sql=…)`
    when the parameter is `covariate_sql` — a 422 at call time, in prose."""
    params = _tool_params()
    call = re.compile(r"\b(" + "|".join(map(re.escape, sorted(params))) + r")\s*\(([^)]*)\)")

    bad = []
    for skill, text in _skill_text():
        for match in call.finditer(text):
            name, args = match.group(1), match.group(2)
            for kw in re.findall(r"(\w+)\s*=", args):
                if kw not in params[name]:
                    bad.append(f"{skill}: {name}({kw}=...)")
    assert not bad, f"skills pass unknown arguments: {sorted(set(bad))}"


# Things that look like a call in the skills but are not MCP tools.
_NON_TOOL_CALLABLES = {
    "count", "sum", "min", "max", "avg", "round", "cast", "coalesce", "date_from_parts",
    "to_date", "row_number", "listagg", "iff", "datediff", "dateadd", "try_cast",
    "regexp_like", "nullif", "lower", "upper", "trim", "concat", "abs", "floor", "ceil",
}
