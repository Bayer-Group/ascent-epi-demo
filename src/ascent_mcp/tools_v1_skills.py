"""
MCP V1 tool: `sync_skills`.

Distribution channel for this repo's Claude Code skills AND slash commands.
The repository is the single source of truth: skills live in
`.claude/skills/`, commands in `.claude/commands/`. This tool reads them and
returns their contents; Claude Code is responsible for writing the files
under the appropriate directories on the user's machine.

The image is built from the repository root, so `.claude/` is present at
runtime and the source directories are served directly.

`bundled_skills/` and `bundled_commands/` still win when non-empty, so a
distributor can override what ships without patching this module.

Why route through the MCP at all? Because a user who has connected the MCP
server has everything they need -- no clone, no separate plugin install, and
the skills always match the server they are talking to.

The MCP itself never writes to the local filesystem — it just serves
content. The agent uses its existing Write tool to materialize the files.
Skills install at `<skills_target_base>/<skill-name>/SKILL.md` (Claude
Code's auto-discovery layout). Commands install at
`<commands_target_base>/<name>.md` (Claude Code's slash-command layout).
"""

from pathlib import Path
from typing import Any, Dict, List, Literal

from ascent_mcp.app_v1 import mcp_v1

_BUNDLE_SKILLS_DIR = Path(__file__).parent / "bundled_skills"
_BUNDLE_COMMANDS_DIR = Path(__file__).parent / "bundled_commands"
# src/ascent_mcp/tools_v1_skills.py -> src/ascent_mcp -> src -> repo root
_REPO_ROOT = Path(__file__).resolve().parents[2]
_REPO_SKILLS_DIR = _REPO_ROOT / ".claude" / "skills"
_REPO_COMMANDS_DIR = _REPO_ROOT / ".claude" / "commands"
_SHA_FILE_NAME = ".skills_sha"


def _has_content(base: Path) -> bool:
    """Whether *base* holds anything worth serving.

    The bundle directories ship with only a README and a .gitignore, both of
    which _collect_files filters out -- so "exists" is not "populated".
    """
    return base.is_dir() and bool(_collect_files(base))


def _source_dirs() -> tuple[Path, Path]:
    """Where to read skills and commands from, bundle overriding the repo."""
    skills = _BUNDLE_SKILLS_DIR if _has_content(_BUNDLE_SKILLS_DIR) else _REPO_SKILLS_DIR
    commands = _BUNDLE_COMMANDS_DIR if _has_content(_BUNDLE_COMMANDS_DIR) else _REPO_COMMANDS_DIR
    return skills, commands


def _read_bundle_version() -> str:
    """Identify what was served, so a user can correlate it with a commit.

    Prefers a SHA written next to an overriding bundle; otherwise the app
    version baked into the image. "unknown" only when neither is available,
    which is the from-source case.
    """
    from ascent_platform.config.runtime import get_runtime_settings

    sha_path = _BUNDLE_SKILLS_DIR / _SHA_FILE_NAME
    if sha_path.exists():
        if sha := sha_path.read_text(encoding="utf-8").strip():
            return sha
    return (get_runtime_settings().APP_VERSION or "").strip() or "unknown"


def _collect_files(
    base: Path,
    skip_filenames: tuple[str, ...] = ("README.md",),
) -> List[Dict[str, str]]:
    """Walk `base`, returning relative-path + UTF-8 content per regular file.

    Excludes dotfiles (`.skills_sha`, `.gitignore`, `.gitkeep`, etc.) and any
    file whose name matches `skip_filenames` at any depth. Returns an empty
    list if `base` is missing or empty — sync_skills treats that as a
    graceful no-op rather than an error so local-dev setups without a bundle
    don't crash.
    """
    if not base.is_dir():
        return []

    files: List[Dict[str, str]] = []
    for path in sorted(base.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(base)
        # Skip dotfiles at any depth.
        if any(part.startswith(".") for part in rel.parts):
            continue
        if path.name in skip_filenames:
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            # Skip binary files defensively — skills/commands should be markdown.
            continue
        files.append({"path": str(rel), "content": content})
    return files


@mcp_v1.tool
async def sync_skills(scope: Literal["user", "project"] = "user") -> Dict[str, Any]:
    """
    Return the Ascent Claude Code bundle — the skills and slash commands
    that ship with this server.

    Install these to teach Claude Code how to drive the Ascent tools: which
    order to call them in, which pipeline a question belongs to, and the
    dialect and conventions the SQL tools expect. The content comes from the
    server's own `.claude/skills/` and `.claude/commands/`, so it always
    matches the server you are connected to — no clone and no plugin install.

    Call it with no arguments to install for every project.

    The tool itself does not touch the user's filesystem — it just returns
    the bundled content. The calling agent should write each entry in
    `skills` to `<skills_target_base>/<path>` and each entry in `commands`
    to `<commands_target_base>/<path>` using its own Write tool.

    Args:
        scope: Where the agent should install the assets.
            - `"user"` (default): `~/.claude/skills/` and `~/.claude/commands/`
              — available in every Claude Code session, regardless of project.
              Right choice for org-wide distribution.
            - `"project"`: `<cwd>/.claude/skills/` and `<cwd>/.claude/commands/`
              — only load when Claude Code is run from this project. Use for
              local testing or project-specific assets.

    Returns:
        Dictionary with:
        - `version`: identifies what was served — the app version of the
          running server, or `"unknown"` when running from source.
        - `scope`: echo of the requested scope.
        - `skills_target_base`: where the agent should write skills
          (`~/.claude/skills` or `.claude/skills`). Paths in `skills` are
          relative to this base.
        - `commands_target_base`: where the agent should write commands
          (`~/.claude/commands` or `.claude/commands`). Paths in `commands`
          are relative to this base.
        - `managed_skills`: list of top-level skill directory names this
          bundle owns. Cleanup should only delete directories under
          `skills_target_base` whose name is in this list and is no longer
          in `skills`.
        - `managed_commands`: list of command filenames (with `.md`
          extension) this bundle owns. Cleanup should only delete files
          under `commands_target_base` whose name is in this list and is
          no longer in `commands`.
        - `skills`: list of `{path, content}` entries. `path` is relative to
          `skills_target_base` (e.g. `"epi-questions/SKILL.md"`).
        - `commands`: list of `{path, content}` entries. `path` is relative to
          `commands_target_base` (e.g. `"sync-skills.md"`).
        - `write_instructions`: human-readable protocol the agent should
          follow.
    """
    skills_dir, commands_dir = _source_dirs()
    skills = _collect_files(skills_dir)
    commands = _collect_files(commands_dir)

    skills_target_base = "~/.claude/skills" if scope == "user" else ".claude/skills"
    commands_target_base = "~/.claude/commands" if scope == "user" else ".claude/commands"

    managed_skills = sorted({entry["path"].split("/", 1)[0] for entry in skills})
    managed_commands = sorted({entry["path"] for entry in commands})

    return {
        "version": _read_bundle_version(),
        "scope": scope,
        "skills_target_base": skills_target_base,
        "commands_target_base": commands_target_base,
        "managed_skills": managed_skills,
        "managed_commands": managed_commands,
        "skills": skills,
        "commands": commands,
        "write_instructions": (
            f"Apply this bundle in two parts.\n"
            f"\n"
            f"## Skills → `{skills_target_base}/`\n"
            f"1. For each entry in `skills`, write `content` to "
            f"`{skills_target_base}/<path>` using the Write tool. Create "
            f"parent directories as needed. Overwrite if the file already "
            f"exists.\n"
            f"2. Cleanup (safe scope): list top-level entries under "
            f"`{skills_target_base}/`. For each directory whose name is in "
            f"`managed_skills` but no longer represented by any `skills` "
            f"entry, delete it. Never touch directories whose name is NOT "
            f"in `managed_skills`.\n"
            f"\n"
            f"## Commands → `{commands_target_base}/`\n"
            f"3. For each entry in `commands`, write `content` to "
            f"`{commands_target_base}/<path>` using the Write tool. Create "
            f"parent directories as needed. Overwrite if the file already "
            f"exists.\n"
            f"4. Cleanup: list files under `{commands_target_base}/`. For "
            f"each file whose name is in `managed_commands` but no longer "
            f"represented by any `commands` entry, delete it. Never touch "
            f"files whose name is NOT in `managed_commands`.\n"
            f"\n"
            f"## Report\n"
            f"5. Tell the user: which files were created, updated, removed; "
            f"the chosen `scope`; and the bundle `version` SHA."
        ),
    }
