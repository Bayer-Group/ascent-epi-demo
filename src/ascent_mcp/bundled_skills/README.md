# Bundled Claude Code skills — optional override

`sync_skills` serves the repository's `.claude/skills/` directly. **This
directory is empty and does not need to be populated.**

It exists as an override: if you drop skill directories here, `sync_skills`
serves these instead of `.claude/skills/`. That is useful for a distributor who
wants to ship a different set without editing `tools_v1_skills.py`. Write a
`.skills_sha` file alongside them to have that string reported as the bundle
`version`.

Contents are untracked (see `.gitignore`); only this README and the
`.gitignore` are committed. The same applies to `../bundled_commands/`.
