# Bundled Claude Code slash commands — optional override

`sync_skills` serves the repository's `.claude/commands/` when present. **This
directory is empty and does not need to be populated.**

It exists as an override: drop `<name>.md` files here and `sync_skills` serves
these instead of `.claude/commands/`.

Contents are untracked (see `.gitignore`); only this README and the
`.gitignore` are committed.
