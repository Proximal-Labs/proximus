# Working on Proximus

Read README.md first. Proximus is an external Harbor agent: it targets upstream
Harbor's public agent interface and never patches Harbor's factory, enums or
environments.

- Keep default behavior stable. New options must be inert when unset, and
  changes to prompts, tools, submission/restore, compaction or retries are
  behavior changes that need tests showing the new requests.
- Keep the package infrastructure-agnostic: no cloud SDKs, storage backends,
  private endpoints or organization-specific defaults.
- Never put credentials in code, tests, examples or agent kwargs.
- Tests run offline with fake providers (`uv run pytest -q`); shell submission
  tests use temporary local Git repositories. Validate built distributions with
  `scripts/verify_wheel.py`.
- Do not claim live provider or sandbox behavior from the offline suite.
