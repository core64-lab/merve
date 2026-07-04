# RFC 0002 — Agent-Facing Surface: Decisions A1–A6 and Implementation Sprint Plan

- **Status:** Implemented (2026-07-04) — W-A1 (AGENTS.md scaffold + `init-agents` + doctor check), W-A2 (`doctor --json`), and W-A3 (`llms.txt` + `docs/agent-guide.md`) landed the same day; A5 (`merve guide`) and A6 (MCP server) deferred as recorded. Spec amendment during implementation: the doctor check is wired into `run_all_checks` (which `merve doctor` actually calls) in addition to `run_project_checks`.
- **Decision owner:** Alexander Herzog
- **Scope:** Make merve-built classifier repositories legible and operable for coding agents (Claude Code, Cursor, Codex, …) and for scripts, building on the machine-readable CLI groundwork from RFC 0001 (D7/D8).
- **Baseline:** post-RFC-0001 state — `merve` CLI with `--json` on read commands, stable exit codes (0/1/2), errors that carry the fix, `merve schema`, OpenAPI with request examples, `/healthz` readiness semantics.

---

## 1. Motivation

An exploration (2026-07-04) found that merve has **no agent-specific surface** —
no AGENTS.md, no llms.txt, no machine-readable diagnosis — even though its CLI
primitives are already unusually agent-friendly. The gap matters at a specific
place: agents are pointed at **classifier repositories built with merve**, not
at the merve repo itself, and `merve init` scaffolds zero guidance files.
Field evidence: a downstream repo hand-wrote a 439-line `OPERATE.md` runbook
which has already drifted badly (old package name, old command name, old URLs).
Hand-written operator guides drift; generated ones can be version-stamped,
regenerated, and checked by `doctor` exactly like the CI workflow (v3) is.

## 2. Decision register

| # | Decision | Resolution |
|---|----------|------------|
| A1 | Agent guidance in classifier repos | `merve init` also scaffolds an **AGENTS.md** (the cross-tool agent-guidance convention). Content is **generated from a template shipped in merve** — never hand-written — parameterized on single/multi config shape and classifier names. |
| A2 | Regeneration & staleness | New command **`merve init-agents`** (mirrors `init-github`: `--path/-C`, `--force/-f`) regenerates AGENTS.md for existing repos. The file carries a template-version stamp in an HTML comment; **`merve doctor` checks it** (missing → SKIPPED + recommendation; stale version → WARNING; never FAILED — advisory only). |
| A3 | Machine-readable diagnosis | **`merve doctor --json`** — the last read command without `--json`. Same D7 contract: one JSON document on stdout, diagnostics/deprecations on stderr, exit code identical to human mode (1 on failed checks, else 0). |
| A4 | Discovery from the merve repo | **`llms.txt`** at the merve repo root (llmstxt.org format) pointing into `docs/`, plus a new **`docs/agent-guide.md`** ("Driving Merve from an agent") documenting the JSON surfaces, exit codes, stdout/stderr split, and error-recovery patterns as a contract. |
| A5 | Live digest command (`merve guide`) | **Deferred.** The digest is already the composition of `tag --status --json` + `list-classifiers --json` + `status --json` + `version --json`; AGENTS.md tells agents exactly that. Revisit only if AGENTS.md feedback shows agents still can't orient. |
| A6 | MCP server | **Deferred indefinitely.** The CLI + JSON + stable exit codes + fix-carrying errors already form a complete tool interface for any shell-capable agent; an MCP wrapper would duplicate it. `docs/agent-guide.md` documents the CLI-as-tool-interface instead. |

Design rule reaffirmed from RFC 0001: **errors carry the fix** (removed flags
exit 2 with the replacement; invalid classifier selection lists valid names).
All new surfaces must follow it.

## 3. Sprint plan — three waves

Waves W-A1 and W-A3 are independent (disjoint files) and run in parallel.
W-A2 depends on W-A1 (both touch `mlserver/cli/project.py` and `mlserver/doctor.py`).
CHANGELOG entries are added once at the end by the orchestrator to avoid
three-way conflicts on `[Unreleased]`.

### W-A1 — AGENTS.md scaffold (A1 + A2)

**New module `mlserver/agents_md.py`:**

- `AGENTS_MD_TEMPLATE_VERSION = "1.0"` (bump on incompatible template changes,
  like `github_actions.py` does with its workflow version).
- `generate_agents_md(classifier_names, default_classifier, multi, mlserver_version) -> str`
  — renders the template in §4 verbatim, selecting the single/multi variant blocks.
- `parse_agents_md_version(path) -> Optional[str]` — extracts the stamp from the
  header comment (`AGENTS.md template version: X.Y`); `None` if absent/unparseable.
- `init_agents_md(project_path=".", force=False) -> tuple[bool, str, dict[str, str]]`
  — same return contract as `init_github_actions`. Detects config shape via
  `mlserver.multi_classifier.detect_multi_classifier_config` /
  `list_available_classifiers`; without a config it renders the
  single-classifier variant using the directory name (sanitized) as the
  classifier name. Never overwrites without `force`.
- The merve version comes from the same source `github_actions.py` uses to
  stamp generated workflows.

**Wiring:**

- `init_mlserver_project` creates `AGENTS.md` after `.gitignore` (same
  skip/force semantics; key `"agents_md"` in `files_created`); CLI `init`
  docstring gains the AGENTS.md bullet.
- New CLI command `init-agents` in `mlserver/cli/project.py`, mirroring
  `init-github` (incl. the hidden `-p` rejection via `_LEGACY_P_AS_PATH`).
- `mlserver/doctor.py`: new `check_agents_md(project_path, verbose) -> CheckResult`
  (name `"AGENTS.md"`; PASSED / WARNING stale / SKIPPED missing) added to
  `run_project_checks`; missing file also adds a report recommendation
  pointing at `merve init-agents`.

**Tests:** new `tests/unit/test_agents_md.py` (generation variants, version
parse round-trip, init_agents_md skip/force, single/multi detection) plus
CliRunner tests for `init-agents` and the `init` integration; extend doctor
tests for the three check states. **Docs:** `docs/cli-reference.md`
(`init` file list + new `init-agents` section) and `tests/INDEX.md`.

### W-A2 — `merve doctor --json` (A3) — after W-A1

- `--json` flag on `doctor`; suppresses all rich output; prints exactly one
  JSON document to stdout:
  `{"success": bool, "checks": [{name, status, message, suggestion, details}], "recommendations": [...], "summary": {passed, warnings, failed, skipped}}`
  with `status` as the `CheckStatus` value string. Exit code unchanged
  (1 when `report.has_errors`, else 0).
- Tests in `tests/unit/test_cli_json_output.py` style; docs: `--json` table +
  doctor section in `docs/cli-reference.md`, doctor row in
  `docs/agent-guide.md`, `tests/INDEX.md`.

### W-A3 — Discovery docs (A4) — parallel with W-A1

- `llms.txt` at repo root (llmstxt.org format: H1, blockquote summary, link
  sections) pointing at the key docs pages by GitHub URL.
- `docs/agent-guide.md` — the contract page for agents/scripts: JSON surfaces
  table, exit codes, stdout/stderr split, error-recovery patterns, canonical
  request shape, AGENTS.md scaffold pointer. Must pass the docs-drift gate
  (`scripts/check_docs_drift.py` scans `docs/`).
- Link from `docs/INDEX.md`.

## 4. AGENTS.md template (normative)

The generated file content, verbatim. `{merve_version}` is the installed merve
version; `{TEMPLATE_VERSION}` is `AGENTS_MD_TEMPLATE_VERSION`. Blocks marked
`[SINGLE]`/`[MULTI]` render depending on the config shape; `{classifier_list}`,
`{default_classifier}`, `{first_classifier}` come from the config.

````markdown
<!-- Generated by merve {merve_version}. Do not edit by hand;
     regenerate with: merve init-agents --force
     AGENTS.md template version: {TEMPLATE_VERSION} -->

# Operating this repository (for coding agents and humans)

This repository serves ML classifiers with [Merve](https://github.com/core64-lab/merve)
(`pip install merve`): Python predictor classes wrapped into a FastAPI inference
API, configured by `mlserver.yaml`. The CLI is `merve` (`mlserver` is a
deprecated alias). The importable Python module is still named `mlserver`, the
config file is `mlserver.yaml`, and env vars are `MLSERVER_*` — only the tool
is called merve; do not rename those.

## The 60-second tour

[SINGLE]- Config: `mlserver.yaml` (single classifier — one top-level `predictor:` section).
[SINGLE]- Serve locally: `merve serve`
[MULTI]- Config: `mlserver.yaml` (multi-classifier — top-level `classifiers:` section).
[MULTI]- Classifiers here: {classifier_list}. Default: {default_classifier}.
[MULTI]- Serve locally: `merve serve --classifier <name>`
- Every server process serves ONE classifier at flat paths: `POST /predict`,
  `POST /predict_proba`, `GET /info`, `GET /healthz`, `GET /status`,
  `GET /metrics` — no version or classifier name in URLs.
- Versions come from git tags only (`merve tag`), canonical format
  `<classifier>/vX.Y.Z`. `classifier.version` in the YAML is display-only.
- Validate any config change before serving: `merve validate --json`.

## Request shape (canonical)

Send input keys at the TOP LEVEL of the JSON body:

```bash
curl -X POST localhost:8000/predict -H 'Content-Type: application/json' \
  -d '{"records": [{"feature_a": 1.0, "feature_b": "x"}]}'
```

Accepted top-level keys: `records` / `instances` (list of feature dicts),
`features` (single feature dict), `ndarray` / `inputs` (nested arrays; require
`api.adapter: ndarray` or `auto`). The legacy `{"payload": {...}}` wrapper is
deprecated (removal targeted for 1.0) — do not write new code with it.

The running server documents itself: `GET /openapi.json` carries request
examples and response schemas; `GET /info` lists endpoints and version metadata.

## Machine-readable CLI

JSON goes to stdout; diagnostics and deprecation notices go to stderr.
Exit codes: `0` success, `1` failure, `2` usage error. Errors carry the fix:
removed/renamed flags exit 2 with the replacement spelled out, and an invalid
classifier selection lists the valid names.

| Command | What you get |
|---------|--------------|
| `merve validate --json` | Config validity + per-check results |
| `merve list-classifiers --json` | Classifier names + default |
| `merve tag --status --json` | Per-classifier version/tag status |
| `merve version --json --detailed` | Classifier + tool versions |
| `merve images --json` | Built container images |
| `merve status --json` | Docker/config/environment status |
| `merve doctor --json` | Full diagnosis with fix suggestions |
| `merve schema` | JSON Schema for `mlserver.yaml` |

## Workflow: change → release → deploy

1. Edit the predictor/artifacts; check with `merve validate` and `merve serve`.
2. Commit, then tag the release: `merve tag <patch|minor|major> --classifier <name>`
   — this writes `<name>/vX.Y.Z`; the released version comes ONLY from this tag.
3. `git push --tags` → the GitHub Actions workflow
   (`.github/workflows/ml-classifier-container-build.yml`) builds and publishes
   the container.
4. Manual container ops: `merve build`, `merve push --classifier <name>
   --registry <url>`, `merve run --classifier <name>`, `merve images`, `merve clean`.

[MULTI]## Multi-classifier rules
[MULTI]
[MULTI]- ONE server process serves ONE classifier. Select at startup with
[MULTI]  `merve serve --classifier <name>` or the `MLSERVER_CLASSIFIER` env var
[MULTI]  (containers/Kubernetes). Precedence: `--classifier` flag >
[MULTI]  `MLSERVER_CLASSIFIER` > config `default_classifier`. An invalid value
[MULTI]  fails startup loudly, listing the valid names (fail-fast by design).
[MULTI]- `merve build` builds ONE commit image bundling every classifier
[MULTI]  (`<repo>:<git-sha>` + `<repo>:latest`, no baked classifier).
[MULTI]- `merve push --classifier X --registry <url>` applies release aliases
[MULTI]  (`<repo>:X-vN.N.N`, `<repo>:X-latest`) on that same image — no rebuild.
[MULTI]- Tag each classifier independently: `merve tag patch --classifier <name>`.

## Health and readiness

`GET /healthz` returns `200 {"status": "ok", "model": ...}` once the model is
loaded and `503 {"status": "loading", "model": null}` before that — point
liveness, readiness, and startup probes at it. `GET /status` exposes
concurrency-slot availability; overflow prediction requests get `503` with a
`Retry-After` header.

## Do / Don't

- DO run `merve validate --json` after any config edit, `merve doctor --json`
  when something is broken.
- DO use top-level request keys; DON'T add the deprecated `payload` wrapper.
- DON'T hand-edit versions; versions come from `merve tag` git tags.
- DON'T rename `mlserver.yaml`, the `mlserver` module, or `MLSERVER_*` env vars.
- Full documentation: https://github.com/core64-lab/merve/tree/main/docs
````

## 5. Acceptance gates (every wave)

1. `pytest tests/` green (adapt existing count-based assertions if a new doctor
   check shifts totals — behavior assertions must not be weakened).
2. `.venv/bin/python scripts/check_docs_drift.py` exits 0 (banned tokens:
   `ml_server`, `cli_v2`, `ainit`, `/readyz`, `/startupz`, `batch_predict`,
   `max_concurrent_requests` — never introduce them in `docs/` or `examples/`).
3. `.venv/bin/ruff format --check mlserver/ tests/` clean; `make lint` clean;
   line length 100.
4. Live verification: run the new commands against a scratch project
   (`merve init` in a temp dir; `merve init-agents`; `merve doctor --json | jq .`).
5. No hand-maintained statistics in docs; `tests/INDEX.md` updated with new
   test files; CHANGELOG entries land once, at the end, under `[Unreleased]`.
