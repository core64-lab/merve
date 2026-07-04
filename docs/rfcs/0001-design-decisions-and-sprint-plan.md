# RFC 0001 — v1 Roadmap: Design Decisions D1–D22 and Implementation Sprint Plan

- **Status:** Implemented (2026-07-04) — all of D1–D22 landed across Waves 0–2; suite at 1135 passing / 76% coverage. Release tags v0.4.0 / v0.5.0 are the user's to push.
- **Decision owner:** Alexander Herzog
- **Scope:** All 22 design decisions from the 2026-07-03 design review, accepted as recommended. This document is the decision register and the execution plan.
- **Baseline:** post-stabilization-sprint state — 897 tests passing / 0 failing, 64% coverage, all previously confirmed bugs fixed.

---

## 1. Decision register

| # | Decision | Resolution (accepted recommendation) |
|---|----------|--------------------------------------|
| D1 | Framework commit in tag name | Remove `-mlserver-<hash>` from tags. Toolchain provenance moves to the dependency pin (requirements) + OCI image labels. Old tags remain readable via a parsing shim. |
| D2 | Tag format | `<classifier>/vX.Y.Z` (slash-namespaced). Migration shim reads legacy `<classifier>-vX.Y.Z-mlserver-<hash>`. |
| D3 | Version source of truth | Git tags are canonical. `classifier.version` in YAML becomes display-only and deprecated (load-time warning); `push --version-source` removed. |
| D4 | Image strategy | Build once per git commit; classifier selected at deploy time via `MLSERVER_CLASSIFIER`; per-classifier release tags applied as registry tag aliases on the same digest. Per-classifier images remain as opt-out for diverging deps. |
| D5 | Image labels | Standard OCI annotations (`org.opencontainers.image.source/revision/version/created`) + custom `dev.merve.*` labels. |
| D6 | CLI structure | Split `cli.py` (1,816 lines) into `mlserver/cli/` command modules, thin over library functions. Every command gets CliRunner tests. |
| D7 | Machine-readable CLI | `--json` on all read commands; documented stable exit codes (0 ok / 1 failed / 2 usage). |
| D8 | Flag consistency | One breaking pass: `-p`=port only, `--path`/`-C`=project dir everywhere, `-v`=verbose only; project/config resolution centralized in a Typer callback context. |
| D9 | Naming collision with Seldon MLServer | Rename distribution + console script to `merve`; keep `mlserver` alias entry point for one transition release. Python module rename deferred (out of scope). |
| D10 | Request envelope | Accept top-level `records`/`instances`/`ndarray`/`inputs` AND the legacy `{"payload": {...}}` wrapper; wrapper logs a deprecation warning; wrapper removal targeted for 1.0. |
| D11 | Response formats | Deprecate `custom` response format and `extract_values` (warnings, docs updated); keep `standard` + `passthrough`. Add `predictor_class` to `ProbaResponse` for consistency. |
| D12 | GlobalSettings | Delete `settings.py` GlobalSettings singleton and `global_config.yaml`. Real knobs become module constants overridable by env/CLI. Presence of a `global_config.yaml` logs a clear "no longer read" warning. |
| D13 | Predictor contract | Add `Predictor` Protocol (`predict`, optional `predict_proba`, optional `load()`); accept single-string spec `predictor: "module:ClassName"`; import via `importlib.util.spec_from_file_location` under `merve._user.<name>` namespace — no more `sys.path.insert` / `del sys.modules`. Old two-field spec keeps working. |
| D14 | Concurrency stance | Keep default `max_concurrent_predictions: 1` (+ 503). Make `Retry-After` configurable. Startup warning when `workers > 1` and metrics enabled. Prominent README note. |
| D15 | Container build | BuildKit (drop `DOCKER_BUILDKIT=0`), multi-stage Dockerfile (build deps out of runtime layer), optional `--platform` multi-arch flag. |
| D16 | Framework install in image | Once releases exist (D18): `pip install merve==X.Y.Z`; wheel-copy path kept only as fallback for unreleased dev builds. |
| D17 | CI for this repo | GitHub Actions: pytest, ruff check + format check, wheel build, coverage floor with ratchet, and drift-grep gates (`ml_server`, `cli_v2`, `ainit`, `/readyz`, `batch_predict`, `max_concurrent_requests` banned outside `docs/archive/`). |
| D18 | Releases | Tag real releases (starting v0.4.0), track CHANGELOG.md in git (fix `.gitignore` whitelist), consumers pin versions. |
| D19 | Style tooling | ruff (lint + format) + light mypy on `mlserver/`; pre-commit config; remove the "black" badge until formatting is enforced (then re-add a ruff badge). |
| D20 | Docs discipline | `docs/` describes `main` only; proposals in `docs/rfcs/` with status headers; hand-maintained statistics are generated or deleted. |
| D21 | Test hygiene | Rewrite or delete the 16 permanently-skipped container tests; skips must carry reason + date; skips older than one sprint are deleted. |
| D22 | Environment hygiene | Dedicated venv for this repo (`.venv`); `trade-likelihood` reinstalls the editable from the real repo path; remove the stale third checkout in its `.venv/src`. |

---

## 2. Sequencing: three waves, two releases

Rationale: the items interlock (rename touches everything; tags touch version_control/cli/github_actions/docs; envelope touches server/schemas/docs/examples/tests). Breaking changes are batched into one release so users migrate once.

| Wave | Theme | Breaking? | Release |
|------|-------|-----------|---------|
| 0 | Foundations & guardrails (D17, D18-scaffold, D19, D20-done, D21, D22) | No | — |
| 1 | Compatible product changes + deprecations (D1–D3 write-side-compatible, D5, D7, D10–D14, D15, D16-prep) | No (deprecations only) | **v0.4.0** |
| 2 | Breaking batch (D4, D6, D8, D9, D16, tag write-format switch, `ruff format` finale) | Yes, batched | **v0.5.0** |

Each wave: parallel subagents with strict disjoint file ownership → orchestrator verification battery → review checkpoint → commit(s) on approval. No wave starts before the previous wave's gate passes.

---

## 3. Wave 0 — Foundations (no behavior changes)

**W0.1 — CI workflow (D17).** `.github/workflows/ci.yml`: jobs = test (pytest, Python 3.11 + 3.13), lint (ruff check + format --check), build (python -m build), drift-grep gate, coverage floor `--cov-fail-under=60`. Runs on push/PR to main. *Tests: the drift-grep script itself gets a unit test (patterns, archive exclusion).*

**W0.2 — Lint baseline (D19).** Add `[tool.ruff]` (line-length 100, lint rule set E,F,W,I,UP,B; format deferred to Wave 2 finale) + `[tool.mypy]` (lenient: ignore-missing-imports, no strict flags yet) to pyproject. Run `ruff check --fix` (safe fixes only) as a single isolated pass; fix remaining errors by hand. Add `.pre-commit-config.yaml` (ruff, ruff-format staged-only off until Wave 2, check-yaml, end-of-file-fixer). Remove the "black" badge from README.

**W0.3 — Release scaffolding (D18).** Fix `.gitignore`: whitelist `CHANGELOG.md` (currently swallowed by `/*.md`). Backfill CHANGELOG with the stabilization sprint under `[Unreleased]`. Document the release procedure in `docs/development.md` (tag `vX.Y.Z` on main → CI builds wheel). No tag pushed yet — v0.4.0 tags at end of Wave 1.

**W0.4 — Container-test rewrite (D21).** Rewrite the 16 skipped tests in `tests/unit/test_container.py` against the current container.py API (mock `subprocess.run`; no docker daemon needed). Delete any that test removed behavior. Target: container.py 56% → ≥70%. Add the skip-policy note to `tests/INDEX.md`.

**W0.5 — Environment (D22).** Create `.venv` in this repo (`python3 -m venv .venv && pip install -e ".[dev]"`); reinstall trade-likelihood's editable from `/Users/peter/Desktop/enmacc/merve`; remove its stale `.venv/src/mlserver-fastapi-wrapper` checkout. Document in `docs/development.md`.

**Wave 0 gate:** full suite green in the new venv; `ruff check` clean; CI workflow file validated with `act`-style dry parse or pushed on a branch when the user chooses; coverage ≥ 66% (container rewrite bump).

---

## 4. Wave 1 — Compatible changes + deprecations (→ v0.4.0)

**W1.1 — Request envelope compatibility (D10).** `server.py`/`schemas.py`: request model accepts BOTH shapes — top-level `records`/`instances`/`ndarray`/`inputs`/`features` and legacy `{"payload": {...}}`. Wrapper usage logs one deprecation warning per process (not per request). OpenAPI examples show top-level form. *Tests: parametrized matrix — {top-level, wrapped} × {records, instances, ndarray, inputs, single-features} × {predict, predict_proba} — asserting identical predictions; deprecation-warning emission test.*

**W1.2 — Response-format deprecations (D11).** Load-time `DeprecationWarning` for `response_format: custom` and `extract_values: true`; `ProbaResponse` gains `predictor_class`. *Tests: warning emission; proba response field.*

**W1.3 — Tag system, compatible half (D1–D3).** `version_control.py`: new canonical parser for `<classifier>/vX.Y.Z` + legacy shim for `<classifier>-vX.Y.Z(-mlserver-<hash>)`; all read paths (status, push validation, version listing) accept both. `mlserver tag` still WRITES legacy format in this wave (write-switch is Wave 2, breaking). `classifier.version` and `--version-source` log deprecation warnings. Delete `validate_version_consistency` version-vs-config cross-checks that D3 obsoletes. *Tests: parametrized old/new/both-present tag parsing (≥20 cases incl. hyphenated classifier names, malformed tags); version_control.py ≥85%.*

**W1.4 — OCI labels (D5).** `container.py`: emit `org.opencontainers.image.{source,revision,version,created}` + `dev.merve.{classifier,mlserver_version}`; keep old labels one release for dashboard continuity. *Tests: label-set assertion on generated Dockerfile.*

**W1.5 — CLI additive UX (D7).** `--json` on `images`, `status`, `list-classifiers`, `validate`, `tag --status`; exit-code table documented in cli-reference. *Tests: CliRunner per command asserting valid JSON schema + exit codes (foundation for the Wave 2 coverage push).*

**W1.6 — GlobalSettings removal (D12).** Delete `SettingsSingleton`/`GlobalSettings`/`global_config.yaml` (git rm); constants move to `mlserver/defaults.py`; loud warning if a `global_config.yaml` is found. Update Makefile/demo references. *Tests: defaults module; warning path; delete test_settings.py sections that die with it.*

**W1.7 — Predictor protocol + import isolation (D13).** New `mlserver/predictor.py`: `Predictor` Protocol + docs; `predictor: "module:ClassName"` string spec accepted alongside the two-field form; loader rewritten to `spec_from_file_location` under `merve._user.*` (no `sys.path` mutation, no `sys.modules` deletion); optional `load()` called in lifespan, `/healthz` distinguishes loaded/not. *Tests: name-collision case (predictor file named `types.py` must not shadow stdlib), string-spec parsing, `load()` lifecycle, legacy two-field spec regression suite; predictor_loader ≥90%.*

**W1.8 — Concurrency polish (D14).** `api.retry_after_seconds` (default 5); startup warning for `workers>1`+metrics; README stance box. *Tests: header value; warning emission.*

**W1.9 — BuildKit + multi-stage (D15, D16-prep).** Remove `DOCKER_BUILDKIT=0`; generated Dockerfile becomes two-stage (builder: gcc/g++ + wheel install into venv; runtime: slim + venv copy + curl only); optional `build --platform`. Framework install: still wheel-copy in this wave (release pin lands in Wave 2 after v0.4.0 exists). *Tests: Dockerfile generation snapshot tests (two-stage structure, no compilers in runtime stage); docker-daemon-gated integration build test (existing skip pattern).*

**W1.10 — Docs sweep for Wave 1.** cli-reference/api-reference/configuration/multi-classifier updated for: dual payload shapes, deprecations, new tag format (read-side), `--json`, protocol spec. CHANGELOG `[0.4.0]` with a deprecation table.

**Wave 1 gate:** full verification battery (envelope matrix live-tested via TestClient + one live server smoke on `../test-classifier`); suite green; coverage ≥ 70%; CHANGELOG complete; **then tag v0.4.0** (user pushes tags — external action, user-executed).

---

## 5. Wave 2 — Breaking batch (→ v0.5.0)

**W2.1 — CLI restructure (D6).** `mlserver/cli/` package: `_context.py` (Typer callback resolving project dir + config once), `serve.py`, `build.py` (build/push/images/clean/run), `versioning.py` (tag/version), `project.py` (init/init-github/validate/doctor/status), `testing.py` (test/schema/list-classifiers). Commands become thin shells; orchestration logic moves into `container.py`/`version_control.py`/new `build_pipeline.py`. *Tests: CliRunner suite per command — help, happy path, 2 failure modes, `--json` where applicable. Coverage target: cli package ≥70% (from 8%).*

**W2.2 — Flag cleanup (D8).** `-p`=port only; `--path`/`-C` everywhere for project dir; `-v`=verbose only (run's volume becomes `--volume` long-only); `--config` uniform. Old flags error with a pointer to the new one (not silently reinterpreted). CHANGELOG migration table.

**W2.3 — Rename (D9).** pyproject: distribution `merve`, scripts `merve = mlserver.cli:main` + `mlserver = mlserver.cli:main` (alias, deprecation notice on startup via argv[0] check). README/docs command examples switch to `merve`. Generated Dockerfiles/workflows emit `merve`. PyPI availability check before any publish (not blocking: installs are git-based today). Python module stays `mlserver` (deferred decision, documented).

**W2.4 — Tag write-switch (D2/D3 breaking half).** `merve tag` writes `<classifier>/vX.Y.Z`; `--version-source` removed; `classifier.version` ignored (warning). Legacy tags remain readable forever (shim from W1.3). Optional `merve tag migrate --dry-run` helper listing legacy tags and their new-format equivalents (no auto-rewrite of git history).

**W2.5 — Build-once pipeline (D4).** New default: `merve build` builds ONE image per commit (`<repo>:<git-sha>` + OCI labels), no baked classifier; `merve push --classifier X` applies registry tag aliases (`<repo>:X-vN.N.N` → same digest) after validation; `merve run --classifier X` passes `MLSERVER_CLASSIFIER` env. `--per-classifier-image` flag preserves the old behavior for diverging deps. `github_actions.py` template v2: single build job + per-classifier alias/push matrix from the pushed tag. *Tests: pipeline unit tests with mocked docker (tag-alias sequences, digest reuse); daemon-gated integration: build once, run twice with different `MLSERVER_CLASSIFIER`, assert `/info` classifier differs while image ID matches.*

**W2.6 — Release-pinned images (D16).** Generated Dockerfile installs `merve==<installed version>` when the running merve is a release; wheel-copy fallback only for dev builds (with a WARNING in build output).

**W2.7 — `ruff format` finale (D19 completion).** Single atomic formatting commit across the repo AFTER all Wave 2 code lands; enable format check in CI and format hook in pre-commit.

**W2.8 — Docs + migration guide.** All docs to `merve` command, new tag format, build-once flow; `docs/migration-0.5.md` (old→new flags, tag format, image strategy, envelope reminder); CHANGELOG `[0.5.0]`.

**Wave 2 gate:** verification battery (CLI matrix via CliRunner, live serve smoke, build-once integration when daemon available, alias-tag correctness); suite green; **coverage ≥ 75% with CI ratchet raised to 75**; migration guide reviewed; then tag v0.5.0.

---

## 6. Test-coverage plan (cross-cutting)

| Module | Now | Target | How |
|--------|-----|--------|-----|
| cli (→ package) | 8% | ≥70% | CliRunner suite (W1.5 foundation, W2.1 full) |
| container.py | 56% | ≥70% | W0.4 rewrite of 16 stale skips + W1.9/W2.5 pipeline tests |
| version_control.py | 78% | ≥85% | W1.3 parametrized tag matrix |
| server.py | 67%* | ≥80% | W1.1 envelope matrix, W1.7 lifecycle, W1.8 warnings |
| predictor_loader/predictor | 76% | ≥90% | W1.7 isolation + collision tests |
| github_actions.py | 66% | ≥75% | W2.5 template v2 snapshot tests |
| **Overall** | **64%** | **≥75%** | CI ratchet: 60 (W0) → 70 (post-W1) → 75 (post-W2) |

*post-stabilization figures. Policies: every new module born ≥90%; every bug fix lands with a regression test (the stabilization sprint's verification scripts get promoted into `tests/` where still missing); skips carry reason+date and die after one sprint (D21); coverage floor only ratchets up, never down.

## 7. Execution model & risks

- Same model as the stabilization sprint: parallel subagents with disjoint file ownership per wave, orchestrator runs the verification battery between waves, review checkpoint before commits. Nothing is pushed anywhere without explicit go-ahead; tag pushes (v0.4.0/v0.5.0) are user-executed.
- **Risks:** rename fallout in user projects (mitigated: alias entry point + migration guide); legacy tag parsing edge cases (mitigated: parametrized matrix + shim kept forever); single-image rollout surprising existing deployments (mitigated: `--per-classifier-image` escape hatch + template v2 only emitted by `init-github`, existing workflows untouched until regenerated); mass-format merge pain (mitigated: formatting is the last commit of Wave 2).
- **Out of scope (recorded, deferred):** Python module rename `mlserver`→`merve`; prometheus multiprocess mode; PyPI publishing; auth on endpoints.
