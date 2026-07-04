# Driving Merve from an Agent

Merve exposes a machine-readable command-line contract, so a coding agent
(Claude Code, Cursor, Codex, …) or a plain script can operate a classifier
repository without any special integration. **The `merve` CLI *is* the tool
interface** — there is no separate SDK and no MCP server to install. RFC 0002
(decision A6) deferred an MCP server indefinitely precisely because the CLI plus
JSON output plus stable exit codes plus fix-carrying errors already form a
complete tool surface for any shell-capable agent; wrapping it would only
duplicate it.

Every repository scaffolded by `merve init` also ships an **`AGENTS.md`** at its
root, carrying guidance specific to *that* repo — its classifier names, its
single- or multi-classifier shape, and the exact commands to run. That file is
generated from a template inside merve (never hand-written), so it can be
version-stamped and kept current: `merve init` creates it, `merve init-agents`
regenerates it for an existing repo, and `merve doctor` flags it when the stamp
goes stale. Read a repo's `AGENTS.md` first; this page is the general contract
behind it.

> Naming: `merve` is the tool and distribution name. The importable Python
> module is still `mlserver`, the config file is `mlserver.yaml`, and env vars
> are `MLSERVER_*` — only the tool was renamed.

## The contract

Every command that supports `--json` prints **exactly one JSON document to
stdout** and nothing else there. All human-facing output — rich tables,
progress, deprecation warnings, diagnostic notices — goes to **stderr**. So an
agent can run `merve <cmd> --json 2>/dev/null | jq …` and parse stdout without
stripping decoration. JSON keys are stable `snake_case`.

The process exit code is the other half of the contract:

| Code | Meaning |
|------|---------|
| `0` | Success |
| `1` | Command failed (invalid config, push failure, missing config, a failed doctor check, …) |
| `2` | Usage error (bad arguments or options — including every removed or renamed flag) |

Branch on the exit code: on `0` read stdout (the JSON result); on nonzero read
stderr (the human message — which carries the fix, see below).

## Machine-readable commands

Each of these accepts `--json` and emits one JSON document on stdout:

| Command | What the JSON contains |
|---------|------------------------|
| `merve version --json` (`--detailed`) | Classifier / model / API version metadata; `--detailed` adds the merve tool version, commit, and install source |
| `merve images --json` | Built container images, with `count` and `classifier` |
| `merve status --json` | Docker / config / Python / venv / GitHub-Actions environment status |
| `merve list-classifiers --json` | Classifier names and the `default_classifier` |
| `merve validate --json` | A `valid` flag plus per-check results; exits nonzero when the config is invalid |
| `merve tag --status --json` | Per-classifier version and tag status (status mode only) |
| `merve doctor --json` | Full diagnosis: each check with status, message, and fix suggestion, plus recommendations and a pass/warn/fail/skip summary |

Two more that agents lean on:

- **`merve schema`** prints a JSON Schema for `mlserver.yaml` to stdout (single,
  multi, or both; `--vscode` also wires up IDE validation). Use it to validate
  or generate config.
- **`merve test`** sends a prediction request to an already-running server
  (`--data`/`--file`, `--url`, `--endpoint`) — a scripted smoke test, not a
  config check.

The full option tables live in the
[CLI reference](./cli-reference.md#machine-readable-output---json); this page
does not repeat them.

## Errors carry the fix

The design rule (from RFC 0001, reaffirmed by RFC 0002): **a failing command
tells you what to do instead, in the error message on stderr.** On any nonzero
exit, read stderr — the recovery action is in the text, so an agent rarely has
to guess. Three patterns you will hit:

**1. A removed flag spelling exits `2` with the replacement.** Old short options
are rejected, never silently reinterpreted:

```
$ merve version -p ./sentiment
Invalid value for '-p': -p no longer means --path here (it is reserved for
--port across the CLI): use -C or --path for the project directory.
# exit 2  → retry with:  merve version -C ./sentiment
```

**2. An invalid classifier selection fails loudly, listing the valid names.** A
bad `--classifier` flag or `MLSERVER_CLASSIFIER` value never falls back
silently:

```
$ MLSERVER_CLASSIFIER=setiment merve serve       # typo
Error: unknown classifier 'setiment'. Available: sentiment, intent, fraud.
# fix the name and retry
```

**3. `merve push` off an untagged commit points you at `merve tag`.** A release
version comes only from a git tag:

```
$ merve push --classifier sentiment --registry gcr.io/project
Error: HEAD is not on a release tag for 'sentiment'.
Create one first:  merve tag patch --classifier sentiment   (then git push --tags)
# or pass --force to reuse the classifier's latest existing release tag
```

*(The messages above are illustrative — exact wording may vary, but the shape is
guaranteed: nonzero exit, the fix on stderr.)*

## Request shape

Send input keys at the **top level** of the JSON body:

```bash
curl -X POST localhost:8000/predict -H 'Content-Type: application/json' \
  -d '{"records": [{"age": 25, "sex": "male", "fare": 72.5}]}'
```

Accepted top-level keys: `records` / `instances` (a list of feature dicts),
`features` (a single feature dict), and `ndarray` / `inputs` (nested arrays,
which require `api.adapter: ndarray` or `auto`). The legacy `{"payload": {...}}`
wrapper still works but is **deprecated** (one warning per process, removal
targeted for 1.0) — do not emit it in new code.

You do not have to hard-code any of this: **the running server describes
itself.** `GET /openapi.json` carries request-body examples (the top-level
shapes first, the deprecated wrapper last) and the response schemas
(`PredictResponse`, `ProbaResponse`); `GET /info` lists the live endpoints and
the deployed version metadata. Fetch those to discover the exact contract of the
server you are talking to. Full schemas are in the
[API reference](./api-reference.md).

## Readiness

Point liveness, readiness, and startup probes at **`GET /healthz`**:

- `200 {"status": "ok", "model": "<PredictorClass>"}` once the model has loaded
  and the server can serve predictions.
- `503 {"status": "loading", "model": null}` before loading finishes — so a
  readiness probe never reports "ok" for a model that cannot yet answer.

Poll `/healthz` until it returns `200` before sending traffic. For live
capacity, `GET /status` reports concurrency-slot availability
(`prediction_slots_available`, `active_predictions`,
`max_concurrent_predictions`). Under load, prediction requests that exceed the
limit are rejected immediately with `503` and a `Retry-After` header — honor it
and back off (or scale out) rather than hammering.

## A typical agent session

1. **Validate** the config first: `merve validate --json` — parse `.valid`, stop
   on `false` and surface the per-check messages.
2. **Start** a server: `merve serve` (single-classifier) or
   `merve serve --classifier <name>` (multi); or run a built container with
   `merve run --classifier <name>`.
3. **Wait for readiness**: poll `GET /healthz` until it returns
   `200 {"status": "ok", ...}`.
4. **Predict**: `POST /predict` (or `/predict_proba`) with top-level keys, e.g.
   `{"records": [ … ]}`. Use `merve test` for a quick scripted probe.
5. **Release**: `merve tag <patch|minor|major> --classifier <name>` →
   `git push --tags` (CI builds the container); or, manually, `merve build` then
   `merve push --classifier <name> --registry <url>`.

For a multi-classifier repo the release rules are specific: **`merve build`
builds ONE commit image bundling every classifier**; **`merve push --classifier
X --registry <url>` applies registry tag aliases on that same image — no
rebuild**; and the classifier is chosen **at deploy time** by setting
`MLSERVER_CLASSIFIER` (or `--classifier`), with precedence flag > env var >
`default_classifier`. See [multi-classifier](./multi-classifier.md).

---

**See also:** the [documentation index](./INDEX.md) for everything else, and
RFC 0002 — [Agent-Facing Surface](./rfcs/0002-agent-surface-sprint.md) — for the
AGENTS.md template and the decisions (A4/A6) behind this page.
