# Migration Guide: → 0.5

This release (RFC 0001, Waves 1–2) renames the tool, changes the tag and image
model, and deprecates a few config surfaces. Everything deprecated in 0.4 still
works in 0.5 with a warning; the breaking changes below are the ones that need
action. The importable module stays `mlserver`, and `mlserver.yaml` and the
`MLSERVER_*` environment variables are unchanged.

## Breaking changes

### 1. The command is now `merve`

The distribution and console command were renamed from `mlserver-fastapi-wrapper`
/ `mlserver` to **`merve`** (the old name collided with Seldon's `mlserver`).

```bash
# before
mlserver serve
# after
merve serve
```

The `mlserver` command still exists as a **deprecated alias** for one release and
prints a warning. Update scripts and CI to `merve`. Generated Dockerfiles now use
`CMD ["merve", "serve", ...]`.

### 2. CLI short flags

- `-p` now means `--port` only. On `version`, `init`, `init-github`, and `doctor`
  it previously meant `--path`; use `-C` (or `--path`) there now.
- `-v` now means `--verbose` only. `run`'s volume flag is long-only: use
  `--volume`, not `-v`.
- `--classifier` / `-c` is unchanged.

Removed short flags are rejected with **exit code 2 and a pointer to the
replacement** (never a bare "No such option", never a silent reinterpretation).

### 3. Version tags are `<classifier>/vX.Y.Z`

`merve tag` now writes the canonical slash format, e.g. `sentiment/v1.2.0`.
The MLServer commit is no longer part of the tag name — it lives in the annotated
tag message and the image's OCI labels. **Legacy tags
(`<classifier>-vX.Y.Z-mlserver-<hash>`) are still read**, so existing history keeps
working; only newly created tags use the new format. Generated CI workflows now
trigger on `*/v*`.

### 4. Build once, deploy per classifier

For multi-classifier repos, `merve build` (no `--classifier`) now builds **one
image per git commit** (`<repo>:<sha>` + `<repo>:latest`) with no baked
classifier. Select the classifier at deploy time:

```bash
merve run --classifier sentiment            # sets MLSERVER_CLASSIFIER=sentiment
merve push --classifier sentiment           # tags <repo>:sentiment-vX.Y.Z on the same image
```

`merve push --classifier X` no longer rebuilds; it applies a registry tag alias to
the commit image after validating the `X/vX.Y.Z` tag at HEAD. To keep the old
one-image-per-classifier behavior (for classifiers with conflicting
dependencies), pass `--per-classifier-image`.

### 5. Removed `push --version-source`

Git tags are the canonical version source, so the pushed version always comes
from the classifier's release tag (the tag at HEAD, or with `--force` the
classifier's latest release tag). Passing `--version-source` exits with code 2
and a pointer to this change. Drop the flag; nothing replaces it.

### 6. Removed `global_config.yaml`

The `GlobalSettings` singleton and `global_config.yaml` were removed. Defaults now
live in code and are overridable via `mlserver.yaml` or the documented `MLSERVER_*`
environment variables. A leftover `global_config.yaml` is ignored with a warning.

## Deprecations (still work, will be removed)

- **Request `payload` wrapper.** Send input keys at the top level:
  `{"records": [...]}` instead of `{"payload": {"records": [...]}}`. Both work in
  0.5; the wrapper warns once per process.
- **`response_format: custom` and `api.extract_values`.** Use `standard` (the
  validated envelope) or `passthrough` (raw predictor output).
- **`classifier.version` in `mlserver.yaml`.** Git tags are the canonical version
  source; `merve tag` computes the next version from the latest tag. The field is
  display-only and logs a deprecation warning at config load. (`push
  --version-source` is *removed*, not deprecated — see breaking change 5.)

## Quick checklist

1. `pip install merve` (or reinstall) and switch `mlserver …` → `merve …`.
2. Fix any `-p`/`-v` short flags used for `--path`/`--volume`.
3. Drop the `payload` wrapper from request bodies.
4. Re-run `merve init-github` if you want the new build-once CI workflow.
5. Remove `global_config.yaml` and any `classifier.version` you were relying on.
