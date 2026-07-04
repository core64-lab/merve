"""
GitHub Actions CI/CD workflow generation for MLServer projects.
"""

import re
import subprocess
from pathlib import Path
from typing import Any, Optional


def get_git_remote_info(project_path: str = ".") -> Optional[dict[str, str]]:
    """Extract GitHub repository information from git remote."""
    try:
        remote_url = (
            subprocess.check_output(
                ["git", "config", "--get", "remote.origin.url"],
                cwd=project_path,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )

        if not remote_url:
            return None

        # Parse GitHub repository from URL
        # Handle both HTTPS and SSH formats:
        # - https://github.com/owner/repo.git
        # - git@github.com:owner/repo.git

        if "github.com" not in remote_url:
            return None

        # Extract owner and repo
        if remote_url.startswith("git@"):
            # SSH format: git@github.com:owner/repo.git
            match = re.search(r"github\.com:([^/]+)/(.+?)(?:\.git)?$", remote_url)
        else:
            # HTTPS format: https://github.com/owner/repo.git
            match = re.search(r"github\.com/([^/]+)/(.+?)(?:\.git)?$", remote_url)

        if match:
            owner = match.group(1)
            repo = match.group(2)
            return {"owner": owner, "repo": repo, "url": remote_url, "is_github": True}

        return None
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _normalize_git_url(url: str) -> str:
    """Normalize a git remote URL to a pip-installable HTTPS form.

    SSH remotes (git@github.com:owner/repo.git) cannot be used with
    'pip install git+<url>' in CI, so convert them to
    https://github.com/owner/repo. HTTPS URLs pass through unchanged.
    """
    url = url.strip()
    match = re.match(r"^(?:ssh://)?git@([^:/]+)[:/](.+?)(?:\.git)?/?$", url)
    if match:
        host, repo_path = match.groups()
        return f"https://{host}/{repo_path}"
    return url


def get_mlserver_source_url(project_path: str = ".") -> str:
    """Determine the MLServer source URL for installation in CI/CD.

    Returns the git URL to use for installing MLServer in the workflow
    (always in pip-installable HTTPS form).
    """
    # Try to detect if we're in a development setup
    try:
        # Check if mlserver module has a git repository
        import mlserver

        mlserver_path = Path(mlserver.__file__).parent.parent

        if (mlserver_path / ".git").exists():
            # We're in a git repository - try to get remote URL
            try:
                remote_url = (
                    subprocess.check_output(
                        ["git", "config", "--get", "remote.origin.url"],
                        cwd=str(mlserver_path),
                        stderr=subprocess.DEVNULL,
                    )
                    .decode()
                    .strip()
                )

                if remote_url:
                    return _normalize_git_url(remote_url)
            except Exception:
                pass
    except Exception:
        pass

    # Default to the main repository
    return "https://github.com/core64-lab/merve.git"


def generate_build_and_push_workflow(
    repo_name: str,
    mlserver_source_url: str,
    python_version: str = "3.11",
    registry_config: Optional[dict] = None,
) -> str:
    """Generate the ml-classifier-container-build.yml workflow content.

    Args:
        repo_name: Name of the repository
        mlserver_source_url: Git URL for MLServer installation
        python_version: Python version to use
        registry_config: Registry configuration from mlserver.yaml deployment.registry section
    """

    # Get the installed merve version for workflow versioning (the build-time
    # _version_info snapshot goes stale on editable installs)
    try:
        from .container import _get_installed_mlserver_version

        mlserver_version = _get_installed_mlserver_version() or "unknown"
        # v3: build-once commit image + canonical tag trigger (RFC 0001 D4)
        workflow_version = "3.0"  # Increment on incompatible workflow changes
    except Exception:
        mlserver_version = "unknown"
        workflow_version = "3.0"

    # Extract registry configuration
    if registry_config is None:
        registry_config = {}

    registry_type = registry_config.get("type", "ghcr")

    # Extract ECR configuration
    ecr_config = registry_config.get("ecr", {})
    ecr_aws_region = ecr_config.get("aws_region", "eu-central-1")
    ecr_registry_id = ecr_config.get("registry_id")
    ecr_repository_prefix = ecr_config.get("repository_prefix", "ml-classifiers")

    # Validate ECR configuration if type is 'ecr'
    if registry_type == "ecr":
        if not ecr_registry_id:
            raise ValueError(
                "ECR registry type selected but registry_id is missing in mlserver.yaml.\n\n"
                "Please add registry_id to mlserver.yaml under deployment.registry.ecr:\n"
                "  registry_id: '123456789012'  # Your AWS account ID"
            )

    # Build AWS/ECR configuration strings for template
    # Role ARN: Configurable via mlserver.yaml (can use GitHub variable or baked value)
    # Registry ID: Bake from mlserver.yaml (fixed for organization)
    # Region: Use environment variable from workflow env section

    # Get GitHub variables configuration
    github_vars_config = registry_config.get("github_variables", {})
    role_arn_var_name = github_vars_config.get("aws_role_arn_var", "AWS_RUNNER_ROLE_ARN")
    role_arn_direct_value = github_vars_config.get("aws_role_arn_value")

    if role_arn_direct_value:
        # Bake direct value (less secure, but simpler for some setups)
        aws_role_arn_value = f'"{role_arn_direct_value}"'
        role_arn_source = "baked value (from mlserver.yaml)"
    else:
        # Use GitHub repository variable (recommended, more secure)
        aws_role_arn_value = f"${{{{ vars.{role_arn_var_name} }}}}"
        role_arn_source = f"GitHub variable '{role_arn_var_name}'"

    if ecr_registry_id:
        ecr_registry_value = f"'{ecr_registry_id}'"
        ecr_registry_url = f"{ecr_registry_id}.dkr.ecr.{ecr_aws_region}.amazonaws.com"
    else:
        ecr_registry_value = "'${{ vars.ECR_REGISTRY_ID }}'"
        ecr_registry_url = (
            "${{ vars.ECR_REGISTRY_ID }}.dkr.ecr." + ecr_aws_region + ".amazonaws.com"
        )

    # Build header with current registry configuration
    if registry_type == "ecr":
        # Build role ARN documentation based on configuration
        if role_arn_direct_value:
            role_arn_doc = f"""#   - Role ARN: {role_arn_direct_value} (baked from mlserver.yaml)
#
# ⚠️  Security Note: Role ARN is baked into workflow file. Consider using GitHub variable instead."""
        else:
            role_arn_doc = f"""#   - Role ARN: From {role_arn_source}
#
# Required GitHub Repository Variable:
#   - {role_arn_var_name}: IAM role ARN for OIDC (e.g., arn:aws:iam::123456789012:role/GitHubActionsRole)
#   Set at: Settings → Secrets and variables → Actions → Variables"""

        config_info = f"""# Registry Configuration: AWS ECR
#   - AWS Region: {ecr_aws_region} (from mlserver.yaml, set in env.AWS_REGION)
#   - Registry ID: {ecr_registry_id} (from mlserver.yaml, baked into workflow)
{role_arn_doc}
#   - Repository: {ecr_repository_prefix} (single repo for all classifiers)
#
# Image naming (build-once / deploy-many, RFC 0001 D4):
#   Commit image:     {{registry}}/{{repo}}:{{git-sha}}  (+ :latest) - one per commit, all classifiers
#   Classifier alias: {{registry}}/{{repo}}:{{classifier}}-v{{version}}  (+ {{classifier}}-latest)
#   Example alias: {ecr_registry_id}.dkr.ecr.{ecr_aws_region}.amazonaws.com/myrepo:sentiment-v1.0.0
#
# Note: The ECR repository must be pre-created.
#
# To update configuration: modify mlserver.yaml and regenerate with merve init-github --force"""
    else:
        config_info = """# Registry Configuration: GitHub Container Registry (GHCR)
#   - Uses GITHUB_TOKEN for authentication
#   - Repository: ghcr.io/{owner}/{repo}
#
# Image naming (build-once / deploy-many, RFC 0001 D4):
#   Commit image:     ghcr.io/{owner}/{repo}:{git-sha}  (+ :latest) - one per commit, all classifiers
#   Classifier alias: ghcr.io/{owner}/{repo}:{classifier}-v{version}  (+ {classifier}-latest)
#   Example alias: ghcr.io/myorg/myrepo:sentiment-v1.0.0"""

    template = f"""# ============================================================================
# ML Classifier Container Build & Publish Workflow (v3: build-once, RFC 0001 D4)
# ============================================================================
# Generated by MLServer {mlserver_version}
# Workflow version: {workflow_version} (build-once commit image + canonical tags)
#
{config_info}
#
# Build-once / deploy-many: ONE commit image is built and pushed per release,
# then the classifier's release is applied as registry tag aliases on the SAME
# image digest (no rebuild). The classifier is selected at run time via the
# MLSERVER_CLASSIFIER environment variable.
#
# To change registry configuration, update mlserver.yaml and regenerate:
#   merve init-github --force
# ============================================================================
name: Build & Publish ML Classifier Container

on:
  push:
    tags:
      - '*/v*'  # canonical release tag <classifier>/vX.Y.Z (RFC 0001 D2)

permissions:
  contents: read
  packages: write   # required to push to GHCR
  id-token: write   # required for ECR OIDC authentication

jobs:
  build-and-push:
    runs-on: ubuntu-latest

    env:
      # Registry type (configured from mlserver.yaml)
      REGISTRY_TYPE: "{registry_type}"
      AWS_REGION: "{ecr_aws_region}"

    steps:
      - name: Checkout
        uses: actions/checkout@v4
        with:
          ref: ${{{{ github.ref }}}}
          fetch-depth: 0
          fetch-tags: true

      # Step 1: Parse the canonical release tag <classifier>/vX.Y.Z
      - name: Parse Canonical Tag
        id: parse
        shell: bash
        run: |
          set -euo pipefail
          TAG_NAME="${{GITHUB_REF#refs/tags/}}"
          echo "Full tag: $TAG_NAME"

          # Canonical tag format (RFC 0001 D2): <classifier>/vX.Y.Z
          # classifier = everything before the first '/v'; version = the rest.
          CLASSIFIER="${{TAG_NAME%%/v*}}"
          VERSION="${{TAG_NAME##*/v}}"
          COMMIT_SHA="${{GITHUB_SHA:0:7}}"

          if [ "$CLASSIFIER" = "$TAG_NAME" ] || [ -z "$VERSION" ]; then
            echo "❌ Tag '$TAG_NAME' is not in canonical <classifier>/vX.Y.Z form"
            exit 1
          fi

          echo "classifier=$CLASSIFIER"   >> "$GITHUB_OUTPUT"
          echo "version=$VERSION"         >> "$GITHUB_OUTPUT"
          echo "commit_sha=$COMMIT_SHA"   >> "$GITHUB_OUTPUT"
          echo "full_tag=$TAG_NAME"       >> "$GITHUB_OUTPUT"

          echo "Parsed: classifier=$CLASSIFIER version=$VERSION commit=$COMMIT_SHA"

      # Step 2: Set up Python
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '{python_version}'

      # Step 3: Install MLServer (canonical tags do not pin the framework commit)
      - name: Install MLServer
        shell: bash
        run: |
          set -euo pipefail
          python -m venv mlserver-venv
          source mlserver-venv/bin/activate
          pip install --upgrade pip
          pip install "git+{mlserver_source_url}"

          # Verify installation (run from /tmp to avoid config parsing)
          cd /tmp && merve version --json && cd -

      # Step 4: Set up Docker Buildx
      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      # Step 5: Build the commit image ONCE (build-once, RFC 0001 D4)
      - name: Build Commit Image
        id: build
        shell: bash
        run: |
          set -euo pipefail
          source mlserver-venv/bin/activate
          echo "Building single commit image bundling all classifiers"
          merve build
          echo "Commit image built: {repo_name}:latest"

      # Step 6: Smoke test the commit image for the released classifier
      - name: Smoke Test Commit Image
        shell: bash
        run: |
          set -euo pipefail
          CLASSIFIER="${{{{ steps.parse.outputs.classifier }}}}"
          echo "Running commit image with MLSERVER_CLASSIFIER=$CLASSIFIER"
          docker run -d -p 8000:8000 -e "MLSERVER_CLASSIFIER=$CLASSIFIER" \\
            --name smoke-test "{repo_name}:latest"

          # Wait for the health endpoint (disable exit-on-error for the retry loop)
          set +e
          for i in {{1..30}}; do
            if curl -fsS "http://localhost:8000/healthz" >/dev/null 2>&1; then
              echo "✅ Health check passed after ${{i}} attempts"
              break
            fi
            echo "Attempt ${{i}}/30 failed, retrying..."
            sleep 2
          done
          set -e

          echo "Final health check:"
          curl -f "http://localhost:8000/healthz"

          echo "Container logs:"
          docker logs smoke-test | tail -20

          docker rm -f smoke-test
          echo "✅ Smoke test passed"

      # Step 7a: Configure AWS credentials (ECR only)
      - name: 🔑 Configure AWS credentials
        if: env.REGISTRY_TYPE == 'ecr'
        uses: aws-actions/configure-aws-credentials@v5
        with:
          role-to-assume: {aws_role_arn_value}
          role-session-name: GitHubActions-${{{{ github.job }}}}
          aws-region: ${{{{ env.AWS_REGION }}}}

      # Step 7b: Login to Amazon ECR (ECR only)
      - name: 🔐 Login to Amazon ECR
        if: env.REGISTRY_TYPE == 'ecr'
        uses: aws-actions/amazon-ecr-login@v2
        with:
          registries: {ecr_registry_value}

      # Step 7c: Log in to GHCR (GHCR only)
      - name: 🔐 Log in to GHCR
        if: env.REGISTRY_TYPE == 'ghcr'
        uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{{{ github.actor }}}}
          password: ${{{{ secrets.GITHUB_TOKEN }}}}

      # Step 8: Push the commit image ONCE (raw <git-sha> + latest)
      - name: Push Commit Image
        id: push
        shell: bash
        run: |
          set -euo pipefail
          REGISTRY_TYPE="${{{{ env.REGISTRY_TYPE }}}}"
          COMMIT_SHA="${{{{ steps.parse.outputs.commit_sha }}}}"

          if [ "$REGISTRY_TYPE" = "ecr" ]; then
            REGISTRY_BASE="{ecr_registry_url}"
          else
            OWNER_LOWER="$(echo "${{{{ github.repository_owner }}}}" | tr '[:upper:]' '[:lower:]')"
            REGISTRY_BASE="ghcr.io/${{OWNER_LOWER}}"
          fi
          echo "Registry base: $REGISTRY_BASE"

          # One commit image per git commit, shared by every classifier
          docker tag "{repo_name}:latest" "${{REGISTRY_BASE}}/{repo_name}:${{COMMIT_SHA}}"
          docker tag "{repo_name}:latest" "${{REGISTRY_BASE}}/{repo_name}:latest"
          docker push "${{REGISTRY_BASE}}/{repo_name}:${{COMMIT_SHA}}"
          docker push "${{REGISTRY_BASE}}/{repo_name}:latest"

          echo "registry_base=$REGISTRY_BASE" >> "$GITHUB_OUTPUT"
          echo "commit_image=${{REGISTRY_BASE}}/{repo_name}:${{COMMIT_SHA}}" >> "$GITHUB_OUTPUT"
          echo "✅ Pushed commit image to $REGISTRY_TYPE"

      # Step 9: Apply the per-classifier release alias on the SAME image digest
      - name: Apply Per-Classifier Release Alias
        id: alias
        shell: bash
        run: |
          set -euo pipefail
          REGISTRY_BASE="${{{{ steps.push.outputs.registry_base }}}}"
          CLASSIFIER="${{{{ steps.parse.outputs.classifier }}}}"
          VERSION="${{{{ steps.parse.outputs.version }}}}"

          # Registry tag aliases derived from the pushed commit image - no rebuild,
          # same digest (RFC 0001 D4 / W2.5): <repo>:<classifier>-v<version> and
          # <repo>:<classifier>-latest.
          ALIAS_VERSION="${{REGISTRY_BASE}}/{repo_name}:${{CLASSIFIER}}-v${{VERSION}}"
          ALIAS_LATEST="${{REGISTRY_BASE}}/{repo_name}:${{CLASSIFIER}}-latest"

          docker tag "{repo_name}:latest" "$ALIAS_VERSION"
          docker tag "{repo_name}:latest" "$ALIAS_LATEST"
          docker push "$ALIAS_VERSION"
          docker push "$ALIAS_LATEST"

          echo "alias_version=$ALIAS_VERSION" >> "$GITHUB_OUTPUT"
          echo "alias_latest=$ALIAS_LATEST"   >> "$GITHUB_OUTPUT"
          echo "✅ Applied release aliases for $CLASSIFIER v$VERSION"

      # Step 10: Summary
      - name: Build Summary
        shell: bash
        run: |
          echo "## 🎉 Build-once & Publish Complete" >> $GITHUB_STEP_SUMMARY
          echo "" >> $GITHUB_STEP_SUMMARY
          echo "**Classifier:** ${{{{ steps.parse.outputs.classifier }}}}" >> $GITHUB_STEP_SUMMARY
          echo "**Version:** ${{{{ steps.parse.outputs.version }}}}" >> $GITHUB_STEP_SUMMARY
          echo "**Tag:** ${{{{ steps.parse.outputs.full_tag }}}}" >> $GITHUB_STEP_SUMMARY
          echo "" >> $GITHUB_STEP_SUMMARY
          echo "**Commit image:** ${{{{ steps.push.outputs.commit_image }}}}" >> $GITHUB_STEP_SUMMARY
          echo "**Release aliases:**" >> $GITHUB_STEP_SUMMARY
          echo "- ${{{{ steps.alias.outputs.alias_version }}}}" >> $GITHUB_STEP_SUMMARY
          echo "- ${{{{ steps.alias.outputs.alias_latest }}}}" >> $GITHUB_STEP_SUMMARY
"""

    return template


def init_github_actions(
    project_path: str = ".",
    python_version: str = "3.11",
    registry: str = "ghcr.io",
    force: bool = False,
) -> tuple[bool, str, dict[str, str]]:
    """
    Initialize GitHub Actions CI/CD for the project.

    Args:
        project_path: Path to the project directory
        python_version: Python version to use
        registry: Registry URL (deprecated - use mlserver.yaml instead)
        force: Force overwrite of existing workflow files

    Returns:
        Tuple of (success, message, files_created)
    """
    project_path = Path(project_path).resolve()

    # Check if we're in a git repository
    if not (project_path / ".git").exists():
        return False, "Not a git repository. Initialize git first with: git init", {}

    # Get git remote info
    git_info = get_git_remote_info(str(project_path))
    if not git_info:
        return (
            False,
            "No GitHub remote found. Add GitHub remote first with: git remote add origin <url>",
            {},
        )

    # Get repository name from directory or git
    from .version import get_repository_name

    repo_name = get_repository_name(str(project_path))

    # Get MLServer source URL
    mlserver_url = get_mlserver_source_url(str(project_path))

    # Read registry configuration from mlserver.yaml if it exists
    registry_config = None
    mlserver_yaml_path = project_path / "mlserver.yaml"
    if mlserver_yaml_path.exists():
        try:
            import yaml

            with open(mlserver_yaml_path) as f:
                config = yaml.safe_load(f)
                if config and "deployment" in config and "registry" in config["deployment"]:
                    registry_config = config["deployment"]["registry"]
        except Exception:
            # If we can't read the config, fall back to defaults
            pass

    # Create .github/workflows directory
    workflows_dir = project_path / ".github" / "workflows"
    workflows_dir.mkdir(parents=True, exist_ok=True)

    # Check if ml-classifier-container-build.yml already exists
    workflow_file = workflows_dir / "ml-classifier-container-build.yml"
    files_created = {}

    if workflow_file.exists() and not force:
        return False, f"Workflow file already exists: {workflow_file}\nUse --force to overwrite", {}

    # Generate and write workflow
    workflow_content = generate_build_and_push_workflow(
        repo_name=repo_name,
        mlserver_source_url=mlserver_url,
        python_version=python_version,
        registry_config=registry_config,
    )

    with open(workflow_file, "w") as f:
        f.write(workflow_content)

    files_created["workflow"] = str(workflow_file.relative_to(project_path))

    # Create a README for the workflows directory
    readme_file = workflows_dir / "README.md"
    if not readme_file.exists():
        readme_content = r"""# GitHub Actions Workflows

## ml-classifier-container-build.yml

Builds ONE container image per commit (build-once / deploy-many, RFC 0001 D4) and
publishes it, then applies the released classifier as registry tag aliases on the
same image digest when a canonical release tag is pushed.

### Supported Registries

1. **GHCR (GitHub Container Registry)** - Default, no additional setup required
2. **ECR (AWS Elastic Container Registry)** - Configured via mlserver.yaml

### Trigger

Workflow is triggered when a canonical release tag matching the pattern `*/v*` is
pushed (RFC 0001 D2).

Example tag: `sentiment/v1.0.0`

### What it does

1. Parses the canonical tag to extract:
   - Classifier name (the part before `/v`)
   - Version number (the part after `/v`)

2. Installs MLServer

3. Builds the single commit image bundling every classifier (no baked classifier)

4. Smoke-tests the image by running it with `MLSERVER_CLASSIFIER=<classifier>`

5. Authenticates with the configured registry (GHCR or ECR)

6. Pushes the commit image once (`<repo>:<git-sha>` and `<repo>:latest`), then
   applies the release as registry tag aliases on the SAME digest:
   - `<repo>:<classifier>-v<version>`
   - `<repo>:<classifier>-latest`

### Usage

Create and push a canonical release tag using:

```bash
merve tag patch --classifier <classifier-name>
git push --tags
```

The workflow will automatically build and publish your container.

### Configuration

Registry configuration is read from `mlserver.yaml` under `deployment.registry` and **baked into the generated workflow file**.

#### GHCR (Default)

No additional setup required. Uses `GITHUB_TOKEN` automatically.

```yaml
deployment:
egistry:
    type: "ghcr"
```

#### ECR

**Step 1**: Configure ECR settings in `mlserver.yaml` under `deployment.registry`:

```yaml
deployment:
  registry:
    type: "ecr"
    ecr:
      aws_region: "eu-central-1"          # AWS region for ECR (required)
      registry_id: "123456789012"         # AWS account ID (required)
      repository_prefix: "ml-classifiers" # Repository prefix (optional, default: "ml-classifiers")
```

**Important**: The build-once commit image and its per-classifier release aliases
share one repository (named after your repo); every alias resolves to the same
image digest:
- Commit image: `<registry>/<repo>:<git-sha>` (+ `:latest`) - must be pre-created in ECR
- Release alias: `<registry>/<repo>:<classifier>-v<version>` (+ `<classifier>-latest`)
- Example alias: `123456789012.dkr.ecr.eu-central-1.amazonaws.com/myrepo:sentiment-v1.0.0`

**Step 2**: Configure GitHub variables (optional - customize variable name in mlserver.yaml):

```yaml
deployment:
  registry:
    github_variables:
      aws_role_arn_var: "AWS_DEV_ROLE_ARN"  # Customize variable name (default: AWS_RUNNER_ROLE_ARN)
```

**Step 3**: Set GitHub repository variable for IAM role:
- Go to: Settings → Secrets and variables → Actions → Variables
- Add variable: `{your_configured_variable_name}` = `arn:aws:iam::123456789012:role/GitHubActionsRole`

**Step 4**: Generate/update the workflow:

```bash
merve init-github --force
git add .github && git commit -m "Update workflow for ECR"
git push
```

**Note**: Registry ID, region, and variable names are configured in mlserver.yaml. This allows environment-specific or organization-specific naming conventions.

### Permissions

- **GHCR**: Requires `packages: write` (automatically granted with `GITHUB_TOKEN`)
- **ECR**: Requires `id-token: write` for OIDC authentication and GitHub repository variable (name configured in mlserver.yaml)
"""
        with open(readme_file, "w") as f:
            f.write(readme_content)

        files_created["readme"] = str(readme_file.relative_to(project_path))

    # Build registry-specific instructions
    registry_type = registry_config.get("type", "ghcr") if registry_config else "ghcr"
    if registry_type == "ecr":
        ecr_info = registry_config.get("ecr", {})
        gh_vars = registry_config.get("github_variables", {})
        var_name = gh_vars.get("aws_role_arn_var", "AWS_RUNNER_ROLE_ARN")
        baked_value = gh_vars.get("aws_role_arn_value")

        if baked_value:
            role_arn_instruction = f"""⚠️  Role ARN baked into workflow: {baked_value}
   Security: Consider using GitHub variable instead for better security"""
        else:
            role_arn_instruction = f"""⚠️  IMPORTANT: Set GitHub repository variable:
   - {var_name}: Your IAM role ARN for OIDC authentication

   Go to: Settings → Secrets and variables → Actions → Variables"""

        registry_instructions = f"""
Registry: AWS ECR
  - Region: {ecr_info.get("aws_region", "eu-central-1")} (baked into workflow)
  - Registry ID: {ecr_info.get("registry_id", "N/A")} (baked into workflow)

{role_arn_instruction}
"""
    else:
        registry_instructions = "\nRegistry: GitHub Container Registry (GHCR)"

    success_message = f"""✅ GitHub Actions CI/CD initialized successfully!

Repository: {git_info["owner"]}/{git_info["repo"]}
MLServer Source: {mlserver_url}
{registry_instructions}

Created files:
  - {files_created.get("workflow", "N/A")}
  - {files_created.get("readme", "N/A")}

Next steps:
  1. Review the generated workflow file
  2. Commit the changes: git add .github && git commit -m "Add CI/CD workflow"
  3. Push to GitHub: git push
  4. Create a version tag: merve tag patch --classifier <name>
  5. Push the tag: git push --tags

The workflow will automatically build and publish your container!
"""

    return True, success_message, files_created


def check_github_actions_setup(project_path: str = ".") -> bool:
    """Check if GitHub Actions workflow is set up."""
    workflow_file = (
        Path(project_path) / ".github" / "workflows" / "ml-classifier-container-build.yml"
    )
    return workflow_file.exists()


def parse_workflow_version(workflow_path: Path) -> tuple[Optional[str], Optional[str]]:
    """
    Parse MLServer version and workflow version from workflow file.

    Returns:
        Tuple of (mlserver_version, workflow_version) or (None, None) if not found
    """
    try:
        with open(workflow_path) as f:
            content = f.read()

        # Look for version markers in first few lines
        mlserver_version = None
        workflow_version = None

        for line in content.split("\n")[:10]:  # Check first 10 lines
            if "# Generated by MLServer" in line:
                # Extract: # Generated by MLServer 0.3.2.dev9
                parts = line.split("MLServer")
                if len(parts) > 1:
                    mlserver_version = parts[1].strip()
            elif "# Workflow version:" in line:
                # Extract: # Workflow version: 1.0 (with optional comment)
                # Extract only the version number, ignore comments in parentheses
                parts = line.split(":")
                if len(parts) > 1:
                    version_str = parts[1].strip()
                    # Extract just the version number (before any space or parenthesis)
                    version_parts = version_str.split()
                    if version_parts:
                        workflow_version = version_parts[0]

        return mlserver_version, workflow_version
    except Exception:
        return None, None


def validate_workflow_compatibility(
    project_path: str = ".", strict: bool = False
) -> tuple[bool, Optional[str], dict[str, str]]:
    """
    Validate that GitHub Actions workflow is compatible with current MLServer version.

    Args:
        project_path: Path to project
        strict: If True, fail on any version mismatch. If False, only warn.

    Returns:
        Tuple of (is_valid, warning_message, details)
    """
    workflow_file = (
        Path(project_path) / ".github" / "workflows" / "ml-classifier-container-build.yml"
    )

    if not workflow_file.exists():
        return False, "Workflow file does not exist", {}

    # Get current installed merve version (same source as workflow generation)
    try:
        from .container import _get_installed_mlserver_version

        current_mlserver_version = _get_installed_mlserver_version() or "unknown"
        current_workflow_version = "3.0"  # Should match generate_build_and_push_workflow()
    except Exception:
        current_mlserver_version = "unknown"
        current_workflow_version = "3.0"

    # Parse versions from workflow file
    file_mlserver_version, file_workflow_version = parse_workflow_version(workflow_file)

    details = {
        "current_mlserver": current_mlserver_version,
        "current_workflow": current_workflow_version,
        "file_mlserver": file_mlserver_version or "unknown",
        "file_workflow": file_workflow_version or "unknown",
    }

    # If no version markers found, this is an old/manually created workflow
    if file_mlserver_version is None or file_workflow_version is None:
        warning = (
            "Workflow file is missing version markers "
            "(generated by old MLServer or manually created). "
            "Consider regenerating with: merve init-github --force"
        )
        return not strict, warning if not strict else None, details

    # Check workflow version compatibility
    if file_workflow_version != current_workflow_version:
        warning = (
            f"Workflow version mismatch! File has v{file_workflow_version}, "
            f"but current MLServer expects v{current_workflow_version}. "
            "Regenerate with: merve init-github --force"
        )
        return False, warning, details

    # Workflow version matches - compatible
    return True, None, details


def check_workflow_mlserver_url(project_path: str = ".") -> tuple[Optional[str], Optional[str]]:
    """
    Extract the MLServer URL from an existing workflow file.

    Returns:
        Tuple of (url_in_workflow, expected_url) or (None, None) if not found
    """
    workflow_file = (
        Path(project_path) / ".github" / "workflows" / "ml-classifier-container-build.yml"
    )

    if not workflow_file.exists():
        return None, None

    try:
        with open(workflow_file) as f:
            content = f.read()

        # Look for the pip install line with git URL
        # Pattern: pip install "git+https://github.com/xxx/yyy.git@...
        # The greedy match captures up to the LAST '@' on the line so that
        # legacy SSH URLs (git+git@github.com:owner/repo.git@ref) which
        # contain an extra '@' are not truncated.
        import re

        match = re.search(r'pip install "git\+(.+)@', content)
        url_in_workflow = match.group(1) if match else None

        # Get expected URL
        expected_url = get_mlserver_source_url(project_path)

        return url_in_workflow, expected_url
    except Exception:
        return None, None


def validate_workflow_comprehensive(
    project_path: str = ".",
) -> tuple[bool, list[str], dict[str, Any]]:
    """
    Comprehensive workflow validation checking version, URL, and configuration.

    Returns:
        Tuple of (is_valid, list_of_warnings, details_dict)
    """
    warnings = []
    details = {}
    is_valid = True

    workflow_file = (
        Path(project_path) / ".github" / "workflows" / "ml-classifier-container-build.yml"
    )

    if not workflow_file.exists():
        return False, ["Workflow file does not exist"], {"exists": False}

    details["exists"] = True

    # Check 1: Version compatibility
    version_valid, version_warning, version_details = validate_workflow_compatibility(project_path)
    details.update(version_details)
    if not version_valid:
        is_valid = False
        if version_warning:
            warnings.append(version_warning)

    # Check 2: MLServer URL
    url_in_workflow, expected_url = check_workflow_mlserver_url(project_path)
    details["url_in_workflow"] = url_in_workflow
    details["expected_url"] = expected_url

    if url_in_workflow and expected_url:
        # Normalize URLs for comparison (SSH vs HTTPS form, trailing slash,
        # and .git suffix)
        url_workflow_normalized = (
            _normalize_git_url(url_in_workflow).rstrip("/").removesuffix(".git")
        )
        url_expected_normalized = _normalize_git_url(expected_url).rstrip("/").removesuffix(".git")

        if url_workflow_normalized != url_expected_normalized:
            is_valid = False
            warnings.append(
                f"MLServer URL mismatch! Workflow uses '{url_in_workflow}' "
                f"but current MLServer is from '{expected_url}'. "
                "Regenerate with: merve init-github --force"
            )

    # Check 3: Look for known outdated patterns
    try:
        with open(workflow_file) as f:
            content = f.read()

        # Check for old naming scheme (IMAGE_BASE with /)
        if 'IMAGE_BASE="${REPOSITORY_PREFIX}/${CLASSIFIER}"' in content:
            is_valid = False
            warnings.append(
                "Workflow uses old ECR naming scheme (prefix/classifier). "
                "Regenerate with: merve init-github --force"
            )

    except Exception:
        pass

    return is_valid, warnings, details
