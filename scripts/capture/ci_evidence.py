"""Capture CI-run evidence from GitHub Actions using the gh CLI."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

from terminal_shot import capture_command

ROOT = Path(__file__).resolve().parents[2]
SCREENSHOTS = ROOT / "screenshots"


def require_gh() -> None:
    """Ensure GitHub CLI is installed and available."""
    if shutil.which("gh") is None:
        raise RuntimeError("GitHub CLI (gh) is not installed or not on PATH.")


def git_repo_slug() -> str:
    """Return the GitHub repo slug (owner/repo) from git remote.origin.url."""
    result = subprocess.run(
        ["git", "config", "--get", "remote.origin.url"],
        capture_output=True,
        text=True,
        check=True,
    )
    url = result.stdout.strip().removesuffix(".git")
    if ":" in url and "github.com" in url and not url.startswith("https://"):
        url = url.replace("git@github.com:", "https://github.com/")
    if url.startswith("https://github.com/"):
        return url.replace("https://github.com/", "")
    raise RuntimeError("Could not determine GitHub repo slug from git remote.origin.url")


def gh_json(*args: str) -> dict | list:
    """Run gh CLI with JSON output and return parsed payload."""
    result = subprocess.run(["gh", *args], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    payload = result.stdout.strip()
    if not payload:
        return {}
    return json.loads(payload)


def main() -> int:
    SCREENSHOTS.mkdir(parents=True, exist_ok=True)
    try:
        require_gh()
        repo = git_repo_slug()
        runs = gh_json(
            "run",
            "list",
            "--limit",
            "1",
            "--json",
            "databaseId,workflowName,status,conclusion,headBranch,url",
        )
    except Exception as exc:  # environment check fallback
        message = str(exc)
        capture_command(
            f"gh auth status || echo '{message}'",
            SCREENSHOTS / "ci_run.png",
            cwd=ROOT,
        )
        capture_command(
            f"gh run list --limit 1 || echo '{message}'",
            SCREENSHOTS / "ci_artifacts.png",
            cwd=ROOT,
        )
        print(message, file=sys.stderr)
        return 0

    if not runs:
        capture_command("gh run list --limit 1", SCREENSHOTS / "ci_run.png", cwd=ROOT)
        capture_command("gh run list --limit 1", SCREENSHOTS / "ci_artifacts.png", cwd=ROOT)
        return 0

    run_id = runs[0]["databaseId"]
    capture_command(
        "gh run view "
        f"{run_id} --json databaseId,workflowName,status,conclusion,headBranch,url,jobs",
        SCREENSHOTS / "ci_run.png",
        cwd=ROOT,
    )
    capture_command(
        f"gh api repos/{repo}/actions/runs/{run_id}/artifacts",
        SCREENSHOTS / "ci_artifacts.png",
        cwd=ROOT,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # final fallback
        print(f"ERR: {exc}", file=sys.stderr)
        raise SystemExit(1)
