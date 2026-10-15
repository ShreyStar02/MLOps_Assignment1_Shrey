"""Run the live evidence collection workflow and exit non-zero on failure."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from terminal_shot import capture_command

ROOT = Path(__file__).resolve().parents[2]
SCREENSHOTS = ROOT / "screenshots"
PYTHON = ROOT / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def run(cmd: list[str], *, cwd: Path | None = None, check: bool = True):
    print(f"+ {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(cwd or ROOT), text=True, capture_output=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    if check and result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}: {' '.join(cmd)}"
        )
    return result


def ensure_venv() -> None:
    venv_dir = ROOT / ".venv"
    pyvenv_cfg = venv_dir / "pyvenv.cfg"
    if not pyvenv_cfg.exists() or not PYTHON.exists():
        if venv_dir.exists():
            for child in sorted(venv_dir.iterdir(), reverse=True):
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()
        subprocess.run(
            [sys.executable, "-m", "venv", str(venv_dir)],
            cwd=str(ROOT),
            check=True,
        )
    run([str(PYTHON), "-m", "pip", "install", "-U", "pip"], check=True)
    run([str(PYTHON), "-m", "pip", "install", "-r", str(ROOT / "requirements.txt")], check=True)
    run([str(PYTHON), "-m", "pip", "install", "-r", str(ROOT / "requirements-dev.txt")], check=True)
    run([str(PYTHON), "-m", "playwright", "install", "chromium"], check=True)


def wait_for_http(url: str, timeout: int = 180) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            import urllib.request
            with urllib.request.urlopen(url, timeout=5) as response:  # noqa: S310
                if response.status < 500:
                    return
        except Exception:
            time.sleep(2)
    raise TimeoutError(f"Timed out waiting for {url}")


def render_mermaid_architecture() -> None:
    source = ROOT / "docs" / "architecture.mmd"
    source.write_text(
        """flowchart LR
  A[UCI dataset] --> B[download_data.py / clean]
  B --> C[EDA]
  B --> D[train.py: GridSearchCV + CV]
  D --> E[(MLflow tracking)]
  D --> F[models/model.joblib]
  F --> G[Docker image: FastAPI]
  H[GitHub Actions: lint, test, train, build] --> G
  G --> I[Kubernetes Deployment + LoadBalancer]
    I --> J["/predict"]
    I --> K["/metrics"] --> L[Prometheus] --> M[Grafana]
""",
        encoding="utf-8",
    )
    npx = shutil.which("npx")
    if npx is None:
        raise RuntimeError("npx is not available. Install Node.js and try again.")
    result = subprocess.run(
        [
            npx,
            "-y",
            "@mermaid-js/mermaid-cli",
            "-i",
            str(source),
            "-o",
            str(SCREENSHOTS / "architecture.png"),
        ],
        cwd=str(ROOT),
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr or result.stdout)


def main() -> int:
    SCREENSHOTS.mkdir(parents=True, exist_ok=True)
    ensure_venv()
    python = PYTHON

    run([str(python), "scripts/download_data.py"], check=True)
    run([str(python), "src/eda.py"], check=True)
    run([str(python), "src/train.py"], check=True)

    compose_file = ROOT / "docker-compose.yml"
    if compose_file.exists():
        subprocess.run(
            ["docker", "compose", "down", "-v", "--remove-orphans"],
            cwd=str(ROOT),
            check=False,
        )
    capture_command("pytest -v", SCREENSHOTS / "pytest_terminal.png", cwd=ROOT)
    run([str(python), "-m", "pytest", "-v"], check=True)
    run([str(python), "-m", "flake8", "src", "api", "scripts", "tests"], check=True)
    run([str(python), "scripts/capture/ci_evidence.py"], check=True)

    capture_command("python src/train.py", SCREENSHOTS / "train_terminal.png", cwd=ROOT)

    subprocess.run(
        ["docker", "build", "-t", "heart-api:latest", "."],
        cwd=str(ROOT),
        check=True,
    )
    capture_command(
        "docker build -t heart-api:latest .",
        SCREENSHOTS / "docker_build.png",
        cwd=ROOT,
    )

    container_name = "heart-api-capture"
    subprocess.run(
        ["docker", "rm", "-f", container_name],
        cwd=str(ROOT),
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    subprocess.run(
        [
            "docker",
            "run",
            "-d",
            "--rm",
            "--name",
            container_name,
            "-p",
            "8000:8000",
            "heart-api:latest",
        ],
        cwd=str(ROOT),
        check=True,
    )
    time.sleep(5)
    run([str(python), "scripts/sample_request.py", "http://localhost:8000", "5"], check=True)
    capture_command(
        "python scripts/sample_request.py http://localhost:8000 5",
        SCREENSHOTS / "docker_predict.png",
        cwd=ROOT,
    )
    capture_command(
        "docker logs heart-api-capture --tail 100",
        SCREENSHOTS / "api_logs.png",
        cwd=ROOT,
    )

    subprocess.run(
        ["docker", "rm", "-f", container_name],
        cwd=str(ROOT),
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    subprocess.run(["kubectl", "apply", "-f", "k8s/deployment.yaml"], cwd=str(ROOT), check=True)
    capture_command("kubectl get pods,svc -o wide", SCREENSHOTS / "k8s_pods_svc.png", cwd=ROOT)
    subprocess.run(
        [
            "kubectl",
            "wait",
            "--for=condition=Ready",
            "pod",
            "-l",
            "app=heart-api",
            "--timeout=180s",
        ],
        cwd=str(ROOT),
        check=True,
    )
    subprocess.run(
        [
            "kubectl",
            "port-forward",
            "svc/heart-api",
            "8001:80",
            "--address",
            "0.0.0.0",
        ],
        cwd=str(ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    time.sleep(8)
    wait_for_http("http://localhost:8001/health")
    capture_command(
        f'"{python}" scripts/sample_request.py http://localhost:8001 1',
        SCREENSHOTS / "k8s_predict.png",
        cwd=ROOT,
    )

    subprocess.run(["docker", "compose", "up", "-d", "--build"], cwd=str(ROOT), check=True)
    wait_for_http("http://localhost:9090/targets")
    wait_for_http("http://localhost:3000")
    run([str(python), "scripts/sample_request.py", "http://localhost:8000", "50"], check=True)

    run([str(python), "scripts/capture/web_shots.py"], check=True)
    render_mermaid_architecture()

    subprocess.run(
        ["docker", "rm", "-f", container_name],
        cwd=str(ROOT),
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    subprocess.run(
        ["docker", "compose", "down", "-v", "--remove-orphans"],
        cwd=str(ROOT),
        check=False,
    )
    print("All evidence capture steps succeeded.")
    print("Created files:")
    for path in sorted(SCREENSHOTS.glob("*.png")):
        print(path.name)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"run_all.py failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
