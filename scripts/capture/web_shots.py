"""Capture browser-based evidence for the MLflow, Prometheus, Grafana and API."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import urlopen

from playwright.async_api import Page, TimeoutError as PlaywrightTimeoutError, async_playwright

ROOT = Path(__file__).resolve().parents[2]
SCREENSHOTS = ROOT / "screenshots"


def wait_for_http(url: str, timeout: int = 180) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urlopen(url, timeout=5) as response:  # noqa: S310
                if response.status < 500:
                    return
        except Exception:
            time.sleep(2)
    raise TimeoutError(f"Timed out waiting for {url}")


def wait_for_prometheus_target(timeout: int = 180) -> None:
    deadline = time.time() + timeout
    url = "http://localhost:9090/api/v1/targets"
    while time.time() < deadline:
        try:
            with urlopen(url, timeout=5) as response:  # noqa: S310
                payload = json.loads(response.read())
            targets = payload.get("data", {}).get("activeTargets", [])
            if any(
                target.get("labels", {}).get("job") == "heart-api"
                and target.get("health") == "up"
                for target in targets
            ):
                return
        except Exception:
            time.sleep(2)
    raise TimeoutError("Prometheus did not report the heart-api target as up")


def find_python() -> Path:
    venv = ROOT / ".venv"
    if os.name == "nt":
        candidate = venv / "Scripts" / "python.exe"
    else:
        candidate = venv / "bin" / "python"
    if candidate.exists():
        return candidate
    return Path(sys.executable)


def start_process(command: list[str], cwd: Path | None = None):
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(ROOT))
    return subprocess.Popen(
        command,
        cwd=str(cwd or ROOT),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )


async def capture_page(
    page: Page,
    url: str,
    screenshot_path: Path,
    *,
    wait_for_selector: str | None = None,
) -> None:
    screenshot_path.parent.mkdir(parents=True, exist_ok=True)
    await page.goto(url, wait_until="domcontentloaded")
    await page.locator("body").wait_for(state="visible")
    if wait_for_selector:
        await page.wait_for_selector(wait_for_selector, timeout=120000)
    await page.screenshot(path=str(screenshot_path), full_page=True)


async def main() -> int:
    SCREENSHOTS.mkdir(parents=True, exist_ok=True)
    for stale_name in (
        "mlflow_compare.png",
        "mlflow_run.png",
        "prometheus_graph.png",
    ):
        (SCREENSHOTS / stale_name).unlink(missing_ok=True)
    python = find_python()

    if not (ROOT / "mlflow.db").exists():
        subprocess.run([str(python), "src/train.py"], cwd=str(ROOT), check=True)

    proc = start_process([
        str(python),
        "-m",
        "mlflow",
        "ui",
        "--backend-store-uri",
        "sqlite:///mlflow.db",
        "--port",
        "5000",
        "--workers",
        "1",
    ])
    try:
        wait_for_http("http://localhost:5000")
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page(viewport={"width": 1600, "height": 900})

            runs = subprocess.run(
                [
                    str(python),
                    "-c",
                    "import json; from pathlib import Path; import mlflow; "
                    "mlflow.set_tracking_uri('sqlite:///mlflow.db'); "
                    "experiment = mlflow.get_experiment_by_name("
                    "'heart-disease-classification'); "
                    "metadata = json.loads(Path('models/metadata.json').read_text()); "
                    "print(str(experiment.experiment_id) + ';' + metadata['run_id'])",
                ],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
                check=True,
            )
            experiment_id, run_id = runs.stdout.strip().split(";", maxsplit=1)
            runs_url = (
                "http://localhost:5000/#/experiments/"
                f"{experiment_id}/runs?workflowType=genai"
            )
            await capture_page(
                page,
                runs_url,
                SCREENSHOTS / "mlflow_runs.png",
                wait_for_selector="text=logistic_regression",
            )
            run_detail_url = (
                f"http://localhost:5000/#/experiments/{experiment_id}/runs/{run_id}"
            )
            await capture_page(
                page,
                run_detail_url,
                SCREENSHOTS / "mlflow_run_detail.png",
                wait_for_selector="text=Metrics",
            )

            await browser.close()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()

    compose = ROOT / "docker-compose.yml"
    if compose.exists():
        try:
            wait_for_http("http://localhost:9090/targets")
            wait_for_http("http://localhost:3000")
            wait_for_http("http://localhost:8000/docs")
            wait_for_prometheus_target()
        except Exception:
            subprocess.run(
                ["docker", "compose", "up", "-d", "--build"],
                cwd=str(ROOT),
                check=True,
            )
        try:
            wait_for_http("http://localhost:9090/targets")
            wait_for_http("http://localhost:3000")
            wait_for_http("http://localhost:8000/docs")
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page(viewport={"width": 1600, "height": 900})
                await capture_page(
                    page,
                    "http://localhost:9090/targets",
                    SCREENSHOTS / "prometheus.png",
                    wait_for_selector="text=1 / 1 up",
                )
                await page.goto(
                    "http://localhost:9090/query?g0.expr=api_requests_total",
                    wait_until="domcontentloaded",
                )
                await page.locator("body").wait_for(state="visible")
                await page.wait_for_selector("text=Result series", timeout=120000)
                await page.screenshot(
                    path=str(SCREENSHOTS / "prometheus_query.png"),
                    full_page=True,
                )

                await page.goto(
                    "http://localhost:3000/login",
                    wait_until="domcontentloaded",
                )
                await page.locator("body").wait_for(state="visible")
                await page.fill('input[name="user"]', "admin")
                await page.fill('input[name="password"]', "admin")
                await page.click('button:has-text("Log in")')
                try:
                    await page.wait_for_selector(
                        'button:has-text("Skip")',
                        timeout=10000,
                    )
                    await page.click('button:has-text("Skip")')
                except PlaywrightTimeoutError:
                    pass
                await page.goto(
                    "http://localhost:3000/d/heart-api/heart-disease-api"
                    "?orgId=1&from=now-15m&to=now&timezone=browser&refresh=5s",
                    wait_until="domcontentloaded",
                )
                await page.wait_for_selector("text=Requests per second", timeout=120000)
                await page.wait_for_selector("text=Predictions by label", timeout=120000)
                await page.wait_for_selector("text=lower_risk", timeout=120000)
                await page.wait_for_selector("text=higher_risk", timeout=120000)
                await page.screenshot(
                    path=str(SCREENSHOTS / "grafana.png"),
                    full_page=True,
                )

                await page.goto("http://localhost:8000/docs", wait_until="domcontentloaded")
                await page.locator("body").wait_for(state="visible")
                await page.screenshot(
                    path=str(SCREENSHOTS / "api_docs.png"),
                    full_page=True,
                )
                await browser.close()
        finally:
            pass

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(asyncio.run(main()))
    except Exception as exc:  # pragma: no cover
        print(f"web_shots failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
