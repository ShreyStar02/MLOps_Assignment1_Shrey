"""Render a shell command and its output into a PNG screenshot."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WIDTH = 1600
DEFAULT_HEIGHT = 900


def get_font(size: int):
    """Return a monospace font suitable for terminal output."""
    candidates = [
        "C:/Windows/Fonts/consola.ttf",
        "C:/Windows/Fonts/CascadiaCode.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationMono-Regular.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def capture_command(
    command: str,
    output_path: Path,
    cwd: Path | None = None,
) -> tuple[int, str]:
    """Run a shell command and save its output as a PNG screenshot."""
    result = subprocess.run(
        command,
        cwd=str(cwd or ROOT),
        shell=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    output = (result.stdout or "")
    if result.stdout and result.stderr:
        output += "\n"
    output += result.stderr or ""
    render_terminal(command, output, result.returncode, output_path)
    return result.returncode, output


def render_terminal(command: str, output: str, exit_code: int, out_path: Path) -> None:
    """Generate a terminal-style PNG preview with a title bar and output."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (DEFAULT_WIDTH, DEFAULT_HEIGHT), color="#0b1220")
    draw = ImageDraw.Draw(image)

    title_font = get_font(20)
    body_font = get_font(16)
    small_font = get_font(14)

    draw.rectangle((0, 0, DEFAULT_WIDTH, 70), fill="#1f2937")
    draw.ellipse((25, 23, 45, 43), fill="#f87171")
    draw.ellipse((48, 23, 68, 43), fill="#fbbf24")
    draw.ellipse((71, 23, 91, 43), fill="#4ade80")
    title_text = command[:90] + ("..." if len(command) > 90 else "")
    draw.text((110, 18), title_text, font=title_font, fill="#e5e7eb")
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    draw.text((DEFAULT_WIDTH - 420, 18), ts, font=small_font, fill="#9ca3af")

    content_top = 90
    content_left = 24
    draw.rectangle(
        (20, content_top, DEFAULT_WIDTH - 20, DEFAULT_HEIGHT - 20),
        fill="#111827",
    )
    draw.text(
        (content_left + 12, content_top + 12),
        f"exit code: {exit_code}",
        font=small_font,
        fill="#7dd3fc",
    )

    lines = output.splitlines() or ["(no output)"]
    y = content_top + 50
    for line in lines[-95:]:
        draw.text((content_left + 18, y), line[:170], font=body_font, fill="#d1fae5")
        y += 18

    if y > DEFAULT_HEIGHT - 40:
        draw.rectangle(
            (DEFAULT_WIDTH - 210, DEFAULT_HEIGHT - 42, DEFAULT_WIDTH - 20, DEFAULT_HEIGHT - 22),
            fill="#0f172a",
        )
        draw.text(
            (DEFAULT_WIDTH - 190, DEFAULT_HEIGHT - 42),
            "TRUNCATED",
            font=small_font,
            fill="#fbbf24",
        )

    image.save(out_path, format="PNG")


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Capture a command output into a PNG screenshot.",
    )
    parser.add_argument("command", help="Shell command to run")
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("-c", "--cwd", type=Path, default=ROOT)
    return parser


def main() -> int:
    parser = create_parser()
    args = parser.parse_args()
    try:
        exit_code, _ = capture_command(args.command, args.output, args.cwd)
        return exit_code
    except Exception as exc:  # pragma: no cover - CLI boundary
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
