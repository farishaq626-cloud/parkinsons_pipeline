"""Compile the Gemini Beamer poster with a detected local LaTeX engine.

The script is intentionally dependency-aware: it checks that the Gemini theme
files are present before starting a multi-pass LaTeX/BibTeX build, preserving
the compiler output in ``poster_build/`` for diagnosis.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "poster.tex"
BUILD_DIR = ROOT / "poster_build"
THEME_FILES = (
    "beamerthemegemini.sty",
    "beamercolorthemegemini.sty",
    "beamerinnerthemegemini.sty",
)
ENGINES = ("pdflatex", "lualatex", "xelatex")
MIKTEX_BIN = (
    Path.home() / "AppData" / "Local" / "Programs" / "MiKTeX" / "miktex" / "bin" / "x64"
)
BUILD_WARNING_PATTERN = re.compile(
    r"^(?:LaTeX|Package .+|Class .+|pdfTeX) Warning:"
    r"|^(?:Over|Under)full \\[hv]box"
    r"|undefined references?"
    r"|Citation .+ undefined",
    flags=re.IGNORECASE | re.MULTILINE,
)


def find_engine() -> str | None:
    """Return an available LaTeX engine from PATH or a standard MiKTeX path."""
    for engine in ENGINES:
        located = shutil.which(engine)
        if located:
            return located
    for engine in ENGINES:
        candidate = MIKTEX_BIN / f"{engine}.exe"
        if candidate.exists():
            return str(candidate)
    return None


def find_bibtex() -> str | None:
    """Return BibTeX from PATH or a standard per-user MiKTeX installation."""
    return shutil.which("bibtex") or (
        str(MIKTEX_BIN / "bibtex.exe") if (MIKTEX_BIN / "bibtex.exe").exists() else None
    )


def generate_qr_assets() -> None:
    """Generate the poster QR assets from their canonical URLs.

    Raises:
        RuntimeError: If the QR-code dependency is unavailable or an asset
            cannot be written.
    """
    try:
        from poster_assets.generate_qr_codes import QR_TARGETS, generate_qr_code

        output_directory = ROOT / "figures"
        output_directory.mkdir(parents=True, exist_ok=True)
        for filename, url in QR_TARGETS.items():
            generate_qr_code(url, output_directory / filename)
    except (ImportError, OSError) as error:
        raise RuntimeError(
            "Unable to generate poster QR assets. Install the pinned project "
            "dependencies with 'pip install -r requirements-lock.txt' and retry."
        ) from error


def run(command: list[str]) -> None:
    """Run a build command and relay clear failure details.

    Args:
        command: Executable and arguments to invoke.

    Raises:
        RuntimeError: If the command exits with a non-zero status.
    """
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )
    if completed.returncode:
        print(completed.stdout)
        print(completed.stderr, file=sys.stderr)
        raise RuntimeError(f"Build command failed: {' '.join(command)}")


def find_build_warnings(log_path: Path) -> list[str]:
    """Return actionable LaTeX warnings from the final build log.

    Args:
        log_path: Final LaTeX log produced after all compilation passes.

    Returns:
        Unique warning lines in their original order.

    Raises:
        FileNotFoundError: If the expected final build log is absent.
    """
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    warnings: list[str] = []
    for match in BUILD_WARNING_PATTERN.finditer(log_text):
        line = log_text[match.start() :].splitlines()[0].strip()
        if line not in warnings:
            warnings.append(line)
    return warnings


def main() -> int:
    """Build ``poster.pdf`` using LaTeX, BibTeX, and two final LaTeX passes.

    Returns:
        Exit status: zero on success and one when a required compiler or theme
        file is unavailable, or when compilation fails.
    """
    if not SOURCE.exists():
        print(f"ERROR: Poster source not found: {SOURCE}", file=sys.stderr)
        return 1

    missing_themes = [name for name in THEME_FILES if not (ROOT / name).exists()]
    if missing_themes:
        print(
            "ERROR: Gemini theme files are required before compilation.",
            file=sys.stderr,
        )
        print("Missing: " + ", ".join(missing_themes), file=sys.stderr)
        print(
            "Download/upload the Gemini theme files to the repository root, "
            "then rerun: python build_poster.py",
            file=sys.stderr,
        )
        return 1

    try:
        generate_qr_assets()
    except RuntimeError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    engine = find_engine()
    if engine is None:
        print(
            "ERROR: No LaTeX engine was found (lualatex, pdflatex, or xelatex).",
            file=sys.stderr,
        )
        print(
            "Install MiKTeX or TeX Live, ensure its bin directory is on PATH, "
            "then rerun: python build_poster.py. Alternatively, upload poster.tex "
            "and the Gemini .sty files to Overleaf.",
            file=sys.stderr,
        )
        return 1

    bibtex = find_bibtex()
    if bibtex is None:
        print(
            "ERROR: BibTeX was not found alongside the LaTeX engine.", file=sys.stderr
        )
        return 1

    if BUILD_DIR.exists():
        shutil.rmtree(BUILD_DIR)
    BUILD_DIR.mkdir()
    latex_command = [
        engine,
        "-interaction=nonstopmode",
        "-halt-on-error",
        f"-output-directory={BUILD_DIR}",
        str(SOURCE),
    ]

    try:
        run(latex_command)
        run([bibtex, str(BUILD_DIR / "poster")])
        run(latex_command)
        run(latex_command)
    except RuntimeError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        print(f"Inspect the build log: {BUILD_DIR / 'poster.log'}", file=sys.stderr)
        return 1

    output = BUILD_DIR / "poster.pdf"
    if not output.exists() or output.stat().st_size == 0:
        print("ERROR: Compilation finished without a usable PDF.", file=sys.stderr)
        return 1

    log_path = BUILD_DIR / "poster.log"
    try:
        warnings = find_build_warnings(log_path)
    except FileNotFoundError:
        print(f"ERROR: Expected build log was not created: {log_path}", file=sys.stderr)
        return 1
    if warnings:
        print("ERROR: Poster build completed with LaTeX warnings:", file=sys.stderr)
        for warning in warnings:
            print(f"  - {warning}", file=sys.stderr)
        return 1

    final_pdf = ROOT / "poster.pdf"
    temporary_pdf = BUILD_DIR / "poster.final.pdf"
    shutil.copy2(output, temporary_pdf)
    temporary_pdf.replace(final_pdf)
    print(f"SUCCESS: Poster created at {final_pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
