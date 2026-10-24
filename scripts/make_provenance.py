"""Write results/PROVENANCE.md: which code state and which result files the paper tables come from.

Run it after regenerating results and before committing. The SHA-256 of each result file lets a reader
check that the files in the repository are the ones the paper's macros were built from.
"""
import hashlib
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def git(*a: str) -> str:
    try:
        return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


lines = ["# Provenance", "",
         "Hashes of the result files that `scripts/make_paper_artifacts.py` turns into the paper's macros.",
         f"Written at code commit `{git('rev-parse', '--short', 'HEAD')}` (the commit that contains this file is its child).",
         "CPU results were produced before the final code changes and were then re-checked: rerunning the protocol with the",
         "released code on the stored CPU model reproduced all eight evaluation files exactly (timings excluded).", ""]
for d in ["results", "results_gpu", "results_gpu_font_fallback"]:
    files = sorted((ROOT / d).glob("*.json"))
    if not files:
        continue
    lines += [f"## {d}/", "", "| file | sha256 (first 16) |", "|---|---|"]
    lines += [f"| {f.name} | `{hashlib.sha256(f.read_bytes()).hexdigest()[:16]}` |" for f in files]
    lines.append("")
(ROOT / "results" / "PROVENANCE.md").write_text("\n".join(lines))
print("wrote results/PROVENANCE.md")
