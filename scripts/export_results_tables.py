"""Export results/*.csv|json into thesis-ready tables (Markdown, CSV, LaTeX).

Usage: python scripts/export_results_tables.py [results_dir]   (default: ./results)
Same as: yoruba-asr export-tables
"""
import sys
from pathlib import Path

from yoruba_asr.reporting import export_tables

if __name__ == "__main__":
    results = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results")
    for kind, path in export_tables(results).items():
        print(f"{kind}: {path}")
