import ast
import re
import sys
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "kaggle"))

import build_notebook  # noqa: E402
from yoruba_asr.pipeline import NOTEBOOK_STAGES, SEQUENCES, STAGE_FUNCTIONS, run_stage  # noqa: E402


def _built(tmp_path):
    out = build_notebook.build(tmp_path / "nb.ipynb")
    return nbformat.read(str(out), as_version=4)


def test_notebook_is_valid_and_parameters_first(tmp_path):
    nb = _built(tmp_path)
    nbformat.validate(nb)
    code = [c for c in nb.cells if c.cell_type == "code"]
    assert "parameters" in code[0].metadata.get("tags", [])
    params = {}
    exec(code[0].source, {}, params)
    assert params["STAGE"] == "prepare" and params["RESUME"] is True
    for name in ("ABLATION_ONLY", "PUSH_TO_HUB", "HUB_REPO_PREFIX", "CPT_MAX_HOURS", "EXTRA_OVERRIDES"):
        assert name in params


def test_every_stage_maps_to_code(tmp_path):
    nb = _built(tmp_path)
    src = nb.cells[1].source
    listed = re.findall(r'"([A-Za-z0-9_]+)"', src.split("#", 1)[1].split("RESUME")[0])
    assert set(listed) == set(NOTEBOOK_STAGES)
    for stage in NOTEBOOK_STAGES:
        assert stage in STAGE_FUNCTIONS or stage in SEQUENCES
        assert run_stage(stage, dry_run=True)
    assert run_stage("augment+tune_mask", dry_run=True)


def test_package_is_embedded(tmp_path):
    nb = _built(tmp_path)
    written = [c.source.splitlines()[0] for c in nb.cells if c.source.startswith("%%writefile")]
    for f in ("src/yoruba_asr/pipeline.py", "src/yoruba_asr/cli.py", "configs/base.yaml", "pyproject.toml",
              "src/yoruba_asr/adapt/cpt.py"):
        assert any(w.endswith(f) for w in written), f
    for c in nb.cells:  # every embedded python module must be syntactically valid
        if c.source.startswith("%%writefile") and c.source.splitlines()[0].endswith(".py"):
            ast.parse(c.source.split("\n", 1)[1])
