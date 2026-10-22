"""Command-line interface: ``yoruba-asr <command>`` (or ``python -m yoruba_asr.cli <command>``)."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Optional

import typer

app = typer.Typer(add_completion=False, help="Yoruba ASR (XLS-R) - data, training, evaluation and inference.",
                  no_args_is_help=True, pretty_exceptions_enable=False)

ConfigOpt = typer.Option(None, "--config", "-c", help="YAML config(s); base.yaml is always merged first.")
SetOpt = typer.Option(None, "--set", help="Override a config value, e.g. --set train.max_steps=10")
ResumeOpt = typer.Option(True, "--resume/--no-resume", help="Resume from checkpoints / skip finished work.")
DryOpt = typer.Option(False, "--dry-run", help="Print the plan without running anything.")


def _cfg(config: Optional[List[Path]], sets: Optional[List[str]], default: str = "base.yaml"):
    from .config import CONFIGS_DIR, load_config

    files = list(config) if config else [CONFIGS_DIR / default]
    return load_config(files, sets)


def _done(title: str, payload) -> None:
    typer.echo(f"\n== {title} finished ==")
    if isinstance(payload, dict):
        short = {k: v for k, v in payload.items() if not isinstance(v, (dict, list)) or k in ("tts",)}
        typer.echo(json.dumps(short, indent=2, ensure_ascii=False, default=str)[:3000])


def _dry(stage: str) -> None:
    from .pipeline import run_stage

    run_stage(stage, dry_run=True)


@app.command()
def audit(config: Optional[List[Path]] = ConfigOpt, set: Optional[List[str]] = SetOpt, dry_run: bool = DryOpt):
    """Corpus audit (clip counts, hours, speakers, tone coverage)."""
    if dry_run:
        return _dry("audit")
    from .pipeline import stage_audit

    stage_audit(_cfg(config, set))
    typer.echo("== audit finished: results/corpus_audit.md ==")


@app.command()
def prepare(config: Optional[List[Path]] = ConfigOpt, set: Optional[List[str]] = SetOpt, resume: bool = ResumeOpt,
            dry_run: bool = DryOpt):
    """Load, normalise, filter, leakage-check, build vocabulary and cache Common Voice."""
    if dry_run:
        return _dry("prepare")
    from .pipeline import stage_prepare

    _done("prepare", stage_prepare(_cfg(config, set), resume=resume))


@app.command()
def augment(config: Optional[List[Path]] = ConfigOpt, set: Optional[List[str]] = SetOpt, resume: bool = ResumeOpt,
            dry_run: bool = DryOpt):
    """Build augmented copies + TTS clips for V2/V4."""
    if dry_run:
        return _dry("augment")
    from .pipeline import stage_augment

    _done("augment", stage_augment(_cfg(config, set, "v2_augmentation.yaml"), resume=resume))


@app.command("tune-mask")
def tune_mask(config: Optional[List[Path]] = ConfigOpt, set: Optional[List[str]] = SetOpt, resume: bool = ResumeOpt,
              skip_mask_tuning: bool = typer.Option(False, "--skip-mask-tuning", help="Use 0.075 without tuning."),
              dry_run: bool = DryOpt):
    """Tune mask_time_prob for V2/V4 on the dev set."""
    if dry_run:
        return _dry("tune_mask")
    from .pipeline import stage_tune_mask

    cfg = _cfg(config, set, "v2_augmentation.yaml")
    if skip_mask_tuning:
        cfg.mask_tuning.skip = True
    _done("tune-mask", stage_tune_mask(cfg, resume=resume))


@app.command()
def cpt(config: Optional[List[Path]] = ConfigOpt, set: Optional[List[str]] = SetOpt, resume: bool = ResumeOpt,
        dry_run: bool = DryOpt):
    """Continued self-supervised pretraining (resumable)."""
    if dry_run:
        return _dry("cpt")
    from .pipeline import stage_cpt

    _done("cpt", stage_cpt(_cfg(config, set, "cpt.yaml"), resume=resume))


@app.command()
def train(config: Optional[List[Path]] = ConfigOpt, set: Optional[List[str]] = SetOpt, resume: bool = ResumeOpt,
          dry_run: bool = DryOpt):
    """Train one variant (the config's `variant`: V1/V2/V3/V4) and evaluate it on the test set."""
    cfg = _cfg(config, set, "v1_baseline.yaml")
    if dry_run:
        return _dry(cfg.variant)
    from .pipeline import stage_train

    _done(f"train {cfg.variant}", stage_train(cfg, resume=resume))


@app.command()
def evaluate(variant: Optional[List[str]] = typer.Option(None, "--variant", help="Variant(s) to evaluate"),
             config: Optional[List[Path]] = ConfigOpt, set: Optional[List[str]] = SetOpt,
             force: bool = typer.Option(False, "--force", help="Re-decode even if metrics exist."),
             dry_run: bool = DryOpt):
    """Evaluate trained variants on the CV test set (+ significance + effects)."""
    if dry_run:
        return _dry("evaluate")
    from .pipeline import stage_evaluate

    stage_evaluate(_cfg(config, set), variants=list(variant) if variant else None, force=force)
    typer.echo("== evaluate finished: results/summary.csv ==")


@app.command()
def significance(config: Optional[List[Path]] = ConfigOpt, set: Optional[List[str]] = SetOpt):
    """Blockwise (speaker) bootstrap significance tests."""
    from .pipeline import stage_significance

    stage_significance(_cfg(config, set))
    typer.echo("== significance finished: results/significance.csv ==")


@app.command()
def effects(config: Optional[List[Path]] = ConfigOpt, set: Optional[List[str]] = SetOpt):
    """Factorial main effects and interaction."""
    from .pipeline import stage_effects

    stage_effects(_cfg(config, set))
    typer.echo("== effects finished: results/factorial_effects.csv ==")


@app.command()
def ablation(config: Optional[List[Path]] = ConfigOpt, set: Optional[List[str]] = SetOpt, resume: bool = ResumeOpt,
             only: Optional[str] = typer.Option(None, "--only", help="Run a single ablation, e.g. no_tts"),
             dry_run: bool = DryOpt):
    """Leave-one-out ablation runs of V2."""
    if dry_run:
        return _dry("ablation")
    from .pipeline import stage_ablation

    stage_ablation(_cfg(config, set, "ablation.yaml"), resume=resume, only=only)
    typer.echo("== ablation finished: results/ablation.csv ==")


@app.command()
def analyse(config: Optional[List[Path]] = ConfigOpt, set: Optional[List[str]] = SetOpt):
    """Error analysis, figures and thesis tables."""
    from .pipeline import stage_analyse

    _done("analyse", stage_analyse(_cfg(config, set)))


@app.command("export-tables")
def export_tables(config: Optional[List[Path]] = ConfigOpt, set: Optional[List[str]] = SetOpt):
    """Write results/thesis_tables.{md,csv,tex}."""
    from .pipeline import stage_export_tables

    _done("export-tables", stage_export_tables(_cfg(config, set)))


@app.command()
def transcribe(audio: Path = typer.Argument(..., help="Audio file (wav/mp3/flac/ogg/m4a)"),
               model: str = typer.Option("models/V4", "--model", help="Model folder or Hugging Face repo id"),
               no_tone: bool = typer.Option(False, "--no-tone", help="Print the transcript without tone marks")):
    """Transcribe an audio file with a trained model."""
    from .inference import Transcriber

    t = Transcriber.from_dir(model) if Path(model).exists() else Transcriber.from_hub(model)
    r = t.transcribe_file(audio)
    typer.echo(r["text_notone"] if no_tone else r["text"])
    typer.echo(f"[{r['model']}, {r['seconds']} s]", err=True)


@app.command()
def stage(name: str = typer.Argument(..., help="Stage name as used in the Kaggle notebook"),
          smoke: bool = typer.Option(False, "--smoke", help="Layer configs/smoke.yaml on top"),
          set: Optional[List[str]] = SetOpt, resume: bool = ResumeOpt,
          only: Optional[str] = typer.Option(None, "--only"), dry_run: bool = DryOpt):
    """Run a notebook stage (prepare, V1, augment+tune_mask, cpt, ..., all_light)."""
    from .config import parse_overrides
    from .pipeline import next_step_message, run_stage

    run_stage(name, parse_overrides(set), smoke=smoke, resume=resume, only=only, dry_run=dry_run)
    typer.echo(next_step_message(name))


@app.command("smoke")
def smoke_cmd(set: Optional[List[str]] = SetOpt, dry_run: bool = DryOpt,
              fixture: Path = typer.Option(None, "--fixture", help="Fixture data folder (default: tests/fixtures/data)"),
              out: Path = typer.Option(None, "--out", help="Work root (default: ./_smoke)")):
    """End-to-end smoke pipeline on the tiny synthetic fixture with a tiny random model (CPU, minutes)."""
    from .config import REPO_ROOT, parse_overrides
    from .pipeline import run_stage

    overrides = parse_overrides(set)
    if not Path("/kaggle/input").exists() and "paths.input_root" not in overrides:
        fixture = fixture or REPO_ROOT / "tests" / "fixtures" / "data"
        if not (fixture / "common_voice").exists():
            import importlib.util

            spec = importlib.util.spec_from_file_location("mk", REPO_ROOT / "scripts" / "make_tiny_fixture.py")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)  # type: ignore[union-attr]
            mod.make_fixture(fixture)
        root = (out or Path("_smoke")).resolve()
        overrides = {"paths.input_root": str(fixture), "paths.work_root": str(root),
                     "paths.tmp_dir": str(root / "tmp"), **overrides}
    run_stage("smoke", overrides, smoke=True, resume=False, dry_run=dry_run)
    typer.echo("== smoke pipeline finished ==")


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    app()


if __name__ == "__main__":
    main()
