"""Configuration loading, merging and dataset path discovery.

Configs are YAML files merged in this order:
    configs/base.yaml  ->  variant/stage config(s)  ->  command-line overrides (--set key=value)

Values may reference other values with ``${dotted.key}`` (e.g. ``${paths.work_dir}/results``).
The merged result is a :class:`Config`: a nested dict that also allows attribute access
(``cfg.train.max_steps``).
"""
from __future__ import annotations

import copy
import os
import re
import shutil
import tarfile
import unicodedata
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import yaml

PACKAGE_ROOT = Path(__file__).resolve().parent
# Repository root (…/yoruba-asr) when running from a source checkout.
REPO_ROOT = PACKAGE_ROOT.parent.parent
CONFIGS_DIR = REPO_ROOT / "configs"

_INTERP = re.compile(r"\$\{([^}]+)\}")
OUTPUT_MARKER = "YORUBA_ASR_OUTPUT.txt"  # not a dotfile, so it survives Kaggle's output saving
AUDIO_EXTS = (".wav", ".flac", ".mp3", ".ogg", ".m4a", ".opus")


# --------------------------------------------------------------------------------------
# Config object
# --------------------------------------------------------------------------------------
class Config(dict):
    """Nested dictionary with attribute access and dotted-key helpers."""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def __deepcopy__(self, memo):
        return Config(copy.deepcopy(dict(self), memo))

    def get_path(self, dotted: str, default: Any = None) -> Any:
        node: Any = self
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def set_path(self, dotted: str, value: Any) -> None:
        node = self
        parts = dotted.split(".")
        for part in parts[:-1]:
            if part not in node or not isinstance(node[part], dict):
                node[part] = Config()
            node = node[part]
        node[parts[-1]] = _wrap(value)

    def to_dict(self) -> dict:
        return _unwrap(self)


def _wrap(obj: Any) -> Any:
    if isinstance(obj, dict) and not isinstance(obj, Config):
        return Config({k: _wrap(v) for k, v in obj.items()})
    if isinstance(obj, Config):
        for k, v in list(obj.items()):
            obj[k] = _wrap(v)
        return obj
    if isinstance(obj, list):
        return [_wrap(v) for v in obj]
    return obj


def _unwrap(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _unwrap(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_unwrap(v) for v in obj]
    return obj


def deep_merge(base: dict, other: dict) -> dict:
    """Recursively merge ``other`` into a copy of ``base`` (``other`` wins)."""
    out = copy.deepcopy(base)
    for key, value in (other or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _resolve_interpolations(cfg: Config) -> None:
    """Replace ``${a.b}`` references until nothing changes (max 10 passes)."""

    def resolve(value: Any) -> Any:
        if isinstance(value, str):
            def repl(m: re.Match) -> str:
                ref = cfg.get_path(m.group(1).strip())
                if ref is None:
                    raise KeyError(f"Config interpolation: unknown key '{m.group(1)}'")
                return str(ref)
            return _INTERP.sub(repl, value)
        if isinstance(value, dict):
            for k in value:
                value[k] = resolve(value[k])
            return value
        if isinstance(value, list):
            return [resolve(v) for v in value]
        return value

    for _ in range(10):
        before = repr(cfg)
        resolve(cfg)
        if repr(cfg) == before:
            break


def parse_overrides(items: Iterable[str] | dict | None) -> dict:
    """Turn ``["train.max_steps=10", ...]`` (or a dict) into ``{"train.max_steps": 10}``."""
    if not items:
        return {}
    if isinstance(items, dict):
        return dict(items)
    out = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Override must look like key=value, got: {item!r}")
        key, raw = item.split("=", 1)
        out[key.strip()] = yaml.safe_load(raw)
    return out


def load_config(
    config_files: str | Path | list[str | Path] | None = None,
    overrides: Iterable[str] | dict | None = None,
    include_base: bool = True,
) -> Config:
    """Load ``base.yaml`` + the given config files + overrides into a :class:`Config`."""
    if config_files is None:
        config_files = []
    if isinstance(config_files, (str, Path)):
        config_files = [config_files]
    files = [Path(f) for f in config_files]

    merged: dict = {}
    if include_base:
        base = None
        for f in files:
            if f.name == "base.yaml":
                base = f
                break
        if base is None:
            candidates = [f.parent / "base.yaml" for f in files] + [CONFIGS_DIR / "base.yaml"]
            base = next((c for c in candidates if c.exists()), None)
        if base is not None:
            merged = yaml.safe_load(base.read_text(encoding="utf-8")) or {}
        files = [f for f in files if f.name != "base.yaml" or f != base]

    for f in files:
        data = yaml.safe_load(Path(f).read_text(encoding="utf-8")) or {}
        merged = deep_merge(merged, data)

    cfg = _wrap(merged)
    for key, value in parse_overrides(overrides).items():
        cfg.set_path(key, value)
    _resolve_interpolations(cfg)
    return cfg


def config_path(name: str) -> Path:
    """Path of a bundled config (``"v1_baseline"`` -> ``configs/v1_baseline.yaml``)."""
    p = Path(name)
    if p.suffix == ".yaml" and p.exists():
        return p
    return CONFIGS_DIR / (name if name.endswith(".yaml") else f"{name}.yaml")


# --------------------------------------------------------------------------------------
# Typed views
# --------------------------------------------------------------------------------------
@dataclass
class Paths:
    input_root: Path
    work_dir: Path
    tmp_dir: Path
    results_dir: Path
    models_dir: Path
    cache_dir: Path
    shared_dir: Path
    manifests_dir: Path
    checkpoints_dir: Path
    augmented_dir: Path

    @classmethod
    def from_config(cls, cfg: Config, create: bool = True) -> "Paths":
        p = cfg.paths
        obj = cls(**{k: Path(p[k]) for k in cls.__dataclass_fields__})
        if create:
            for name in cls.__dataclass_fields__:
                if name != "input_root":
                    getattr(obj, name).mkdir(parents=True, exist_ok=True)
            marker = obj.work_dir / OUTPUT_MARKER
            if not marker.exists():
                marker.write_text("yoruba_asr work directory\n", encoding="utf-8")
        return obj


@dataclass
class DataPaths:
    """Dataset folders found by :func:`discover_paths`."""

    common_voice: Path | None = None
    slr86: list[Path] = field(default_factory=list)
    iroyin: Path | None = None
    iroyin_manifest: Path | None = None
    bibletts: Path | None = None
    musan: Path | None = None
    rirs: Path | None = None
    previous_outputs: list[Path] = field(default_factory=list)
    extracted: list[Path] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "common_voice": str(self.common_voice) if self.common_voice else None,
            "slr86": [str(p) for p in self.slr86],
            "iroyin": str(self.iroyin) if self.iroyin else None,
            "iroyin_manifest": str(self.iroyin_manifest) if self.iroyin_manifest else None,
            "bibletts": str(self.bibletts) if self.bibletts else None,
            "musan": str(self.musan) if self.musan else None,
            "rirs": str(self.rirs) if self.rirs else None,
            "previous_outputs": [str(p) for p in self.previous_outputs],
            "extracted_archives": [str(p) for p in self.extracted],
            "notes": self.notes,
        }


# --------------------------------------------------------------------------------------
# Path discovery
# --------------------------------------------------------------------------------------
def _plain(name: str) -> str:
    """Lower-case, accent-free version of a folder name (ÌròyìnSpeech -> iroyinspeech)."""
    s = unicodedata.normalize("NFD", name)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    return s.lower()


def _is_archive(name: str) -> bool:
    n = name.lower()
    return n.endswith((".tar.gz", ".tgz", ".tar", ".zip", ".tar.bz2", ".tar.xz"))


def _archive_stem(name: str) -> str:
    for ext in (".tar.gz", ".tar.bz2", ".tar.xz", ".tgz", ".tar", ".zip"):
        if name.lower().endswith(ext):
            return name[: -len(ext)]
    return name


def extract_archive(archive: Path, dest_root: Path) -> Path:
    """Extract ``archive`` into ``dest_root/<stem>`` once (a ``.done`` marker skips re-extraction)."""
    dest = dest_root / _archive_stem(archive.name)
    done = dest / ".done"
    if done.exists():
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    if archive.name.lower().endswith(".zip"):
        with zipfile.ZipFile(archive) as zf:
            for member in zf.namelist():
                target = (dest / member).resolve()
                if not str(target).startswith(str(dest.resolve())):
                    raise RuntimeError(f"Unsafe path in archive {archive}: {member}")
            zf.extractall(dest)
    else:
        with tarfile.open(archive) as tf:
            try:
                tf.extractall(dest, filter="data")  # Python >= 3.12 (and patched 3.10/3.11)
            except TypeError:
                for member in tf.getmembers():
                    target = (dest / member.name).resolve()
                    if not str(target).startswith(str(dest.resolve())):
                        raise RuntimeError(f"Unsafe path in archive {archive}: {member.name}")
                tf.extractall(dest)
    done.write_text("ok", encoding="utf-8")
    return dest


def _walk(root: Path, skip: set[Path]):
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        d = Path(dirpath)
        if d in skip:
            dirnames[:] = []
            continue
        if OUTPUT_MARKER in filenames or (d / "shared" / "vocab.json").exists():
            # A previous notebook output: never treat it as raw data.
            yield d, "previous_output", filenames
            dirnames[:] = []
            continue
        # Do not descend into huge audio folders.
        dirnames[:] = [x for x in dirnames if x not in ("clips", "__pycache__", ".git")]
        yield d, "dir", filenames


def _first_slr86_id(line_index: Path) -> str:
    try:
        with open(line_index, encoding="utf-8") as fh:
            first = fh.readline()
    except Exception:
        return ""
    return re.split(r"[\t ]", first.strip(), maxsplit=1)[0]


def discover_paths(cfg: Config, verbose: bool = True) -> DataPaths:
    """Find dataset folders under ``paths.input_root`` by marker files.

    * Common Voice: a folder containing ``train.tsv`` + ``dev.tsv`` + ``test.tsv``.
    * SLR86: folders containing ``line_index.tsv`` (ids like ``yof_…``/``yom_…``).
    * ÌròyìnSpeech: a folder whose name contains "iroyin" (or ``data.iroyin.manifest``).
    * BibleTTS / MUSAN / RIRs: folders named ``*bibletts*``, ``musan``, ``*rir*``.
    * Previous outputs: folders containing the ``.yoruba_asr_output`` marker.
    Archives (``.tar.gz``/``.zip``…) are extracted to ``tmp_dir/extracted`` and searched too.
    Kaggle dataset slugs are never hard-coded.
    """
    found = DataPaths()
    input_root = Path(cfg.paths.input_root)
    tmp_dir = Path(cfg.paths.tmp_dir)
    work_dir = Path(cfg.paths.work_dir).resolve() if cfg.paths.get("work_dir") else None
    roots = [input_root] if input_root.exists() else []
    if not roots:
        found.notes.append(f"input_root {input_root} does not exist")

    skip: set[Path] = set()
    if work_dir is not None:
        skip.add(work_dir)
    cv_candidates: list[Path] = []

    def scan(root: Path, archives: list[Path]):
        for d, kind, files in _walk(root, skip):
            if kind == "previous_output":
                if work_dir is None or d.resolve() != work_dir:
                    found.previous_outputs.append(d)
                continue
            fset = set(files)
            name = _plain(d.name)
            if {"train.tsv", "dev.tsv", "test.tsv"} <= fset:
                cv_candidates.append(d)
            if "line_index.tsv" in fset and _first_slr86_id(d / "line_index.tsv").startswith("yo"):
                found.slr86.append(d)
            if "iroyin" in name and found.iroyin is None:
                found.iroyin = d
            if "bibletts" in name and found.bibletts is None:
                found.bibletts = d
            if name == "musan" and found.musan is None:
                found.musan = d
            if ("rir" in name and "rirs_noises" not in name) and found.rirs is None:
                found.rirs = d
            if name == "rirs_noises" and found.rirs is None:
                found.rirs = d
            for f in files:
                if _is_archive(f):
                    archives.append(d / f)

    archives: list[Path] = []
    for root in roots:
        scan(root, archives)

    if cfg.paths.get("extract_archives", True) and archives:
        extract_root = tmp_dir / "extracted"
        for arc in archives:
            try:
                dest = extract_archive(arc, extract_root)
                found.extracted.append(dest)
                if verbose:
                    print(f"[discover] extracted {arc.name} -> {dest}")
            except Exception as exc:  # corrupted/unsupported archive
                found.notes.append(f"could not extract {arc}: {exc}")
        for dest in found.extracted:
            scan(dest, [])

    # Explicit paths from the config win over auto-detection.
    d = cfg.data
    if d.common_voice.get("dir"):
        cv_candidates.insert(0, Path(d.common_voice.dir))
    if d.slr86.get("dir"):
        found.slr86 = [p.parent for p in Path(d.slr86.dir).rglob("line_index.tsv")] or found.slr86
    if d.iroyin.get("dir"):
        found.iroyin = Path(d.iroyin.dir)
    if d.iroyin.get("manifest"):
        found.iroyin_manifest = Path(d.iroyin.manifest)
    if d.bibletts.get("dir"):
        found.bibletts = Path(d.bibletts.dir)
    if d.get("musan", {}).get("dir"):
        found.musan = Path(d.musan.dir)
    if d.get("rirs", {}).get("dir"):
        found.rirs = Path(d.rirs.dir)

    if cv_candidates:
        # Prefer the folder with the most clips (handles duplicates: archive + extracted copy).
        def n_clips(p: Path) -> int:
            c = p / "clips"
            return sum(1 for _ in c.iterdir()) if c.exists() else 0

        cv_candidates = list(dict.fromkeys(cv_candidates))
        found.common_voice = max(cv_candidates, key=n_clips)
        if len(cv_candidates) > 1:
            found.notes.append(f"several Common Voice folders found; using {found.common_voice}")

    found.slr86 = sorted(set(found.slr86))
    if not d.slr86.get("enabled", True):
        found.slr86 = []
    elif not found.slr86 and d.slr86.get("download", False):
        # Not attached: download it (no Kaggle upload needed; Internet must be on).
        from .data.download import download_openslr86

        try:
            found.slr86 = download_openslr86(tmp_dir / "downloads" / "openslr86")
            found.notes.append("OpenSLR 86 downloaded automatically")
        except Exception as exc:
            found.notes.append(f"OpenSLR 86 download failed: {exc}")
    if not d.iroyin.get("enabled", True):
        found.iroyin, found.iroyin_manifest = None, None
    if not d.bibletts.get("enabled", False):
        found.bibletts = None

    if verbose:
        print("[discover] " + ", ".join(
            f"{k}={v}" for k, v in found.summary().items() if v and k not in ("notes", "extracted_archives")
        ))
        for n in found.notes:
            print(f"[discover] note: {n}")
    return found


def copy_previous_outputs(found: DataPaths, work_dir: Path,
                          subdirs: tuple[str, ...] = ("shared", "results", "manifests", "cache", "models",
                                                      "checkpoints", "augmented")) -> list[str]:
    """Copy outputs of earlier notebook versions into ``work_dir`` (chains Kaggle sessions).

    Files that already exist with the same size are skipped, so this is cheap to repeat.
    """
    copied: list[str] = []
    for prev in found.previous_outputs:
        for sub in subdirs:
            src = prev / sub
            if not src.exists():
                continue
            for path in src.rglob("*"):
                if path.is_dir():
                    continue
                rel = path.relative_to(prev)
                dst = work_dir / rel
                if dst.exists() and dst.stat().st_size == path.stat().st_size:
                    continue
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, dst)
            copied.append(f"{prev}/{sub}")
    marker = work_dir / OUTPUT_MARKER
    if not marker.exists():
        work_dir.mkdir(parents=True, exist_ok=True)
        marker.write_text("yoruba_asr work directory\n", encoding="utf-8")
    return copied
