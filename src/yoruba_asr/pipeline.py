"""Stage orchestration: prepare -> V1 -> augment -> tune_mask -> V2 -> cpt -> V3 -> V4 -> evaluate -> analyse.

Every stage reads/writes under ``paths.work_dir`` and is safe to re-run (completed work is skipped
when ``resume=True``). The CLI and the Kaggle notebook both call :func:`run_stage`.
"""
from __future__ import annotations

import shutil
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from . import VARIANTS
from .config import (CONFIGS_DIR, Config, DataPaths, Paths, deep_merge, discover_paths, load_config)
from .data.audit import build_audit, write_audit
from .data.common_voice import clip_durations, load_sentence_pool, load_split
from .data.generic import ManifestError, list_audio
from .data.iroyin import load_iroyin
from .data.leakage import (assert_no_eval_audio, forbidden_sentences, leakage_key, remove_forbidden,
                           speaker_overlap)
from .data.slr86 import load_slr86
from .text import count_intraword, has_digits, has_tone_mark, normalize_text, strip_tones
from .utils import (RunInfo, get_logger, gpu_info, load_json, now_iso, save_json, seed_everything)

log = get_logger("yoruba_asr")

STAGE_CONFIGS = {"V1": "v1_baseline.yaml", "V2": "v2_augmentation.yaml", "V3": "v3_domain_adaptation.yaml",
                 "V4": "v4_hybrid.yaml", "cpt": "cpt.yaml", "ablation": "ablation.yaml",
                 "tune_mask": "v2_augmentation.yaml", "augment": "v2_augmentation.yaml"}

NEXT_STAGE = {
    "smoke": ("prepare", "Attach Common Voice (+ SLR86, ÌròyìnSpeech, MUSAN if you have them)."),
    "prepare": ("V1", "Attach the data datasets + the output of this run."),
    "V1": ("augment+tune_mask", "Attach the data datasets + the output of this run."),
    "augment": ("tune_mask", "Attach the data + the output of this run."),
    "augment+tune_mask": ("V2", "Attach the data + the output of this run."),
    "tune_mask": ("V2", "Attach the data + the output of this run."),
    "V2": ("cpt", "Attach the data + the output of this run. CPT needs two sessions."),
    "cpt": ("cpt (again until completed) then V3", "Attach the data + the output of this run."),
    "V3": ("V4", "Attach the data + the output of this run."),
    "V4": ("evaluate", "Attach the output of this run."),
    "evaluate": ("ablation", "Attach the data + the output of this run (one ablation per session if needed)."),
    "ablation": ("ablation (next run) or analyse", "Attach the output of this run."),
    "analyse": ("done", "Download /kaggle/working/results for the thesis and models/ for the app."),
    "all_light": ("prepare", "The light run is only a sanity check; now run the real stages."),
}


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
@dataclass
class Ctx:
    cfg: Config
    paths: Paths
    found: DataPaths | None

    @property
    def results(self) -> Path:
        return self.paths.results_dir

    def manifest(self, name: str) -> Path:
        return self.paths.manifests_dir / f"{name}.csv"


def make_ctx(cfg: Config, discover: bool = True) -> Ctx:
    seed_everything(int(cfg.seed))
    paths = Paths.from_config(cfg)
    found = discover_paths(cfg) if discover else None
    return Ctx(cfg, paths, found)


def stage_config(stage: str, overrides=None, smoke: bool = False, extra_files: list | None = None) -> Config:
    files = [CONFIGS_DIR / "base.yaml"]
    if stage in STAGE_CONFIGS:
        files.append(CONFIGS_DIR / STAGE_CONFIGS[stage])
    files += [Path(f) for f in (extra_files or [])]
    if smoke:
        files.append(CONFIGS_DIR / "smoke.yaml")
    return load_config(files, overrides)


def _ensure_block(cfg: Config, name: str, file: str) -> None:
    import yaml

    defaults = (yaml.safe_load((CONFIGS_DIR / file).read_text(encoding="utf-8")) or {}).get(name, {})
    cfg[name] = Config(deep_merge(defaults, dict(cfg.get(name) or {})))
    from .config import _wrap

    cfg[name] = _wrap(cfg[name])


def _limit(df: pd.DataFrame, n) -> pd.DataFrame:
    return df.head(int(n)).reset_index(drop=True) if n else df


def _read_manifest(path: Path, required: bool = True) -> pd.DataFrame:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"{path} not found - run the 'prepare' stage first (or attach its output).")
        return pd.DataFrame()
    return pd.read_csv(path, dtype={"speaker": str, "gender": str, "age": str}, keep_default_na=False,
                       encoding="utf-8")


def add_text_columns(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    df = df.copy()
    keep = bool(cfg.text.keep_tones)
    df["text_tone"] = df["sentence"].astype(str).map(lambda s: normalize_text(s, keep_tones=keep))
    df["text_notone"] = df["text_tone"].map(strip_tones)
    reason = pd.Series([None] * len(df), index=df.index, dtype=object)
    if cfg.data.drop_digits:
        reason[df["sentence"].astype(str).map(has_digits)] = "digits"
    reason[(df["text_tone"].str.strip() == "") & reason.isna()] = "empty_text"
    df["drop_reason"] = reason
    return df


def resolve_mask(cfg: Config, ctx: Ctx) -> float:
    m = cfg.get("mask_time_prob", cfg.model.mask_time_prob)
    if m != "tuned":
        return float(m)
    info = load_json(ctx.results / "V2" / "mask_tuning.json")
    if info and info.get("chosen") is not None:
        return float(info["chosen"])
    if cfg.mask_tuning.get("skip"):
        return float(cfg.mask_tuning.default_if_skipped)
    raise FileNotFoundError("results/V2/mask_tuning.json not found: run the 'tune_mask' stage first "
                            "(or set mask_tuning.skip=true to use 0.075).")


def _processor(ctx: Ctx):
    from .vocab import load_shared_processor

    return load_shared_processor(ctx.paths.shared_dir, ctx.results)


def _cv_cached(ctx: Ctx, split: str):
    from .train.datasets import CachedASRDataset, load_cached_split

    p = ctx.paths.cache_dir / f"cv_{split}"
    if not p.exists():
        raise FileNotFoundError(f"{p} not found - run the 'prepare' stage first (or attach its output).")
    return CachedASRDataset(load_cached_split(p))


def _manifest_dataset(ctx: Ctx, df: pd.DataFrame, processor):
    from .train.datasets import ManifestASRDataset

    if df is None or len(df) == 0:
        return None
    return ManifestASRDataset(df, processor, top_db=float(ctx.cfg.data.trim_top_db))


def _previous(ctx: Ctx) -> list[Path]:
    return ctx.found.previous_outputs if ctx.found else []


# --------------------------------------------------------------------------------------
# loading corpora (shared by audit + prepare)
# --------------------------------------------------------------------------------------
class MissingAudioError(FileNotFoundError):
    pass


def check_cv_audio(cv: dict, cfg: Config, max_missing_share: float = 0.2) -> dict:
    """Make sure the clips listed in the TSVs exist; find them anywhere under the input if needed.

    Raises :class:`MissingAudioError` with a listing of what the dataset contains when more than
    ``max_missing_share`` of train/dev/test clips cannot be found.
    """
    from .data.common_voice import dataset_tree_report, resolve_clip_paths

    roots = [cfg.paths.input_root, str(Path(cfg.paths.tmp_dir) / "extracted")]
    report = {}
    for split in ("train", "dev", "test", "other"):
        if len(cv[split]) == 0:
            continue
        cv[split], report[split] = resolve_clip_paths(cv[split], roots)
        r = report[split]
        if r["remapped"]:
            log.info("cv_%s: %d clips were not in clips/ and were found elsewhere in the input", split, r["remapped"])
        if r["missing"]:
            log.warning("cv_%s: %d of %d clips are missing from the attached data", split, r["missing"], len(cv[split]))
    bad = {s: r for s, r in report.items() if s != "other" and r["missing"] > max_missing_share * len(cv[s])}
    if bad:
        found = next(iter(report.values())).get("audio_files_found_under_input", "?")
        raise MissingAudioError(
            "\n" + "=" * 90 + "\nCOMMON VOICE AUDIO IS MISSING. The TSV files were found, but most MP3 clips were not:\n"
            + "\n".join(f"  {s}: {r['missing']} of {len(cv[s])} clips missing" for s, r in bad.items())
            + f"\n  audio files found anywhere under {cfg.paths.input_root}: {found} (Common Voice 27.0 Yoruba has 4,970)\n"
            "\nWhat the attached input contains:\n" + dataset_tree_report(cfg.paths.input_root) +
            "\n\nFIX: delete the Kaggle dataset and upload it again as ONE file: the original archive\n"
            "  '1789491097449-cv-corpus-27.0-2026-09-11-yo.tar.gz' (about 175 MB). Do not upload the extracted\n"
            "  folder (the one on your PC is incomplete). Wait until Kaggle finishes processing, check that the\n"
            "  dataset page shows a 'clips' folder with 4,970 files, then attach it again (see docs/KAGGLE_GUIDE.md).\n"
            + "=" * 90)
    return report


def _load_cv_primary(cfg: Config, cv_dir: Path) -> dict:
    lim = cfg.data.limit
    cv = {s: add_text_columns(_limit(load_split(cv_dir, s), lim.get(s)), cfg) for s in ("train", "dev", "test")}
    cv["other"] = _limit(load_split(cv_dir, "other"), lim.get("other"))
    check_cv_audio(cv, cfg)
    return cv


def _load_slr86_primary(cfg: Config, folders: list[Path]) -> dict:
    """OpenSLR 86 as the main corpus: speaker-disjoint (and sentence-disjoint) train/dev/test."""
    from .data.slr86 import speaker_split

    df = load_slr86(folders)
    if df.empty:
        raise FileNotFoundError(f"OpenSLR 86 folders {folders} contain no usable recordings.")
    sp = cfg.data.slr86.get("split", {}) or {}
    parts = speaker_split(df, float(sp.get("dev_share", 0.1)), float(sp.get("test_share", 0.1)), int(cfg.seed),
                          key=leakage_key)
    lim = cfg.data.limit
    cv = {s: add_text_columns(_limit(parts[s], lim.get(s)), cfg) for s in ("train", "dev", "test")}
    cv["other"] = df.iloc[:0].copy()
    empty = [s for s in ("train", "dev", "test") if len(cv[s]) == 0]
    if empty:
        raise ValueError(f"OpenSLR 86 split produced empty {empty} sets ({df['speaker'].nunique()} speakers found). "
                         "Attach both yo_ng_female and yo_ng_male (or let the notebook download them).")
    log.info("OpenSLR 86 as primary corpus: train %d / dev %d / test %d utterances, speakers %d / %d / %d",
             *(len(cv[s]) for s in ("train", "dev", "test")), *(cv[s]["speaker"].nunique() for s in ("train", "dev", "test")))
    return cv


def choose_primary(cfg: Config, found: DataPaths) -> tuple[str, dict, str]:
    """Return (primary corpus name, train/dev/test/other frames, note)."""
    mode = str(cfg.data.get("primary", "auto"))
    have_cv, have_slr = found is not None and found.common_voice is not None, bool(found and found.slr86)
    if mode == "slr86" or (mode == "auto" and not have_cv):
        if not have_slr:
            raise FileNotFoundError(
                f"No training data found under {cfg.paths.input_root}: attach Common Voice 27.0 Yoruba, or turn "
                "Internet On so OpenSLR 86 can be downloaded automatically (data.slr86.download: true).")
        return "slr86", _load_slr86_primary(cfg, found.slr86), "OpenSLR 86 (Common Voice not used)"
    if not have_cv:
        raise FileNotFoundError(
            f"Common Voice Yoruba was not found under {cfg.paths.input_root}. Upload the Common Voice 27.0 Yoruba "
            "archive as a PRIVATE Kaggle dataset and attach it, or set PRIMARY_DATASET = 'slr86'.")
    try:
        return "common_voice", _load_cv_primary(cfg, found.common_voice), "Common Voice 27.0"
    except MissingAudioError as exc:
        if mode == "auto" and have_slr:
            log.warning("=" * 90 + "\nCommon Voice audio is missing, so OpenSLR 86 is used as the main corpus "
                        "instead (PRIMARY_DATASET='auto').\n" + str(exc).splitlines()[2] + "\n" + "=" * 90)
            return "slr86", _load_slr86_primary(cfg, found.slr86), "OpenSLR 86 (fallback: Common Voice audio missing)"
        raise


def load_corpora(ctx: Ctx) -> dict:
    cfg, found = ctx.cfg, ctx.found
    lim = cfg.data.limit
    primary, cv, primary_note = choose_primary(cfg, found)
    cv_dir = found.common_voice if primary == "common_voice" else None
    extra_counts = {s: len(load_split(cv_dir, s)) for s in ("validated", "invalidated")} if cv_dir else {}
    forbidden = forbidden_sentences(cv["dev"], cv["test"])

    ood, ood_notes, gate = [], {}, {}
    if found.slr86 and primary != "slr86":
        ood.append(load_slr86(found.slr86))
    if found.iroyin is not None or found.iroyin_manifest is not None:
        try:
            ood.append(load_iroyin(found.iroyin, found.iroyin_manifest))
        except ManifestError as exc:
            ood_notes["iroyin"] = str(exc)
            log.warning("ÌròyìnSpeech skipped: %s", exc)
    ood_df = pd.concat(ood, ignore_index=True) if ood else pd.DataFrame(
        columns=["path", "sentence", "speaker", "corpus", "split", "gender", "age"])
    leak_removed = {}
    kept_parts = []
    for corpus, g in ood_df.groupby("corpus") if len(ood_df) else []:
        toneless = float(np.mean([not has_tone_mark(s) for s in g["sentence"]])) if len(g) else 0.0
        needs_check = toneless > float(cfg.data.tone_gate_max_toneless_share)
        force = bool(cfg.data.get(corpus, {}).get("force_include", False))
        gate[corpus] = {"share_toneless": round(toneless, 4), "needs_manual_check": needs_check,
                        "force_include": force, "included": (not needs_check) or force}
        if needs_check:
            log.warning("=" * 70 + f"\n{corpus}: {toneless:.0%} of transcripts have NO tone marks -> "
                        "needs_manual_check. Listen to a 50-utterance sample before using it. "
                        f"{'Included because force_include=true.' if force else 'EXCLUDED (set data.' + corpus + '.force_include: true to include).'}\n" + "=" * 70)
            if not force:
                continue
        g = add_text_columns(g, cfg)
        g, n_removed = remove_forbidden(g, forbidden)
        leak_removed[corpus] = n_removed
        kept_parts.append(g)
    ood_df = pd.concat(kept_parts, ignore_index=True) if kept_parts else ood_df.assign(
        text_tone="", text_notone="", drop_reason=None)
    ood_df = _limit(ood_df, lim.get("ood"))

    tts, tts_info = build_tts_pool(cfg, cv["train"], forbidden, cv_dir)
    if "removed_forbidden" in tts_info:
        leak_removed["tts_pool"] = tts_info["removed_forbidden"]
    return {"cv": cv, "cv_dir": cv_dir, "primary": primary, "primary_note": primary_note, "tts_info": tts_info,
            "extra_counts": extra_counts, "forbidden": forbidden, "ood": ood_df,
            "ood_notes": ood_notes, "tone_gate": gate, "leak_removed": leak_removed, "tts_sentences": tts}


def build_tts_pool(cfg: Config, train: pd.DataFrame, forbidden: set, cv_dir) -> tuple[pd.DataFrame, dict]:
    """TTS text pool: CV validated sentences (or, with OpenSLR 86 as primary, its training sentences),
    never dev/test sentences (tone-stripped match), no digits. Returns (sentences, step counts)."""
    from .vocab import canonical_charset

    if cv_dir is not None:
        pool = load_sentence_pool(cv_dir, [cfg.paths.input_root, str(Path(cfg.paths.tmp_dir) / "extracted")])
        info = {"source": pool.attrs.get("pool_source")}
    else:
        pool = train[["sentence"]].drop_duplicates().reset_index(drop=True)
        info = {"source": "primary training sentences (OpenSLR 86)"}
    info["loaded"] = len(pool)
    empty = pd.DataFrame(columns=["sentence", "text_tone", "text_notone"])
    if not len(pool):
        log.warning("TTS sentence pool is empty (%s)", info["source"])
        return empty, info
    pool = add_text_columns(pool, cfg)
    pool = pool[pool["drop_reason"].isna()]
    info["after_text_filter"] = len(pool)
    pool, info["removed_forbidden"] = remove_forbidden(pool, forbidden)
    # Only sentences written with characters already used by the training text (or the canonical Yoruba
    # set), so odd encodings (e.g. U+031F "plus below" instead of the sub-dot) never enter the vocabulary.
    allowed = canonical_charset() | set("".join(train["text_tone"])) | {" "}
    pool = pool[pool["text_tone"].map(lambda t: set(t) <= allowed)].drop_duplicates("text_tone").copy()
    info["after_charset_filter"] = len(pool)
    train_keys = set(train["sentence"].map(leakage_key))
    pool["recorded_in_train"] = pool["sentence"].map(leakage_key).isin(train_keys)
    pool = pool.sample(frac=1.0, random_state=int(cfg.seed))
    pool = pool.sort_values("recorded_in_train", kind="stable")  # unrecorded sentences first
    cap = int(np.ceil(len(train) * float(cfg.tts.max_per_real_clip) * 1.5))
    tts = pool.head(cap).reset_index(drop=True)
    info["selected"] = len(tts)
    log.info("TTS sentence pool: %s", info)
    return tts, info


def _unlabelled_pool(ctx: Ctx, data: dict, durations_cv: dict) -> pd.DataFrame:
    from .train.datasets import probe_manifest

    cfg = ctx.cfg
    other = data["cv"]["other"][["path", "speaker", "corpus"]].copy()
    other["corpus"] = "common_voice_other"
    parts = [other]
    if data.get("primary") == "slr86":  # primary training audio (never dev/test) is valid unlabelled audio
        parts.append(data["cv"]["train"][["path", "speaker", "corpus"]])
    ood = data["ood"]
    if len(ood):
        parts.append(ood[["path", "speaker", "corpus"]])
    if ctx.found.bibletts is not None:
        files = list_audio(ctx.found.bibletts)
        parts.append(pd.DataFrame({"path": files, "speaker": "bibletts", "corpus": "bibletts"}))
    pool = pd.concat(parts, ignore_index=True).drop_duplicates("path")
    pool = _limit(pool, cfg.data.limit.get("unlabelled"))
    # Durations: CV from clip_durations.tsv (fast) else file headers.
    known = pool["path"].map(lambda p: durations_cv.get(Path(p).name))
    probed = probe_manifest(pool[known.isna()], 0.0, 1e9) if known.isna().any() else pool.iloc[:0].assign(duration=[])
    pool["duration"] = known
    if len(probed):
        pool.loc[known.isna(), "duration"] = probed["duration"].values
    pool = pool[pool["duration"].notna() & (pool["duration"].astype(float) >= float(cfg.data.min_seconds))]
    return pool.reset_index(drop=True)


# --------------------------------------------------------------------------------------
# stages
# --------------------------------------------------------------------------------------
def stage_audit(cfg: Config, **_) -> dict:
    """Corpus audit from transcripts + file headers (no feature extraction)."""
    from .train.datasets import probe_manifest

    ctx = make_ctx(cfg)
    data = load_corpora(ctx)
    rows = []
    for split, df in data["cv"].items():
        if split == "other":
            df = add_text_columns(df, cfg)
        p = probe_manifest(df, cfg.data.min_seconds, cfg.data.max_seconds)
        p["drop_reason"] = df["drop_reason"].where(df["drop_reason"].notna(), p["drop_reason"])
        rows.append(p)
    if len(data["ood"]):
        p = probe_manifest(data["ood"], cfg.data.min_seconds, cfg.data.max_seconds)
        p["drop_reason"] = data["ood"]["drop_reason"].where(data["ood"]["drop_reason"].notna(), p["drop_reason"])
        rows.append(p)
    audit = build_audit(pd.concat(rows, ignore_index=True), extra={
        "cv_other_counts": data["extra_counts"], "ood_tone_gate": data["tone_gate"], "ood_notes": data["ood_notes"],
        "note": "audit stage uses raw (untrimmed) durations; 'prepare' reports trimmed durations"})
    write_audit(audit, ctx.results)
    print(f"Audit written to {ctx.results / 'corpus_audit.md'}")
    return audit


def stage_prepare(cfg: Config, resume: bool = True, **_) -> dict:
    """Load, normalise, filter, leakage-check, build the shared vocabulary and cache Common Voice."""
    from .train.datasets import build_cached_split, load_cached_split, probe_manifest
    from .vocab import build_vocab, make_processor, save_vocab, unknown_chars, vocab_sha256

    ctx = make_ctx(cfg)
    run = RunInfo(ctx.results / "stages" / "prepare_run_info.json", "prepare")
    data = load_corpora(ctx)
    cv, ood, tts = data["cv"], data["ood"], data["tts_sentences"]
    durations_cv = clip_durations(data["cv_dir"]) if data["cv_dir"] else {}

    # Shared vocabulary from CV train + OOD train + TTS sentences (never CV dev/test).
    vocab_texts = (cv["train"].loc[cv["train"]["drop_reason"].isna(), "text_tone"].tolist()
                   + (ood.loc[ood["drop_reason"].isna(), "text_tone"].tolist() if len(ood) else [])
                   + tts["text_tone"].tolist())
    vocab_path = ctx.paths.shared_dir / "vocab.json"
    if resume and vocab_path.exists() and (ctx.results / "vocab_info.json").exists():
        log.info("Reusing existing shared vocabulary %s", vocab_path)
        vinfo = load_json(ctx.results / "vocab_info.json")
    else:
        vinfo = save_vocab(build_vocab(vocab_texts), ctx.paths.shared_dir, ctx.results,
                           sources={"cv_train": int(cv["train"]["drop_reason"].isna().sum()),
                                    "ood_train": int(ood["drop_reason"].isna().sum()) if len(ood) else 0,
                                    "tts_sentences": len(tts)})
    processor = make_processor(vocab_path)
    vocab = processor.tokenizer.get_vocab()
    unk = {s: unknown_chars(cv[s]["text_tone"], vocab) for s in ("dev", "test")}
    vinfo["unknown_chars_in_cv_dev_test"] = unk
    save_json(vinfo, ctx.results / "vocab_info.json")

    # Common Voice audio -> cached HF datasets (input_values, input_length, labels).
    sha = vocab_sha256(ctx.paths.shared_dir)
    audit_rows, manifests = [], {}
    for split in ("train", "dev", "test"):
        df = cv[split]
        kept_text = df[df["drop_reason"].isna()]
        out = ctx.paths.cache_dir / f"cv_{split}"
        cache_info = load_json(out / "cache_info.json") or {}
        if resume and out.exists() and cache_info.get("vocab_sha256") == sha and cache_info.get("rows_in") == len(kept_text):
            ds = load_cached_split(out)
            drop_log = cache_info.get("drop_log", {})
            log.info("cv_%s: reusing cache (%d clips)", split, len(ds))
        else:
            if out.exists():
                shutil.rmtree(out)
            log.info("cv_%s: decoding %d clips ...", split, len(kept_text))
            ds, drop_log = build_cached_split(kept_text, processor, out, ctx.paths.tmp_dir, cfg.data.min_seconds,
                                              cfg.data.max_seconds, cfg.data.trim_top_db)
            save_json({"vocab_sha256": sha, "rows_in": len(kept_text), "rows_out": len(ds), "drop_log": drop_log,
                       "built_at": now_iso()}, out / "cache_info.json")
        meta = ds.select_columns(["path", "duration"]).to_pandas()
        dur_map = dict(zip(meta["path"], meta["duration"]))
        full = df.copy()
        full["duration"] = full["path"].map(dur_map)
        full["drop_reason"] = full["drop_reason"].where(full["drop_reason"].notna(), full["path"].map(drop_log))
        hdr = probe_manifest(full[["path"]], 0, 1e9)
        full["sample_rate"] = hdr["sample_rate"].values
        audit_rows.append(full)
        kept = full[full["drop_reason"].isna()].copy()
        manifests[f"cv_{split}"] = kept
        kept.drop(columns=["drop_reason"]).to_csv(ctx.manifest(f"cv_{split}"), index=False, encoding="utf-8")

    # OOD labelled (B1) - durations from headers; audio is loaded on the fly at training time.
    if len(ood):
        o = probe_manifest(ood, cfg.data.min_seconds, cfg.data.max_seconds)
        o["drop_reason"] = ood["drop_reason"].where(ood["drop_reason"].notna(), o["drop_reason"])
        audit_rows.append(o)
        ood_kept = o[o["drop_reason"].isna()].copy()
    else:
        ood_kept = pd.DataFrame(columns=["path", "sentence", "text_tone", "text_notone", "speaker", "corpus", "duration"])
    ood_kept.drop(columns=["drop_reason"], errors="ignore").to_csv(ctx.manifest("ood_train"), index=False, encoding="utf-8")

    # Unlabelled pool for CPT (CV other + OOD audio + BibleTTS), never CV dev/test.
    unl = _unlabelled_pool(ctx, data, durations_cv)
    unl.to_csv(ctx.manifest("unlabelled"), index=False, encoding="utf-8")
    other = add_text_columns(cv["other"], cfg)
    other["duration"] = other["path"].map(lambda p: durations_cv.get(Path(p).name))
    audit_rows.append(other.assign(drop_reason=other["drop_reason"]))
    tts.to_csv(ctx.manifest("tts_sentences"), index=False, encoding="utf-8")

    # Leakage report.
    pools = {"cv_train": manifests["cv_train"], "ood_train": ood_kept, "unlabelled": unl}
    in_pools = assert_no_eval_audio(pools, cv["dev"], cv["test"])
    tts_leaks = int(tts["sentence"].map(leakage_key).isin(data["forbidden"]).sum()) if len(tts) else 0
    assert tts_leaks == 0, "TTS sentences overlap CV dev/test"
    overlap = speaker_overlap(cv["train"], cv["dev"], cv["test"])
    leak = {"forbidden_sentences": len(data["forbidden"]), "removed_by_corpus": data["leak_removed"],
            "cv_dev_test_audio_in_pools": in_pools, "tts_sentences_in_forbidden": tts_leaks,
            "speaker_overlap": overlap, "speaker_overlap_ok": all(v == 0 for v in overlap.values()),
            "checked_at": now_iso()}
    save_json(leak, ctx.results / "leakage_report.json")
    if not leak["speaker_overlap_ok"]:
        log.warning("CV train/dev/test speakers overlap: %s (recorded in the audit)", overlap)

    # Audit.
    intraword = {"apostrophes": 0, "hyphens": 0}
    for df in cv.values():
        for s in df["sentence"]:
            for k, v in count_intraword(str(s)).items():
                intraword[k] += v
    rows = pd.concat(audit_rows, ignore_index=True)
    audit = build_audit(rows, extra={
        "primary_dataset": data["primary_note"],
        "cv_other_counts": data["extra_counts"], "ood_tone_gate": data["tone_gate"], "ood_notes": data["ood_notes"],
        "intraword_punctuation_removed": intraword, "digit_policy": "utterances containing digits are dropped",
        "speaker_overlap": overlap, "unlabelled_pool_hours": round(float(unl["duration"].sum()) / 3600, 2),
        "unlabelled_pool_clips": len(unl), "tts_sentence_pool": len(tts),
        "unknown_chars_cv_dev_test": unk})
    write_audit(audit, ctx.results)

    summary = {
        "primary_dataset": data["primary_note"],
        "cv_train": len(manifests["cv_train"]), "cv_dev": len(manifests["cv_dev"]), "cv_test": len(manifests["cv_test"]),
        "cv_train_hours": round(float(manifests["cv_train"]["duration"].sum()) / 3600, 3),
        "ood_train": len(ood_kept), "ood_hours": round(float(ood_kept["duration"].astype(float).sum()) / 3600, 2) if len(ood_kept) else 0,
        "unlabelled": len(unl), "unlabelled_hours": round(float(unl["duration"].sum()) / 3600, 2),
        "tts_sentences": len(tts), "tts_pool_steps": data["tts_info"], "vocab_size": vinfo["size"], "vocab_sha256": sha,
        "discovered": ctx.found.summary(),
    }
    save_json(summary, ctx.results / "data_summary.json")
    run.finish(steps=None)
    print("\n== prepare summary ==")
    for k, v in summary.items():
        if k != "discovered":
            print(f"  {k}: {v}")
    return summary


def rebuild_tts_sentences(ctx: Ctx, train: pd.DataFrame) -> pd.DataFrame:
    """Rebuild an empty TTS sentence list (e.g. prepare ran before validated_sentences.tsv was found).

    Uses the same rules as 'prepare': dev/test sentences of the primary corpus are forbidden and only
    characters already in the shared vocabulary's source text are allowed. The vocabulary is unchanged.
    """
    summary = load_json(ctx.results / "data_summary.json", default={}) or {}
    use_cv = str(summary.get("primary_dataset", "Common Voice")).startswith("Common Voice")
    cv_dir = ctx.found.common_voice if (use_cv and ctx.found) else None
    if cv_dir is not None:
        dev, test = load_split(cv_dir, "dev"), load_split(cv_dir, "test")
    else:
        dev, test = _read_manifest(ctx.manifest("cv_dev")), _read_manifest(ctx.manifest("cv_test"))
    tts, info = build_tts_pool(ctx.cfg, train, forbidden_sentences(dev, test), cv_dir)
    tts.to_csv(ctx.manifest("tts_sentences"), index=False, encoding="utf-8")
    save_json(info, ctx.results / "V2" / "tts_pool_rebuilt.json")
    log.info("TTS sentence list rebuilt in the augment stage: %d sentences", len(tts))
    return tts


def stage_augment(cfg: Config, resume: bool = True, **_) -> dict:
    """Offline augmented copies (2 per CV train clip), pitch check and MMS-TTS synthesis."""
    from .augment.build_augmented import build_augmented_set, pitch_check
    from .augment.tts import synthesize_set

    ctx = make_ctx(cfg)
    run = RunInfo(ctx.results / "stages" / "augment_run_info.json", "augment")
    train = _read_manifest(ctx.manifest("cv_train"))
    a, d = cfg.augment, cfg.data
    v2_results = ctx.results / "V2"
    musan = ctx.found.musan if ctx.found else None
    rirs = ctx.found.rirs if ctx.found else None

    pc = pitch_check(train, a, int(cfg.seed), float(d.trim_top_db), int(a.pitch_check_clips))
    save_json(pc, v2_results / "pitch_check.json")
    log.info("pitch check: median relative F0 change %.4f (limit %.2f) -> %s", pc["median_relative_diff"],
             pc["limit"], "PASS" if pc["passed"] else "FAIL")
    if not pc["passed"]:
        raise AssertionError(f"Tempo perturbation changed pitch by {pc['median_relative_diff']:.2%} "
                             f"(limit {pc['limit']:.0%}). See results/V2/pitch_check.json")

    manifest, stats = build_augmented_set(train, ctx.paths.augmented_dir / "main", a, int(cfg.seed),
                                          list(a.transforms), d.min_seconds, d.max_seconds, d.trim_top_db,
                                          musan, rirs, name="main")
    tts_info = {"enabled_effective": False, "reason": "tts.enabled=false"}
    if cfg.tts.enabled:
        sentences = _read_manifest(ctx.manifest("tts_sentences"), required=False)
        if len(sentences) == 0:
            sentences = rebuild_tts_sentences(ctx, train)
        cap = int(len(train) * float(cfg.tts.max_per_real_clip))
        if len(sentences):
            _, tts_info = synthesize_set(sentences, ctx.paths.augmented_dir / "tts", v2_results, cfg.tts.model,
                                         int(cfg.seed), cap, d.min_seconds, d.max_seconds, d.trim_top_db,
                                         int(cfg.tts.listening_sample))
        else:
            tts_info = {"enabled_effective": False, "reason": "no TTS sentence pool (validated_sentences.tsv missing)"}
    save_json(tts_info, v2_results / "tts_status.json")
    info = run.finish(steps=None)
    out = {"augmented": stats, "pitch_check_passed": pc["passed"], "tts": {k: tts_info.get(k) for k in
                                                                           ("enabled_effective", "synthesised", "skipped", "reason", "error")},
           "gpu_hours": info["total_gpu_hours"]}
    save_json(out, v2_results / "augment_summary.json")
    print(f"augmented clips: {stats['written']}, TTS clips: {tts_info.get('synthesised', 0)}, "
          f"TTS effective: {tts_info.get('enabled_effective')}")
    return out


def _v2_train_parts(ctx: Ctx, processor, aug_set: str = "main", use_tts: bool = True):
    """CV train (cached) + augmented copies + TTS clips."""
    from .train.datasets import ConcatASRDataset

    cv_train = _cv_cached(ctx, "train")
    aug_m = ctx.paths.augmented_dir / aug_set / "manifest_augmented.csv"
    if not aug_m.exists():
        raise FileNotFoundError(f"{aug_m} not found - run the 'augment' stage first.")
    aug = _read_manifest(aug_m)
    tts_m = ctx.paths.augmented_dir / "tts" / "manifest_tts.csv"
    tts = _read_manifest(tts_m, required=False) if use_tts else pd.DataFrame()
    ds = ConcatASRDataset([cv_train, _manifest_dataset(ctx, aug, processor), _manifest_dataset(ctx, tts, processor)])
    hours = (sum(cv_train.lengths) / 16000 + float(aug["duration"].sum()) +
             (float(tts["duration"].sum()) if len(tts) else 0.0)) / 3600
    return ds, {"cv_train": len(cv_train), "augmented": len(aug), "tts": len(tts), "train_hours": round(hours, 3)}


def stage_tune_mask(cfg: Config, resume: bool = True, **_) -> dict:
    """Three short V2 runs with mask_time_prob in {0.05, 0.075, 0.10}; choose the lowest dev WER."""
    from .modeling import resolve_model_name
    from .train.finetune import run_finetune

    ctx = make_ctx(cfg, discover=True)
    out_path = ctx.results / "V2" / "mask_tuning.json"
    mt = cfg.mask_tuning
    state = load_json(out_path, default={}) if resume else {}
    if mt.skip:
        state = {"skipped": True, "chosen": float(mt.default_if_skipped),
                 "note": "mask tuning skipped (--skip-mask-tuning); default value used"}
        save_json(state, out_path)
        print(f"mask tuning skipped -> {state['chosen']}")
        return state
    processor = _processor(ctx)
    train_ds, parts = _v2_train_parts(ctx, processor)
    dev = _cv_cached(ctx, "dev")
    results = state.get("candidates", {})
    for p in mt.candidates:
        key = f"{float(p):g}"
        if key in results and results[key].get("status") == "complete":
            continue
        tag = f"mask_{key}"
        s = run_finetune(cfg, tag=tag, train_ds=train_ds, eval_ds=dev, processor=processor,
                         init_from=resolve_model_name(cfg), mask_time_prob=float(p), max_steps=int(mt.max_steps),
                         model_dir=ctx.paths.models_dir / "mask_tuning" / key, results_dir=ctx.results / "V2" / tag,
                         resume=resume, previous_outputs=_previous(ctx))
        results[key] = {"mask_time_prob": float(p), "dev_wer": s.get("best_dev_wer"), "status": s["status"],
                        "global_step": s.get("global_step")}
        save_json({"candidates": results, "max_steps": int(mt.max_steps), "chosen": None}, out_path)
        if s["status"] != "complete":
            print("mask tuning interrupted by the time budget - run 'tune_mask' again.")
            return {"candidates": results, "status": "incomplete"}
        shutil.rmtree(ctx.paths.models_dir / "mask_tuning" / key, ignore_errors=True)
    best = min(results.values(), key=lambda r: (r["dev_wer"] if r["dev_wer"] is not None else 1e9))
    state = {"candidates": results, "max_steps": int(mt.max_steps), "chosen": best["mask_time_prob"],
             "selection": "lowest dev WER", "train_parts": parts, "finished_at": now_iso()}
    save_json(state, out_path)
    print(f"chosen mask_time_prob = {state['chosen']} ({results})")
    return state


def stage_cpt(cfg: Config, resume: bool = True, **_) -> dict:
    from .adapt.cpt import CPTError, run_cpt
    from .modeling import resolve_model_name

    _ensure_block(cfg, "cpt", "cpt.yaml")
    ctx = make_ctx(cfg)
    unl = _read_manifest(ctx.manifest("unlabelled"))
    try:
        summary = run_cpt(cfg, unl, resolve_model_name(cfg), ctx.paths.models_dir / "cpt_encoder",
                          ctx.paths.checkpoints_dir / "cpt", ctx.results / "V3", resume=resume)
    except CPTError as exc:
        q = Path(cfg.paths.work_dir) / "OPEN_QUESTIONS_runtime.md"
        q.write_text(f"# CPT stopped\n\n{exc}\n", encoding="utf-8")
        raise
    print(f"CPT: {summary['completed_steps']} steps, stop reason = {summary['stop_reason']}, "
          f"completed = {summary['completed']}")
    if not summary["completed"]:
        print("CPT is not finished: run STAGE='cpt' again (attach this output) until completed=True.")
    return summary


def write_model_card(model_dir: Path, variant: str, cfg: Config, metrics: dict | None, extra: dict) -> None:
    from .reporting import VARIANT_NAMES

    m = metrics or {}
    lines = [
        f"# Yoruba ASR - {variant}: {VARIANT_NAMES.get(variant, variant)}", "",
        "Fine-tuned from `facebook/wav2vec2-xls-r-300m` (wav2vec 2.0, CTC head, greedy decoding, no LM).", "",
        "## Training", "",
        f"- Training set: {extra.get('training_set')} ({extra.get('train_parts')})",
        f"- Initialisation: {extra.get('init')}; stage B1: {extra.get('stage_b1')}",
        f"- max_steps={cfg.train.max_steps}, lr={cfg.train.learning_rate}, effective batch="
        f"{cfg.train.per_device_train_batch_size * cfg.train.gradient_accumulation_steps}, "
        f"mask_time_prob={extra.get('mask_time_prob')}, seed={cfg.seed}",
        f"- Shared vocabulary SHA-256: {extra.get('vocab_sha256')}", "",
        "## Test results (Common Voice 27.0 Yoruba test, %)", "",
        f"WER {m.get('wer', 'n/a')} | CER {m.get('cer', 'n/a')} | WER (no tone) {m.get('wer_no_tone', 'n/a')} | "
        f"TER {m.get('ter', 'n/a')}", "",
        "## Licences and limitations", "",
        "- Common Voice: CC0 (data may not be re-hosted). OpenSLR 86: CC BY-SA 4.0. ÌròyìnSpeech text: CC BY 4.0.",
        "- Synthetic speech (V2/V4) from `facebook/mms-tts-yor` (CC BY-NC 4.0): research use only.",
        "- Read speech, Standard Yoruba, limited number of speakers; not validated for spontaneous speech.", "",
        "## References", "",
        "Baevski et al. (2020) wav2vec 2.0; Babu et al. (2022) XLS-R; Pratap et al. (2024) MMS; Ardila et al. (2020) "
        "Common Voice; Gutkin et al. (2020) Yoruba corpus; Ògúnrẹ̀mí et al. (2024) ÌròyìnSpeech.",
    ]
    (model_dir / "model_card.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate_variant(ctx: Ctx, variant: str, model_dir: Path, extra: dict | None = None) -> dict:
    from .evaluate.decode import decode_cached, evaluate_predictions, load_model_and_processor
    from .train.datasets import load_cached_split
    from .vocab import vocab_sha256

    model, processor, device = load_model_and_processor(model_dir)
    test = load_cached_split(ctx.paths.cache_dir / "cv_test")
    preds = decode_cached(model, processor, test, device, int(ctx.cfg.eval.batch_size))
    status = load_json(model_dir / "training_complete.json", default={}) or {}
    info = {"train_clips": status.get("train_clips"), "train_hours": status.get("train_hours"),
            "gpu_hours": status.get("gpu_hours_total"), "mask_time_prob": status.get("mask_time_prob"),
            "checkpoint": str(model_dir), "vocab_sha256": vocab_sha256(ctx.paths.shared_dir), **(extra or {})}
    metrics = evaluate_predictions(preds, variant, ctx.results / variant, info)
    del model
    return metrics


def stage_train(cfg: Config, resume: bool = True, variant: str | None = None, **_) -> dict:
    """Train one variant (V1-V4) and evaluate it on the CV test set."""
    from .adapt.staged import run_stage_b1
    from .modeling import resolve_model_name
    from .train.finetune import run_finetune
    from .vocab import vocab_sha256

    variant = variant or cfg.get("variant")
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant}")
    needs_raw = bool(cfg.get("stage_b1"))
    ctx = make_ctx(cfg, discover=True)
    processor = _processor(ctx)
    mask = resolve_mask(cfg, ctx)
    dev = _cv_cached(ctx, "dev")
    if cfg.training_set == "cv":
        train_ds = _cv_cached(ctx, "train")
        parts = {"cv_train": len(train_ds), "train_hours": round(sum(train_ds.lengths) / 16000 / 3600, 3)}
    else:
        train_ds, parts = _v2_train_parts(ctx, processor)

    stage_b1 = "not used"
    if cfg.init == "cpt":
        cpt_dir = ctx.paths.models_dir / "cpt_encoder"
        if not (cpt_dir / "config.json").exists():
            raise FileNotFoundError(f"{cpt_dir} not found - run the 'cpt' stage first.")
        cinfo = load_json(cpt_dir / "cpt_complete.json", default={}) or {}
        if not cinfo.get("completed"):
            log.warning("CPT encoder is NOT marked completed (%s); continuing with the current encoder.",
                        cinfo.get("stop_reason"))
        init = str(cpt_dir)
        if needs_raw:
            ood = _read_manifest(ctx.manifest("ood_train"), required=False)
            b1 = run_stage_b1(cfg, ood_train=_manifest_dataset(ctx, ood, processor), cv_dev=dev, processor=processor,
                              cpt_dir=cpt_dir, models_dir=ctx.paths.models_dir, results_dir=ctx.results,
                              resume=resume, previous_outputs=_previous(ctx))
            stage_b1 = b1["stage_b1"]
            if stage_b1 == "incomplete":
                print("Stage B1 stopped by the time budget - run this stage again.")
                return b1
            init = b1["init_for_b2"]
            save_json({k: v for k, v in b1.items() if k != "newly_initialised"}, ctx.results / variant / "stage_b1.json")
    else:
        init = resolve_model_name(cfg)

    extra = {"variant": variant, "training_set": cfg.training_set, "init": cfg.init, "stage_b1": stage_b1,
             "train_parts": parts, "train_hours": parts.get("train_hours"), "mask_time_prob": mask}
    summary = run_finetune(cfg, tag=variant, train_ds=train_ds, eval_ds=dev, processor=processor, init_from=init,
                           mask_time_prob=mask, max_steps=int(cfg.train.max_steps),
                           model_dir=ctx.paths.models_dir / variant, results_dir=ctx.results / variant,
                           resume=resume, previous_outputs=_previous(ctx), extra_info=extra)
    if summary["status"] != "complete":
        return summary
    model_dir = ctx.paths.models_dir / variant
    metrics = evaluate_variant(ctx, variant, model_dir, {"stage_b1": stage_b1})
    write_model_card(model_dir, variant, cfg, metrics, {**extra, "vocab_sha256": vocab_sha256(ctx.paths.shared_dir)})
    if variant in ("V3", "V4") and cfg.adapt.get("cleanup_b1_after_v4"):
        from .train.finetune import is_complete

        if is_complete(ctx.paths.models_dir / "V3") and is_complete(ctx.paths.models_dir / "V4"):
            shutil.rmtree(ctx.paths.models_dir / "B1", ignore_errors=True)
    return {**summary, "test": metrics}


def _available_variants(ctx: Ctx) -> list[str]:
    from .train.finetune import is_complete

    return [v for v in VARIANTS if is_complete(ctx.paths.models_dir / v)]


def _load_predictions(ctx: Ctx) -> dict[str, pd.DataFrame]:
    out = {}
    for v in VARIANTS:
        p = ctx.results / v / "predictions.csv"
        if p.exists():
            out[v] = pd.read_csv(p, dtype={"speaker": str}, keep_default_na=False)
    return out


def stage_significance(cfg: Config, **_) -> pd.DataFrame:
    from .evaluate.bootstrap import significance_table, system_ci

    ctx = make_ctx(cfg, discover=False)
    preds = _load_predictions(ctx)
    e = cfg.eval
    pairs = [tuple(p) for p in e.pairs]
    table = significance_table(preds, pairs, int(e.bootstrap_resamples), float(e.ci), int(cfg.seed))
    if len(table):
        table.to_csv(ctx.results / "significance.csv", index=False)
    else:
        (ctx.results / "significance.csv").unlink(missing_ok=True)
    cis = {v: {m: system_ci(df, m, int(e.bootstrap_resamples), float(e.ci), int(cfg.seed))
               for m in ("wer", "cer", "ter")} for v, df in preds.items()}
    save_json(cis, ctx.results / "system_cis.json")
    if len(table):
        print(table[["comparison", "metric", "delta", "ci_low", "ci_high", "significant"]].to_string(index=False))
    else:
        print("significance: fewer than two variants evaluated yet")
    return table


def stage_effects(cfg: Config, **_) -> pd.DataFrame | None:
    from .evaluate.effects import effects_table

    ctx = make_ctx(cfg, discover=False)
    metrics = {v: load_json(ctx.results / v / "metrics.json") for v in VARIANTS}
    metrics = {k: v for k, v in metrics.items() if v}
    try:
        table = effects_table(metrics)
    except ValueError as exc:
        print(f"effects: {exc}")
        return None
    table.to_csv(ctx.results / "factorial_effects.csv", index=False)
    print(table.to_string(index=False))
    return table


def stage_evaluate(cfg: Config, variants: list[str] | None = None, force: bool = False, **_) -> pd.DataFrame:
    """(Re)evaluate trained variants on the CV test set; write summary, significance and effects."""
    from .reporting import export_tables

    ctx = make_ctx(cfg, discover=False)
    variants = variants or _available_variants(ctx)
    rows = []
    for v in variants:
        mpath = ctx.results / v / "metrics.json"
        if force or not mpath.exists():
            evaluate_variant(ctx, v, ctx.paths.models_dir / v)
        m = load_json(mpath)
        if m:
            rows.append({k: m.get(k) for k in ("variant", "n_utts", "wer", "cer", "wer_no_tone", "ter",
                                               "ter_matched_words", "train_clips", "train_hours", "gpu_hours",
                                               "mask_time_prob", "checkpoint", "vocab_sha256")})
    summary = pd.DataFrame(rows)
    summary.to_csv(ctx.results / "summary.csv", index=False)
    if len(summary):
        print(summary[["variant", "wer", "cer", "wer_no_tone", "ter"]].to_string(index=False))
    stage_significance(cfg)
    stage_effects(cfg)
    export_tables(ctx.results)
    return summary


def stage_ablation(cfg: Config, resume: bool = True, only: str | None = None, **_) -> pd.DataFrame:
    """Leave-one-out V2 runs (1,500 steps). ``only`` selects a single run for one Kaggle session."""
    from .augment.build_augmented import build_augmented_set
    from .evaluate.ablation import resolve_run, update_table
    from .evaluate.decode import decode_cached, load_model_and_processor
    from .evaluate.metrics import all_metrics
    from .modeling import resolve_model_name
    from .train.datasets import load_cached_split
    from .train.finetune import run_finetune

    _ensure_block(cfg, "ablation", "ablation.yaml")
    ctx = make_ctx(cfg)
    table_path = ctx.results / "ablation.csv"
    tuned = resolve_mask(Config({**cfg, "mask_time_prob": "tuned"}), ctx)
    processor = _processor(ctx)
    dev = _cv_cached(ctx, "dev")
    runs = dict(cfg.ablation.runs)
    names = [only] if only else list(runs)
    unknown = [n for n in names if n not in runs]
    if unknown:
        raise ValueError(f"unknown ablation run(s) {unknown}; choose from {list(runs)}")
    done = pd.read_csv(table_path) if table_path.exists() else pd.DataFrame(columns=["run", "status"])
    table = done
    for name in names:
        if resume and ((done["run"] == name) & (done["status"] == "complete")).any():
            log.info("ablation %s already done", name)
            continue
        spec = resolve_run(name, runs[name], list(cfg.augment.transforms), tuned)
        if spec["aug_set"] != "main":
            train = _read_manifest(ctx.manifest("cv_train"))
            d = cfg.data
            build_augmented_set(train, ctx.paths.augmented_dir / spec["aug_set"], cfg.augment, int(cfg.seed),
                                spec["transforms"], d.min_seconds, d.max_seconds, d.trim_top_db,
                                ctx.found.musan if ctx.found else None, ctx.found.rirs if ctx.found else None,
                                name=spec["aug_set"])
        train_ds, parts = _v2_train_parts(ctx, processor, spec["aug_set"], spec["use_tts"])
        tag = f"abl_{name}"
        mdir = ctx.paths.models_dir / "ablation" / name
        s = run_finetune(cfg, tag=tag, train_ds=train_ds, eval_ds=dev, processor=processor,
                         init_from=resolve_model_name(cfg), mask_time_prob=spec["mask_time_prob"],
                         max_steps=int(cfg.ablation.max_steps), model_dir=mdir, results_dir=ctx.results / "ablation" / name,
                         resume=resume, previous_outputs=_previous(ctx))
        row = {"run": name, "description": spec["description"], "transforms": "+".join(spec["transforms"]),
               "use_tts": spec["use_tts"], "mask_time_prob": spec["mask_time_prob"],
               "max_steps": int(cfg.ablation.max_steps), "train_clips": len(train_ds), "status": s["status"]}
        if s["status"] == "complete":
            model, proc, device = load_model_and_processor(mdir)
            for split in ("dev", "test"):
                preds = decode_cached(model, proc, load_cached_split(ctx.paths.cache_dir / f"cv_{split}"), device,
                                      int(cfg.eval.batch_size))
                m = all_metrics(preds["reference"].tolist(), preds["hypothesis"].tolist())
                row.update({f"{split}_wer": m["wer"], f"{split}_ter": m["ter"]})
                if split == "test":
                    row.update({"test_cer": m["cer"], "test_wer_no_tone": m["wer_no_tone"]})
                    preds.to_csv(ctx.results / "ablation" / name / "test_predictions.csv", index=False)
            del model
            shutil.rmtree(mdir, ignore_errors=True)
        table = update_table(table_path, row)
        if s["status"] != "complete":
            print(f"ablation {name} interrupted by the time budget - run it again.")
            break
    print(table.to_string(index=False) if len(table) else "no ablation results yet")
    return table


def stage_analyse(cfg: Config, **_) -> dict:
    from .evaluate.error_analysis import analyse_variant, load_cis, write_figures
    from .reporting import export_tables

    ctx = make_ctx(cfg, discover=False)
    preds = _load_predictions(ctx)
    out = {}
    for v, df in preds.items():
        out[v] = analyse_variant(df, v, ctx.results / v / "error_analysis", int(cfg.eval.n_examples), int(cfg.seed))
        print(f"{v}: tonal share of WER = {out[v]['tonal_word_error_share']:.2f} points")
    summary = pd.read_csv(ctx.results / "summary.csv") if (ctx.results / "summary.csv").exists() else None
    if summary is not None and not (ctx.results / "system_cis.json").exists() and len(preds):
        stage_significance(cfg)
    figs = write_figures(ctx.results, summary, load_cis(ctx.results))
    tables = export_tables(ctx.results)
    print(f"figures: {figs}\ntables: {tables['md']}")
    return {"variants": list(out), "figures": figs}


def stage_export_tables(cfg: Config, **_) -> dict:
    from .reporting import export_tables

    ctx = make_ctx(cfg, discover=False)
    out = export_tables(ctx.results)
    print(f"written: {', '.join(str(p) for p in out.values())}")
    return {k: str(v) for k, v in out.items()}


# --------------------------------------------------------------------------------------
# dispatcher
# --------------------------------------------------------------------------------------
LIGHT_OVERRIDES = {
    "paths.work_dir": "${paths.work_root}/light", "run_name": "light", "train.max_steps": 300,
    "train.eval_steps": 100, "train.save_steps": 100, "mask_tuning.skip": True, "cpt.max_steps": 300,
    "cpt.max_total_hours": 1.0, "adapt.b1_max_steps": 200, "tts.max_per_real_clip": 0.2,
}

SEQUENCES = {
    "smoke": ["prepare", "augment", "tune_mask", "cpt", "V1", "V2", "V3", "V4", "evaluate", "analyse"],
    "all_light": ["prepare", "V1", "augment", "tune_mask", "V2", "cpt", "V3", "V4", "evaluate", "analyse"],
}

STAGE_FUNCTIONS: dict[str, Callable] = {
    "audit": stage_audit, "prepare": stage_prepare, "augment": stage_augment, "tune_mask": stage_tune_mask,
    "cpt": stage_cpt, "V1": stage_train, "V2": stage_train, "V3": stage_train, "V4": stage_train,
    "evaluate": stage_evaluate, "significance": stage_significance, "effects": stage_effects,
    "ablation": stage_ablation, "analyse": stage_analyse, "export_tables": stage_export_tables,
}
NOTEBOOK_STAGES = ["smoke", "prepare", "V1", "augment", "tune_mask", "V2", "cpt", "V3", "V4", "evaluate",
                   "ablation", "analyse", "all_light"]

PLANS = {
    "prepare": "discover data -> load CV/SLR86/ÌròyìnSpeech -> normalise (NFC) -> drop digits -> leakage filter -> "
               "tone gate -> shared vocab (hash) -> cache CV train/dev/test -> manifests -> audit + leakage report",
    "augment": "pitch check (tempo keeps F0) -> 2 augmented copies per CV train clip (FLAC) -> MMS-TTS coverage "
               "check -> TTS clips (<= 1 per real clip) -> listening sample",
    "tune_mask": "3 x V2 runs (1,000 steps) with mask_time_prob 0.05/0.075/0.10 -> choose lowest dev WER",
    "cpt": "continued wav2vec 2.0 pretraining on unlabelled audio (checkpoint every 500 steps, resumable)",
    "V1": "fine-tune XLS-R on CV train (3,000 steps) -> best dev WER -> test evaluation",
    "V2": "fine-tune XLS-R on CV train + augmented + TTS (tuned mask) -> test evaluation",
    "V3": "CPT encoder -> B1 (OOD labelled) -> B2 (CV train) -> test evaluation",
    "V4": "CPT encoder -> B1 (shared with V3) -> B2 on V2 training set (tuned mask) -> test evaluation",
    "evaluate": "decode test set for every trained variant -> summary.csv -> bootstrap significance -> effects",
    "ablation": "leave-one-out V2 runs (1,500 steps each), one per session with ABLATION_ONLY",
    "analyse": "error analysis per variant, figures (300 dpi), thesis tables",
    "smoke": "tiny model end-to-end: " + " -> ".join(SEQUENCES["smoke"]),
    "all_light": "short real-data sanity run: " + " -> ".join(SEQUENCES["all_light"]),
}


def run_stage(stage: str, overrides: dict | None = None, smoke: bool = False, resume: bool = True,
              only: str | None = None, dry_run: bool = False, extra_files: list | None = None) -> dict:
    """Run one stage, a ``+``-joined list of stages (``augment+tune_mask``) or a sequence (smoke/all_light)."""
    if "+" in stage:
        return {s: run_stage(s, overrides, smoke, resume, only, dry_run, extra_files) for s in stage.split("+")}
    if stage in SEQUENCES:
        ov = dict(overrides or {})
        if stage == "all_light":
            ov = {**LIGHT_OVERRIDES, **ov}
        results = {}
        for s in SEQUENCES[stage]:
            print(f"\n{'=' * 30} {stage}: {s} {'=' * 30}")
            results[s] = run_stage(s, ov, smoke or stage == "smoke", resume, only, dry_run, extra_files)
        return results
    if stage not in STAGE_FUNCTIONS:
        raise ValueError(f"unknown stage {stage!r}; choose from {sorted(set(STAGE_FUNCTIONS) | set(SEQUENCES))}")
    if dry_run:
        print(f"[dry-run] {stage}: {PLANS.get(stage, STAGE_FUNCTIONS[stage].__doc__ or '')}")
        return {"stage": stage, "dry_run": True, "function": STAGE_FUNCTIONS[stage].__name__}
    cfg = stage_config(stage, overrides, smoke, extra_files)
    if stage in ("cpt",):
        _ensure_block(cfg, "cpt", "cpt.yaml")
    kwargs = {"resume": resume, "only": only}
    if stage in VARIANTS:
        kwargs["variant"] = stage
    result = STAGE_FUNCTIONS[stage](cfg, **kwargs)
    return result if isinstance(result, dict) else {"stage": stage, "done": True}


def next_step_message(stage: str) -> str:
    nxt, attach = NEXT_STAGE.get(stage, ("see docs/KAGGLE_GUIDE.md", ""))
    return f"NEXT: set STAGE = \"{nxt}\". {attach}"


def environment_report(cfg: Config | None = None) -> dict:
    from .utils import disk_free_gb

    cfg = cfg or stage_config("prepare")
    info = gpu_info()
    info["disk_free_gb_work"] = disk_free_gb(cfg.paths.work_root)
    info["disk_free_gb_tmp"] = disk_free_gb(cfg.paths.tmp_dir)
    return info


def safe_run(fn: Callable, *a, **k):
    try:
        return fn(*a, **k)
    except Exception:
        traceback.print_exc()
        raise
