"""ÌròyìnSpeech (Ògúnrẹ̀mí et al., 2024) loader, built on the generic manifest loader.

The exact release layout is not fixed in the project specification, so the loader:
1. uses ``data.iroyin.manifest`` when given;
2. otherwise auto-detects a CSV/TSV/JSONL in the ÌròyìnSpeech folder that has an audio-path column
   and a transcript column;
3. otherwise raises :class:`ManifestError` explaining how to create the manifest.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .generic import MANIFEST_HELP, ManifestError, find_manifest, load_manifest


def load_iroyin(folder: str | Path | None = None, manifest: str | Path | None = None) -> pd.DataFrame:
    if manifest is None:
        if folder is None:
            raise ManifestError("ÌròyìnSpeech: no folder or manifest given. " + MANIFEST_HELP)
        manifest = find_manifest(folder)
        if manifest is None:
            raise ManifestError(
                f"ÌròyìnSpeech: no manifest with audio/transcript columns found under {folder}. " + MANIFEST_HELP)
    return load_manifest(manifest, corpus="iroyin", split="ood_train", audio_root=folder)
