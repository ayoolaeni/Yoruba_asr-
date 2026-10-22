"""Automatic download of the Google Yoruba speech corpus (OpenSLR 86, CC BY-SA 4.0).

No Kaggle upload is needed: the notebook downloads ~0.9 GB from OpenSLR mirrors (Internet must be On).
Files (verified on the OpenSLR 86 page, 2026-10-20): yo_ng_female.zip (462 MB), yo_ng_male.zip (445 MB),
each a flat archive with line_index.tsv + 48 kHz mono WAVs.
"""
from __future__ import annotations

import shutil
import time
import urllib.request
import zipfile
from pathlib import Path

from ..utils import get_logger

log = get_logger(__name__)
MIRRORS = ["https://openslr.elda.org/resources/86", "https://openslr.trmal.net/resources/86",
           "https://openslr.magicdatatech.com/resources/86", "https://www.openslr.org/resources/86"]
ARCHIVES = ["yo_ng_female.zip", "yo_ng_male.zip"]


def _fetch(name: str, dest: Path, retries: int = 3) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    last: Exception | None = None
    for mirror in MIRRORS:
        for attempt in range(retries):
            try:
                log.info("downloading %s/%s", mirror, name)
                with urllib.request.urlopen(f"{mirror}/{name}", timeout=60) as r, open(tmp, "wb") as fh:
                    shutil.copyfileobj(r, fh, length=1 << 20)
                tmp.replace(dest)
                return
            except Exception as exc:  # try again, then the next mirror
                last = exc
                time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Could not download {name} from any OpenSLR mirror ({last}). "
                       "Check that Internet is On in the Kaggle notebook settings.")


def download_openslr86(dest_root: str | Path) -> list[Path]:
    """Download + extract both archives into ``dest_root/yo_ng_{female,male}/`` (once). Returns the folders."""
    dest_root = Path(dest_root)
    dest_root.mkdir(parents=True, exist_ok=True)
    folders = []
    for name in ARCHIVES:
        folder = dest_root / name.replace(".zip", "")
        if (folder / ".done").exists():
            folders.append(folder)
            continue
        archive = dest_root / name
        if not archive.exists():
            _fetch(name, archive)
        folder.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as zf:
            for member in zf.namelist():
                if not str((folder / member).resolve()).startswith(str(folder.resolve())):
                    raise RuntimeError(f"unsafe path in {name}: {member}")
            zf.extractall(folder)
        archive.unlink()  # keep disk usage low
        (folder / ".done").write_text("ok", encoding="utf-8")
        folders.append(folder)
        log.info("OpenSLR 86: %s ready (%d files)", folder.name, sum(1 for _ in folder.iterdir()))
    return folders
