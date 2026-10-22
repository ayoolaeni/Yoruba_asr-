"""Audio loading, resampling, silence trimming and duration filtering."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .utils import get_logger

log = get_logger(__name__)
SR = 16000


class AudioLoadError(RuntimeError):
    pass


def _to_mono(x: np.ndarray) -> np.ndarray:
    if x.ndim == 2:
        # soundfile returns (frames, channels)
        x = x.mean(axis=1)
    return x


def resample(x: np.ndarray, orig_sr: int, sr: int = SR) -> np.ndarray:
    if orig_sr == sr:
        return x.astype(np.float32, copy=False)
    import librosa

    return librosa.resample(x.astype(np.float32), orig_sr=orig_sr, target_sr=sr).astype(np.float32)


def load_audio(path: str | Path, sr: int = SR) -> np.ndarray:
    """Decode MP3/WAV/FLAC/OGG/M4A, convert to mono float32, resample to ``sr``.

    The signal is peak-normalised only if it would clip (|x| > 1).
    Raises :class:`AudioLoadError` for corrupt or empty files.
    """
    path = str(path)
    x = None
    orig_sr = None
    err: Exception | None = None
    try:
        import soundfile as sf

        x, orig_sr = sf.read(path, dtype="float32", always_2d=False)
    except Exception as exc:  # unsupported format -> fall back to librosa/audioread
        err = exc
        try:
            import librosa

            x, orig_sr = librosa.load(path, sr=None, mono=True)
        except Exception as exc2:
            raise AudioLoadError(f"cannot decode {path}: {exc2} (soundfile: {err})") from exc2
    x = _to_mono(np.asarray(x, dtype=np.float32))
    if x.size == 0:
        raise AudioLoadError(f"empty audio: {path}")
    if not np.all(np.isfinite(x)):
        raise AudioLoadError(f"non-finite samples: {path}")
    x = resample(x, int(orig_sr), sr)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak > 1.0:
        x = x / peak * 0.999
    return x.astype(np.float32)


def trim_silence(x: np.ndarray, top_db: float = 30) -> np.ndarray:
    """Trim leading/trailing silence (librosa ``effects.trim``)."""
    import librosa

    if x.size == 0:
        return x
    y, _ = librosa.effects.trim(x, top_db=top_db)
    return y if y.size else x


def duration(x: np.ndarray, sr: int = SR) -> float:
    return float(len(x)) / sr


def probe(path: str | Path) -> tuple[float, int] | None:
    """(duration seconds, native sample rate) from the file header, or None if unreadable."""
    try:
        import soundfile as sf

        info = sf.info(str(path))
        if info.frames <= 0:
            return None
        return info.frames / info.samplerate, int(info.samplerate)
    except Exception:
        try:
            import librosa

            return float(librosa.get_duration(path=str(path))), int(librosa.get_samplerate(str(path)))
        except Exception:
            return None


def duration_reason(dur: float, min_s: float, max_s: float) -> str | None:
    """None if the clip is kept, else the reason it is dropped."""
    if dur < min_s:
        return "too_short"
    if dur > max_s:
        return "too_long"
    return None


def load_trimmed(path: str | Path, sr: int = SR, top_db: float = 30) -> np.ndarray:
    return trim_silence(load_audio(path, sr), top_db=top_db)


def safe_load_trimmed(path: str | Path, sr: int = SR, top_db: float = 30) -> np.ndarray | None:
    """Like :func:`load_trimmed` but logs and returns None for corrupt files."""
    try:
        return load_trimmed(path, sr, top_db)
    except Exception as exc:
        log.warning("skipping corrupt audio %s: %s", path, exc)
        return None


def save_flac(path: str | Path, x: np.ndarray, sr: int = SR) -> None:
    """Write 16-bit FLAC (clip-safe)."""
    import soundfile as sf

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak > 0.999:
        x = x / peak * 0.999
    sf.write(str(path), x.astype(np.float32), sr, subtype="PCM_16", format="FLAC")


def split_on_silence(x: np.ndarray, sr: int = SR, max_seconds: float = 15.0, top_db: float = 35) -> list[np.ndarray]:
    """Split long audio into chunks of at most ``max_seconds``, cutting at silences."""
    import librosa

    max_len = int(max_seconds * sr)
    if len(x) <= max_len:
        return [x]
    intervals = librosa.effects.split(x, top_db=top_db)
    if len(intervals) == 0:
        intervals = np.array([[0, len(x)]])
    chunks: list[np.ndarray] = []
    start, end = None, None
    for s, e in intervals:
        if start is None:
            start, end = s, e
        elif e - start <= max_len:
            end = e
        else:
            chunks.append(x[start:end])
            start, end = s, e
    if start is not None:
        chunks.append(x[start:end])
    out: list[np.ndarray] = []
    for c in chunks:  # hard-split any chunk that is still too long
        for i in range(0, len(c), max_len):
            piece = c[i: i + max_len]
            if len(piece) > int(0.2 * sr):
                out.append(piece)
    return out or [x[:max_len]]
