"""Audio transforms. Every transform takes a ``numpy.random.Generator`` so results are reproducible.

| transform | pitch      | use           |
|-----------|------------|---------------|
| tempo     | preserved  | main          |
| noise     | preserved  | main          |
| reverb    | preserved  | main          |
| gain      | preserved  | main          |
| speed     | SHIFTED    | ablation only |
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.signal import fftconvolve

from ..utils import get_logger

log = get_logger(__name__)
SR = 16000
MAIN_TRANSFORMS = ("tempo", "noise", "reverb", "gain")


# ---------------------------------------------------------------- tempo / speed
def tempo(x: np.ndarray, factor: float) -> np.ndarray:
    """Time-stretch by ``factor`` (>1 = faster) with pitch preserved (phase vocoder)."""
    import librosa

    return librosa.effects.time_stretch(x.astype(np.float32), rate=float(factor)).astype(np.float32)


def speed(x: np.ndarray, factor: float, sr: int = SR) -> np.ndarray:
    """Speed perturbation by resampling: changes duration AND pitch (ablation only)."""
    import librosa

    return librosa.resample(x.astype(np.float32), orig_sr=sr, target_sr=int(round(sr / float(factor)))).astype(np.float32)


# ---------------------------------------------------------------- noise
def colored_noise(n: int, rng: np.random.Generator, beta: float = 1.0) -> np.ndarray:
    """Noise with a 1/f^beta power spectrum (0 = white, 1 = pink, 2 = brown)."""
    white = rng.standard_normal(n)
    spec = np.fft.rfft(white)
    f = np.fft.rfftfreq(n)
    f[0] = f[1] if len(f) > 1 else 1.0
    spec = spec / (f ** (beta / 2.0))
    y = np.fft.irfft(spec, n=n)
    return (y / (np.std(y) + 1e-9)).astype(np.float32)


def mix_at_snr(x: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    ps = float(np.mean(x ** 2)) + 1e-12
    pn = float(np.mean(noise ** 2)) + 1e-12
    scale = np.sqrt(ps / (pn * 10 ** (snr_db / 10.0)))
    return (x + scale * noise).astype(np.float32)


class NoiseBank:
    """Noise source: MUSAN files when available, otherwise generated coloured noise."""

    def __init__(self, files: list[str] | None = None, max_files: int = 2000):
        self.files = sorted(files or [])[:max_files]
        self._cache: dict[str, np.ndarray] = {}

    @classmethod
    def from_dir(cls, musan_dir: str | Path | None) -> "NoiseBank":
        if not musan_dir:
            return cls([])
        root = Path(musan_dir)
        files = [str(p) for p in root.rglob("*") if p.suffix.lower() in (".wav", ".flac")]
        log.info("noise bank: %d files from %s", len(files), root)
        return cls(files)

    def sample(self, n: int, rng: np.random.Generator) -> tuple[np.ndarray, str]:
        if self.files:
            path = self.files[int(rng.integers(len(self.files)))]
            if path not in self._cache:
                from ..audio import load_audio

                try:
                    self._cache[path] = load_audio(path, SR)
                except Exception:
                    self._cache[path] = np.zeros(0, np.float32)
                if len(self._cache) > 64:
                    self._cache.pop(next(iter(self._cache)))
            src = self._cache[path]
            if len(src) > SR // 4:
                if len(src) < n:
                    src = np.tile(src, int(np.ceil(n / len(src))))
                start = int(rng.integers(0, len(src) - n + 1))
                return src[start: start + n].astype(np.float32), f"musan:{Path(path).name}"
        beta = float(rng.choice([0.0, 1.0, 2.0]))
        return colored_noise(n, rng, beta), f"colored(beta={beta:g})"


def add_noise(x: np.ndarray, rng: np.random.Generator, bank: NoiseBank, snr_range=(10, 30)) -> tuple[np.ndarray, str]:
    snr = float(rng.uniform(*snr_range))
    noise, name = bank.sample(len(x), rng)
    return mix_at_snr(x, noise, snr), f"noise({name},snr={snr:.1f})"


# ---------------------------------------------------------------- reverberation
def synthetic_rir(rt60: float, rng: np.random.Generator, sr: int = SR) -> np.ndarray:
    """Exponentially decaying noise RIR (fallback when pyroomacoustics is unavailable)."""
    n = int(rt60 * 1.2 * sr)
    t = np.arange(n) / sr
    rir = rng.standard_normal(n) * np.exp(-6.9 * t / rt60)
    rir[0] = 1.0
    return (rir / np.max(np.abs(rir))).astype(np.float32)


def simulate_rir(rng: np.random.Generator, rt60_range=(0.2, 0.8), sr: int = SR) -> tuple[np.ndarray, float]:
    """Random shoebox room RIR with pyroomacoustics (image-source method)."""
    rt60 = float(rng.uniform(*rt60_range))
    try:
        import pyroomacoustics as pra
    except ImportError:
        return synthetic_rir(rt60, rng, sr), rt60
    for _ in range(10):
        dims = rng.uniform([3.0, 3.0, 2.5], [10.0, 8.0, 4.0])
        try:
            e_abs, max_order = pra.inverse_sabine(rt60, dims)
            room = pra.ShoeBox(dims, fs=sr, materials=pra.Material(e_abs), max_order=min(int(max_order), 17))
            src = rng.uniform([0.5, 0.5, 1.0], dims - 0.5)
            mic = rng.uniform([0.5, 0.5, 1.0], dims - 0.5)
            room.add_source(src)
            room.add_microphone(mic.reshape(3, 1))
            room.compute_rir()
            rir = np.asarray(room.rir[0][0], dtype=np.float32)
            if rir.size and np.all(np.isfinite(rir)) and np.max(np.abs(rir)) > 0:
                return rir / np.max(np.abs(rir)), rt60
        except Exception:
            continue
    return synthetic_rir(rt60, rng, sr), rt60


class RIRBank:
    """Fixed, seeded bank of RIRs (simulated, or real OpenSLR RIRs when provided)."""

    def __init__(self, rirs: list[np.ndarray], labels: list[str]):
        self.rirs, self.labels = rirs, labels

    @classmethod
    def build(cls, size: int, seed: int, rt60_range=(0.2, 0.8), rir_dir: str | Path | None = None) -> "RIRBank":
        rirs, labels = [], []
        if rir_dir:
            from ..audio import load_audio

            files = sorted(p for p in Path(rir_dir).rglob("*") if p.suffix.lower() in (".wav", ".flac"))
            files = [p for p in files if "noise" not in p.name.lower()][: max(size, 1) * 4]
            for p in files[:size]:
                try:
                    r = load_audio(p, SR)
                    if r.size:
                        rirs.append(r / (np.max(np.abs(r)) + 1e-9)); labels.append(f"real:{p.name}")
                except Exception:
                    continue
        rng = np.random.default_rng([seed, 7919])
        while len(rirs) < size:
            r, rt60 = simulate_rir(rng, rt60_range)
            rirs.append(r); labels.append(f"sim(rt60={rt60:.2f})")
        return cls(rirs, labels)


def reverb(x: np.ndarray, rir: np.ndarray) -> np.ndarray:
    """Convolve with an RIR, align the direct path and keep the original length and RMS."""
    d = int(np.argmax(np.abs(rir)))
    y = fftconvolve(x, rir)[d: d + len(x)]
    if len(y) < len(x):
        y = np.pad(y, (0, len(x) - len(y)))
    rms_x, rms_y = np.sqrt(np.mean(x ** 2)) + 1e-9, np.sqrt(np.mean(y ** 2)) + 1e-9
    return (y * rms_x / rms_y).astype(np.float32)


# ---------------------------------------------------------------- gain
def gain(x: np.ndarray, db: float) -> np.ndarray:
    """Apply gain in dB, then scale down only if the result would clip."""
    y = x * (10 ** (db / 20.0))
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 0.99:
        y = y / peak * 0.99
    return y.astype(np.float32)


# ---------------------------------------------------------------- chains
def choose_chain(rng: np.random.Generator, transforms: list[str], p: float = 0.5) -> list[str]:
    """Each transform is included with probability ``p``; at least one is always applied."""
    chain = [t for t in transforms if rng.random() < p]
    if not chain:
        chain = [transforms[int(rng.integers(len(transforms)))]]
    order = ["speed", "tempo", "reverb", "noise", "gain"]
    return sorted(chain, key=order.index)


def apply_chain(x: np.ndarray, chain: list[str], rng: np.random.Generator, cfg_aug, noise_bank: NoiseBank,
                rir_bank: RIRBank) -> tuple[np.ndarray, list[str]]:
    applied = []
    y = x
    for t in chain:
        if t == "tempo":
            f = float(rng.choice(cfg_aug["tempo_factors"]))
            y = tempo(y, f); applied.append(f"tempo({f})")
        elif t == "speed":
            f = float(rng.choice(cfg_aug["speed_factors"]))
            y = speed(y, f); applied.append(f"speed({f})")
        elif t == "reverb":
            k = int(rng.integers(len(rir_bank.rirs)))
            y = reverb(y, rir_bank.rirs[k]); applied.append(f"reverb({rir_bank.labels[k]})")
        elif t == "noise":
            y, desc = add_noise(y, rng, noise_bank, tuple(cfg_aug["snr_db"])); applied.append(desc)
        elif t == "gain":
            db = float(rng.uniform(*cfg_aug["gain_db"]))
            y = gain(y, db); applied.append(f"gain({db:+.1f}dB)")
        else:
            raise ValueError(f"unknown transform {t}")
    return y, applied


def median_f0(x: np.ndarray, sr: int = SR) -> float:
    """Median F0 (Hz) over voiced frames using librosa.pyin."""
    import librosa

    f0, voiced, _ = librosa.pyin(x.astype(np.float32), fmin=60, fmax=500, sr=sr, frame_length=1024)
    f0 = f0[np.isfinite(f0)]
    return float(np.median(f0)) if f0.size else float("nan")
