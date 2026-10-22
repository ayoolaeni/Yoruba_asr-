import numpy as np
import pytest
import soundfile as sf

from yoruba_asr.audio import (AudioLoadError, duration_reason, load_audio, safe_load_trimmed, split_on_silence,
                              trim_silence)


def test_resample_and_mono(tmp_path):
    sr = 44100
    t = np.arange(sr) / sr
    stereo = np.stack([0.5 * np.sin(2 * np.pi * 220 * t), 0.5 * np.sin(2 * np.pi * 330 * t)], axis=1)
    p = tmp_path / "stereo.wav"
    sf.write(p, stereo, sr)
    x = load_audio(p, 16000)
    assert x.ndim == 1
    assert abs(len(x) - 16000) <= 2
    assert x.dtype == np.float32


def test_peak_normalised_only_if_clipping(tmp_path):
    p = tmp_path / "quiet.flac"
    sf.write(p, 0.1 * np.ones(16000, dtype=np.float32), 16000)
    assert np.isclose(np.max(np.abs(load_audio(p))), 0.1, atol=1e-3)


def test_duration_filter():
    assert duration_reason(0.3, 0.5, 15) == "too_short"
    assert duration_reason(16.0, 0.5, 15) == "too_long"
    assert duration_reason(3.0, 0.5, 15) is None


def test_trim_silence():
    x = np.concatenate([np.zeros(16000), 0.5 * np.sin(np.arange(16000) / 5), np.zeros(16000)]).astype(np.float32)
    assert len(trim_silence(x)) < 20000


def test_corrupt_file_is_skipped(tmp_path):
    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"not audio at all")
    with pytest.raises(AudioLoadError):
        load_audio(bad)
    assert safe_load_trimmed(bad) is None


def test_split_long_audio():
    sr = 16000
    speech = 0.3 * np.sin(np.arange(sr * 8) / 4)
    x = np.concatenate([speech, np.zeros(sr), speech, np.zeros(sr), speech]).astype(np.float32)
    chunks = split_on_silence(x, sr, max_seconds=15)
    assert len(chunks) >= 2 and all(len(c) <= 15 * sr for c in chunks)
