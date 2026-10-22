import numpy as np

from yoruba_asr.inference import Transcriber


def test_transcriber_returns_text(trained_model, fixture_dir):
    t = Transcriber.from_dir(trained_model, device="cpu")
    wav = next((fixture_dir / "common_voice").rglob("*.wav"))
    r = t.transcribe_file(wav)
    assert set(r) >= {"text", "text_notone", "seconds"}
    assert isinstance(r["text"], str)
    batch = t.transcribe_batch([str(wav), str(wav)], batch_size=2)
    assert len(batch) == 2 and batch[0]["text"] == batch[1]["text"]


def test_long_audio_is_chunked(trained_model):
    t = Transcriber.from_dir(trained_model, device="cpu")
    sr = 16000
    speech = (0.3 * np.sin(np.arange(sr * 9) / 3)).astype(np.float32)
    x = np.concatenate([speech, np.zeros(sr, np.float32), speech, np.zeros(sr, np.float32), speech])
    r = t.transcribe_array(x, sr)
    assert r["chunks"] >= 2
    # integer PCM input and other sampling rates are accepted
    r2 = t.transcribe_array((speech[:sr * 2] * 32767).astype(np.int16), sr)
    assert isinstance(r2["text"], str)
