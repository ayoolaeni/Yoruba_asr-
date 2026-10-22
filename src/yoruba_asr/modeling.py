"""Model construction: XLS-R CTC models and the tiny random model used for CPU smoke tests."""
from __future__ import annotations

import warnings
from pathlib import Path

from .config import Config
from .utils import get_logger

log = get_logger(__name__)
# The frozen CNN feature encoder is still wrapped by gradient checkpointing; torch then warns that its
# (frozen) inputs need no gradient. Verified harmless: all trainable layers receive gradients.
warnings.filterwarnings("ignore", message="None of the inputs have requires_grad=True")
TINY_NAME = "tiny-random"


def tiny_wav2vec2_config():
    """A very small Wav2Vec2 config (hidden 32, 2 layers, 3 conv layers, 80x downsampling)."""
    from transformers import Wav2Vec2Config

    return Wav2Vec2Config(
        vocab_size=8, hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=37,
        conv_dim=(32, 32, 32), conv_stride=(5, 4, 4), conv_kernel=(10, 4, 4), num_feat_extract_layers=3,
        conv_bias=True, feat_extract_norm="layer", do_stable_layer_norm=True,
        num_conv_pos_embeddings=16, num_conv_pos_embedding_groups=2,
        num_codevector_groups=2, num_codevectors_per_group=8, codevector_dim=16, proj_codevector_dim=16,
        num_negatives=10, diversity_loss_weight=0.1, mask_time_prob=0.05, mask_time_length=4,
        hidden_dropout=0.0, attention_dropout=0.0, activation_dropout=0.0, feat_proj_dropout=0.0,
        final_dropout=0.0, layerdrop=0.0,
    )


def ensure_tiny_init(work_dir: str | Path, seed: int = 42) -> Path:
    """Create (once) a tiny random ``Wav2Vec2ForPreTraining`` checkpoint that plays the role of XLS-R."""
    import torch
    from transformers import Wav2Vec2ForPreTraining

    path = Path(work_dir) / "tiny_init"
    if (path / "config.json").exists():
        return path
    torch.manual_seed(seed)
    model = Wav2Vec2ForPreTraining(tiny_wav2vec2_config())
    model.save_pretrained(path)
    return path


def resolve_model_name(cfg: Config) -> str:
    name = cfg.model.name
    if name == TINY_NAME:
        return str(ensure_tiny_init(cfg.paths.work_dir, cfg.seed))
    return name


def ctc_kwargs(cfg: Config, processor, mask_time_prob: float) -> dict:
    m = cfg.model
    tiny = cfg.model.name == TINY_NAME
    return dict(
        attention_dropout=m.attention_dropout, hidden_dropout=m.hidden_dropout,
        feat_proj_dropout=m.feat_proj_dropout, layerdrop=m.layerdrop,
        mask_time_prob=float(mask_time_prob), mask_time_length=4 if tiny else m.mask_time_length,
        mask_feature_prob=m.mask_feature_prob,
        ctc_loss_reduction=m.ctc_loss_reduction, ctc_zero_infinity=m.ctc_zero_infinity,
        pad_token_id=processor.tokenizer.pad_token_id, vocab_size=len(processor.tokenizer),
    )


def load_ctc_model(init_from: str, cfg: Config, processor, mask_time_prob: float):
    """``Wav2Vec2ForCTC.from_pretrained`` with the project's settings; logs newly initialised weights."""
    from transformers import Wav2Vec2ForCTC

    model, info = Wav2Vec2ForCTC.from_pretrained(init_from, output_loading_info=True,
                                                 **ctc_kwargs(cfg, processor, mask_time_prob))
    if info.get("missing_keys"):
        log.info("Newly initialised weights (expected for a new CTC head): %s", info["missing_keys"])
    model.config.mask_time_prob = float(mask_time_prob)
    if cfg.model.freeze_feature_encoder:
        model.freeze_feature_encoder()
    if cfg.model.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    return model, info
