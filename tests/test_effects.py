import pytest

from yoruba_asr.evaluate.effects import effects_table, factorial_effects


def test_factorial_formulas():
    e = factorial_effects(v1=40.0, v2=36.0, v3=34.0, v4=28.0)
    assert e["augmentation_main_effect"] == pytest.approx(((36 - 40) + (28 - 34)) / 2)  # -5
    assert e["adaptation_main_effect"] == pytest.approx(((34 - 40) + (28 - 36)) / 2)  # -7
    assert e["interaction"] == pytest.approx((28 - 34) - (36 - 40))  # -2


def test_no_interaction_when_additive():
    assert factorial_effects(30, 25, 20, 15)["interaction"] == 0


def test_effects_table_requires_all_variants():
    m = {v: {"wer": 1.0, "cer": 1.0, "wer_no_tone": 1.0, "ter": 1.0} for v in ("V1", "V2", "V3", "V4")}
    t = effects_table(m)
    assert list(t["metric"]) == ["wer", "cer", "wer_no_tone", "ter"]
    with pytest.raises(ValueError):
        effects_table({"V1": m["V1"]})
