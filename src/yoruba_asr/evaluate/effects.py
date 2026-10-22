"""2 x 2 factorial effects from the four variants.

* Augmentation main effect = mean[(V2 - V1), (V4 - V3)]
* Adaptation main effect   = mean[(V3 - V1), (V4 - V2)]
* Interaction              = (V4 - V3) - (V2 - V1)
Negative values mean the treatment lowers the error rate.
"""
from __future__ import annotations

import pandas as pd

DEFAULT_METRICS = ("wer", "cer", "wer_no_tone", "ter")


def factorial_effects(v1: float, v2: float, v3: float, v4: float) -> dict:
    return {
        "augmentation_main_effect": ((v2 - v1) + (v4 - v3)) / 2,
        "adaptation_main_effect": ((v3 - v1) + (v4 - v2)) / 2,
        "interaction": (v4 - v3) - (v2 - v1),
    }


def effects_table(metrics: dict[str, dict], keys=DEFAULT_METRICS) -> pd.DataFrame:
    rows = []
    if not all(v in metrics for v in ("V1", "V2", "V3", "V4")):
        missing = [v for v in ("V1", "V2", "V3", "V4") if v not in metrics]
        raise ValueError(f"Factorial effects need all four variants; missing {missing}")
    for k in keys:
        vals = [float(metrics[v][k]) for v in ("V1", "V2", "V3", "V4")]
        rows.append({"metric": k, "V1": vals[0], "V2": vals[1], "V3": vals[2], "V4": vals[3],
                     **factorial_effects(*vals)})
    return pd.DataFrame(rows)
