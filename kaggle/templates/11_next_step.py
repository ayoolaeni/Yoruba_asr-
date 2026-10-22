# ---- Next step ------------------------------------------------------------------------------------
from IPython.display import Markdown, display

_msg = pipeline.next_step_message(STAGE.split("+")[-1] if "+" in STAGE and STAGE != "augment+tune_mask" else STAGE)
_incomplete = isinstance(RESULT, dict) and (RESULT.get("status") == "incomplete" or RESULT.get("completed") is False)
if _incomplete:
    _msg = (f"The stage **{STAGE}** stopped at the session time budget and saved a checkpoint. "
            f"Run **the same STAGE again** with RESUME=True and attach this version's output.")
display(Markdown(f"### Next step\n\n{_msg}\n\nIn the next version: *Add Input → Your Work → this notebook "
                 "→ the version you just ran* (keep the data datasets attached), then edit `STAGE` and "
                 "*Save & Run All*."))
print(_msg)
