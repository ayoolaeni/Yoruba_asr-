# ---- Install pinned extra packages (skips packages already present in the Kaggle image) -------
REQUIREMENTS = """{{REQUIREMENTS}}"""
import importlib.metadata as _md, re as _re

def _needs(req: str) -> bool:
    name, _, ver = req.partition("==")
    try:
        have = _md.version(name.strip())
    except _md.PackageNotFoundError:
        return True
    return bool(ver) and have != ver.strip()

_reqs = [r.split("#")[0].strip() for r in REQUIREMENTS.splitlines() if r.split("#")[0].strip()]
_todo = [r for r in _reqs if _needs(r)]
if ON_KAGGLE and _todo:
    print("pip install:", " ".join(_todo))
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *_todo], check=True)
else:
    print("nothing to install" if not _todo else f"local test: skipping pip install of {_todo}")
