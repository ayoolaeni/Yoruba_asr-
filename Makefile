PY ?= python

.PHONY: install test smoke notebook notebook-test app fixture demo-assets

install:
	$(PY) -m pip install -r requirements.txt && $(PY) -m pip install --no-deps -e .

fixture:
	$(PY) scripts/make_tiny_fixture.py

test:
	$(PY) -m pytest

smoke:            ## prepare -> augment -> cpt -> V1..V4 -> evaluate -> significance -> effects -> analyse (tiny, CPU)
	$(PY) -m yoruba_asr.cli smoke

notebook:         ## regenerate kaggle/yoruba_asr_kaggle.ipynb from the package source
	$(PY) kaggle/build_notebook.py

notebook-test:    ## build + execute the notebook locally with STAGE="smoke"
	$(PY) kaggle/build_notebook.py --test

demo-assets:
	$(PY) scripts/make_demo_assets.py

app:
	$(PY) app/app.py
