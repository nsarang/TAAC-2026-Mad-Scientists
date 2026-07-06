SHELL := /usr/bin/env bash

CONDA := conda
CONDA_ENVIRONMENT ?= mad-scientists
DEACTIVATE_VENV := if [ -n "$$VIRTUAL_ENV" ]; then export PATH=$$(echo "$$PATH" | sed -e "s|:$$VIRTUAL_ENV/bin||g" -e "s|$$VIRTUAL_ENV/bin:||g"); unset VIRTUAL_ENV; fi
CONDA_INIT := . $$($(CONDA) info | awk '/base environment/ {print $$4}')/etc/profile.d/$(CONDA).sh
CONDA_ACTIVATE := $(DEACTIVATE_VENV) && $(CONDA_INIT) && $(CONDA) activate $(CONDA_ENVIRONMENT)

DOCS_PORT ?= 58432
NOTEBOOKS_PORT ?= 3456
BUNDLE_OUT ?= dist/submission

# Build Cython extensions in-place
build-ext:
	$(CONDA_ACTIVATE) && \
	python setup.py build_ext --inplace

# Remove compiled .so and Cython build artifacts
clean-ext:
	find . -name '*.so' -path '*/core/*' -exec rm -f {} +
	rm -f core/utils/_hashing.c
	rm -rf build/ cython_debug/

# Remove all generated files (bytecode, caches, coverage, eggs)
clean: clean-ext
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -rf {} +
	find . -name '.coverage*' -exec rm -rf {} +
	find . -name '.pytest_cache' -exec rm -rf {} +
	find . -name 'coverage.*' -exec rm -rf {} +
	find . -name '*.egg-info' -exec rm -rf {} +
	find . -name '.eggs' -exec rm -rf {} +

# Install core application dependencies
install-deps:
	$(CONDA_ACTIVATE) && \
	uv pip install -r requirements/app.txt

# Install core + GPU dependencies (CUDA, etc.)
install-deps-gpu:
	$(CONDA_ACTIVATE) && \
	uv pip install -r requirements/gpu.txt

# Install core + dev dependencies (linters, test tools)
install-deps-dev:
	$(MAKE) install-deps
	$(CONDA_ACTIVATE) && uv pip install -r requirements/dev.txt

# Create the conda environment from environment.yaml
create-environment:
	$(CONDA_INIT) && \
	$(CONDA) env create --name $(CONDA_ENVIRONMENT) --file environment.yaml
	$(MAKE) install-deps-dev

# Update the conda environment to match environment.yaml
update-environment:
	$(CONDA_INIT) && \
	$(CONDA) env update --name $(CONDA_ENVIRONMENT) --file environment.yaml
	$(MAKE) install-deps-dev

# Run all pre-commit hooks on every file
pre-commit:
	$(CONDA_ACTIVATE) && \
	pre-commit run --all-files

# Run pytest with parallel workers, coverage, and verbose output
# Pass extra args: make pytest ARGS="--lf" or make pytest ARGS="-k bundler"
pytest:
	$(CONDA_ACTIVATE) && \
	pytest -vv --tb=long -n auto --durations=0 --cov=core --cov-report=term-missing $(ARGS); \
	rc=$$?; if [ $$rc -eq 5 ]; then echo "No tests collected (exit 5) — OK"; exit 0; fi; exit $$rc

# Run only tests marked @pytest.mark.nightly
pytest-nightly:
	$(MAKE) pytest ARGS="-m nightly"

# Rerun only previously failed tests
pytest-lf:
	$(MAKE) pytest ARGS="--lf"

# Run failed tests first, then the rest
pytest-ff:
	$(MAKE) pytest ARGS="--ff"

# Run linting then tests
test: pre-commit pytest


# Bundle execute.py + config into a submission package
# Optional: PRIOR_LOG=experiment_logs/run1.txt to warm-start TPE from a previous run
bundle-submission:
ifndef CONFIG
	$(error CONFIG is required. Usage: make bundle-submission CONFIG="configs/hyformer_v2/default.yaml" or CONFIG="configs/a.yaml configs/b.yaml")
endif
	rm -rf $(BUNDLE_OUT)
	$(CONDA_ACTIVATE) && \
	python tools/bundle.py scripts/execute.py \
		-o $(BUNDLE_OUT)/execute.py \
		$$(for f in $(CONFIG); do echo -c $$f; done) \
		-a scripts/submission/run.sh \
		-a scripts/submission/infer.py
	@$(CONDA_ACTIVATE) && python -m tools.save_configs \
		--out $(BUNDLE_OUT) \
		--commit $$(git rev-parse --short HEAD) \
		-- $(CONFIG)
	# Flatten extra-gpu.txt (resolve -r includes) into one requirements file
	awk '/^-r /{system("cat requirements/"substr($$0,4)); next} !/^#|^$$/{print}' requirements/extra-gpu.txt | sort -u > $(BUNDLE_OUT)/extra.txt
ifdef PRIOR_LOG
	$(CONDA_ACTIVATE) && python tools/extract_prior.py $(PRIOR_LOG) -o $(BUNDLE_OUT)/prior.json
endif

LEGACY_OUT ?= dist/legacy

# Self-contained legacy bundle: train.py (bundled execute.py + configs + deps
# inlined) and infer.py.  Output works with legacy/submission/run.sh as-is.
bundle-legacy-lazy:
	rm -rf $(LEGACY_OUT)
	$(MAKE) bundle-submission
	$(CONDA_ACTIVATE) && \
	python tools/inline_deps.py \
		--execute-py $(BUNDLE_OUT)/execute.py \
		--infer-py scripts/submission/infer.py \
		--extra-txt $(BUNDLE_OUT)/extra.txt \
		$$(for f in $(BUNDLE_OUT)/config_*.yaml; do echo --config $$f; done) \
		--out-dir $(LEGACY_OUT)
	cp legacy/submission/run.sh $(LEGACY_OUT)/run.sh
ifdef PRIOR_LOG
	cp $(BUNDLE_OUT)/prior.json $(LEGACY_OUT)/prior.json
endif
	@echo "Legacy bundle ready: $(LEGACY_OUT)/"
	@ls -lh $(LEGACY_OUT)/

# Kill any mkdocs process on DOCS_PORT
docs-kill:
	@pids=$$(lsof -ti :$(DOCS_PORT) | tr '\n' ' '); \
	if [ -z "$$pids" ]; then \
		echo "Nothing listening on port $(DOCS_PORT)"; \
	elif ! ps -p $$(echo $$pids | tr ' ' ',') | grep -q mkdocs; then \
		echo "Port $(DOCS_PORT) is held by a non-mkdocs process (pids $$pids)" >&2; exit 1; \
	else \
		kill -9 $$pids; \
	fi

# Build static docs site
docs-build:
	$(CONDA_ACTIVATE) && \
	cd docs && mkdocs build

# Serve docs in foreground with live reload
docs-serve-fg: docs-kill
	$(CONDA_ACTIVATE) && \
	cd docs && mkdocs serve --open -a localhost:$(DOCS_PORT)

# Serve docs in background
docs-serve:
	nohup $(MAKE) docs-serve-fg > /dev/null 2>&1 &
	@echo "mkdocs serving at http://localhost:$(DOCS_PORT), browser will open once built"

# Install torchrec (CPU/MPS) with fbgemm_gpu stub
install-torchrec:
	$(CONDA_ACTIVATE) && \
	uv pip install torchrec==1.4.0 --no-deps && \
	uv pip install iopath torchmetrics==1.0.3 tensordict==0.12.4 tqdm && \
	uv pip install -e tools/fbgemm_gpu_stub && \
	python tools/fbgemm_gpu_stub/patch_torchrec.py

# Launch marimo notebook editor
marimo-notebooks:
	$(CONDA_ACTIVATE) && \
	marimo edit notebooks/ --host 127.0.0.1 --port $(NOTEBOOKS_PORT) --no-token
