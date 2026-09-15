PYTHON ?= python3
VERIFY_ARGS ?=

.PHONY: verify check test figures

verify:
	$(PYTHON) -B scripts/verify.py $(VERIFY_ARGS)

check:
	$(PYTHON) -B scripts/anon_scan.py
	$(PYTHON) -B scripts/check_integrity.py
	$(PYTHON) -B scripts/check_git_metadata.py

test:
	$(PYTHON) -B -m unittest discover -s tests -v

figures:
	$(PYTHON) -B -c "import sys; sys.path.insert(0, 'docs/paper2/figures'); import make_figures as m; m.render_accounting_scatter()"
	$(PYTHON) -B docs/paper2/figures/svg_to_tikz.py --width-pt 240 docs/paper2/figures/accounting-scatter.svg
