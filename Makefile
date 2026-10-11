# ArityGuard: code and results. `make help`.
.PHONY: help data verify smoke tables clean
PY := python3
AGENTS := "Qwen/Qwen2.5-7B-Instruct" "Qwen/Qwen2.5-14B-Instruct"

help:
	@echo "  make verify   22 exhaustive / Monte Carlo checks of the certificate (no model)"
	@echo "  make smoke    the whole pipeline end to end on a mock agent"
	@echo "  make tables   rebuild every table of the paper from results/ into tables/"
	@echo "  make data     fetch the public datasets and upstream repos (to re-run experiments)"

data:
	bash scripts/download/all.sh

verify:
	cd code && $(PY) -m tests.test_certificate

smoke:
	cd code && $(PY) -m arq.smoke && rm -rf results/MOCK_smoke

tables:
	mkdir -p tables
	cd code && $(PY) -m arq.report --results ../results --out ../tables \
	  --agents $(AGENTS) --e1-results ../results/relevance_filtered

clean:
	rm -rf tables code/results/MOCK_smoke
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
