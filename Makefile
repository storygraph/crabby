.PHONY: test-transe
test-transe:
	$(call run_in_venv,"cmd/test_transe.py")

.PHONY: test-relex
test-relex:
	$(call run_in_venv,"cmd/test_relex.py")

.PHONY: test-unit
test-unit:
	@echo "Running crabby unit tests..."
	$(call run_in_venv,-m unittest discover tests/unit/crabby)

.PHONY: setup
setup:
	$(call pip install -r requirements.txt)

.PHONY: fetch-lang-mdl
fetch-lang-mdl:
	@./scripts/fetch_lang_model.sh


# 1 - script path
define run_in_venv
	@(export DATA_DIR="${PWD}/data" && export FT_MDL="${PWD}/data/ft_cc.en.300_freqprune_100K_20K_pq_100.bin" && . .venv/bin/activate && PYTHONPATH="${PYTHONPATH}:${PWD}" && python3 $(1))
endef
