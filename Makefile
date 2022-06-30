.PHONY: test-transe
test-transe:
	$(call run_in_venv,"cmd/test_transe.py")

.PHONY: test-unit
test-unit:
	$(call run_in_venv,-m unittest discover tests/unit/crabby)


# 1 - script path
define run_in_venv
	@echo "Running crabby unit tests..."
	@(export DATA_DIR="${PWD}/data" && . .venv/bin/activate && PYTHONPATH="${PYTHONPATH}:${PWD}" && python3 $(1))
endef
