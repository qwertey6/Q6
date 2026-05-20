.PHONY: all corpus harness detector report clean reproduce

PYTHON ?= python3
CORPUS_DIR := corpus
RESULTS_DIR := results
REPORT_DIR := report/out

all: corpus harness report

# --- Milestone 1: corpus -----------------------------------------------------
corpus: $(CORPUS_DIR)/sources/.fetched $(CORPUS_DIR)/generated/.materialized $(CORPUS_DIR)/MANIFEST.csv

$(CORPUS_DIR)/sources/.fetched:
	@bash $(CORPUS_DIR)/fetch_sources.sh
	@touch $@

$(CORPUS_DIR)/generated/.materialized: $(CORPUS_DIR)/sources/.fetched
	@bash $(CORPUS_DIR)/build_trace_videos.sh
	@$(PYTHON) $(CORPUS_DIR)/build_extended_corpus.py
	@touch $@

$(CORPUS_DIR)/MANIFEST.csv: $(CORPUS_DIR)/sources/.fetched $(CORPUS_DIR)/generated/.materialized
	@$(PYTHON) $(CORPUS_DIR)/build_manifest.py

# --- Milestone 2: harness ----------------------------------------------------
harness: corpus
	@$(PYTHON) -m harness.runner --corpus $(CORPUS_DIR)/MANIFEST.csv --out $(RESULTS_DIR)
	@$(PYTHON) -m harness.scoring --results $(RESULTS_DIR) --manifest $(CORPUS_DIR)/MANIFEST.csv --out $(RESULTS_DIR)/scores

# --- Milestone 3: our detector (built as part of harness via adapter) --------
detector:
	@$(PYTHON) -m detector --self-test

# --- Milestone 4: report -----------------------------------------------------
report: harness
	@$(PYTHON) report/generate_report.py --scores $(RESULTS_DIR)/scores --manifest $(CORPUS_DIR)/MANIFEST.csv --out $(REPORT_DIR)

# --- Reproducibility ---------------------------------------------------------
reproduce:
	@docker build -t pse-bench .
	@docker run --rm -v $$PWD:/workspace pse-bench make all

clean:
	rm -rf $(RESULTS_DIR) $(REPORT_DIR) $(CORPUS_DIR)/generated
	rm -f $(CORPUS_DIR)/sources/.fetched $(CORPUS_DIR)/generated/.materialized
